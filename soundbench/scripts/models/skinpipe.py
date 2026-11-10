"""One reed/bore, a leaking chamber, a circular head and five reciprocal courses.

The reed scattering, bag plan, membrane bank and phase-compensated string
loops use the studio's wind and banjo physics. No pitch-triggered halo source
exists: courses receive only head velocity (or an explicitly scored pluck).
"""
from __future__ import annotations
import hashlib
import json
import math
import numpy as np
from numba import njit
from scipy.signal import butter, sosfilt, lfilter
from soundbench.scripts.models.mallorca_winds import ROOT, plan, frequency, bag_wander, _cc_lane, _read, reed_wave
from soundbench.scripts.models.banjo import MEMBRANE_KEYS, membrane_modes, modal_coefficients, _coefficients
from soundbench.scripts.paths import preset_path, read_preset

CONFIG = preset_path('skinpipe')
SCHEMA = {
    'head': MEMBRANE_KEYS | {'radius_m'},
    'strings': set('t60 loss_mix dispersion head_drive head_return damper_t60 pluck_gain pluck_position position_jitter pick_ms pluck_lowpass_hz'.split()),
}
KEYS = set('sample_rate seed output_gain lowpass_hz noise loss_pole pressure attack_s release_s cut_s tail_seconds pressure_pitch_cents vibrato_cents vibrato_hz wobble_cents squawk_cents sag_cents tuning_cents delay_scale bore_correction squawk_noise courses dry_gain skin_gain string_gain chamber_drive chamber_feedback chamber_highpass_hz head strings'.split())


def validate(c):
    if not isinstance(c, dict) or set(c) != KEYS:
        raise ValueError(f'skinpipe config keys must be {sorted(KEYS)}')
    for key, keys in SCHEMA.items():
        if not isinstance(c[key], dict) or set(c[key]) != keys:
            raise ValueError(f'{key} keys must be {sorted(keys)}')
    def finite(value):
        if isinstance(value, dict): return all(finite(v) for v in value.values())
        if isinstance(value, list): return all(finite(v) for v in value)
        return type(value) in (int, float) and math.isfinite(value)
    if not finite(c): raise ValueError('parameters must be finite numbers')
    if c['sample_rate'] != 48000 or type(c['seed']) is not int or c['seed'] < 0:
        raise ValueError('skinpipe requires 48 kHz and nonnegative integer seed')
    if not isinstance(c['courses'], list) or len(c['courses']) != 5 or any(type(p) is not int or not 36 <= p <= 84 for p in c['courses']) or len(set(c['courses'])) != 5:
        raise ValueError('courses needs five distinct integer MIDI pitches 36..84')
    if not isinstance(c['tuning_cents'], list) or len(c['tuning_cents']) != 12 or any(abs(v) > 100 for v in c['tuning_cents']):
        raise ValueError('tuning_cents needs twelve offsets within +/-100 cents')
    if not isinstance(c['bore_correction'], list) or len(c['bore_correction']) != 2: raise ValueError('bore_correction needs two values')
    for key in ('attack_s', 'release_s', 'cut_s'):
        if not 0 < c[key] <= 1: raise ValueError(f'invalid {key}')
    if not 1 <= c['tail_seconds'] <= 30 or not 0 < c['output_gain'] <= 2:
        raise ValueError('invalid tail or output_gain')
    if not (0 < c['lowpass_hz'] < 20000 and 0 <= c['loss_pole'] < .95 and .5 < c['delay_scale'] < 2
            and 0 < c['pressure'] < 2 and 0 <= c['noise'] <= .3 and 0 <= c['squawk_noise'] <= .1
            and 0 <= c['wobble_cents'] <= 20 and 0 < c['chamber_highpass_hz'] <= 200
            and 0 <= c['chamber_feedback'] <= 200 and 0 <= c['pressure_pitch_cents'] <= 180):
        raise ValueError('invalid wind or chamber parameters')
    if any(c[k] < 0 for k in ('dry_gain','skin_gain','string_gain','chamber_drive','vibrato_cents','vibrato_hz','squawk_cents','sag_cents')):
        raise ValueError('gains and gesture depths must be nonnegative')
    h, s = c['head'], c['strings']
    if not (0 < h['radius_m'] < 1 and 0 < h['bridge_radius'] < 1 and 0 < h['fundamental_hz'] < h['max_hz'] <= 8000
            and h['t60_fundamental'] > 0 and h['damping_exponent'] >= 0):
        raise ValueError('invalid membrane head')
    for key in ('t60','loss_mix','dispersion'):
        if not isinstance(s[key], list) or len(s[key]) != 5: raise ValueError(f'strings.{key} needs five values')
    if any(x <= 0 for x in s['t60']) or any(not 0 <= x <= .5 for x in s['loss_mix']) or any(not -.5 <= x <= 0 for x in s['dispersion']):
        raise ValueError('invalid string losses/dispersion')
    if not (0 <= s['head_drive'] <= .2 and 0 <= s['head_return'] <= 2 and 0 < s['damper_t60'] <= .3
            and 0 < s['pluck_position'] < .5 and 0 <= s['position_jitter'] < 1
            and 0 < s['pick_ms'] <= 5 and 0 < s['pluck_lowpass_hz'] < 24000 and s['pluck_gain'] >= 0):
        raise ValueError('invalid bridge, damper or pluck parameters')
    return c


def load_config(path=CONFIG):
    return validate(read_preset(path))


def stream_seed(seed, track, stream, index=0, onset=0):
    """Stable identity, independent of Python hash randomization and track order."""
    identity = json.dumps([seed, str(track), stream, int(index), float(onset)], separators=(',', ':'))
    return int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], 'little')


def bag_lane(rows, n, c, track):
    """Per-note squeeze trajectories joined without a pressure discontinuity."""
    wander = np.zeros(n)
    sr = c['sample_rate']
    rows = sorted(rows, key=lambda r: r['time'])
    for k, r in enumerate(rows):
        a = round(r['time']*sr)
        b = round(rows[k+1]['time']*sr) if k+1 < len(rows) else n
        seed = stream_seed(c['seed'],track,'bag',k,r['time'])
        # Read beyond the zero-valued first knot, then join to the player's hand.
        trajectory = bag_wander(b-a+2*sr,sr,seed)[2*sr:]
        blend = min(b-a,round(.04*sr))
        u = np.linspace(0,1,blend);u = u*u*(3-2*u)
        trajectory[:blend] = (wander[a-1] if a else 0)*(1-u)+trajectory[:blend]*u
        wander[a:b] = trajectory
    return wander


def stochastic_lanes(rows, n, c, track):
    """Independent reed and exposed-air streams for each fingering onset."""
    noise, air = np.zeros(n), np.zeros(n)
    rows = sorted(rows, key=lambda r: r['time'])
    for k, r in enumerate(rows):
        a = round(r['time']*c['sample_rate'])
        b = round(rows[k+1]['time']*c['sample_rate']) if k+1 < len(rows) else n
        noise[a:b] = np.random.default_rng(stream_seed(c['seed'],track,'reed',k,r['time'])).uniform(-1,1,b-a)
        air[a:b] = np.random.default_rng(stream_seed(c['seed'],track,'air',k,r['time'])).standard_normal(b-a)
    return noise, air


@njit(cache=True)
def _engine(f, natural_f, pressure, loud, noise, lean_lane, damper_lane, events, pulses,
            sr, pole, delay_scale, chamber_drive, chamber_feedback, hp_hz,
            modal_f, modal_times, modal_weights, opens, times, mix, dispersion,
            head_drive, head_return, damper_t60, diagnostics):
    n = len(f)
    dry, skin, strings = np.zeros(n), np.zeros(n), np.zeros(n)
    courses = np.zeros((5, n if diagnostics else 0))
    bore = np.zeros(4096)
    size = int(sr/(440*2**((min(opens)-69)/12)))+128
    buf, injection = np.zeros((5,size)), np.zeros((5,size))
    ap_y, disp_x, disp_y, loss_prev = np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5)
    ints, aps, gains, damp_gains = np.zeros(5,np.int64), np.zeros(5), np.zeros(5), np.zeros(5)
    for s in range(5):
        ints[s], aps[s], gains[s] = _coefficients(opens[s],sr,mix[s],dispersion[s],times[s])
        damp_gains[s] = math.exp(-math.log(1000)/(damper_t60*(440*2**((opens[s]-69)/12))))
    y1, y2 = np.zeros(len(modal_f)), np.zeros(len(modal_f))
    reference1, reference2 = y1.copy(), y2.copy()
    mc, mr, mi = y1.copy(), y1.copy(), y1.copy()
    lp = dc = prev = breath = head_previous = hp = hp_prev = lean = damp = 0.0
    feedback_velocity = 0.0
    bore_power = 0.0
    port_x1 = port_x2 = port_y1 = port_y2 = 0.0
    port_b = port_a1 = port_a2 = 0.0
    hp_pole = math.exp(-2*math.pi*hp_hz/sr)
    lean_alpha, damp_alpha = 1-math.exp(-1/(.04*sr)), 1-math.exp(-1/(.03*sr))
    ev = 0
    waves = np.zeros(5)
    for i in range(n):
        pos, bp = i % size, i % 4096
        while ev < len(events) and events[ev,0] <= i:
            s = int(events[ev,1])
            # A new thumb replaces its course's stored wave, as on the banjo.
            buf[s,:] = 0
            injection[s,:] = 0
            ap_y[s] = disp_x[s] = disp_y[s] = loss_prev[s] = 0
            comb = max(1,int(events[ev,2]*sr/(440*2**((opens[s]-69)/12))))
            for j in range(pulses.shape[1]):
                injection[s,(pos+j)%size] += pulses[ev,j]
                injection[s,(pos+j+comb)%size] -= pulses[ev,j]
            ev += 1
        lean += lean_alpha*(lean_lane[i]-lean)
        damp += damp_alpha*(damper_lane[i]-damp)
        # Control-rate coefficient updates retain every modal history.
        if i % 64 == 0:
            port_w = 2*math.pi*natural_f[i]/sr
            port_alpha = math.sin(port_w)/16.0
            port_b = port_alpha/(1+port_alpha)
            port_a1 = -2*math.cos(port_w)/(1+port_alpha)
            port_a2 = (1-port_alpha)/(1+port_alpha)
            for m in range(len(modal_f)):
                theta = 2*math.pi*modal_f[m]*(1+.4*lean)/sr
                t60 = modal_times[m]*(1-.75*lean)
                radius = math.exp(-math.log(1000)/(sr*t60))
                mc[m], mr[m] = 2*radius*math.cos(theta), radius*radius
                mi[m] = .5*(t60/.12)*(1-radius)*math.sqrt(1+radius*radius-2*radius*math.cos(2*theta))
        breath += .002*(pressure[i]-breath)
        w = 2*math.pi*f[i]/sr
        phase = math.atan2(pole*math.sin(w),1-pole*math.cos(w))/w
        delay = max(3.,sr/f[i]*delay_scale-phase-1)
        returned = _read(bore,bp,delay)
        lp = (1-pole)*returned+pole*lp
        bore_power += .002*(returned*returned-bore_power)
        port_limit = .15*math.sqrt(bore_power+1e-12)
        reaction = port_limit*math.tanh(chamber_feedback*port_y1/port_limit)
        incoming = .97*lp + reaction*min(1.0, breath)
        wave = reed_wave(.65*breath*(1+noise[i]), incoming)
        dc = wave-prev+.995*dc
        prev = wave
        bore[bp] = dc
        dry[i] = returned*.3*loud[i]
        hp = returned-hp_prev+hp_pole*hp
        hp_prev = returned
        force = chamber_drive*hp
        chamber_force = force
        common = 0.0
        for s in range(5):
            read = (pos-ints[s])%size
            x = buf[s,read]
            y = aps[s]*x+buf[s,(read-1)%size]-aps[s]*ap_y[s]
            ap_y[s] = y
            d = dispersion[s]*y+disp_x[s]-dispersion[s]*disp_y[s]
            disp_x[s], disp_y[s] = y,d
            gain = gains[s]*(1-damp)+damp_gains[s]*damp
            waves[s] = ((1-mix[s])*d+mix[s]*loss_prev[s])*gain
            loss_prev[s] = d
            common += waves[s]
        for s in range(5):
            pluck = injection[s,pos]
            injection[s,pos] = 0
            buf[s,pos] = waves[s] + head_drive*head_previous + pluck
            force += head_return*(waves[s]+pluck)
            if diagnostics: courses[s,i] = waves[s]
        head = 0.0
        reference_head = 0.0
        for m in range(len(modal_f)):
            value = mc[m]*y1[m]-mr[m]*y2[m]+mi[m]*force
            if abs(value) < 1e-20: value = 0
            y2[m],y1[m] = y1[m],value
            head += modal_weights[m]*value
            reference = mc[m]*reference1[m]-mr[m]*reference2[m]+mi[m]*chamber_force
            if abs(reference) < 1e-20: reference = 0.0
            reference2[m], reference1[m] = reference1[m], reference
            reference_head += modal_weights[m]*reference
        # The calibrated bare closed-head bore is the reference impedance.
        # Only the bridge-loaded perturbation is added to its returning wave.
        feedback_velocity = head-reference_head
        # The reed's fundamental mechanical admittance rejects octave capture.
        port = port_b*(feedback_velocity-port_x2)-port_a1*port_y1-port_a2*port_y2
        port_x2, port_x1 = port_x1, feedback_velocity
        port_y2, port_y1 = port_y1, port
        head_previous = head
        skin[i], strings[i] = head,common
    return dry,skin,strings,courses


def render(rows, seconds, config=None, controls=(), plucks=(), *, track_id='skinpipe', details=False):
    c = validate(config if config is not None else load_config())
    if not math.isfinite(seconds) or seconds <= 0: raise ValueError('seconds must be finite and positive')
    sr, count = c['sample_rate'], round(seconds*c['sample_rate'])
    wc = dict(c, drone_pitch=50, wobble_cents=0)
    f,p,loud,_,_,transient = plan(rows,seconds,wc,'xeremia',controls,details=True,compass=(57,71),cut_s=c['cut_s'])
    wander = bag_lane(rows,count,c,track_id)
    pressure_ratio = _cc_lane(controls,2,count,sr,96)/96
    f *= 2**(c['wobble_cents']*c['pressure_pitch_cents']/180*pressure_ratio*wander/1200)
    p *= 1+c['wobble_cents']/180* wander
    natural_f = f.copy()
    a,b = c['bore_correction']
    f /= 2**((a+b*f)/1200)
    noise,air = stochastic_lanes(rows,count,c,track_id)
    noise *= c['noise']+6*c['squawk_noise']*transient
    lean = _cc_lane(controls,74,count,sr,0)/127
    damper = (_cc_lane(controls,64,count,sr,127)<64).astype(float)
    ratios,weights,decay = membrane_modes(c['head'],radius_m=c['head']['radius_m'])
    bc = {'engine': {'sample_rate':sr}, 'head': {'fundamental_hz':c['head']['fundamental_hz'],
          'ratios':ratios,'weights':weights,'t60_seconds':decay},
          'pot': {'frequency_hz':100.,'t60_seconds':.1,'weight':0.}}
    _,_,_,mw = modal_coefficients(bc)
    # No pot mode: the only body is the chamber and membrane.
    modal_f = c['head']['fundamental_hz']*np.array(ratios)
    s = c['strings']
    opens = np.array([q+c['tuning_cents'][q%12]/100 for q in c['courses']])
    events, pulses = [], []
    pulse_size = math.ceil(sr*s['pick_ms']/1000)+64
    for k,r in enumerate(sorted(plucks,key=lambda r:r['time'])):
        if r['pitch'] not in c['courses']:
            raise ValueError(f'course pluck {pitch_name(r["pitch"])} (MIDI {r["pitch"]}) is not a course')
        if not (0 <= r['time'] < r['off'] <= seconds and 1 <= r['velocity'] <= 127):
            raise ValueError(f'invalid pluck: {r}')
        string = c['courses'].index(r['pitch'])
        rng = np.random.default_rng(stream_seed(c['seed'],track_id,'pluck',k,r['time']))
        events.append([round(r['time']*sr),string,s['pluck_position']*rng.uniform(1-s['position_jitter'],1+s['position_jitter'])])
        length = max(2,round(sr*s['pick_ms']/1000))
        pulse = np.sin(np.pi*(np.arange(length)+.5)/length)
        force = np.zeros(pulse_size)
        force[:length] = pulse/pulse.sum()*(r['velocity']/127)**1.3*s['pluck_gain']
        alpha = 1-math.exp(-2*math.pi*s['pluck_lowpass_hz']*max(r['velocity']/127,.01)**.5/sr)
        pulses.append(lfilter([alpha],[1,-(1-alpha)],force))
    dry,skin,strings,courses = _engine(f,natural_f,p*c['pressure'],loud,noise,lean,damper,
        np.asarray(events).reshape(-1,3),np.asarray(pulses).reshape(-1,pulse_size),
        sr,c['loss_pole'],c['delay_scale'],c['chamber_drive'],c['chamber_feedback'],c['chamber_highpass_hz'],
        modal_f,np.array(decay),mw[:-1],opens,np.array(s['t60']),np.array(s['loss_mix']),np.array(s['dispersion']),
        s['head_drive'],s['head_return'],s['damper_t60'],details)
    dry = sosfilt(butter(3,c['lowpass_hz'],fs=sr,output='sos'),dry)
    air = sosfilt(butter(2,[1800,14000],btype='bandpass',fs=sr,output='sos'),air)/np.sqrt(.52)
    dry += air*p*c['squawk_noise']*transient
    paths = dict(dry=dry*c['dry_gain']*c['output_gain'],skin=skin*c['skin_gain']*c['output_gain'],
                 strings=strings*c['string_gain']*c['output_gain'])
    audio = paths['dry']+paths['skin']+paths['strings']
    if not np.isfinite(audio).all() or np.max(abs(audio),initial=0) >= .9:
        raise ValueError('unstable or excessive peak skinpipe render')
    return (audio,dict(paths,courses=courses)) if details else audio


def pitch_name(p):
    return f'{["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"][int(p)%12]}{int(p)//12-1}'
