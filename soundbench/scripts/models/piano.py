"""Sample-free piano: passive bridge-coupled unisons and fitted body radiation.

A normalized complex bridge admittance couples stiff-string rotating-wave
states reciprocally. Exact eigenmodes carry prompt sound and aftersound.
Filtered stochastic residual radiation and a fitted low body mode reproduce
non-partial energy and key-bottom knock; no sampled waveforms are loaded.
"""
from functools import lru_cache
import json
import math
from pathlib import Path

import numpy as np
from numba import njit
from scipy.signal import butter, sosfilt, sosfreqz
from .piano_bridge import coupled_poles, weight_initial_excitation
from soundbench.scripts.paths import data_path, preset_path, read_preset

CONFIG = preset_path('piano')


def load_config(path=CONFIG):
    cfg = read_preset(path)
    e = cfg['engine']
    # Model data: an absolute path, or a file in the piano's data folder.
    data = {k: Path(v) if Path(v).is_absolute() else data_path('piano', v) for k, v in e.items() if k.endswith('_data')}
    cfg['data'] = read_data(str(data['modal_data']))
    cfg['release_data'] = json.loads(data['release_data'].read_text())['profiles']
    for name in ('coupled', 'body'):
        if name+'_data' in e:
            cfg[name] = read_auxiliary(str(data[name+'_data']))
    if e['sample_rate'] != 48000 or not 0 < e['output_gain'] <= 4:
        raise ValueError('piano requires 48 kHz and a positive bounded gain')
    if not 0 < cfg['damper']['release_s'] <= 2 or not 0 < e['max_note_seconds'] <= 60:
        raise ValueError('invalid piano release or maximum note lifetime')
    for key, low, high in [('gain', 0, 2), ('noise_gain', 0, 2), ('decay_scale', .25, 6)]:
        value = cfg.get('body_settings', {}).get(key, 1)
        if not np.isfinite(value) or not low <= value <= high:
            raise ValueError('invalid body adjustment')
    validate_adjustments(cfg.get('adjustments', {}))
    return cfg


def validate_adjustments(a):
    """Bound optional recording-fit adjustments without altering source data."""
    bounds = {'tuning_cents': (-50, 50), 'inharmonicity_scale': (.25, 4),
              'detune_scale': (.25, 4), 'brightness_db_octave': (-6, 6),
              'velocity_brightness': (-3, 3), 'attack_gain': (0, 2)}
    allowed = set(bounds) | {'pitch_cents', 'pitch_detune_cents', 'pitch_string_weights', 'frequency_anchor', 'partial_gain_db', 'prompt_rate_scale', 'late_rate_scale'}
    if set(a) - allowed:
        raise ValueError('unknown piano adjustment')
    for key, (low, high) in bounds.items():
        if key in a and (not np.isfinite(a[key]) or not low <= a[key] <= high):
            raise ValueError(f'invalid piano adjustment: {key}')
    for key in ('prompt_rate_scale', 'late_rate_scale'):
        values = np.asarray(a.get(key, [1, 1, 1]), float)
        if values.shape != (3,) or not np.all(np.isfinite(values)) or np.any((values < .25) | (values > 6)):
            raise ValueError(f'invalid piano adjustment: {key}')
    if a.get('frequency_anchor', 'slow') not in ('slow', 'dominant'):
        raise ValueError('invalid coupled frequency anchor')
    gains = np.asarray(a.get('partial_gain_db', [0]), float)
    if gains.ndim != 1 or not 1 <= len(gains) <= 96 or not np.all(np.isfinite(gains)) or np.any(abs(gains)>24):
        raise ValueError('invalid partial excitation gains')
    for pitch, weights in a.get('pitch_string_weights', {}).items():
        count = 1 if int(pitch)<31 else 2 if int(pitch)<43 else 3
        weights = np.asarray(weights, float)
        if not 21 <= int(pitch) <= 108 or weights.shape != (count,) or not np.all(np.isfinite(weights)) or np.any((weights<0) | (weights>4)) or not np.any(weights>0):
            raise ValueError('invalid per-string excitation weights')
    for pitch, cents in a.get('pitch_cents', {}).items():
        if not 21 <= int(pitch) <= 108 or not np.isfinite(cents) or abs(cents) > 50:
            raise ValueError('invalid per-pitch tuning')
    for pitch, cents in a.get('pitch_detune_cents', {}).items():
        if not 21 <= int(pitch) <= 108 or not np.isfinite(cents) or not .05 <= cents <= 12:
            raise ValueError('invalid per-pitch unison spacing')


def rate_adjustment(frequencies, config, kind):
    values = config.get('adjustments', {}).get(kind + '_rate_scale', [1, 1, 1])
    return np.exp(np.interp(np.log(np.maximum(frequencies, 1)), np.log([250, 1500, 6000]), np.log(values)))


def brightness_adjustment(frequencies, velocity, config):
    a = config.get('adjustments', {})
    tilt = a.get('brightness_db_octave', 0) + a.get('velocity_brightness', 0) * (velocity - 64) / 63
    gains = np.asarray(a.get('partial_gain_db', []), float)
    ordinal = np.zeros(len(frequencies)); ordinal[:min(len(gains), len(ordinal))] = gains[:len(ordinal)]
    return 10 ** ((np.clip(np.log2(np.maximum(frequencies, 50) / 1500), -5, 4) * tilt + ordinal) / 20)


@lru_cache(maxsize=8)
def read_data(path):
    d = json.loads(Path(path).read_text())
    if d['schema'] != 1:
        raise ValueError('unknown piano modal schema')
    for p in d['profiles']:
        for k in ('slow_amp', 'fast_amp', 'attack_amp', 'slow_rate', 'fast_rate', 'stereo'):
            p[k] = np.asarray(p[k], dtype=float)
            if not np.all(np.isfinite(p[k])) or np.any(p[k] < 0):
                raise ValueError('invalid modal coefficients')
    return d


@lru_cache(maxsize=8)
def read_auxiliary(path):
    data=json.loads(Path(path).read_text())
    if data['schema']!=1:
        raise ValueError('invalid piano auxiliary schema')
    return data


def interpolate(data, key, pitch, velocity, logarithmic=True):
    profiles=data['profiles']; pitches=np.array([p['pitch'] for p in profiles])
    hi=int(np.clip(np.searchsorted(pitches,pitch),1,len(pitches)-1)); lo=hi-1
    u=float(np.clip((pitch-pitches[lo])/(pitches[hi]-pitches[lo]),0,1))
    vs=np.array(data['velocities']); vh=int(np.clip(np.searchsorted(vs,velocity),1,len(vs)-1)); vl=vh-1
    v=float(np.clip((velocity-vs[vl])/(vs[vh]-vs[vl]),0,1))
    values=[]
    for p in (profiles[lo],profiles[hi]):
        a=np.asarray(p[key],float)
        values.append(np.exp((1-v)*np.log(a[vl]+1e-12)+v*np.log(a[vh]+1e-12)) if logarithmic else (1-v)*a[vl]+v*a[vh])
    return np.exp((1-u)*np.log(values[0]+1e-12)+u*np.log(values[1]+1e-12)) if logarithmic else (1-u)*values[0]+u*values[1]


def coefficients(pitch, velocity, config, with_attack=False):
    if not 21 <= pitch <= 108 or not 1 <= velocity <= 127:
        raise ValueError('piano range is MIDI 21–108; note-on velocity is 1–127')
    d = config['data']; profiles = d['profiles']
    pitches = np.array([p['pitch'] for p in profiles])
    right = int(np.clip(np.searchsorted(pitches, pitch), 1, len(pitches)-1))
    left = right-1
    weight = np.clip((pitch-pitches[left])/(pitches[right]-pitches[left]), 0, 1)
    vs = np.array(d['velocities'])
    hi = int(np.clip(np.searchsorted(vs, velocity), 1, len(vs)-1)); lo = hi-1
    blend = np.clip((velocity-vs[lo])/(vs[hi]-vs[lo]), 0, 1)
    def interpolate(key):
        # Log interpolation keeps rates/gains positive and velocity continuous.
        values = []
        for p in (profiles[left], profiles[right]):
            values.append(np.exp((1-blend)*np.log(p[key][lo]+1e-12)+blend*np.log(p[key][hi]+1e-12)))
        return np.exp((1-weight)*np.log(values[0]+1e-12)+weight*np.log(values[1]+1e-12))
    b = math.exp((1-weight)*math.log(profiles[left]['B']+1e-8)+weight*math.log(profiles[right]['B']+1e-8))
    # The library plays equal temperament at the first partial. The samples'
    # stretch/tuning is measured, but is not silently imposed on scored notes.
    adjustment = config.get('adjustments', {})
    b *= adjustment.get('inharmonicity_scale', 1)
    cents = adjustment.get('tuning_cents', 0) + adjustment.get('pitch_cents', {}).get(str(pitch), 0)
    f0 = 440*2**((pitch-69)/12 + cents/1200)
    k = np.arange(1, d['modes']+1)
    frequencies = f0*k*np.sqrt((1+b*k*k)/(1+b))
    amp_s, amp_f = interpolate('slow_amp'), interpolate('fast_amp')
    # SFZ uses amp_veltrack=73. Apply that continuous MIDI gain to fitted raw
    # sample gains, without the source map's hard layer transitions.
    gain = (velocity/127)**1.46
    if velocity < vs[0]:
        gain *= velocity/vs[0]
    shape = brightness_adjustment(frequencies, velocity, config)
    result = (frequencies, amp_s*gain*shape, amp_f*gain*shape,
              interpolate('slow_rate')*rate_adjustment(frequencies, config, 'late'),
              interpolate('fast_rate')*rate_adjustment(frequencies, config, 'prompt'), interpolate('stereo'), b)
    return (*result, interpolate('attack_amp')*gain*shape*adjustment.get('attack_gain', 1)) if with_attack else result


@njit(cache=True)
def _hammer(velocity, sr, mass, stiffness, exponent, hysteresis):
    """Dissipative power-law contact; force is returned with unit impulse."""
    dt = 1/(sr*4)
    v0 = .15 + 3.5*(velocity/127)**1.5
    x, v = 0.0, v0
    force = np.zeros(2048)
    count = 0
    for i in range(len(force)*4):
        elastic = stiffness*max(x, 0.0)**exponent
        f = elastic*max(0.0, 1+hysteresis*v/v0)
        v -= f/mass*dt
        x += v*dt
        force[i//4] += f/4
        count = i//4+1
        if x <= 0 and i > 4:
            break
    result = force[:count]
    total = result.sum()
    if total > 0:
        result /= total
    return result


def hammer_force(velocity, config):
    h = config['hammer']
    return _hammer(velocity, config['engine']['sample_rate'], h['mass_kg'], h['stiffness'], h['exponent'], h['hysteresis'])


def controller_curve(events, seconds, control, default=0):
    """1-ms controller curve; events are (seconds, CC number, value)."""
    n = int(np.ceil(seconds*1000))+2
    result = np.full(n, default/127, dtype=float)
    for time, cc, value in sorted(events, key=lambda e: e[0]):
        if cc == control:
            if not math.isfinite(time) or time < 0 or not 0 <= value <= 127:
                raise ValueError('invalid piano controller event')
            result[min(round(time*1000), n):] = value/127
    return result


@njit(cache=True)
def _add_modes(out, start, force, freq, slow_amp, fast_amp, slow_rate, fast_rate,
               stereo, damp, sr, detune, count_strings, commuted):
    length = min(len(damp)*48, len(out)-start)
    attack_envelope = np.cumsum(force)
    for k in range(len(freq)):
        if freq[k] >= sr*.45 or slow_amp[k]+fast_amp[k] < 1e-7:
            continue
        # The fitted radiated gain compensates the effective-load hammer's
        # transfer, keeping the pulse causal without double-filtering the fit.
        hr, hi = 0.0, 0.0
        omega = 2*np.pi*freq[k]/sr
        for j in range(len(force)):
            hr += force[j]*math.cos(omega*j)
            hi -= force[j]*math.sin(omega*j)
        compensation = 1.0 if commuted else 1/max(math.sqrt(hr*hr+hi*hi), .025)
        sign = 1.0 if math.sin((k+1)*np.pi*.117) >= 0 else -1.0
        # Prompt motion is the strongly radiating eigenmode. Aftersound is
        # split across the unison's weak eigenmodes; their frequencies beat.
        for s in range(count_strings):
            center = count_strings//2
            cents = (s-center)*detune
            weight = 1.0 if count_strings == 1 else (.8 if s == center else .2/(count_strings-1))
            w = omega*2**(cents/1200)
            cs, sn = math.cos(w), math.sin(w)
            rs = math.exp(-slow_rate[k]/sr)
            rf = math.exp(-fast_rate[k]/sr)
            xs, ys, xf, yf = 0.0, 0.0, 0.0, 0.0
            a_s = slow_amp[k]*weight*compensation*sign
            a_f = fast_amp[k]*compensation*sign if s == center else 0.0
            for i in range(length):
                drive = (1.0 if i == 0 else 0.0) if commuted else (force[i] if i < len(force) else 0.0)
                xs, ys = rs*(cs*xs-sn*ys)+a_s*drive, rs*(sn*xs+cs*ys)
                xf, yf = rf*(cs*xf-sn*yf)+a_f*drive, rf*(sn*xf+cs*yf)
                block = i//48
                amount = damp[block]
                onset = attack_envelope[i] if commuted and i < len(force) else 1.0
                value = (ys+yf)*amount*onset
                out[start+i, 0] += value*stereo[k, 0]
                out[start+i, 1] += value*stereo[k, 1]
                if i > len(force) and i % 480 == 0 and (abs(xs)+abs(ys)+abs(xf)+abs(yf))*amount < 1e-7:
                    break


def coupled_coefficients(pitch,velocity,freq,config):
    data=config['coupled']
    profiles=data['profiles']; ps=np.array([p['pitch'] for p in profiles])
    adjustment = config.get('adjustments', {})
    cents = adjustment.get('pitch_detune_cents', {}).get(str(pitch), float(np.interp(pitch,ps,[p['mistuning_cents'] for p in profiles])) * adjustment.get('detune_scale', 1))
    count=1 if pitch<31 else 2 if pitch<43 else 3
    slow=interpolate(data,'late_rate',pitch,velocity)*rate_adjustment(freq,config,'late')
    prompt=interpolate(data,'prompt_rate',pitch,velocity)*rate_adjustment(freq,config,'prompt')
    a_s=interpolate(data,'slow_amp',pitch,velocity)*brightness_adjustment(freq,velocity,config)
    a_f=interpolate(data,'prompt_amp',pitch,velocity)*brightness_adjustment(freq,velocity,config)
    gain=(velocity/127)**1.46
    if velocity<data['velocities'][0]: gain*=velocity/data['velocities'][0]
    poles=np.zeros((len(freq),3),complex); residues=np.zeros_like(poles)
    for k,hz in enumerate(freq):
        if hz>10500: continue
        vals,info=coupled_poles(hz,count,cents,slow[k],prompt[k])
        n=len(vals)
        sign=1 if math.sin((k+1)*np.pi*.117)>=0 else -1
        excitation = np.zeros(n, complex)
        excitation[info['slow']] = a_s[k]*gain*sign
        excitation[info['fast']] += a_f[k]*gain*sign
        weights = adjustment.get('pitch_string_weights', {}).get(str(pitch))
        if weights is not None and count>1:
            excitation = weight_initial_excitation(info, excitation, weights)
        anchor = int(np.argmax(abs(excitation))) if adjustment.get('frequency_anchor', 'slow') == 'dominant' else info['slow']
        center_shift = vals[anchor].imag
        poles[k,:n]=np.exp((vals+1j*(2*np.pi*hz-center_shift))/48000)
        residues[k,:n] = excitation
    return poles,residues


@njit(cache=True)
def _add_coupled(out,start,force,poles,residues,stereo,damp,sr):
    """Exact eigensolution of the passive reciprocal bridge string system."""
    length=min(len(damp)*48,len(out)-start)
    onset=np.cumsum(force)
    for k in range(len(poles)):
        for j in range(3):
            if abs(residues[k,j])<1e-7: continue
            z=residues[k,j]
            for i in range(length):
                z*=poles[k,j]
                a=damp[i//48]*(onset[i] if i<len(onset) else 1)
                value=z.imag*a
                out[start+i,0]+=value*stereo[k,0]
                out[start+i,1]+=value*stereo[k,1]
                if i%480==0 and i>len(force) and abs(z)*a<1e-7: break


@lru_cache(maxsize=8)
def body_filters(edges,sr):
    result=[]
    for low,high in zip(edges[:-1],edges[1:]):
        sos=butter(2,(low,high),fs=sr,btype='bandpass',output='sos')
        _,h=sosfreqz(sos,worN=65536)
        result.append((sos,1/max(np.sqrt(np.mean(abs(h)**2)),1e-12)))
    return result


def body_audio(pitch,velocity,seconds,config,start,frequencies):
    data=config['body']; sr=config['engine']['sample_rate']
    rms=interpolate(data,'rms',pitch,velocity)
    settings = config['body_settings']
    noise_gain = settings.get('noise_gain', 1)
    decay_scale = settings.get('decay_scale', 1)
    n=round(seconds*sr)
    if n<=0: return np.zeros((0,2))
    warm=sr//2
    rng=np.random.default_rng(config['engine']['seed']+start*131+pitch*127)
    t=np.arange(n)/sr
    result=np.zeros(n)
    # Independent diffuse-radiation bands have fitted power, not fitted phase.
    for band,(sos,gain) in enumerate(body_filters(tuple(data['band_edges']),sr)):
        noise=rng.standard_normal(n+warm)
        filtered=sosfilt(sos,noise)[warm:]*gain
        envelope=np.interp(t*decay_scale,data['times'],rms[band])*noise_gain
        result+=filtered*envelope
    # Preserve the residual's exclusion of narrow sinusoidal carrier bands.
    # Without these notches, broad noise could bias apparent fundamental pitch.
    pad=sr
    fft_n=n+2*pad
    spectrum=np.fft.rfft(np.pad(result,(pad,pad)))
    for hz in frequencies:
        if hz>10500: break
        width=max(3,hz*.0025)
        lo=max(0,int(np.ceil((hz-2*width)*fft_n/sr)))
        hi=min(len(spectrum),int(np.floor((hz+2*width)*fft_n/sr))+1)
        distance=abs(np.arange(lo,hi)*sr/fft_n-hz)
        weight=np.clip((2*width-distance)/width,0,1)
        spectrum[lo:hi]*=.5+.5*np.cos(np.pi*weight)
    result=np.fft.irfft(spectrum,n=fft_n)[pad:pad+n]
    # Carrier exclusions must not radiate after their fitted diffuse envelope
    # has vanished. Suppress projection tails only below -60 dB of peak RMS.
    power_envelope=np.interp(t*decay_scale,data['times'],np.sqrt(np.sum(rms*rms,axis=0)))*noise_gain
    result*=np.clip(power_envelope/max(power_envelope.max()*1e-3,1e-15),0,1)
    hz,tau,sine,cosine=interpolate(data,'knock',pitch,velocity,logarithmic=False)
    result+=(sine*np.sin(2*np.pi*hz*t)+cosine*np.cos(2*np.pi*hz*t))*np.exp(-t/tau)
    result*=np.minimum(t/.003,1)
    fade=min(n,4800); result[-fade:]*=np.linspace(1,0,fade)
    gain=(velocity/127)**1.46*config['body_settings']['gain']
    if velocity<data['velocities'][0]: gain*=velocity/data['velocities'][0]
    # Radiation is centred; the measured stereo partial gains carry the
    # pitched AB image. This avoids claiming fitted body spatial phase.
    return np.repeat((result*gain)[:,None],2,axis=1)


@njit(cache=True)
def _sympathetic(audio, pedal, gain, sr):
    """Small causal chromatic resonator bank driven by bridge-output proxy."""
    x = (audio[:, 0]+audio[:, 1])*.5
    for pitch in range(36, 97):
        hz = 440*2**((pitch-69)/12)
        w = 2*np.pi*hz/sr
        cs, sn = math.cos(w), math.sin(w)
        real, imag = 0.0, 0.0
        for i in range(len(audio)):
            p = pedal[min(i//48, len(pedal)-1)]
            rate = .7 + 35*(1-p)**2
            r = math.exp(-rate/sr)
            real, imag = r*(cs*real-sn*imag)+x[i]*p/sr, r*(sn*real+cs*imag)
            value = imag*gain*8
            audio[i, 0] += value
            audio[i, 1] += value


def render(notes, seconds, config=None, controls=()):
    """Render note dictionaries (time, off, pitch, velocity) and plain MIDI CCs.

    Note releases are physical key releases, not MIDI-pedal-extended endpoints.
    All channels passed here form one instrument. Controller conflicts should
    be resolved by the caller, rather than hidden in a note scheduler.
    """
    if config is None:
        config = load_config()
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('positive finite render duration required')
    sr = config['engine']['sample_rate']
    out = np.zeros((round(seconds*sr), 2))
    pedal = controller_curve(controls, seconds, 64)
    soft = controller_curve(controls, seconds, 67)
    for row in sorted(notes, key=lambda n: (n['time'], n['pitch'])):
        t, off = float(row['time']), float(row['off'])
        if not all(math.isfinite(v) for v in (t, off)) or t < 0 or off < t or off > seconds:
            raise ValueError('invalid piano note times')
        pitch, velocity = int(row['pitch']), int(row['velocity'])
        freq, a_s, a_f, r_s, r_f, stereo, _, attack = coefficients(pitch, velocity, config, with_attack=True)
        force = hammer_force(velocity, config)
        start = round(t*sr)
        if start >= len(out):
            continue
        length = min(seconds-t, config['engine']['max_note_seconds'])
        n = max(1, round(length*1000))
        times = t+np.arange(n)/1000
        p = pedal[np.minimum((times*1000).astype(int), len(pedal)-1)]
        rates = np.where(times >= off, (1-p)**config['damper']['half_pedal_exponent']*6.9078/config['damper']['release_s'], 0)
        if pitch >= config['damper']['undamped_from']:
            rates[:] = 0
        damp = np.exp(-np.cumsum(rates)/1000)
        quiet = np.flatnonzero(damp < 1e-6)
        if len(quiet):
            damp = damp[:min(len(damp), quiet[0]+100)]
            n = len(damp)
        # End-of-render and bounded voice retirement have a click-free fade.
        fade = min(n, 100)
        damp[-fade:] *= np.linspace(1, 0, fade)
        touch = soft[min(round(t*1000), len(soft)-1)]
        shape = (1-touch+touch*config['soft_pedal']['gain'])*np.exp(-touch*(1-config['soft_pedal']['brightness'])*freq/2500)
        a_s *= shape; a_f *= shape; attack *= shape
        count_strings = 1 if pitch < 31 else 2 if pitch < 43 else 3
        if 'coupled' in config:
            poles,residues=coupled_coefficients(pitch,velocity,freq,config)
            _add_coupled(out,start,force,poles,residues*shape[:,None],stereo,damp,sr)
        else:
            _add_modes(out, start, force, freq, a_s, a_f, r_s, r_f, stereo, damp, sr,
                       config.get('adjustments', {}).get('pitch_detune_cents', {}).get(str(pitch), config['strings']['detune_cents']*config.get('adjustments', {}).get('detune_scale', 1)), count_strings, config['engine']['commuted_excitation'])
        _add_modes(out, start, force, freq, np.zeros_like(attack), attack, r_s, np.full_like(r_s, 40), stereo, damp, sr, 0.0, 1, config['engine']['commuted_excitation'])
        if 'body' in config and config['body_settings']['gain']:
            body=body_audio(pitch,velocity,min(length,8),config,start,freq)
            size=min(len(body),len(out)-start)
            bt=t+np.arange(size)/sr
            pp=pedal[np.minimum((bt*1000).astype(int),len(pedal)-1)]
            body_damp=np.exp(-np.cumsum(np.where(bt>=off,(1-pp)**2*6.9078/1.3,0))/sr)
            out[start:start+size]+=body[:size]*body_damp[:,None]*(1-touch+touch*config['soft_pedal']['gain'])
        if off < seconds:
            profile = config['release_data'][pitch-21]
            duration = min(seconds-off, profile['peak_time']+profile['tau']*10)
            ns = round(duration*sr)
            if ns > 0:
                rng = np.random.default_rng(config['engine']['seed']+start*131+pitch*127+velocity)
                noise = rng.standard_normal(ns)
                noise = sosfilt(butter(2, profile['cutoff_hz'], fs=sr, output='sos'), noise)
                noise /= max(np.std(noise), 1e-12)
                rt = np.arange(ns)/sr
                peak_time = max(profile['peak_time'], .003)
                envelope = np.minimum(rt/peak_time, 1)*np.exp(-np.maximum(rt-peak_time, 0)/profile['tau'])
                gain = profile['rms']*10**(-37/20)*(velocity/127)**1.64*10**(-2*(off-t)/20)
                idx = round(off*sr)
                end = min(len(out), idx+ns)
                out[idx:end] += (noise[:end-idx]*envelope[:end-idx]*gain)[:, None]
    if config['resonance']['gain']:
        _sympathetic(out, pedal, config['resonance']['gain'], sr)
    # Global static gain; no note normalization or hidden limiter.
    out *= config['engine']['output_gain']
    return out
