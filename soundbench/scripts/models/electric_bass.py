"""Four waveguide strings, magnetic spatial taps, and a bass amp/cab.

The string loop and contractive frequency-dependent losses are shared with
models.guitar. Pickup taps are a traveling-wave approximation, not a full
magnetic field or split-coil hum simulation. No recorded samples enter render.
"""
from functools import lru_cache
import json
import math
from pathlib import Path

import numpy as np
from numba import njit
from scipy.signal import butter, lfilter, resample_poly, sosfilt
from soundbench.scripts.models.banjo import _render, _output_guard
from soundbench.scripts.models.string_loss import design_loss, PITCHES
from soundbench.scripts.paths import preset_path, read_preset

ROOT = Path(__file__).resolve().parents[2]
CONFIG = preset_path('electric-bass')
SR = 48000


def load_config(path=CONFIG):
    c = read_preset(path)
    if set(c) != {'engine', 'strings', 'excitation', 'pickup', 'amp'}:
        raise ValueError('expected engine, strings, excitation, pickup, amp')
    for table in c.values():
        for v in table.values():
            if not np.all(np.isfinite(np.asarray(v, dtype=float))):
                raise ValueError('configuration must be finite')
    e, s, x, p, a = (c[k] for k in ('engine','strings','excitation','pickup','amp'))
    if e['sample_rate'] != SR or e['output_gain'] <= 0 or not 0 <= e['tail_seconds'] <= 10 or not isinstance(e['seed'], int) or e['seed'] < 0:
        raise ValueError('invalid engine')
    if np.shape(s['open_midi']) != (4,) or min(s['open_midi']) < 21 or max(s['open_midi'])+s['max_fret'] > 96 or not 0 <= s['max_fret'] <= 24:
        raise ValueError('invalid strings')
    hz, times = np.array(s['decay_hz']), np.array(s['decay_t60'])
    if hz.ndim != 1 or len(hz) < 2 or hz[0] <= 0 or hz[-1] > SR/2 or np.any(np.diff(hz) <= 0) or times.shape != (4,len(hz)) or np.any(times <= 0):
        raise ValueError('invalid decay curve')
    if min(s['release_t60'], s['choke_t60'], s['pitch_smoothing_ms']) <= 0 or not -.5 <= s['dispersion'] <= 0:
        raise ValueError('invalid string damping/dispersion')
    if not 0 < x['position'] < .5 or not 0 <= x['jitter'] < .2 or not 0 < x['finger_ms'] <= 5 or not 0 < x['pick_ms'] <= 5 or not 20 <= x['lowpass_hz'] < SR/2 or x['velocity_exponent'] <= 0 or not 0 <= x['noise'] <= .1:
        raise ValueError('invalid excitation')
    if not (len(p['position']) == len(p['weight']) == len(p['aperture'])) or not len(p['position']) or any(not 0 < q < .5 for q in p['position']) or any(not 0 < q < .1 for q in p['aperture']) or min(p['weight']) < 0 or sum(p['weight']) <= 0:
        raise ValueError('invalid pickup')
    if a['input_gain'] <= 0 or not 0 <= a['drive'] <= 20 or not 20 <= a['cab_hz'] < 20000 or not 1 <= a['compressor_ratio'] <= 20 or not -80 <= a['compressor_threshold_db'] <= 0 or min(a['attack_ms'],a['release_ms']) <= 0:
        raise ValueError('invalid amp')
    return c


def assign_string(pitch, config, explicit=None):
    opens = config['strings']['open_midi']
    if explicit is not None and (not isinstance(explicit,int) or explicit not in range(1,5)):
        raise ValueError('strings are 1..4, high to low')
    choices = [explicit-1] if explicit else range(4)
    playable = [s for s in choices if 0 <= pitch-opens[s] <= config['strings']['max_fret']]
    if not playable:
        raise ValueError(f'MIDI {pitch} is outside the four-string bass')
    return min(playable, key=lambda s: pitch-opens[s])


@lru_cache(maxsize=8)
def _bank(serialized):
    s = json.loads(serialized)
    bank = np.empty((4,len(PITCHES),65))
    for lane in range(4):
        for i,pitch in enumerate(PITCHES):
            bank[lane,i] = design_loss(s['decay_hz'], np.log(1000)/np.array(s['decay_t60'][lane]), 440*2**((pitch-69)/12))
    return bank


@njit(cache=True)
def _compress_gains(y, threshold, ratio, attack, release):
    out = np.ones(len(y))
    env = 0.
    for n in range(len(y)):
        level = abs(y[n])
        alpha = attack if level > env else release
        env = alpha*env+(1-alpha)*level
        gain = (env/threshold)**(1/ratio-1) if env > threshold else 1.
        out[n] = gain
    return out


def amp_stage(y, config):
    """One shared amp, with instantaneous gain attribution to source stems.

    Summing the stems equals applying the nonlinear stage to their sum.
    Distortion attribution isn't a unique physical observable.
    """
    a = config['amp']
    mono = y.ndim == 1
    y = np.atleast_2d(y)
    samples = y.shape[1]
    # Oversampling puts most new saturation harmonics below the output Nyquist.
    if a['drive'] > 0:
        z = resample_poly(y, 4, 1, axis=1)
        drive = a['drive']; total = z.sum(axis=0)
        gain = np.ones_like(total)
        nonzero = abs(total) > 1e-20
        gain[nonzero] = np.tanh(drive*total[nonzero])/(drive*total[nonzero])
        y = resample_poly(z*gain, 1, 4, axis=1)[:,:samples]
    y = sosfilt(butter(2, 28, 'highpass', fs=SR, output='sos'), y, axis=1)
    y = sosfilt(butter(2, a['cab_hz'], fs=SR, output='sos'), y, axis=1)
    if a['compressor_ratio'] > 1:
        gain = _compress_gains(y.sum(axis=0), 10**(a['compressor_threshold_db']/20), a['compressor_ratio'],
                      math.exp(-1/(SR*a['attack_ms']/1000)), math.exp(-1/(SR*a['release_ms']/1000)))
        y *= gain
    return y[0] if mono else y


def render(events, seconds, owners=1, config=None):
    """11-column shared-string events; kinds 0 pluck, 1 off, 2 slide,
    3 hammer, 4 pull, 5 hand-damp, 6 ghost, 7 palm-muted pluck.

    Column 9 on attacks: 0 finger / 1 pick; column 8: pluck position override.
    Every note-off damps, including open strings. Tokens protect replacements.
    A slide targets a held token; hammer/pull transfer a held string to a new
    token. Callers must provide held sources for legato and explicit releases.
    """
    c = config or load_config()
    if not math.isfinite(seconds) or seconds < 0 or not isinstance(owners,int) or owners < 1:
        raise ValueError('invalid render dimensions')
    rows = np.asarray(events,float).reshape(-1,11).copy()
    if not np.all(np.isfinite(rows)):
        raise ValueError('events must be finite')
    for r in rows:
        if r[0] < 0 or r[1] not in range(8) or r[2] not in range(4) or r[3] not in range(owners) or not 0 <= r[5] <= 127 or r[6] < 0 or r[7] not in (0,1) or r[10] < 0 or r[10] != round(r[10]):
            raise ValueError('invalid bass event')
        if r[1] in (0,2,3,4,6,7):
            assign_string(r[4],c,int(r[2])+1)
        if r[1] in (0,3,4,6,7) and r[9] not in (0,1):
            raise ValueError('attack hardness is 0 finger / 1 pick')
    rows = rows[np.argsort(rows[:,0],kind='stable')]
    s,x,e = c['strings'], c['excitation'], c['engine']
    expanded, pulses, noises = [], [], np.zeros((owners*4,round(seconds*SR)))
    for r in rows:
        r = r.copy()
        lane = int(r[2]); owner = int(r[3])*4+lane
        r[3] = owner
        pulse = np.zeros(512)
        if r[1] in (0,3,4,6,7):
            kind = int(r[1])
            rng = np.random.default_rng(np.random.SeedSequence([e['seed'],int(r[10]),lane,owner]))
            r[8] = (r[8] or x['position'])*rng.uniform(1-x['jitter'],1+x['jitter'])
            if not 0 < r[8] < .5:
                raise ValueError('pluck position must be 0..0.5 exclusive')
            width = x['pick_ms'] if r[9] else x['finger_ms']
            pole = math.exp(-2/(SR*width/1000))
            amp = (r[5]/127)**x['velocity_exponent']
            amp *= .28 if kind == 3 else .4 if kind == 4 else .16 if kind == 6 else 1.
            cutoff = x['lowpass_hz']*(1.8 if r[9] else 1.)
            alpha = 1-math.exp(-2*math.pi*min(cutoff,20000)/SR)
            pulse = amp*lfilter([alpha],[1,-(1-alpha)], (1-pole)*pole**np.arange(512))
            length = 960 if kind == 6 else 192
            burst = lfilter([1,-1],[1,-.75],rng.normal(size=length))*np.exp(-np.arange(length)/(160 if kind == 6 else 35))
            burst /= max(np.sqrt(np.sum(burst**2)),1e-12)
            on = int(r[0]); end = min(noises.shape[1],on+length)
            if end > on:
                noises[owner,on:end] += burst[:end-on]*x['noise']*amp*(5 if kind == 6 else 1)
            if kind in (6,7):
                r[1] = 0
                damp = r.copy(); damp[0] += round((.006 if kind == 6 else .025)*SR)
                damp[1] = 1; damp[9] = .018 if kind == 6 else .13
                expanded.append(damp); pulses.append(np.zeros(512))
        elif r[1] == 1:
            r[9] = s['release_t60']
        expanded.append(r); pulses.append(pulse)
    rows = np.array(expanded,float).reshape(-1,11)
    pulses = np.array(pulses,float).reshape(-1,512)
    order = np.argsort(rows[:,0],kind='stable'); rows,pulses = rows[order],pulses[order]
    audio = _render(rows,pulses,round(seconds*SR),owners*4,SR,np.array(s['open_midi'],float),
        np.full(4,4.),np.zeros(4),np.full(4,s['dispersion']),s['release_t60'],s['choke_t60'],
        1-math.exp(-1/(s['pitch_smoothing_ms']*SR/1000)),0.,0.,
        np.zeros(0),np.zeros(0),np.zeros(0),np.zeros(0),0.,1.,1.,1e-18,
        _bank(json.dumps(s,sort_keys=True)),np.zeros((4,3)))
    audio += noises
    out = np.zeros((owners,audio.shape[1]))
    p = c['pickup']
    for owner in range(owners):
        for lane in range(4):
            y = audio[owner*4+lane]
            period = SR/(440*2**((s['open_midi'][lane]-69)/12))
            pickup = np.zeros_like(y)
            for position, aperture, weight in zip(p['position'],p['aperture'],p['weight']):
                # Fixed bridge distance means a fixed time separation on each
                # string; its comb moves in harmonic number with the fret.
                delay = position*period
                integer = int(delay); fraction = delay-integer
                delayed = np.zeros_like(y)
                if integer < len(y):
                    delayed[integer:] += (1-fraction)*y[:len(y)-integer]
                if integer+1 < len(y):
                    delayed[integer+1:] += fraction*y[:len(y)-integer-1]
                width = max(1,round(aperture*period))
                pickup += weight*lfilter(np.ones(width)/width,[1],y-delayed)
            out[owner] += pickup/sum(p['weight'])
    out = amp_stage(out*c['amp']['input_gain'],c)*e['output_gain']
    return _output_guard(out,.85,.98)


def note(on,off,pitch,velocity,string,token,kind=0,hardness=0,owner=0):
    if not 0 <= on < off or not 1 <= velocity <= 127:
        raise ValueError('note needs positive duration and velocity 1..127')
    return [[round(on*SR),kind,string-1,owner,pitch,velocity,.008*SR,1,0,hardness,token],
            [round(off*SR),1,string-1,owner,0,0,0,0,0,0,token]]
