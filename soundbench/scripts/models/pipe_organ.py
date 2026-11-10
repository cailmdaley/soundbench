"""Church pipe organ: per-pipe additive spectra, speech chiff, wind and tuning scatter.

Every sounding pipe is one stop (family and footage) at one pitch. Its spectrum,
tuning offset, voicing ripple and phases are fixed by (seed, stop, pitch), so a
pipe is the same pipe every time it speaks; the noise (chiff, wind) varies by
note onset. Velocity is ignored, as on a real organ; dynamics are registration
and, through CC11, a swell box (gain and top end together). Notes render one at
a time into float32 buffers.
"""
import math
import numpy as np
from scipy.signal import butter, sosfilt
from .keys_common import SR, ROOT, load_config as _load, validate, curve, smooth, finish

CONFIG = ROOT/'instruments/models/pipe-organ.toml'
FAMILY_KEYS = {'power', 'cutoff_hz', 'even', 'ripple_db', 'attack_s', 'release_s',
               'chiff_harmonic', 'chiff_db', 'chiff_ms', 'break_hz'}


def load_config(path=CONFIG):
    cfg = _load('pipe-organ', path)
    if not cfg['registration']:
        raise ValueError('pipe organ needs registrations')
    for name, reg in cfg['registration'].items():
        lo, hi = reg['range']
        if not 12 <= lo < hi <= 108 or not reg['stop']:
            raise ValueError(f'invalid registration {name!r}')
        for stop in reg['stop']:
            if stop['family'] not in cfg['family']:
                raise ValueError(f'registration {name!r}: unknown family {stop["family"]!r}')
            if not all(0.1 <= f <= 32 for f in footages(stop)):
                raise ValueError(f'registration {name!r}: footage out of range')
    for fam in cfg['family'].values():
        if set(fam) - FAMILY_KEYS:
            raise ValueError('unknown pipe family parameter')
        if not .005 <= fam['attack_s'] <= .3 or not .02 <= fam['release_s'] <= .5:
            raise ValueError('implausible pipe speech or release time')
    if cfg['engine']['registration'] not in cfg['registration']:
        raise ValueError('default registration is not defined')
    return cfg


def footages(stop):
    f = stop['footage']
    return list(f) if isinstance(f, list) else [f]


def pitch_hz(pitch, cfg):
    return cfg['engine']['a4_hz']*2**((pitch-69)/12)


def spectrum(f0, fam, rng):
    """Harmonic numbers and unit-RMS amplitudes of one pipe."""
    top = min(SR*.45, 12*fam['cutoff_hz'])
    h = np.arange(1, max(1, min(80, int(top/f0)))+1)
    a = h**-fam['power']/np.sqrt(1+(h*f0/fam['cutoff_hz'])**4)
    a = a*np.where(h % 2 == 0, fam['even'], 1.0)
    a = a*10**(rng.normal(0, fam['ripple_db']/2, len(h))/20)
    keep = a > 1e-3*a.max()
    h, a = h[keep], a[keep]
    return h, a/math.sqrt(np.sum(a**2)/2)


def band_noise(rng, n, centre, sr=SR):
    lo, hi = centre/1.45, min(centre*1.45, sr*.45)
    x = sosfilt(butter(2, [lo, hi], 'bandpass', fs=sr, output='sos'), rng.standard_normal(n))
    return x/(np.std(x)+1e-12)


def attack_tau(tau0, h):
    return tau0*(1+.35*math.sqrt(h-1))


def ramp_samples(tau0, hmax):
    """Samples after which even the slowest harmonic's attack has settled (< 1e-5 step)."""
    return int(12*attack_tau(tau0, hmax)*SR)


def pipe(f0, on, total, fam, cfg, rs, rn):
    """One pipe sounding for `on` samples inside a buffer of `total` samples."""
    t = np.arange(total)/SR
    harmonics, amps = spectrum(f0, fam, rs)
    phases = rs.uniform(0, 2*np.pi, len(harmonics))
    tau0 = fam['attack_s']*(130/max(f0, 30))**.3
    m = min(total, ramp_samples(tau0, harmonics[-1]))
    y = np.zeros(total)
    for h, a, ph in zip(harmonics, amps, phases):
        s = a*np.sin(2*np.pi*f0*h*t+ph)
        s[:m] *= 1-np.exp(-t[:m]/attack_tau(tau0, h))
        y += s
    # Speech: a short burst of band-limited noise around the 2nd-3rd harmonic.
    d = fam['chiff_ms']/1000; c = min(int(4*d*SR), total)
    centre = min(max(f0*fam['chiff_harmonic'], 250), 4500)
    env = (1-np.exp(-t[:c]/.003))*np.exp(-t[:c]/(d/2.5))
    y[:c] += 10**(fam['chiff_db']/20)*math.sqrt(2)*env*band_noise(rn, c, centre)
    wind = sosfilt(butter(1, cfg['wind']['lowpass_hz'], fs=SR, output='sos'), rn.standard_normal(total))
    y += 10**(cfg['wind']['level_db']/20)*math.sqrt(2)*wind/(np.std(wind)+1e-12)*(1-np.exp(-t/.05))
    r = np.arange(total)-on
    return y*np.where(r > 0, np.exp(-np.maximum(r, 0)/(fam['release_s']*SR)), 1.0)


def stereo_gains(pitch):
    """Chromatic chest layout: alternate pitches sit left and right of centre."""
    pan = .3*(((pitch*7) % 12)-5.5)/5.5
    angle = (pan+1)*math.pi/4
    return math.cos(angle), math.sin(angle)


def sounding_pipes(pitch, stops, cfg):
    for si, stop in enumerate(stops):
        fam = cfg['family'][stop['family']]
        for ri, foot in enumerate(footages(stop)):
            ratio = 8/foot; f0 = pitch_hz(pitch, cfg)*ratio
            while fam.get('break_hz') and f0 > fam['break_hz']:
                f0 /= 2
            yield si, ri, fam, f0, 10**(stop['db']/20)/math.sqrt(len(footages(stop)))


def swell(y, controls, cfg):
    s = cfg['swell']; n = len(y)
    c = smooth(curve(controls, 11, n, 127)/127, s['smooth_s'])
    low = sosfilt(butter(1, s['lowpass_hz'], fs=SR, output='sos'), y, axis=0)
    bright = s['brightness_floor']+(1-s['brightness_floor'])*c**1.5
    gain = 10**(s['closed_db']*(1-c)/20)
    return (low+(y-low)*bright[:, None])*gain[:, None]


def render(notes, seconds, config=None, controls=()):
    cfg = config or load_config()
    name = cfg.get('active', cfg['engine']['registration'])
    reg = cfg['registration'][name]
    validate(notes, seconds, controls, *reg['range'])
    n = round(seconds*SR); out = np.zeros((n, 2), np.float32); seed = cfg['engine']['seed']
    cents = cfg['engine']['tuning_scatter_cents']
    onsets = {}
    for note in notes:
        onsets.setdefault(note['pitch'], []).append(round(note['time']*SR))
    for note in notes:
        start = round(note['time']*SR); off = round(note['off']*SR); pitch = note['pitch']
        # One pipe per pitch: a re-struck pipe cuts the previous instance (same seed, same phases).
        later = [t for t in onsets[pitch] if t > start]
        cut = min(later) if later else None
        left, right = stereo_gains(pitch)
        for si, ri, fam, f0, level in sounding_pipes(pitch, reg['stop'], cfg):
            rs = np.random.default_rng([seed, si, ri, pitch])
            f0 *= 2**(rs.normal(0, cents)/1200)
            end = min(n, off+round(8*fam['release_s']*SR))
            if end <= start:
                continue
            rn = np.random.default_rng([seed, si, ri, pitch, start])
            if cut is not None and cut < end:
                end = max(cut, start+1)
            y = (level*pipe(f0, off-start, end-start, fam, cfg, rs, rn))
            if cut is not None and cut <= end:
                fade = min(len(y), round(.005*SR)); y[len(y)-fade:] *= np.linspace(1, 0, fade)
            y = y.astype(np.float32)
            out[start:end, 0] += left*y; out[start:end, 1] += right*y
    if any(c == 11 for _, c, _ in controls):
        out = swell(out, controls, cfg)
    return finish(out, cfg['engine']['output_gain']).astype(np.float32)
