# /// script
# dependencies = ["numpy", "scipy", "numba", "soundfile"]
# ///
"""Print pitch/energy diagnostics for wind waveguides."""
from soundbench.scripts.models.mallorca_winds import load_config, render, frequency, _reed
import numpy as np
from scipy.signal import find_peaks
import soundfile as sf
from pathlib import Path


def pitch(y, sr, target):
    x = y[-sr//2:]
    size = 262144
    s = np.abs(np.fft.rfft(x*np.hanning(len(x)), n=size))
    f = np.fft.rfftfreq(size, 1/sr)
    peaks, _ = find_peaks(s)
    peaks = sorted(peaks, key=lambda p: s[p], reverse=True)[:8]
    p = min(peaks, key=lambda p: abs(np.log2(f[p]/target)))
    a, b, c = np.log(s[p-1:p+2]+1e-20)
    offset = .5*(a-c)/(a-2*b+c)
    return (p+offset)*sr/size, [(round(f[q],1), round(float(s[q]),2)) for q in peaks]


if __name__ == "__main__":
    Path('build/winds-diagnose').mkdir(parents=True, exist_ok=True)
    c = load_config('xeremia')
    for p in [36, 43, 48, 55, 60]:
        f = frequency(p, c)
        x = _reed(np.full(48000, f), np.ones(48000), np.random.default_rng(1).uniform(-.01, .01, 48000), 48000, .65, False, 1.0, np.zeros(48000))
        actual, _ = pitch(x, 48000, f)
        print('drone', p, 'cents', 1200*np.log2(actual/f))
    for inst, notes in (("flabiol", [74, 81, 86, 93, 98]), ("xeremia", [59, 60, 64, 67, 72])):
        c = load_config(inst)
        c['wobble_cents'] = c['squawk_cents'] = c['sag_cents'] = 0
        c['drone_gain'] = 0
        for p in notes:
            y = render([dict(time=0, off=1.5, pitch=p, velocity=90)], 1.6, c, inst)
            f, peaks = pitch(y[:60000], 48000, frequency(p,c))
            print(inst, p, 'rms',np.sqrt(np.mean(y*y)), 'peak',max(abs(y)), 'cents',1200*np.log2(f/frequency(p,c)), 'peaks',peaks)
            sf.write(f'build/winds-diagnose/{inst}-{p}.wav',y,48000)
