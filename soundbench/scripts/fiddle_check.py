#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy", "numba", "soundfile"]
# ///
"""Numerical inspection of the fiddle junction's tuning and periodic motion."""
import sys
from pathlib import Path
import numpy as np
from scipy.signal import periodogram, find_peaks
from soundbench.scripts.models.fiddle import load_config, render


def estimate_pitch(y, target):
    f, power = periodogram(y, 48000, nfft=262144)
    mask = (f > target*.9) & (f < target*1.1)
    indices = np.flatnonzero(mask)
    k = indices[np.argmax(power[mask])]
    l = np.log(power[k-1:k+2]+1e-30)
    offset = .5*(l[0]-l[2])/(l[0]-2*l[1]+l[2])
    return (k+offset)*48000/262144


if __name__ == '__main__':
    import soundfile as sf
    c = load_config()
    for p in [55, 57, 62, 64, 69, 71, 76, 81, 88, 96]:
        y, d = render([dict(time=0, off=1, pitch=p, velocity=90)], 2.5, config=c, diagnostics=True)
        target = 440*2**((p-69)/12)
        freq = estimate_pitch(y[0, 12000:40000, 0], target)
        s = d['assignment'][0]['string']-1
        print(p, 'cents', round(1200*np.log2(freq/target), 2), 'RMS', np.sqrt(np.mean(y*y)), 'peak', abs(y).max(), 'stick', np.mean(d['stick'][s, 12000:40000]))
    Path('build/fiddle-probe').mkdir(parents=True, exist_ok=True)
    sf.write('build/fiddle-probe/check.wav', y[0], 48000)
