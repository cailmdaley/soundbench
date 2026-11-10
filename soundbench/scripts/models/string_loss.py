"""Contractive FIR waveguide loss, fitted to log decay at string partials.

A minimum-phase FIR avoids spending the delay budget on a linear-phase filter.
A Lipschitz bound between frequency samples certifies contraction on the entire
unit circle, not just the design grid. Pitch compensation uses the actual FIR
phase at the fundamental. The bank interpolates adjacent MIDI pitches; convex
interpolation preserves contraction.
"""
from functools import lru_cache
import json
import numpy as np
from scipy.signal import firwin2, minimum_phase, freqz

TAPS = 65
PITCHES = np.arange(21., 110.)


def design_loss(frequency, rates, fundamental, taps=TAPS, fs=48000):
    grid = np.linspace(0, fs/2, 4097)
    rate = np.interp(np.log(np.maximum(grid, 1)), np.log(frequency), rates)
    target = np.exp(-rate / fundamental)
    linear = firwin2(2*taps-1, grid, target**2, fs=fs)
    h = minimum_phase(linear, n_fft=32768)
    # |dH/dw| <= sum k*|h[k]|. Every w is within pi/N of a grid point.
    count = 32768
    _, response = freqz(h, worN=count, whole=True)
    bound = np.max(abs(response)) + np.pi/count*np.sum(np.arange(len(h))*abs(h))
    if bound >= 1-1e-7:
        h *= (1-1e-7)/bound
    return h


def loss_bank(strings):
    keys = ('decay_hz', 'decay_t60', 'polarization_rate')
    return _loss_bank(json.dumps({k: strings[k] for k in keys if k in strings}))


@lru_cache(maxsize=8)
def _loss_bank(serialized):
    strings = json.loads(serialized)
    frequency = np.array(strings['decay_hz'])
    rates = np.log(1000)/np.array(strings['decay_t60'])
    multipliers = strings.get('polarization_rate', [1.]*5)
    bank = np.empty((10, len(PITCHES), TAPS))
    for lane in range(10):
        s = lane % 5
        for i, pitch in enumerate(PITCHES):
            f0 = 440*2**((pitch-69)/12)
            bank[lane, i] = design_loss(frequency, np.maximum(rates[s]*(multipliers[s] if lane>=5 else 1), np.log(1000)/3), f0)
    return bank
