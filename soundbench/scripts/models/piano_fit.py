#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy", "soundfile", "jax", "optax", "matplotlib"]
# ///
"""Fit Salamander modal observations with two-stage decay and a short attack in JAX.

The exported radiated modal gains include the board, microphones and hammer;
this is not an identification of their separate mechanical impedances.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re

import numpy as np
import soundfile as sf
from scipy.signal import stft, find_peaks, resample_poly
from scipy.optimize import least_squares
from soundbench.scripts.paths import data_path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'soundfonts/SalamanderGrandPiano-SFZ+FLAC-V3+20200602'
LAYERS = [13.5, 30.5, 35.5, 40, 45, 48.5, 53.5, 60.5, 68.5, 76.5, 84.5, 92.5, 100.5, 108.5, 116.5, 124]
MODES = 96


def source_map():
    text = (SOURCE / 'SalamanderGrandPiano-V3+20200602.sfz').read_text()
    rows = []
    for sample in re.findall(r'<region> sample=(samples/[A-G]#?\d+v\d+\.flac)', text):
        name, octave, layer = re.search(r'/([A-G]#?)(\d+)v(\d+)\.flac', sample).groups()
        pitch = (int(octave)+1)*12 + {'C':0, 'D#':3, 'F#':6, 'A':9}[name]
        rows.append((pitch, int(layer), SOURCE / sample))
    return sorted(rows)


def load(path, seconds=5):
    y, sr = sf.read(path, frames=round(sf.info(path).samplerate * seconds), always_2d=True)
    if sr != 24000:
        from math import gcd
        g = gcd(sr, 24000)
        y = resample_poly(y, 24000 // g, sr // g, axis=0)
    length = round(seconds*24000)
    y = np.pad(y[:length], ((0, max(0, length-len(y))), (0, 0)))
    return y, 24000


def stiffness(y, sr, pitch):
    """Robust stiff-string regression against resolved early spectral peaks."""
    x = y[int(.1*sr):int(2.1*sr)].mean(axis=1)
    nfft = 262144
    mag = abs(np.fft.rfft(x * np.hanning(len(x)), nfft))
    f = np.fft.rfftfreq(nfft, 1/sr)
    peaks, _ = find_peaks(mag)
    nominal = 440 * 2 ** ((pitch - 69) / 12)
    ns, fs, weights = [], [], []
    # Progressive search follows the stiff-string curve, rather than insisting
    # high partials lie inside a harmonic-width window.
    b = 0.0001
    for k in range(1, 25):
        expected = nominal * k * np.sqrt(1+b*k*k)
        window = max(nominal*.22, expected*.014)
        candidates = peaks[(abs(f[peaks]-expected) < window) & (f[peaks] < 10000)]
        if not len(candidates):
            continue
        p = candidates[np.argmax(mag[candidates])]
        if mag[p] < mag.max() * .015:
            continue
        # Parabolic interpolation in log magnitude.
        a, c, d = np.log(np.maximum(mag[p-1:p+2], 1e-20))
        delta = .5*(a-d)/(a-2*c+d)
        ns.append(k); fs.append((p+delta)*sr/nfft); weights.append(mag[p])
        if len(ns) >= 2:
            n = np.array(ns); measured = np.array(fs)
            result = least_squares(lambda z: 1200*np.log2(z[0]*n*np.sqrt(1+z[1]*n*n)/measured),
                                   [nominal, b], bounds=([nominal*.97, 0], [nominal*1.04, .02]),
                                   loss='soft_l1', f_scale=3)
            nominal, b = result.x
    if len(ns)==1 and ns[0]==1:
        nominal=fs[0]/np.sqrt(1+b)
    return float(nominal), float(b), {'partials': ns, 'frequencies': fs}


def envelopes(y, sr, f0, b, nperseg=8192):
    f, t, z = stft(y, fs=sr, nperseg=nperseg, noverlap=nperseg-512, axis=0, boundary=None)
    # scipy layout: frequency, channel, time. Squared-band sum is insensitive
    # to the slow unison beating that would defeat a single-bin measurement.
    amp = np.zeros((MODES, len(t), 2))
    for k in range(1, MODES+1):
        hz = f0*k*np.sqrt(1+b*k*k)
        if hz > 10500:
            continue
        band = abs(f-hz) < max(8, sr/nperseg*1.3, hz*.004)
        amp[k-1] = np.sqrt(np.sum(abs(z[band])**2, axis=0).T / 1.5)*2
    return t, amp


def fit_envelopes(observed, times, steps):
    import jax
    import jax.numpy as jnp
    import optax
    # A_s exp(-r_s t) + A_p exp(-r_p t) + A_attack exp(-40 t), r_p > r_s > 0.
    # Log-amplitude residuals emphasize decay, with a relative -40 dB floor.
    y = jnp.asarray(observed)
    t = jnp.asarray(times)[None, :]
    scale = jnp.maximum(jnp.max(y, axis=1, keepdims=True), 1e-7)
    normalized = y / scale
    initial = np.tile(np.array([-.5, -.5, -2.0, -1.5, .5]), (len(observed), 1))
    params = jnp.asarray(initial)
    def decode(p):
        a = jax.nn.softplus(p[:, :3])
        slow = .08 + jax.nn.softplus(p[:, 3])
        fast = slow + jax.nn.softplus(p[:, 4])
        return a, slow, fast
    def predict(p):
        a, slow, fast = decode(p)
        return (a[:, 0:1]*jnp.exp(-slow[:, None]*t) + a[:, 1:2]*jnp.exp(-fast[:, None]*t)
                + a[:, 2:3]*jnp.exp(-40*t))
    def loss(p):
        pred = predict(p)
        return jnp.mean((jnp.log(pred+.01)-jnp.log(normalized+.01))**2)
    optimizer = optax.adam(.035)
    state = optimizer.init(params)
    @jax.jit
    def step(p, state):
        value, grads = jax.value_and_grad(loss)(p)
        updates, state = optimizer.update(grads, state)
        return optax.apply_updates(p, updates), state, value
    before = float(loss(params))
    for _ in range(steps):
        params, state, value = step(params, state)
    a, slow, fast = map(np.asarray, decode(params))
    a = a * np.asarray(scale)
    pred = np.asarray(predict(params)) * np.asarray(scale)
    return a, slow, fast, pred, before, float(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=700)
    args = parser.parse_args()
    out = ROOT/'build/piano-fit'; out.mkdir(parents=True, exist_ok=True)
    rows = source_map()
    notes = sorted(set(r[0] for r in rows))
    tuning = {}
    for pitch in notes:
        path = next(r[2] for r in rows if r[:2] == (pitch, 8))
        y, sr = load(path)
        tuning[pitch] = stiffness(y, sr, pitch)
        print('frequency', pitch, tuning[pitch][:2], flush=True)
    obs, metadata = [], []
    for pitch, layer, path in rows:
        y, sr = load(path)
        f0, b, _ = tuning[pitch]
        t, env = envelopes(y, sr, f0, b)
        early_t, early = envelopes(y, sr, f0, b, nperseg=2048)
        select = early_t < t[0]
        t = np.concatenate((early_t[select], t))
        env = np.concatenate((early[:, select], env), axis=1)
        # Mean-channel envelope defines decay; stereo balance is a separate gain.
        obs.append(env.mean(axis=2).T); metadata.append((pitch, layer, env))
    observations = np.array(obs)
    # time x mode -> mode x time, flatten the note/layer/modal dimensions.
    flat = observations.transpose(0, 2, 1).reshape(-1, len(t))
    a, slow, fast, pred, before, after = fit_envelopes(flat, t, args.steps)
    a = a.reshape(len(rows), MODES, 3)
    slow = slow.reshape(len(rows), MODES); fast = fast.reshape(len(rows), MODES)
    profiles = []
    for pitch in notes:
        indices = [i for i, r in enumerate(rows) if r[0] == pitch]
        f0, b, peaks = tuning[pitch]
        profiles.append({'pitch': pitch, 'f0': f0, 'B': b,
                         'slow_amp': a[indices, :, 0].tolist(), 'fast_amp': a[indices, :, 1].tolist(),
                         'attack_amp': a[indices, :, 2].tolist(),
                         'slow_rate': slow[indices].tolist(), 'fast_rate': fast[indices].tolist(),
                         'stereo': [np.clip(metadata[i][2].mean(axis=1)/(observations[i].mean(axis=0)[:, None]+1e-10), .2, 2).tolist() for i in indices]})
    export = {'schema': 1, 'modes': MODES, 'velocities': LAYERS, 'profiles': profiles,
              'source': 'Alexander Holm, Salamander Grand Piano V3, CC BY 3.0',
              'source_sfz_sha256': hashlib.sha256((SOURCE/'SalamanderGrandPiano-V3+20200602.sfz').read_bytes()).hexdigest()}
    target = data_path('piano', 'modal.json')
    target.write_text(json.dumps(export, separators=(',', ':'))+'\n')
    np.savez_compressed(out/'observations.npz', observed=observations, predicted=pred.reshape(len(rows), MODES, -1).transpose(0, 2, 1), times=t, pitches=[r[0] for r in rows], layers=[r[1] for r in rows])
    metrics = {'clips': len(rows), 'notes': len(notes), 'layers': 16, 'steps': args.steps,
               'before_log_envelope_mse': before, 'after_log_envelope_mse': after,
               'tuning': {p: {'f0': x[0], 'B': x[1], **x[2]} for p, x in tuning.items()}}
    (out/'fit.json').write_text(json.dumps(metrics, indent=2)+'\n')
    print(json.dumps({k:v for k,v in metrics.items() if k != 'tuning'}, indent=2))

if __name__ == '__main__':
    main()
