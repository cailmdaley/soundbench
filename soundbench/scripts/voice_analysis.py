"""Measurements of rendered voice audio: LPC formants and an f0 track."""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.signal import resample_poly


def lpc_formants(audio, sr=48000, count=3, order=None, frame_s=0.05, hop_s=0.01,
                 analysis_sr=12000, max_bandwidth=500.0):
    """Median formant frequencies over frames, from autocorrelation LPC.

    Audio is resampled to 12 kHz, pre-emphasised (0.97) and Hamming-windowed;
    the LPC order is 4 + analysis_sr/1000 (16 at 12 kHz, enough to separate
    the close F2 and F3 of the r-coloured vowel). Roots with bandwidth below
    max_bandwidth and frequency above 90 Hz are formant candidates; the
    lowest `count` per frame are kept and their medians returned.
    """
    x = resample_poly(np.asarray(audio, float), analysis_sr, sr)
    x = np.append(x[0], x[1:] - 0.97 * x[:-1])
    order = order or 4 + analysis_sr // 1000
    size, hop = round(frame_s * analysis_sr), round(hop_s * analysis_sr)
    window = np.hamming(size)
    found = []
    for start in range(0, len(x) - size, hop):
        frame = x[start:start + size] * window
        r = np.correlate(frame, frame, "full")[size - 1:size + order]
        if r[0] <= 0:
            continue
        a = solve_toeplitz(r[:order], r[1:order + 1])
        roots = np.roots(np.concatenate([[1], -a]))
        roots = roots[np.imag(roots) > 0]
        freq = np.angle(roots) * analysis_sr / (2 * np.pi)
        bw = -analysis_sr / np.pi * np.log(np.abs(roots))
        keep = np.sort(freq[(freq > 90) & (bw < max_bandwidth)])
        if len(keep) >= count:
            found.append(keep[:count])
    if not found:
        raise ValueError("no LPC frames with enough formant candidates")
    return [float(v) for v in np.median(np.array(found), axis=0)]


def f0_track(audio, expected_midi, sr=48000, frame_s=0.04, hop_s=0.005, search_cents=250):
    """Frame-wise f0 in cents relative to expected_midi (autocorrelation, parabolic peak).

    The lag search is limited to ±search_cents around the expected pitch.
    Returns frame centre times (s) and cents.
    """
    f = 440 * 2 ** ((expected_midi - 69) / 12)
    size, hop = round(frame_s * sr), round(hop_s * sr)
    lo = int(np.floor(sr / (f * 2 ** (search_cents / 1200))))
    hi = int(np.ceil(sr / (f * 2 ** (-search_cents / 1200)))) + 1
    window = np.hanning(size)
    times, cents = [], []
    for start in range(0, len(audio) - size, hop):
        frame = audio[start:start + size] * window
        frame = frame - frame.mean()
        spec = np.fft.rfft(frame, 2 * size)
        corr = np.fft.irfft(np.abs(spec) ** 2)[:size]
        if corr[0] <= 0:
            continue
        seg = corr[lo:hi + 1]
        i = int(np.argmax(seg)) + lo
        if i <= lo or i >= hi:
            continue
        d = 0.5 * (corr[i - 1] - corr[i + 1]) / (corr[i - 1] - 2 * corr[i] + corr[i + 1])
        times.append((start + size / 2) / sr)
        cents.append(1200 * np.log2(sr / (i + d) / f))
    return np.array(times), np.array(cents)


def vibrato(times, cents):
    """Mean pitch, vibrato rate (Hz) and depth (± cents, sinusoid amplitude).

    The mean covers a whole number of vibrato cycles; the rate is the peak of
    the detrended track's spectrum; depth is sqrt(2) times its RMS.
    """
    hop = times[1] - times[0]
    x = cents - cents.mean()
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)), 8 * len(x)))
    freqs = np.fft.rfftfreq(8 * len(x), hop)
    band = (freqs > 2) & (freqs < 12)
    rate = float(freqs[band][np.argmax(spec[band])])
    cycles = int(len(x) * hop * rate)
    n = max(1, round(cycles / rate / hop))
    mean = float(np.mean(cents[:n]))
    depth = float(np.sqrt(2) * np.std(cents[:n]))
    return mean, rate, depth
