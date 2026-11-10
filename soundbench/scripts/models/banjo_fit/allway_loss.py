"""Measured-background, noise-aware objective and residual diagnostics.

The training loss compares compressed magnitudes, so it remains insensitive to
STFT phase. Complex residuals are exposed only as phase-sensitive diagnostics.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from . import chain, loss


SR = chain.SR
_EPS = 1e-8
_BANDS = (("low", 0., 500.), ("mid", 500., 4000.),
          ("high", 4000., 12000.), ("ultrahigh", 12000., 24000.))
_PERIODS = (("attack", 0., .03), ("sustain", .03, .3), ("tail", .3, np.inf))


def _noise_sigma(notes, window):
    """Expected raw FFT coefficient power converted to its RMS amplitude."""
    frequencies = np.fft.rfftfreq(window, 1 / SR)
    hann_power = np.sum(np.hanning(window) ** 2)
    endpoints = np.ones(len(frequencies))
    endpoints[0] = 2.
    if window % 2 == 0:
        endpoints[-1] = 2.
    rows = []
    for note in notes:
        psd = np.asarray(note["background_psd"], dtype=float)
        if psd.shape != (24,) or not np.all(np.isfinite(psd)) or np.any(psd < 0):
            raise ValueError("background_psd must contain 24 finite nonnegative anchors")
        scale = float(note["background_scale"])
        if not np.isfinite(scale) or scale < 0:
            raise ValueError("background_scale must be finite and nonnegative")
        interpolated = np.exp(np.interp(np.log(np.maximum(frequencies, 1.)),
                                        np.log(chain.BACKGROUND_HZ),
                                        np.log(np.maximum(psd, 1e-300))))
        power = interpolated * SR * hann_power / 2. * endpoints * scale ** 2
        rows.append(np.sqrt(np.maximum(power, 1e-30)))
    return np.asarray(rows)


def _partial_centers_and_flags(target, notes, window=8192, threshold=3.):
    """Return per-note ridge frequencies and whether each was data-resolved."""
    y = np.asarray(target)
    if y.ndim != 2 or y.shape[0] != len(notes):
        raise ValueError("target must have shape (len(notes), samples)")
    spec = np.asarray(loss.magnitude(jnp.asarray(y), window))
    f = np.fft.rfftfreq(window, 1 / SR)
    noise = _noise_sigma(notes, window)
    centers, resolved = [], []
    hop = window // 4
    for i, note in enumerate(notes):
        # Middle-of-note power suppresses the broadband pluck without assuming
        # the model's phase or a harmonic stiffness law.
        a, b = int(.03 * SR / hop), int(.3 * SR / hop)
        ref = np.mean(spec[i, a:max(a + 1, min(b, spec.shape[1]))], axis=0)
        sigma = noise[i]
        nominal = float(note["frequency"])
        row, flags = [], []
        for harmonic in range(1, int(SR / 2 / nominal) + 1):
            expected = harmonic * nominal
            lo, hi = np.searchsorted(f, (max(0., expected * .96), min(SR / 2, expected * 1.04)))
            candidates = np.arange(lo, max(lo + 1, hi))
            candidates = candidates[candidates < len(f)]
            if not len(candidates):
                break
            idx = candidates[np.argmax(ref[candidates])]
            is_peak = ref[idx] >= (ref[idx-1] if idx else -np.inf) and ref[idx] >= (ref[idx+1] if idx+1 < len(ref) else -np.inf)
            ok = bool(is_peak and ref[idx] >= threshold * sigma[idx])
            if ok:
                if idx == 0 or idx + 1 == len(ref):
                    offset = 0.
                else:
                    left, peak, right = np.log(np.maximum(ref[idx-1:idx+2], 1e-30))
                    offset = .5 * (left-right) / (left-2*peak+right) if left-2*peak+right else 0.
                row.append(float(f[idx] + np.clip(offset, -.5, .5) * (SR/window)))
            else:
                row.append(float(expected))
            flags.append(ok)
        centers.append(np.asarray(row))
        resolved.append(np.asarray(flags, dtype=bool))
    return centers, resolved


def partial_centers(target, notes, window=8192, *, threshold=3.):
    """Return per-note resolved inharmonic ridge frequencies in Hz.

    Each nominal harmonic searches a neighborhood of ±4%; a peak must exceed
    three times the measured-background FFT RMS. Unresolved partials fall back
    to the nominal harmonic frequency. The fallback is useful for stable masks,
    and callers can compare against the reference peaks via ``prepare`` metadata.
    """
    centers, _ = _partial_centers_and_flags(target, notes, window, threshold)
    return centers


def prepare(target, notes):
    """Prepare fixed reference magnitudes, noise scales and off-ridge mask.

    Returns a dict consumed by :func:`objective`. PSD anchors are interpolated
    logarithmically over the FFT frequency grid. Noise floors use only a tiny
    numerical bound; measured PSD sets the compression scale.
    """
    if len(notes) != target.shape[0]:
        raise ValueError("notes and target batch dimensions differ")
    spectra = tuple(loss.magnitude(target, w) for w in loss.WINDOWS)
    sigmas = tuple(jnp.asarray(_noise_sigma(notes, w), dtype=jnp.float32)[:, None, :]
                   for w in loss.WINDOWS)
    centers, resolved = _partial_centers_and_flags(target, notes)
    f = np.fft.rfftfreq(4096, 1 / SR)
    mask = np.ones((len(notes), len(f)), dtype=bool)
    for i, row in enumerate(centers):
        for center in row:
            mask[i] &= abs(f-center) >= max(2 * SR / 4096, .002 * center)
    return dict(target=spectra, noise_sigma=sigmas, notes=notes,
                partial_centers=centers, partial_resolved=resolved,
                ridge_mask=jnp.asarray(mask, dtype=jnp.float32))


def objective(prediction, prepared):
    """Noise-compressed magnitude loss plus conservative off-ridge excess.

    The L1 term compares ``asinh(|X| / sigma_noise)``. A small spectral
    convergence term preserves broad energy shape without letting loud bins
    alone set the fit. The off-ridge term penalizes only predicted excess in
    the 4096-sample spectrum, outside reference-supported partial centers.
    """
    robust, convergence = 0., 0.
    for w, reference, sigma in zip(loss.WINDOWS, prepared["target"], prepared["noise_sigma"]):
        estimate = loss.magnitude(prediction, w)
        ref_z = jnp.arcsinh(reference / jnp.maximum(sigma, 1e-20))
        est_z = jnp.arcsinh(estimate / jnp.maximum(sigma, 1e-20))
        robust += jnp.mean(jnp.abs(est_z-ref_z))
        error_power = jnp.sum((est_z-ref_z)**2, axis=(1, 2))
        convergence += jnp.mean((jnp.sqrt(error_power+_EPS)-jnp.sqrt(_EPS)) /
                                 (jnp.sqrt(jnp.sum(ref_z**2, axis=(1, 2)))+_EPS))
    sigma = prepared["noise_sigma"][loss.WINDOWS.index(4096)]
    reference = prepared["target"][loss.WINDOWS.index(4096)]
    estimate = loss.magnitude(prediction, 4096)
    ref_z = jnp.arcsinh(reference / jnp.maximum(sigma, 1e-20))
    est_z = jnp.arcsinh(estimate / jnp.maximum(sigma, 1e-20))
    excess = jnp.maximum(est_z-ref_z, 0.) * prepared["ridge_mask"][:, None, :]
    ridge = jnp.sum(excess**2) / (jnp.sum(prepared["ridge_mask"]) * excess.shape[1] + _EPS)
    return robust / len(loss.WINDOWS) + .25 * convergence / len(loss.WINDOWS) + .05 * ridge


def _complex_stft(y, window):
    y = np.asarray(y, dtype=float)
    hop = window // 4
    padded = np.pad(y, ((0, 0), (window // 2, window // 2)))
    starts = np.arange(0, y.shape[-1], hop)
    frames = padded[:, starts[:, None] + np.arange(window)]
    return np.fft.rfft(frames * np.hanning(window), axis=-1)


def _summary(values):
    """Finite-frame mean and SEM; overlapping STFT frames are not independent."""
    x = np.asarray(values, dtype=float).reshape(-1)
    if not len(x):
        return dict(mean=None, sem=None, samples=0)
    return dict(mean=float(np.mean(x)), sem=float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.,
                samples=int(len(x)))


def noise_units(target, prediction, notes, window=2048):
    """Return noise-scaled residual maps and regional summaries.

    ``magnitude_map`` is absolute magnitude difference divided by the measured
    complex FFT RMS noise scale. ``complex_residual_map`` retains phase and is
    explicitly phase-sensitive; neither summary is a chi-square or a goodness
    of fit claim. SEM is computed over the selected finite set of frames/bins;
    overlap and spectral correlation make it a descriptive finite-sample error
    scale, not an independent-bin uncertainty.
    """
    yt, yp = np.asarray(target), np.asarray(prediction)
    if yt.shape != yp.shape or yt.ndim != 2 or yt.shape[0] != len(notes):
        raise ValueError("target and prediction must share shape (len(notes), samples)")
    xt, xp = _complex_stft(yt, window), _complex_stft(yp, window)
    sigma = _noise_sigma(notes, window)[:, None, :]
    mag = (np.abs(xt)-np.abs(xp)) / np.maximum(sigma, 1e-20)
    complex_map = (xt-xp) / np.maximum(sigma, 1e-20)
    times = np.arange(mag.shape[1]) * (window // 4) / SR
    freq = np.fft.rfftfreq(window, 1/SR)
    result = dict(window=window, magnitude_map=mag, complex_residual_map=complex_map,
                  map_labels=dict(magnitude="absolute magnitude difference / measured FFT-noise RMS",
                                  complex="phase-sensitive complex residual / measured FFT-noise RMS"),
                  interpretation="Residual diagnostics, not chi-square or a noise-floor goodness claim.",
                  uncertainty="SEM over finite frames/bins; overlapping frames and spectral bins are correlated.",
                  notes=[])
    for i, note in enumerate(notes):
        regions = {}
        for period, start, end in _PERIODS:
            tm = (times >= start) & (times < end)
            for band, lo, hi in _BANDS:
                fm = (freq >= lo) & (freq < hi)
                key = f"{period}/{band}"
                if np.any(tm) and np.any(fm):
                    vals = mag[i][np.ix_(tm, fm)]
                    cvals = np.abs(complex_map[i][np.ix_(tm, fm)])
                    regions[key] = dict(magnitude_abs_noise=_summary(np.abs(vals)),
                                        magnitude_signed_noise=_summary(vals),
                                        complex_abs_noise_phase_sensitive=_summary(cvals))
                else:
                    regions[key] = dict(magnitude_abs_noise=_summary([]),
                                        magnitude_signed_noise=_summary([]),
                                        complex_abs_noise_phase_sensitive=_summary([]))
        result["notes"].append(dict(pitch=note.get("pitch"), frequency=float(note["frequency"]), regions=regions))
    return result
