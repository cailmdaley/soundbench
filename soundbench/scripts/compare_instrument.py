#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "librosa>=0.10.2,<0.12",
#   "matplotlib>=3.8",
#   "mido>=1.3,<2",
#   "numpy>=1.26",
#   "pyloudnorm>=0.1.1",
#   "scipy>=1.11",
# ]
# ///
"""Compare an sfizz stem with a licensed recording of the same instrument.

The spectrum is averaged in IEC-style one-third-octave bands after both excerpts
are loudness-normalized to -23 LUFS. For archival 78s, pass
``--reference-type 78`` (100 Hz–5 kHz); the same band limit is applied to both
signals before spectral, transient, centroid and decay measurements.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mido
import numpy as np
import pyloudnorm as pyln
from scipy import signal
from scipy.integrate import trapezoid
from soundbench.scripts.audio_pitch import isolated_pitch_report

SAMPLE_RATE = 24_000
TARGET_LUFS = -23.0
HOP_LENGTH = 240
N_FFT = 2048


def db(value: float) -> float | None:
    if not math.isfinite(float(value)) or value <= 0:
        return None
    return float(20 * math.log10(value))


def load_excerpt(path: Path, time_range: tuple[float, float] | None) -> tuple[np.ndarray, float, float]:
    start, end = time_range if time_range else (0.0, None)
    if start < 0 or (end is not None and end <= start):
        raise ValueError(f"invalid time range for {path}: {time_range}")
    duration = None if end is None else end - start
    audio, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True, offset=start, duration=duration)
    if not len(audio):
        raise ValueError(f"empty audio excerpt: {path} [{start}, {end}]")
    actual_end = start + len(audio) / sr
    return audio.astype(np.float64, copy=False), float(start), float(actual_end)


def band_limit(audio: np.ndarray, low_hz: float, high_hz: float, sr: int) -> np.ndarray:
    if not (0 < low_hz < high_hz < sr / 2):
        raise ValueError(f"bandwidth must be inside (0, {sr / 2:g}) Hz; got {low_hz:g}–{high_hz:g} Hz")
    sos = signal.butter(4, (low_hz, high_hz), btype="bandpass", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, audio)


def loudness_normalize(audio: np.ndarray, sr: int) -> tuple[np.ndarray, float, str]:
    try:
        measured = float(pyln.Meter(sr).integrated_loudness(audio))
        if not math.isfinite(measured):
            raise ValueError("non-finite integrated loudness")
        method = "ITU-R BS.1770 integrated loudness"
    except (ValueError, ZeroDivisionError):
        rms = float(np.sqrt(np.mean(np.square(audio))))
        measured = (db(rms) or -120.0) - 0.691
        method = "RMS fallback for a below-gate excerpt"
    gain = 10 ** ((TARGET_LUFS - measured) / 20)
    return audio * gain, measured, method


def third_octave_spectrum(audio: np.ndarray, sr: int, low_hz: float, high_hz: float) -> tuple[list[float], list[float]]:
    nperseg = min(8192, len(audio))
    frequencies, psd = signal.welch(
        audio, fs=sr, window="hann", nperseg=nperseg,
        noverlap=nperseg // 2, detrend="constant", scaling="density",
    )
    centers, levels = [], []
    for octave_step in range(-18, 13):
        center = 1000.0 * 2 ** (octave_step / 3)
        lower, upper = center / 2 ** (1 / 6), center * 2 ** (1 / 6)
        if center < low_hz or center > high_hz or lower < low_hz or upper > high_hz:
            continue
        selected = (frequencies >= lower) & (frequencies < upper)
        bins = np.flatnonzero(selected)
        if not len(bins):
            continue
        if len(bins) == 1:
            width = sr / nperseg
            power = float(psd[bins[0]] * width)
        else:
            power = float(trapezoid(psd[bins], frequencies[bins]))
        centers.append(float(center))
        level = db(math.sqrt(max(power, 0.0)))
        levels.append(-120.0 if level is None else level)
    return centers, levels


def midi_note_events(path: Path, excerpt_start: float, excerpt_end: float) -> list[dict]:
    """Read note on/off groups from a rendered one-track stem MIDI."""
    midi = mido.MidiFile(path)
    tempo = 500_000
    seconds = 0.0
    active: dict[tuple[int, int], list[float]] = {}
    notes: list[tuple[float, float, int]] = []
    for message in mido.merge_tracks(midi.tracks):
        seconds += mido.tick2second(message.time, midi.ticks_per_beat, tempo)
        if message.type == "set_tempo":
            tempo = message.tempo
        elif message.type == "note_on" and message.velocity > 0:
            active.setdefault((message.channel, message.note), []).append(seconds)
        elif message.type in ("note_off", "note_on"):
            key = (message.channel, message.note)
            pending = active.get(key, [])
            if pending:
                notes.append((pending.pop(0), seconds, message.note))
    last = max((off for _, off, _ in notes), default=excerpt_end)
    for (channel, pitch), starts in active.items():
        notes.extend((start, max(last, start + 0.1), pitch) for start in starts)

    notes = [row for row in notes if excerpt_start <= row[0] < excerpt_end]
    notes.sort()
    groups: list[dict] = []
    for start, off, pitch in notes:
        local_start = start - excerpt_start
        local_off = max(local_start, min(off - excerpt_start, excerpt_end - excerpt_start))
        if groups and local_start - groups[-1]["start"] <= 0.035:
            groups[-1]["pitches"].append(int(pitch))
            groups[-1]["note_offs"].append(local_off)
        else:
            groups.append({"start": local_start, "pitches": [int(pitch)], "note_offs": [local_off]})
    for event in groups:
        event["note_off"] = float(np.median(event.pop("note_offs")))
    for index, event in enumerate(groups):
        event["end"] = groups[index + 1]["start"] if index + 1 < len(groups) else excerpt_end - excerpt_start
        event["source"] = "MIDI"
    return groups


def detected_note_events(audio: np.ndarray, sr: int) -> list[dict]:
    """Detect pluck/strum onsets in a reference excerpt."""
    hop = HOP_LENGTH
    strength = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop, n_fft=N_FFT)
    if not np.any(strength):
        return []
    frames = librosa.onset.onset_detect(
        onset_envelope=strength, sr=sr, hop_length=hop, units="frames",
        backtrack=False, wait=max(1, round(0.085 * sr / hop)), delta=0.1,
    )
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop)
    times = [float(t) for t in times if 0 <= t < len(audio) / sr]
    # Include a clearly active excerpt's first pluck when peak picking starts late.
    if len(audio) >= round(0.12 * sr):
        first_rms = float(np.sqrt(np.mean(np.square(audio[:round(0.06 * sr)]))))
        full_peak = float(np.max(np.abs(audio)))
        if first_rms > full_peak * 0.015 and (not times or times[0] > 0.12):
            times.insert(0, 0.0)
    merged: list[float] = []
    for time in times:
        if not merged or time - merged[-1] >= 0.07:
            merged.append(time)
    return [
        {
            "start": start,
            "note_off": merged[index + 1] if index + 1 < len(merged) else len(audio) / sr,
            "end": merged[index + 1] if index + 1 < len(merged) else len(audio) / sr,
            "pitches": [],
            "source": "onset detector",
        }
        for index, start in enumerate(merged)
    ]


def rms_envelope(audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    frame_length = round(0.05 * sr)
    values = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=HOP_LENGTH, center=True)[0]
    times = librosa.frames_to_time(np.arange(len(values)), sr=sr, hop_length=HOP_LENGTH)
    return times, values


def spectral_centroid(audio: np.ndarray, sr: int, low_hz: float, high_hz: float) -> tuple[np.ndarray, np.ndarray]:
    spectrum = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH, center=True))
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    selected = (frequencies >= low_hz) & (frequencies <= high_hz)
    weights = spectrum[selected]
    denominator = weights.sum(axis=0)
    centroid = np.divide(
        frequencies[selected] @ weights,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 1e-12,
    )
    times = librosa.frames_to_time(np.arange(len(centroid)), sr=sr, hop_length=HOP_LENGTH)
    return times, centroid


def window_rms(audio: np.ndarray, sr: int, start: float, end: float) -> float | None:
    left = max(0, round(start * sr))
    right = min(len(audio), round(end * sr))
    if right - left < max(32, round(0.008 * sr)):
        return None
    return float(np.sqrt(np.mean(np.square(audio[left:right]))))


def event_measurements(
    audio: np.ndarray, sr: int, events: list[dict],
    envelope_times: np.ndarray, envelope: np.ndarray,
    centroid_times: np.ndarray, centroids: np.ndarray,
) -> dict:
    duration = len(audio) / sr
    envelope_db = np.asarray([db(value) or -120.0 for value in envelope])
    floor_db = float(np.percentile(envelope_db, 10)) if len(envelope_db) else -120.0
    attack_ratios, attack_crests, attack_centroids, sustain_centroids = [], [], [], []
    t20_values, t30_values = [], []
    decay_curves = []
    centroid_curves = []
    curve_grid = np.arange(0.0, min(3.0, duration), 0.01)
    centroid_grid = np.arange(0.0, 0.31, HOP_LENGTH / sr)
    decay_note_rows = []
    isolated_candidate_count = 0

    for event in events:
        onset = float(event["start"])
        event_end = float(min(event.get("end", duration), duration))
        if onset < 0 or onset >= duration:
            continue
        note_off = float(min(max(event.get("note_off", event_end), onset), duration))

        attack_end = min(onset + 0.055, event_end, duration)
        sustain_start = min(onset + 0.075, event_end)
        sustain_end = min(onset + 0.25, event_end, duration)
        attack_rms = window_rms(audio, sr, onset, attack_end)
        sustain_rms = window_rms(audio, sr, sustain_start, sustain_end)
        if attack_rms and sustain_rms:
            attack_ratios.append(20 * math.log10(max(attack_rms, 1e-12) / max(sustain_rms, 1e-12)))
            left = max(0, round(onset * sr))
            right = min(len(audio), round(attack_end * sr))
            if right > left:
                attack_crests.append(20 * math.log10(max(float(np.max(np.abs(audio[left:right]))), 1e-12) / max(attack_rms, 1e-12)))

        attack_mask = (centroid_times >= onset) & (centroid_times < min(onset + 0.06, event_end))
        sustain_mask = (centroid_times >= onset + 0.08) & (centroid_times < min(onset + 0.25, event_end))
        if np.any(attack_mask):
            attack_centroids.append(float(np.median(centroids[attack_mask])))
        if np.any(sustain_mask):
            sustain_centroids.append(float(np.median(centroids[sustain_mask])))

        relative = centroid_grid + onset
        valid = (relative < event_end) & (relative < duration)
        curve = np.full_like(centroid_grid, np.nan)
        if np.any(valid):
            curve[valid] = np.interp(relative[valid], centroid_times, centroids)
        centroid_curves.append(curve)

        available_end = max(event_end, note_off)
        decay_end = min(available_end, onset + 3.0, duration)
        if decay_end - onset > 0.15:
            relative_grid = onset + curve_grid
            curve_valid = (relative_grid < decay_end) & (relative_grid < duration)
            env_curve = np.full_like(curve_grid, np.nan)
            if np.any(curve_valid):
                vals = np.interp(relative_grid[curve_valid], envelope_times, envelope)
                early_env = envelope[(envelope_times >= onset) & (envelope_times <= min(onset + 0.08, duration))]
                peak = float(np.max(early_env)) if len(early_env) else float(np.max(vals))
                env_curve[curve_valid] = 20 * np.log10(np.maximum(vals, 1e-12) / max(peak, 1e-12))
            decay_curves.append(env_curve)

        if event_end - onset >= 0.7:
            isolated_candidate_count += 1
            decay_mask = (envelope_times >= onset + 0.04) & (envelope_times < decay_end)
            if np.any(decay_mask):
                local_times = envelope_times[decay_mask] - onset
                local_db = envelope_db[decay_mask]
                early = (local_times >= 0) & (local_times <= 0.10)
                note_t20 = note_t30 = None
                if np.any(early):
                    early_indices = np.flatnonzero(early)
                    peak_index = int(early_indices[np.argmax(local_db[early])])
                    peak_db = float(local_db[peak_index])
                    relative_db = local_db - peak_db

                    def crossing(threshold: float, after_time: float | None = None) -> float | None:
                        if peak_db + threshold < floor_db + 5:
                            return None
                        hold = max(2, round(0.03 / (HOP_LENGTH / sr)))
                        for index in range(max(1, peak_index + 1), len(relative_db) - hold + 1):
                            if after_time is not None and local_times[index - 1] < after_time:
                                continue
                            if (
                                relative_db[index - 1] > threshold
                                and np.all(relative_db[index:index + hold] <= threshold)
                                and np.all(local_db[index:index + hold] >= floor_db + 5)
                            ):
                                previous, current = relative_db[index - 1:index + 1]
                                fraction = (threshold - previous) / (current - previous) if current != previous else 1.0
                                return float(local_times[index - 1] + fraction * (local_times[index] - local_times[index - 1]))
                        return None

                    start_decay = crossing(-5.0)
                    end_t20 = crossing(-25.0, start_decay)
                    end_t30 = crossing(-35.0, end_t20) if end_t20 is not None else None
                    if start_decay is not None and end_t20 is not None and end_t20 > start_decay:
                        note_t20 = end_t20 - start_decay
                        t20_values.append(note_t20)
                    if start_decay is not None and end_t30 is not None and end_t30 > start_decay:
                        note_t30 = end_t30 - start_decay
                        t30_values.append(note_t30)
                if note_t20 is not None or note_t30 is not None:
                    decay_note_rows.append({
                        "start_s": round(onset, 4),
                        "pitches_midi": event.get("pitches", []),
                        "t20_s": note_t20,
                        "t30_s": note_t30,
                    })

    def median(values: list[float]) -> float | None:
        return float(np.median(values)) if values else None

    def curve_summary(curves: list[np.ndarray], grid: np.ndarray) -> dict:
        if not curves:
            return {"time_s": grid.tolist(), "median": [], "p25": [], "p75": []}
        matrix = np.vstack(curves)
        summaries = {key: [] for key in ("median", "p25", "p75")}
        for column in matrix.T:
            values = column[np.isfinite(column)]
            summaries["median"].append(float(np.median(values)) if len(values) else None)
            summaries["p25"].append(float(np.percentile(values, 25)) if len(values) else None)
            summaries["p75"].append(float(np.percentile(values, 75)) if len(values) else None)
        return {"time_s": grid.tolist(), **summaries}

    return {
        "event_count": len(events),
        "attack_to_sustain_db": median(attack_ratios),
        "attack_crest_db": median(attack_crests),
        "attack_centroid_hz": median(attack_centroids),
        "sustain_centroid_hz": median(sustain_centroids),
        "centroid_change_hz": (
            median(sustain_centroids) - median(attack_centroids)
            if attack_centroids and sustain_centroids else None
        ),
        "t20_s": median(t20_values),
        "t20_notes": len(t20_values),
        "t30_s": median(t30_values),
        "t30_notes": len(t30_values),
        "isolated_decay_candidate_count": isolated_candidate_count,
        "isolated_decay_notes": decay_note_rows,
        "decay_curve": curve_summary(decay_curves, curve_grid),
        "centroid_curve": curve_summary(centroid_curves, centroid_grid),
        "noise_floor_dbfs_after_normalization": floor_db,
    }


def prepare(path: Path, time_range, low_hz: float, high_hz: float) -> dict:
    audio, start, end = load_excerpt(path, time_range)
    filtered = band_limit(audio, low_hz, high_hz, SAMPLE_RATE)
    normalized, measured_loudness, method = loudness_normalize(filtered, SAMPLE_RATE)
    centers, spectrum = third_octave_spectrum(normalized, SAMPLE_RATE, low_hz, high_hz)
    envelope_times, envelope = rms_envelope(normalized, SAMPLE_RATE)
    centroid_times, centroids = spectral_centroid(normalized, SAMPLE_RATE, low_hz, high_hz)
    return {
        "path": str(path),
        "time_range_s": [round(start, 3), round(end, 3)],
        "duration_s": round(len(audio) / SAMPLE_RATE, 3),
        "input_loudness_lufs": round(measured_loudness, 3),
        "normalization_method": method,
        "audio": normalized,
        "third_octave_hz": centers,
        "third_octave_db": spectrum,
        "envelope_times": envelope_times,
        "envelope": envelope,
        "centroid_times": centroid_times,
        "centroids": centroids,
    }


def plot_comparison(
    output: Path, instrument: str, reference_label: str,
    render: dict, reference: dict, render_features: dict, reference_features: dict,
    low_hz: float, high_hz: float,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 12), constrained_layout=True)
    axes[0].semilogx(render["third_octave_hz"], render["third_octave_db"], "o-", label="Rendered stem")
    axes[0].semilogx(reference["third_octave_hz"], reference["third_octave_db"], "o-", label=reference_label)
    axes[0].set_xlim(low_hz, high_hz)
    axes[0].set_ylabel("1/3-octave level (dB, normalized)")
    axes[0].set_title("Long-term average spectrum")
    axes[0].legend()
    axes[0].grid(True, which="both", alpha=0.25)

    for features, label in (
        (render_features, "Rendered stem"),
        (reference_features, reference_label),
    ):
        curve = features["decay_curve"]
        times = np.asarray(curve["time_s"])
        median = np.asarray([np.nan if value is None else value for value in curve["median"]])
        p25 = np.asarray([np.nan if value is None else value for value in curve["p25"]])
        p75 = np.asarray([np.nan if value is None else value for value in curve["p75"]])
        if np.any(np.isfinite(median)):
            axes[1].plot(times, median, label=label)
            axes[1].fill_between(times, p25, p75, alpha=0.14)
    axes[1].set_xlim(0, 3.0)
    axes[1].set_ylim(-65, 5)
    axes[1].set_xlabel("Time after detected note onset (s)")
    axes[1].set_ylabel("Level relative to attack peak (dB)")
    axes[1].set_title("Median note decay envelope (shading: interquartile range)")
    axes[1].grid(alpha=0.25)
    if axes[1].lines:
        axes[1].legend()

    for features, label in ((render_features, "Rendered stem"), (reference_features, reference_label)):
        curve = features["centroid_curve"]
        times = np.asarray(curve["time_s"])
        median = np.asarray([np.nan if value is None else value for value in curve["median"]])
        p25 = np.asarray([np.nan if value is None else value for value in curve["p25"]])
        p75 = np.asarray([np.nan if value is None else value for value in curve["p75"]])
        if np.any(np.isfinite(median)):
            axes[2].plot(times, median, label=label)
            axes[2].fill_between(times, p25, p75, alpha=0.14)
    axes[2].axvspan(0.0, 0.055, color="grey", alpha=0.1, label="attack window")
    axes[2].axvspan(0.08, 0.25, color="gold", alpha=0.1, label="sustain window")
    axes[2].set_xlim(0, 0.30)
    axes[2].set_xlabel("Time after detected note onset (s)")
    axes[2].set_ylabel("Spectral centroid (Hz)")
    axes[2].set_title("Note-life spectral centroid")
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    def fmt(value):
        return "n/a" if value is None else f"{value:.2f}"

    attack_line = (
        f"Attack/sustain dB: render {fmt(render_features['attack_to_sustain_db'])}, "
        f"reference {fmt(reference_features['attack_to_sustain_db'])}  ·  "
        f"attack crest dB: render {fmt(render_features['attack_crest_db'])}, "
        f"reference {fmt(reference_features['attack_crest_db'])}"
    )
    decay_line = (
        f"T20 s: render {fmt(render_features['t20_s'])} (n={render_features['t20_notes']}), "
        f"reference {fmt(reference_features['t20_s'])} (n={reference_features['t20_notes']})  ·  "
        f"T30 s: render {fmt(render_features['t30_s'])} (n={render_features['t30_notes']}), "
        f"reference {fmt(reference_features['t30_s'])} (n={reference_features['t30_notes']})"
    )
    fig.suptitle(
        f"{instrument}: rendered instrument vs {reference_label}\n{attack_line}\n{decay_line}",
        fontsize=10,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def analyze(args) -> dict:
    if not args.render.is_file() or not args.reference.is_file():
        raise FileNotFoundError(f"render/reference missing: {args.render}, {args.reference}")
    render_range = tuple(args.render_range) if args.render_range else None
    reference_range = tuple(args.reference_range) if args.reference_range else None
    low_hz, high_hz = args.bandwidth or ((100.0, 5000.0) if args.reference_type == "78" else (35.0, 10_000.0))
    render = prepare(args.render, render_range, low_hz, high_hz)
    reference = prepare(args.reference, reference_range, low_hz, high_hz)
    render_events = (
        midi_note_events(args.render_midi, render["time_range_s"][0], render["time_range_s"][1])
        if args.render_midi else []
    )
    if not render_events:
        render_events = detected_note_events(render["audio"], SAMPLE_RATE)
    reference_events = detected_note_events(reference["audio"], SAMPLE_RATE)
    render_features = event_measurements(
        render["audio"], SAMPLE_RATE, render_events,
        render["envelope_times"], render["envelope"],
        render["centroid_times"], render["centroids"],
    )
    reference_features = event_measurements(
        reference["audio"], SAMPLE_RATE, reference_events,
        reference["envelope_times"], reference["envelope"],
        reference["centroid_times"], reference["centroids"],
    )
    pitch_report = isolated_pitch_report(render["audio"], render_events, SAMPLE_RATE) if args.render_midi else None
    centers = render["third_octave_hz"]
    deltas = [a - b for a, b in zip(render["third_octave_db"], reference["third_octave_db"])]
    report = {
        "instrument": args.instrument,
        "reference_type": args.reference_type,
        "bandwidth_hz": [float(low_hz), float(high_hz)],
        "target_loudness_lufs": TARGET_LUFS,
        "loudness_normalization": "ITU-R BS.1770 integrated loudness; both excerpts normalized to the same target",
        "decay_method": "Measured time between sustained -5 and -25/-35 dB crossings from the attack peak; only notes with at least 0.7 s before the next onset are eligible, and results are omitted when the noise floor prevents the crossing.",
        "render": {
            "path": str(args.render),
            "midi": str(args.render_midi) if args.render_midi else None,
            "time_range_s": render["time_range_s"],
            "input_loudness_lufs": render["input_loudness_lufs"],
            "event_features": render_features,
            "pitch_accuracy": pitch_report,
        },
        "reference": {
            "path": str(args.reference),
            "time_range_s": reference["time_range_s"],
            "input_loudness_lufs": reference["input_loudness_lufs"],
            "event_features": reference_features,
        },
        "long_term_spectrum": {
            "center_hz": centers,
            "render_db": render["third_octave_db"],
            "reference_db": reference["third_octave_db"],
            "render_minus_reference_db": deltas,
        },
        "summary": {
            "spectrum_rmse_db": float(np.sqrt(np.mean(np.square(deltas)))) if deltas else None,
            "pitch_max_abs_cents": pitch_report["max_abs_cents"] if pitch_report else None,
            "pitch_notes": pitch_report["count"] if pitch_report else 0,
            "spectrum_delta_median_db": float(np.median(deltas)) if deltas else None,
            "spectrum_delta_max_abs_db": float(np.max(np.abs(deltas))) if deltas else None,
            "centroid_attack_hz": {
                "render": render_features["attack_centroid_hz"],
                "reference": reference_features["attack_centroid_hz"],
            },
            "centroid_sustain_hz": {
                "render": render_features["sustain_centroid_hz"],
                "reference": reference_features["sustain_centroid_hz"],
            },
            "attack_to_sustain_db": {
                "render": render_features["attack_to_sustain_db"],
                "reference": reference_features["attack_to_sustain_db"],
            },
            "attack_crest_db": {
                "render": render_features["attack_crest_db"],
                "reference": reference_features["attack_crest_db"],
            },
            "decay_note_counts": {
                "render": {"t20": render_features["t20_notes"], "t30": render_features["t30_notes"]},
                "reference": {"t20": reference_features["t20_notes"], "t30": reference_features["t30_notes"]},
            },
            "t20_s": {"render": render_features["t20_s"], "reference": reference_features["t20_s"]},
            "t30_s": {"render": render_features["t30_s"], "reference": reference_features["t30_s"]},
        },
        "plot": str(args.output),
    }
    plot_comparison(
        args.output, args.instrument, args.reference_label or args.reference.stem,
        render, reference, render_features, reference_features, low_hz, high_hz,
    )
    json_path = args.output.with_suffix(".json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
    print(f"plot: {args.output}\nmetrics: {json_path}")
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instrument", required=True, choices=("banjo", "guitar", "bass"))
    parser.add_argument("--render", type=Path, required=True, help="rendered mono/stereo stem WAV")
    parser.add_argument("--render-midi", type=Path, help="stem MIDI, used to locate exact rendered note onsets")
    parser.add_argument("--reference", type=Path, required=True, help="licensed reference audio clip")
    parser.add_argument("--reference-type", choices=("modern", "78"), default="modern")
    parser.add_argument("--reference-label", help="legend label, e.g. 'reference recording'")
    parser.add_argument("--reference-range", type=float, nargs=2, metavar=("START", "END"), help="seconds in the source recording")
    parser.add_argument("--render-range", type=float, nargs=2, metavar=("START", "END"), help="seconds in the rendered stem")
    parser.add_argument("--bandwidth", type=float, nargs=2, metavar=("LOW_HZ", "HIGH_HZ"), help="override default analysis band")
    parser.add_argument("--output", type=Path, required=True, help="overlay PNG; a sibling JSON report is also written")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        analyze(args)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise SystemExit(f"compare_instrument.py: {error}") from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
