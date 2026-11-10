#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["librosa>=0.10", "matplotlib>=3.8", "mido>=1.3", "numpy", "scipy", "soundfile"]
# ///
"""Measure roughness, harmony fit, tune masking, and groove in rendered audio.

Usage:
  uv run scripts/listen.py MIX.mp3 [--midi performed.mid | --score score.ly]
      [--stems [NAME=]STEM.wav ...] [--melody STEM]
      [--harmony chords.json] [--out listen.png] [--json listen.json]

With no paths after --stems, stem WAVs are read from qa.json beside the mix.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict, deque
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mido
import numpy as np
from scipy.signal import find_peaks

SR = 22_050
HOP = 256
GROOVE_HOP = 128
PC_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
NOTE_PC = {name: index for index, name in enumerate(PC_NAMES)}
NOTE_PC.update({"Db": 1, "Eb": 3, "Gb": 6, "Ab": 8, "Bb": 10})


def load_audio(path: str | Path, start: float | None = None,
               end: float | None = None) -> tuple[np.ndarray, int]:
    offset = max(0.0, start or 0.0)
    duration = None if end is None else max(0.0, end - offset)
    y, sr = librosa.load(path, sr=SR, mono=True, offset=offset, duration=duration)
    if not len(y):
        raise ValueError(f"audio window is empty: {path}")
    return y, sr


def _tempo_map(midi: mido.MidiFile, total_tick: int):
    changes: dict[int, int] = {0: 500_000}
    signatures: dict[int, tuple[int, int]] = {0: (4, 4)}
    track_data = []
    for track_index, track in enumerate(midi.tracks):
        tick = 0
        name = f"track-{track_index + 1}"
        raw_notes: list[tuple[int, int, int, int]] = []
        active: dict[tuple[int, int], deque[tuple[int, int]]] = defaultdict(deque)
        for order, message in enumerate(track):
            tick += message.time
            if message.type == "track_name":
                name = message.name or name
            elif message.type == "set_tempo":
                changes[tick] = message.tempo
            elif message.type == "time_signature":
                signatures[tick] = (message.numerator, message.denominator)
            elif message.type == "note_on" and message.velocity > 0:
                active[(message.channel, message.note)].append((tick, message.velocity))
            elif message.type == "note_off" or (
                message.type == "note_on" and message.velocity == 0
            ):
                queue = active[(message.channel, message.note)]
                if queue:
                    onset, velocity = queue.popleft()
                    raw_notes.append((onset, max(onset, tick), message.note, velocity))
        for (_, pitch), queue in active.items():
            for onset, velocity in queue:
                raw_notes.append((onset, max(onset, total_tick), pitch, velocity))
        track_data.append({"name": name, "index": track_index, "raw_notes": raw_notes})

    tempo_ticks = sorted(changes)
    tempo_values = [changes[tick] for tick in tempo_ticks]
    cumulative = [0.0]
    for index in range(1, len(tempo_ticks)):
        cumulative.append(
            cumulative[-1] + (tempo_ticks[index] - tempo_ticks[index - 1])
            * tempo_values[index - 1] / (midi.ticks_per_beat * 1_000_000)
        )

    def seconds_at(tick: int | float) -> float:
        index = max(0, bisect_right(tempo_ticks, tick) - 1)
        return cumulative[index] + (tick - tempo_ticks[index]) * tempo_values[index] / (
            midi.ticks_per_beat * 1_000_000
        )

    return signatures, track_data, seconds_at


def read_midi(path: str | Path, offset: float = 0.0) -> dict[str, Any]:
    midi = mido.MidiFile(path)
    total_tick = 0
    for track in midi.tracks:
        total_tick = max(total_tick, sum(message.time for message in track))
    signatures, track_data, seconds_at = _tempo_map(midi, total_tick)
    for track in track_data:
        track["notes"] = [
            {"start": seconds_at(start) + offset, "end": seconds_at(end) + offset,
             "pitch": pitch, "velocity": velocity}
            for start, end, pitch, velocity in track.pop("raw_notes")
        ]

    signature_ticks = sorted(signatures)
    beats = []
    tick = 0
    bar = 1
    beat_number = 1
    last_signature = signatures[signature_ticks[0]]
    while tick < total_tick:
        sig_index = max(0, bisect_right(signature_ticks, tick) - 1)
        numerator, denominator = signatures[signature_ticks[sig_index]]
        signature = (numerator, denominator)
        if signature != last_signature:
            if beat_number != 1:
                bar += 1
            beat_number = 1
            last_signature = signature
        beat_ticks = max(1, round(midi.ticks_per_beat * 4 / denominator))
        next_change = next((value for value in signature_ticks if value > tick), total_tick)
        end_tick = min(tick + beat_ticks, next_change, total_tick)
        if end_tick <= tick:
            tick = next_change
            continue
        beats.append({
            "start": seconds_at(tick) + offset,
            "end": seconds_at(end_tick) + offset,
            "bar": bar,
            "beat": beat_number,
            "beats_per_bar": numerator,
            "denominator": denominator,
        })
        tick = end_tick
        if tick == next_change and tick < total_tick:
            if beat_number != 1:
                bar += 1
            beat_number = 1
            last_signature = signatures[next_change]
        else:
            beat_number += 1
            if beat_number > numerator:
                beat_number = 1
                bar += 1
    return {"beats": beats, "tracks": track_data, "duration": seconds_at(total_tick) + offset}


def compile_score(path: str | Path) -> Path:
    score = Path(path).resolve()
    with tempfile.TemporaryDirectory(prefix="listen-score-") as directory:
        output = Path(directory) / score.stem
        process = subprocess.run(
            ["lilypond", f"--output={output}", str(score)], cwd=score.parent,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        midi_path = output.with_suffix(".midi")
        if process.returncode or not midi_path.is_file():
            raise ValueError(f"could not compile score MIDI with LilyPond:\n{process.stdout}")
        target = Path(tempfile.gettempdir()) / f"listen-score-{score.stem}-{midi_path.stat().st_mtime_ns}.midi"
        target.write_bytes(midi_path.read_bytes())
    return target


def spectral_roughness(y: np.ndarray, sr: int, hop: int = HOP,
                       n_fft: int = 8192, max_peaks: int = 32,
                       return_chroma: bool = False):
    """Sethares roughness and harmonic-reduced pitch-class energy from spectral peaks."""
    spectrum = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop, window="hann"))
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    values = np.zeros(spectrum.shape[1], dtype=float)
    chroma = np.zeros((12, spectrum.shape[1]), dtype=float) if return_chroma else None
    for frame_index, magnitudes in enumerate(spectrum.T):
        peak_indices, _ = find_peaks(magnitudes, distance=2)
        if not len(peak_indices):
            continue
        peak_indices = peak_indices[(frequencies[peak_indices] >= 35)
                                    & (frequencies[peak_indices] <= 10_000)]
        if not len(peak_indices):
            continue
        floor = magnitudes.max() * 10 ** (-36 / 20)
        peak_indices = peak_indices[magnitudes[peak_indices] >= floor]
        if not len(peak_indices):
            continue
        if len(peak_indices) > max_peaks:
            keep = np.argpartition(magnitudes[peak_indices], -max_peaks)[-max_peaks:]
            peak_indices = peak_indices[keep]
        peak_indices = peak_indices[np.argsort(frequencies[peak_indices])]
        freqs = frequencies[peak_indices]
        amps = magnitudes[peak_indices]
        normalized = amps / max(amps.sum(), 1e-12)
        left, right = np.triu_indices(len(freqs), k=1)
        if len(left):
            low = np.minimum(freqs[left], freqs[right])
            separation = np.abs(freqs[left] - freqs[right])
            scale = 0.24 / (0.021 * low + 19.0)
            dissonance = np.exp(-3.5 * scale * separation) - np.exp(-5.75 * scale * separation)
            values[frame_index] = np.sum(
                normalized[left] * normalized[right] * np.maximum(dissonance, 0)
            )
        if chroma is not None:
            fundamentals: list[tuple[float, float]] = []
            for frequency, amplitude in zip(freqs, amps):
                if frequency > 5_000:
                    continue
                harmonic = any(
                    2 <= round(frequency / base) <= 12
                    and abs(frequency - round(frequency / base) * base)
                    <= max(2 * sr / n_fft, base * 0.025)
                    and base_amp >= amplitude * 0.08
                    for base, base_amp in fundamentals
                )
                if not harmonic:
                    fundamentals.append((float(frequency), float(amplitude)))
                    midi_pitch = 69 + 12 * math.log2(float(frequency) / 440.0)
                    pitch_class = int(round(midi_pitch)) % 12
                    chroma[pitch_class, frame_index] += amplitude ** 2
    times = np.arange(len(values)) * hop / sr
    return (times, values, chroma) if return_chroma else (times, values)


def _pc_values(values: Any) -> set[int]:
    if isinstance(values, str):
        match = re.fullmatch(r"([A-Ga-g])([#b]?)(.*)", values.strip())
        if not match:
            raise ValueError(f"invalid chord symbol: {values!r}")
        root = NOTE_PC[match.group(1).upper() + match.group(2)]
        quality = match.group(3).lower().replace(" ", "")
        qualities = {
            "": (0, 4, 7), "maj": (0, 4, 7), "m": (0, 3, 7), "min": (0, 3, 7),
            "7": (0, 4, 7, 10), "dom7": (0, 4, 7, 10), "maj7": (0, 4, 7, 11),
            "m7": (0, 3, 7, 10), "min7": (0, 3, 7, 10), "mmaj7": (0, 3, 7, 11),
            "dim": (0, 3, 6), "°": (0, 3, 6), "m7b5": (0, 3, 6, 10),
            "ø": (0, 3, 6, 10), "aug": (0, 4, 8), "+": (0, 4, 8),
            "sus2": (0, 2, 7), "sus4": (0, 5, 7),
        }
        if quality not in qualities:
            raise ValueError(f"unsupported chord quality in {values!r}; provide pitch classes instead")
        return {(root + interval) % 12 for interval in qualities[quality]}
    if not isinstance(values, list):
        raise ValueError(f"chord pitch classes must be a list or chord symbol, got {values!r}")
    pcs = set()
    for value in values:
        if isinstance(value, int) and 0 <= value <= 11:
            pcs.add(value)
        elif isinstance(value, str):
            token = value.strip()
            if token.isdigit() and 0 <= int(token) <= 11:
                pcs.add(int(token))
            elif token in NOTE_PC:
                pcs.add(NOTE_PC[token])
            else:
                raise ValueError(f"unknown pitch class: {value!r}")
        else:
            raise ValueError(f"invalid pitch class: {value!r}")
    if not pcs:
        raise ValueError("chord pitch-class list cannot be empty")
    return pcs


def load_chords(path: str | Path) -> dict[tuple[int, int], set[int]]:
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        entries = []
        for key, value in data.items():
            match = re.fullmatch(r"(\d+)\s*[:/]\s*(\d+)", str(key))
            if not match:
                raise ValueError(f"chord key must be 'bar:beat', got {key!r}")
            entries.append((int(match.group(1)), int(match.group(2)), value))
    elif isinstance(data, list):
        entries = [(int(item["bar"]), int(item["beat"]), item.get("pcs", item.get("chord")))
                   for item in data]
    else:
        raise ValueError("chord file must be a JSON object or list")
    return {(bar, beat): _pc_values(value) for bar, beat, value in entries}


def expected_harmony(midi_info: dict[str, Any] | None,
                     beats: list[dict[str, Any]], harmony_tracks: list[str] | None,
                     chords: dict[tuple[int, int], set[int]] | None):
    result = []
    selected_tracks = []
    if midi_info and chords is None:
        for track in midi_info["tracks"]:
            if not track["notes"]:
                continue
            if harmony_tracks and not any(
                selector.casefold() in track["name"].casefold()
                for selector in harmony_tracks
            ):
                continue
            selected_tracks.append(track)
        if harmony_tracks and not selected_tracks:
            raise ValueError(f"no MIDI tracks matched harmony selectors: {', '.join(harmony_tracks)}")
    for beat in beats:
        if chords is not None:
            pcs = chords.get((beat["bar"], beat["beat"]), set())
        elif midi_info:
            pcs = {
                note["pitch"] % 12
                for track in selected_tracks
                for note in track["notes"]
                if note["start"] < beat["end"] and note["end"] > beat["start"]
            }
        else:
            pcs = set()
        result.append((beat, pcs))
    source = "explicit chord list" if chords is not None else (
        "MIDI pitch content" + (f" ({', '.join(harmony_tracks)})" if harmony_tracks else "")
        if midi_info else None
    )
    return result, source


def chroma_fit(chroma: np.ndarray, times: np.ndarray,
               expected: list[tuple[dict[str, Any], set[int]]],
               threshold: float) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    rows = []
    for beat, pcs in expected:
        mask = (times >= beat["start"]) & (times < beat["end"])
        frame_energy = chroma[:, mask].sum(axis=1) if mask.any() else np.zeros(12)
        total = float(frame_energy.sum())
        if not pcs or total <= 1e-15:
            continue
        outside = [pc for pc in range(12) if pc not in pcs]
        misfit = float(frame_energy[outside].sum() / total)
        ranked = sorted(outside, key=lambda pc: frame_energy[pc], reverse=True)
        offending = [PC_NAMES[pc] for pc in ranked
                     if frame_energy[pc] >= max(frame_energy[outside], default=0) * 0.25
                     and frame_energy[pc] / total >= 0.01][:4]
        rows.append({
            "bar": beat["bar"], "beat": beat["beat"], "start_seconds": round(beat["start"], 3),
            "misfit_fraction": round(misfit, 5), "expected": [PC_NAMES[pc] for pc in sorted(pcs)],
            "offending": offending, "flagged": misfit >= threshold,
        })
    if not rows:
        return None, rows
    flagged = [row for row in rows if row["flagged"]]
    worst = max(rows, key=lambda row: row["misfit_fraction"])
    by_bar: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        by_bar[row["bar"]].append(row["misfit_fraction"])
    summary = {
        "threshold_fraction": threshold,
        "source": None,
        "beats_analyzed": len(rows),
        "flagged_beats": len(flagged),
        "flagged_fraction": len(flagged) / len(rows),
        "mean_misfit_fraction": float(np.mean([row["misfit_fraction"] for row in rows])),
        "method": "pitch-class energy from independent spectral peaks; harmonic partials are suppressed",
        "worst": worst,
        "bars": [{"bar": bar, "mean_misfit_fraction": float(np.mean(values)),
                  "max_misfit_fraction": max(values)} for bar, values in sorted(by_bar.items())],
        "beats": rows,
    }
    return summary, rows


def beat_curve_summary(times: np.ndarray, values: np.ndarray,
                       beats: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for beat in beats:
        mask = (times >= beat["start"]) & (times < beat["end"])
        if mask.any():
            value = float(np.mean(values[mask]))
            summaries.append({"bar": beat["bar"], "beat": beat["beat"],
                              "start_seconds": round(beat["start"], 3), "mean": value})
    return summaries


def roughness_summary(times: np.ndarray, values: np.ndarray,
                      beats: list[dict[str, Any]]) -> dict[str, Any]:
    if not len(values):
        return {"frames": 0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0,
                "peaks": [], "beats": []}
    p50, p90, p95, p99 = np.percentile(values, [50, 90, 95, 99])
    min_distance = max(1, round(0.6 / max(np.median(np.diff(times)) if len(times) > 1 else 1, 1e-6)))
    peaks, _ = find_peaks(values, distance=min_distance,
                          prominence=max(float(p95 - p50) * 0.08, 1e-7))
    peaks = [int(index) for index in peaks if values[index] >= p90]
    if not peaks:
        peaks = [int(np.argmax(values))]
    peak_rows = [{"time_seconds": round(float(times[index]), 3),
                  "roughness": round(float(values[index]), 6)}
                 for index in sorted(peaks, key=lambda index: values[index], reverse=True)[:12]]
    return {
        "method": "Sethares-weighted pairwise spectral peaks; normalized peak amplitudes",
        "frames": len(values), "p50": float(p50), "p90": float(p90),
        "p95": float(p95), "p99": float(p99), "max": float(values.max()), "peaks": peak_rows,
        "beats": beat_curve_summary(times, values, beats),
    }


def erb_centers_and_weights(sr: int, n_fft: int, bands: int = 32,
                            low_hz: float = 80, high_hz: float = 10_000):
    def rate(frequency):
        return 21.4 * np.log10(1 + 0.00437 * frequency)
    def inverse(value):
        return (10 ** (value / 21.4) - 1) / 0.00437
    center_rates = np.linspace(rate(low_hz), rate(high_hz), bands)
    centers = inverse(center_rates)
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    rates = rate(np.maximum(frequencies, 0))
    spacing = center_rates[1] - center_rates[0]
    weights = np.maximum(0.0, 1.0 - np.abs(rates[None, :] - center_rates[:, None]) / spacing)
    weights[:, frequencies < low_hz] = 0
    weights[:, frequencies > high_hz] = 0
    sums = weights.sum(axis=1, keepdims=True)
    weights = np.divide(weights, sums, out=np.zeros_like(weights), where=sums > 0)
    return centers, weights


def stem_specs(audio: str | Path, stem_arguments: list[str] | None,
               qa_path: str | Path | None) -> list[dict[str, Any]]:
    specs = []
    if stem_arguments:
        for argument in stem_arguments:
            if "=" in argument:
                name, path = argument.split("=", 1)
            else:
                path = argument
                name = Path(path).stem
            specs.append({"name": name, "path": Path(path), "aliases": [name, Path(path).stem]})
    elif stem_arguments is not None or qa_path:
        metadata = Path(qa_path) if qa_path else Path(audio).parent / "qa.json"
        if not metadata.is_file():
            raise ValueError(f"cannot read stems: no QA file at {metadata}; pass WAV paths to --stems")
        qa = json.loads(metadata.read_text())
        for key, item in qa.get("stems", {}).items():
            wav = Path(item["wav"])
            if not wav.is_absolute():
                wav = metadata.parent / wav
            specs.append({"name": key, "path": wav,
                          "aliases": [key, item.get("track_name", ""), wav.stem],
                          "track_index": item.get("track_index")})
    return specs


def resolve_stem(selector: str, specs: list[dict[str, Any]]) -> dict[str, Any]:
    needle = selector.casefold()
    matches = [spec for spec in specs
               if any(needle == alias.casefold() or needle in alias.casefold()
                      for alias in spec["aliases"] if alias)]
    if len(matches) != 1:
        choices = ", ".join(spec["name"] for spec in specs)
        raise ValueError(f"melody stem {selector!r} matched {len(matches)} stems; available: {choices}")
    return matches[0]


def resolve_melody(selector, specs):
    """Resolve one player or '+'-joined players carrying one melodic role."""
    members=[resolve_stem(name.strip(),specs) for name in selector.split('+')]
    if len(members)==1:return members[0]
    if len({m['name'] for m in members})!=len(members):raise ValueError('duplicate melody player')
    return dict(name='+'.join(m['name'] for m in members),members=members)


def parse_melody_sections(melody: str | None, sections: list[str],
                          specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    if melody:
        result.append({"start": -math.inf, "end": math.inf,
                       "stem": resolve_melody(melody, specs)})
    for value in sections:
        try:
            start, end, name = value.split(":", 2)
            start_value, end_value = float(start), float(end)
            if end_value <= start_value:
                raise ValueError
        except ValueError as error:
            raise ValueError("--melody-section must be START:END:STEM with END > START") from error
        result.append({"start": start_value, "end": end_value,
                       "stem": resolve_melody(name, specs)})
    if not result:
        raise ValueError("tune masking needs --melody or at least one --melody-section")
    return result


def analyze_masking(specs: list[dict[str, Any]], melody: str | None,
                    sections: list[str], sr: int, audio_start: float,
                    audio_end: float | None, midi_offset: float,
                    margin_db: float, audibility_threshold: float,
                    min_passage_s: float, n_fft: int = 4096, hop: int = HOP):
    if len(specs) < 1:
        return None, None
    segments = parse_melody_sections(melody, sections, specs)
    first_source = max(0.0, (audio_start or 0.0) - midi_offset)
    source_end = None if audio_end is None else max(first_source, audio_end - midi_offset)
    centers, weights = erb_centers_and_weights(sr, n_fft)
    powers = {}
    frame_times = None
    for spec in specs:
        y, _ = load_audio(spec["path"], first_source if first_source else None, source_end)
        stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)) ** 2
        bands = weights @ stft
        times = np.arange(bands.shape[1]) * hop / sr + first_source + midi_offset
        if frame_times is None:
            frame_times = times
        else:
            common = min(len(frame_times), len(times))
            frame_times = frame_times[:common]
            for key in powers:
                powers[key] = powers[key][:, :common]
            bands = bands[:, :common]
        powers[spec["name"]] = bands
    assert frame_times is not None
    common = min(len(frame_times), *(array.shape[1] for array in powers.values()))
    frame_times = frame_times[:common]
    powers = {name: array[:, :common] for name, array in powers.items()}
    ratios = np.full((len(centers), common), np.nan)
    audibility = np.full(common, np.nan)
    melody_labels = []
    epsilon = 1e-15
    for section in segments:
        spec = section["stem"]
        melody_labels.append(spec["name"])
        names=[m['name'] for m in spec.get('members',[spec])]
        mel_power = sum(powers[name] for name in names)
        other_names = [name for name in powers if name not in names]
        other_power = sum((powers[name] for name in other_names), np.zeros_like(mel_power))
        ratio = 10 * np.log10((mel_power + epsilon) / (other_power + epsilon))
        select = (frame_times >= max(section["start"], audio_start or 0.0)) & (
            frame_times < min(section["end"], audio_end if audio_end is not None else math.inf)
        )
        ratios[:, select] = ratio[:, select]
        for index in np.flatnonzero(select):
            frame = mel_power[:, index]
            maximum = frame.max()
            if maximum <= epsilon:
                audibility[index] = 0.0
                continue
            loudest = frame >= maximum * 10 ** (-20 / 10)
            audibility[index] = float(np.mean(ratio[loudest, index] >= margin_db)) if loudest.any() else 0.0
    valid = np.isfinite(audibility)
    if not valid.any():
        return None, None
    masked = valid & (audibility < audibility_threshold)
    passages = []
    index = 0
    while index < len(masked):
        if not masked[index]:
            index += 1
            continue
        end = index + 1
        while end < len(masked) and masked[end]:
            end += 1
        duration = (end - index) * hop / sr
        if duration >= min_passage_s:
            passages.append({"start_seconds": round(float(frame_times[index]), 3),
                             "end_seconds": round(float(frame_times[min(end - 1, len(frame_times) - 1)]), 3),
                             "duration_seconds": round(duration, 3)})
        index = end
    summary = {
        "melody_stems": list(dict.fromkeys(melody_labels)),
        "margin_db": margin_db,
        "audibility_threshold": audibility_threshold,
        "mean_audibility_fraction": float(np.nanmean(audibility)),
        "masked_fraction": float(masked.sum() / valid.sum()),
        "masked_passages": passages,
        "erb_centers_hz": [round(float(value), 1) for value in centers],
    }
    return summary, {"times": frame_times, "centers": centers, "ratio_db": ratios,
                     "audibility": audibility}


def _grid_from_bpm(start: float, end: float, bpm: float) -> list[dict[str, Any]]:
    duration = 60.0 / bpm
    beats = []
    index = 0
    while start + index * duration < end:
        beats.append({"start": start + index * duration,
                      "end": start + (index + 1) * duration,
                      "bar": index // 4 + 1, "beat": index % 4 + 1,
                      "beats_per_bar": 4, "denominator": 4})
        index += 1
    return beats


def estimate_beat_grid(onset_envelope: np.ndarray, sr: int, hop: int,
                       start: float, end: float, offset: float = 0.0):
    _, frames = librosa.beat.beat_track(onset_envelope=onset_envelope, sr=sr,
                                        hop_length=hop, trim=False)
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop) + offset
    times = times[(times >= start) & (times < end)]
    if len(times) < 2:
        return []
    durations = np.diff(times)
    fallback = float(np.median(durations))
    return [{"start": float(time),
             "end": float(times[index + 1]) if index + 1 < len(times) else float(time + fallback),
             "bar": index // 4 + 1, "beat": index % 4 + 1,
             "beats_per_bar": 4, "denominator": 4}
            for index, time in enumerate(times)]


def detect_onsets(y: np.ndarray, sr: int, hop: int = GROOVE_HOP):
    envelope = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop,
                                             n_fft=2048, aggregate=np.median)
    if not len(envelope) or envelope.max() <= 1e-12:
        return np.array([]), envelope
    threshold = max(float(np.percentile(envelope, 35)), float(envelope.max()) * 0.035)
    peak_indices, properties = find_peaks(
        envelope, height=threshold,
        distance=max(1, round(0.045 * sr / hop)),
        prominence=max(float(envelope.max()) * 0.025, 1e-9),
    )
    times = librosa.frames_to_time(peak_indices, sr=sr, hop_length=hop)
    return times, envelope


def _grid_positions(beats: list[dict[str, Any]], subdivisions: int = 2):
    positions = []
    for beat in beats:
        duration = beat["end"] - beat["start"]
        for subdivision in range(subdivisions):
            positions.append({
                "time": beat["start"] + duration * subdivision / subdivisions,
                "bar": beat["bar"], "beat": beat["beat"],
                "beats_per_bar": beat["beats_per_bar"],
                "subdivision": subdivision,
                "slot": (beat["bar"], (beat["beat"] - 1) * subdivisions + subdivision),
                "profile_index": (beat["beat"] - 1) * subdivisions + subdivision,
                "beat_start": beat["start"], "beat_end": beat["end"],
            })
    return positions


def assign_onsets(onset_times: np.ndarray, envelope: np.ndarray, hop: int, sr: int,
                  grid_positions: list[dict[str, Any]], offset: float, source_start: float):
    if not len(grid_positions):
        return []
    grid_times = np.array([position["time"] for position in grid_positions])
    assigned = []
    for time in onset_times + source_start + offset:
        index = int(np.argmin(np.abs(grid_times - time)))
        position = grid_positions[index]
        tolerance = (position["beat_end"] - position["beat_start"]) * 0.23
        if abs(time - position["time"]) > tolerance:
            continue
        frame = min(len(envelope) - 1,
                    max(0, round((time - offset - source_start) * sr / hop)))
        assigned.append({**position, "grid_time": position["time"], "time": float(time),
                         "residual": float(time - position["time"]),
                         "strength": float(envelope[frame])})
    # A strum may generate several attacks for a single slot; keep the clearest grid-near attack.
    unique = {}
    for event in assigned:
        old = unique.get(event["slot"])
        if old is None or abs(event["residual"]) < abs(old["residual"]):
            unique[event["slot"]] = event
    return sorted(unique.values(), key=lambda item: item["time"])


def midi_track_for_stem(spec: dict[str, Any], midi_info: dict[str, Any] | None):
    if not midi_info:
        return None
    track_index = spec.get("track_index")
    if track_index is not None:
        match = next((track for track in midi_info["tracks"] if track["index"] == track_index), None)
        if match:
            return match
    aliases = [alias.casefold() for alias in spec.get("aliases", []) if alias]
    matches = [track for track in midi_info["tracks"] if track["notes"] and any(
        alias == track["name"].casefold() or alias in track["name"].casefold()
        or track["name"].casefold() in alias for alias in aliases
    )]
    return matches[0] if len(matches) == 1 else None


def match_audio_to_midi(onset_times: np.ndarray, envelope: np.ndarray, sr: int, hop: int,
                        grid_positions: list[dict[str, Any]], midi_track: dict[str, Any],
                        offset: float, source_start: float):
    if not len(onset_times):
        return []
    grid_times = np.array([position["time"] for position in grid_positions])
    targets = []
    for note in sorted(midi_track["notes"], key=lambda item: item["start"]):
        start = note["start"]
        if targets and start - targets[-1][-1] <= 0.025:
            targets[-1].append(start)
        else:
            targets.append([start])
    observed = onset_times + source_start + offset
    events = []
    for group in targets:
        target = float(np.mean(group))
        grid_index = int(np.argmin(np.abs(grid_times - target)))
        position = grid_positions[grid_index]
        grid_tolerance = (position["beat_end"] - position["beat_start"]) * 0.23
        if abs(target - position["time"]) > grid_tolerance:
            continue
        observed_index = int(np.argmin(np.abs(observed - target)))
        actual = float(observed[observed_index])
        if abs(actual - target) > max(0.12, grid_tolerance):
            continue
        frame = min(len(envelope) - 1,
                    max(0, round((actual - offset - source_start) * sr / hop)))
        events.append({**position, "grid_time": position["time"], "time": actual,
                       "residual": actual - position["time"],
                       "strength": float(envelope[frame])})
    unique = {}
    for event in events:
        old = unique.get(event["slot"])
        if old is None or abs(event["residual"]) < abs(old["residual"]):
            unique[event["slot"]] = event
    return sorted(unique.values(), key=lambda item: item["time"])


def swing_ratios_from_events(events: list[dict[str, Any]], beats: list[dict[str, Any]],
                             subdivisions: int = 2):
    by_position = {
        (event["bar"], event["beat"], event["subdivision"]): event
        for event in events
    }
    ratios = []
    offbeat_position = subdivisions // 2
    if offbeat_position == 0 or subdivisions % 2:
        return ratios
    for index, beat in enumerate(beats[:-1]):
        following = beats[index + 1]
        first = by_position.get((beat["bar"], beat["beat"], 0))
        offbeat = by_position.get((beat["bar"], beat["beat"], offbeat_position))
        next_downbeat = by_position.get((following["bar"], following["beat"], 0))
        if first and offbeat and next_downbeat:
            long = offbeat["time"] - first["time"]
            short = next_downbeat["time"] - offbeat["time"]
            if long > 0 and short > 0:
                ratios.append(long / short)
    return ratios



def analyze_groove(specs: list[dict[str, Any]], beats: list[dict[str, Any]],
                   sr: int, audio_start: float, audio_end: float | None,
                   midi_offset: float, subdivisions: int = 2,
                   grid_source: str = "MIDI time signature and tempo map",
                   midi_info: dict[str, Any] | None = None):
    if not specs:
        return None, None
    audio_start = audio_start or 0.0
    grid_positions = _grid_positions(beats, subdivisions)
    if not grid_positions:
        return None, None
    per_stem = {}
    assigned_by_stem = {}
    plot_curves = []
    source_start = max(0.0, audio_start - midi_offset)
    for spec in specs:
        y, _ = load_audio(spec["path"], source_start or None,
                          None if audio_end is None else max(0.0, audio_end - midi_offset))
        onset_times, envelope = detect_onsets(y, sr)
        midi_track = midi_track_for_stem(spec, midi_info)
        if midi_track:
            assigned = match_audio_to_midi(onset_times, envelope, sr, GROOVE_HOP,
                                           grid_positions, midi_track, midi_offset, source_start)
        else:
            assigned = assign_onsets(onset_times, envelope, GROOVE_HOP, sr,
                                     grid_positions, midi_offset, source_start)
        assigned_by_stem[spec["name"]] = assigned
        if len(envelope):
            curve_times = librosa.frames_to_time(np.arange(len(envelope)), sr=sr,
                                                  hop_length=GROOVE_HOP) + source_start + midi_offset
            plot_curves.append((curve_times, envelope, spec["name"]))
        profile_size = max((position["profile_index"] for position in grid_positions), default=0) + 1
        profile_values: list[list[float]] = [[] for _ in range(profile_size)]
        for event in assigned:
            profile_values[event["profile_index"]].append(event["strength"])
        profile = [float(np.mean(values)) if values else 0.0 for values in profile_values]
        active_profile = np.asarray([value for value in profile if value > 0])
        accent_cv = (float(active_profile.std() / active_profile.mean())
                     if len(active_profile) > 1 and active_profile.mean() > 0 else 0.0)
        events_by_bar: dict[int, list[dict[str, Any]]] = defaultdict(list)
        residuals_by_position: dict[tuple[int, int], list[float]] = defaultdict(list)
        accents_by_bar: dict[int, dict[int, float]] = defaultdict(dict)
        for event in assigned:
            events_by_bar[event["bar"]].append(event)
            residuals_by_position[(event["beat"], event["subdivision"])].append(
                event["residual"] * 1000
            )
            accents_by_bar[event["bar"]][event["profile_index"]] = event["strength"]
        timing_variances = [float(np.var(values)) for values in residuals_by_position.values()
                            if len(values) > 1]
        interval_residuals: dict[tuple[tuple[int, int], tuple[int, int]], list[float]] = defaultdict(list)
        for bar_events in events_by_bar.values():
            bar_events.sort(key=lambda event: event["grid_time"])
            for previous, current in zip(bar_events, bar_events[1:]):
                transition = (
                    (previous["beat"], previous["subdivision"]),
                    (current["beat"], current["subdivision"]),
                )
                interval_residuals[transition].append(
                    (current["time"] - previous["time"]
                     - current["grid_time"] + previous["grid_time"]) * 1000
                )
        interval_variances = [float(np.var(values)) for values in interval_residuals.values()
                              if len(values) > 1]
        bar_means = {
            bar: float(np.mean(list(profile.values())))
            for bar, profile in accents_by_bar.items() if profile
        }
        normalized_accents = {
            bar: {position: value / bar_means[bar] for position, value in profile.items()}
            for bar, profile in accents_by_bar.items() if bar in bar_means and bar_means[bar] > 0
        }
        accent_pattern_cvs = []
        for position in range(profile_size):
            values = np.asarray([
                profile.get(position, 0.0) for profile in normalized_accents.values()
            ])
            if len(values) > 1 and values.mean() > 0:
                accent_pattern_cvs.append(float(values.std() / values.mean()))
        ioi_variance = float(np.mean(interval_variances)) if interval_variances else 0.0
        timing_variance = float(np.mean(timing_variances)) if timing_variances else 0.0
        accent_pattern_cv = float(np.mean(accent_pattern_cvs)) if accent_pattern_cvs else 0.0
        mechanicalness = (
            1 / (1 + ioi_variance / 100 + timing_variance / 100 + accent_pattern_cv ** 2)
            if len(events_by_bar) > 1 else None
        )
        ratios = swing_ratios_from_events(assigned, beats, subdivisions)
        accent_positions = [
            f"{index // subdivisions + 1}"
            + ("&" if index % subdivisions == 1 and subdivisions == 2
               else (f"+{index % subdivisions}/{subdivisions}"
                     if index % subdivisions else ""))
            for index in range(profile_size)
        ]
        per_stem[spec["name"]] = {
            "onsets": int(len(onset_times)),
            "grid_matched_onsets": len(assigned),
            "swing_ratio_mean": float(np.mean(ratios)) if ratios else None,
            "swing_pairs": len(ratios),
            "accent_profile": profile,
            "accent_positions": accent_positions,
            "accent_profile_cv": accent_cv,
            "accent_pattern_cv_across_bars": accent_pattern_cv,
            "timing_residual_variance_ms2": timing_variance,
            "inter_onset_interval_pattern_variance_ms2": ioi_variance,
            "pattern_bars": len(events_by_bar),
            "mechanicalness": mechanicalness,
        }
    pairwise = []
    names = list(assigned_by_stem)
    for left_index, left_name in enumerate(names):
        left = {event["slot"]: event for event in assigned_by_stem[left_name]}
        for right_name in names[left_index + 1:]:
            right = {event["slot"]: event for event in assigned_by_stem[right_name]}
            common = sorted(set(left) & set(right))
            differences = [(left[slot]["time"] - right[slot]["time"]) * 1000 for slot in common]
            if differences:
                pairwise.append({
                    "left": left_name, "right": right_name, "matched_slots": len(differences),
                    "mean_asynchrony_ms": float(np.mean(differences)),
                    "sd_asynchrony_ms": float(np.std(differences)),
                })
    all_onsets = np.concatenate([values for values in (
        np.asarray([event["time"] for event in events])
        for events in assigned_by_stem.values() if events
    )]) if any(assigned_by_stem.values()) else np.array([])
    ratios = [ratio for events in assigned_by_stem.values()
              for ratio in swing_ratios_from_events(events, beats, subdivisions)]
    values = [item["mechanicalness"] for item in per_stem.values()
              if item["mechanicalness"] is not None]
    summary = {
        "metric_grid": grid_source,
        "subdivisions_per_beat": subdivisions,
        "per_stem": per_stem,
        "asynchrony": pairwise,
        "swing_ratio_mean": float(np.mean(ratios)) if ratios else None,
        "swing_pairs": len(ratios),
        "mechanicalness_mean": float(np.mean(values)) if values else None,
        "mechanicalness_definition": "1/(1 + repeated-transition interval variance/100 + repeated-grid-position timing variance/100 + across-bar accent-pattern CV^2); high means more mechanically repeatable",
    }
    return summary, {"curves": plot_curves, "onsets": all_onsets}


def plot_report(audio_name: str, y: np.ndarray, sr: int, audio_start: float,
                rough_times: np.ndarray, roughness: np.ndarray,
                harmony_rows: list[dict[str, Any]], chroma: np.ndarray | None,
                frame_hop: int, masking_curve: dict[str, Any] | None,
                groove_curve: dict[str, Any] | None, out: str | Path):
    spec_hop = max(HOP, frame_hop)
    cqt = np.abs(librosa.cqt(y, sr=sr, hop_length=spec_hop,
                             fmin=librosa.note_to_hz("A0"), n_bins=88 * 3,
                             bins_per_octave=36))
    spec_db = librosa.amplitude_to_db(cqt, ref=np.max)
    frequencies = librosa.cqt_frequencies(n_bins=88 * 3,
                                           fmin=librosa.note_to_hz("A0"), bins_per_octave=36)
    spec_db += librosa.A_weighting(frequencies)[:, None]
    spec_db -= np.max(spec_db)
    spec_times = audio_start + np.arange(spec_db.shape[1]) * spec_hop / sr
    panels = 3 + int(bool(harmony_rows)) + (2 if masking_curve is not None else 0) + int(groove_curve is not None)
    height_ratios = [4, 1.0, 1.0] + ([1.0] if harmony_rows else [])
    if masking_curve is not None:
        height_ratios.extend((2.0, 1.0))
    if groove_curve is not None:
        height_ratios.append(1.0)
    fig, axes = plt.subplots(panels, 1, figsize=(18, 3 + 1.8 * panels), sharex=True,
                             gridspec_kw={"height_ratios": height_ratios})
    axes = np.atleast_1d(axes)
    cursor = 0
    extent = [spec_times[0] if len(spec_times) else audio_start,
              spec_times[-1] if len(spec_times) else audio_start + len(y) / sr,
              21, 108]
    axes[cursor].imshow(spec_db, origin="lower", aspect="auto", extent=extent,
                        cmap="magma", vmin=-80, vmax=0, interpolation="nearest")
    axes[cursor].set_ylabel("pitch (MIDI)")
    axes[cursor].set_title(Path(audio_name).name + "  (A-weighted CQT)")
    cursor += 1

    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=frame_hop)[0]
    loud_times = audio_start + np.arange(len(rms)) * frame_hop / sr
    axes[cursor].plot(loud_times, 20 * np.log10(np.maximum(rms, 1e-9) / max(rms.max(), 1e-9)),
                      lw=0.65, color="black")
    axes[cursor].set_ylim(-80, 2)
    axes[cursor].set_ylabel("loudness\ndB rel")
    cursor += 1

    axes[cursor].plot(audio_start + rough_times, roughness, lw=0.65, color="tab:red")
    if len(roughness):
        cut = np.percentile(roughness, 90)
        selected = roughness >= cut
        axes[cursor].scatter(audio_start + rough_times[selected], roughness[selected],
                             s=8, color="darkred", label="top decile")
    axes[cursor].set_ylabel("roughness")
    cursor += 1

    if harmony_rows:
        mids = np.array([row["start_seconds"] for row in harmony_rows])
        values = np.array([row["misfit_fraction"] for row in harmony_rows])
        flags = np.array([row["flagged"] for row in harmony_rows])
        axes[cursor].step(mids, values, where="post", lw=0.8, color="tab:blue")
        if flags.any():
            axes[cursor].scatter(mids[flags], values[flags], color="red", s=12, zorder=3)
        axes[cursor].set_ylim(0, 1)
        axes[cursor].set_ylabel("harmony\noutside")
        cursor += 1

    if masking_curve is not None:
        times = masking_curve["times"]
        centers = masking_curve["centers"]
        ratios = masking_curve["ratio_db"]
        axes[cursor].imshow(ratios, origin="lower", aspect="auto",
                            extent=[times[0], times[-1], centers[0], centers[-1]],
                            cmap="coolwarm", vmin=-18, vmax=18, interpolation="nearest")
        axes[cursor].set_ylabel("ERB Hz\nmelody/rest dB")
        cursor += 1
        axes[cursor].plot(times, masking_curve["audibility"], color="tab:green", lw=0.7)
        axes[cursor].axhline(0.5, color="red", lw=0.5, ls="--")
        axes[cursor].fill_between(times, 0, masking_curve["audibility"],
                                  where=masking_curve["audibility"] < 0.5,
                                  color="red", alpha=0.2)
        axes[cursor].set_ylim(0, 1)
        axes[cursor].set_ylabel("tune\naudible bands")
        cursor += 1

    if groove_curve is not None:
        for index, (times, envelope, name) in enumerate(groove_curve["curves"]):
            axes[cursor].plot(times, envelope, lw=0.6, alpha=0.75, label=name)
        if groove_curve["curves"]:
            axes[cursor].legend(loc="upper right", fontsize=7, ncol=3)
        axes[cursor].set_ylabel("onset\nstrength")
    axes[-1].set_xlabel("time (s)")
    axes[-1].set_xlim(audio_start, audio_start + len(y) / sr)
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)


def verdict(summary: dict[str, Any]) -> str:
    rough = summary.get("roughness", {})
    harmony = summary.get("harmony")
    masking = summary.get("tune_masking")
    parts = [f"Listen: roughness p99 {rough.get('p99', 0):.3f}"]
    if harmony:
        worst = harmony.get("worst", {})
        name = ", ".join(worst.get("offending", [])) or "none"
        source = "MIDI PCs" if str(harmony.get("source", "")).startswith("MIDI") else "chord list"
        parts.append(
            f"harmony misfit {100 * harmony.get('mean_misfit_fraction', 0):.0f}% mean; "
            f"{100 * harmony.get('flagged_fraction', 0):.0f}% beats flagged "
            f"(worst bar {worst.get('bar', '?')} beat {worst.get('beat', '?')}: {name}; {source})"
        )
    else:
        parts.append("harmony n/a")
    if masking:
        parts.append(f"tune masked {100 * masking.get('masked_fraction', 0):.0f}%")
    else:
        parts.append("tune masked n/a")
    return "; ".join(parts)


def analyze(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    audio_start = args.start or 0.0
    y, sr = load_audio(args.audio, args.start, args.end)
    frame_hop = HOP * (2 if args.quick else 1)
    rough_local_times, roughness, chroma = spectral_roughness(
        y, sr, hop=frame_hop, return_chroma=True,
    )
    rough_times = rough_local_times + audio_start
    midi_path = args.midi
    temporary_score_midi = None
    if args.score:
        temporary_score_midi = compile_score(args.score)
        midi_path = str(temporary_score_midi)
    midi_info = read_midi(midi_path, args.midi_offset) if midi_path else None
    beats = midi_info["beats"] if midi_info else []
    if args.bpm and not beats:
        beats = _grid_from_bpm(audio_start, audio_start + len(y) / sr, args.bpm)
    specs = stem_specs(args.audio, args.stems, args.qa)
    chords = load_chords(args.harmony) if args.harmony else None
    expected, harmony_source = expected_harmony(midi_info, beats, args.harmony_tracks, chords)
    frame_times = audio_start + np.arange(roughness.size) * frame_hop / sr
    harmony, harmony_rows = chroma_fit(
        chroma, frame_times, expected, args.harmony_threshold,
    ) if expected else (None, [])
    if harmony:
        harmony["source"] = harmony_source
    rough_summary = roughness_summary(rough_times, roughness, beats)
    masking = masking_curve = None
    groove = groove_curve = None
    if not args.quick and specs and (args.melody or args.melody_section):
        masking, masking_curve = analyze_masking(
            specs, args.melody, args.melody_section, sr, audio_start, args.end,
            args.midi_offset, args.masking_margin_db, args.audibility_threshold,
            args.masked_min_seconds,
        )
    if not args.quick and specs:
        if not beats:
            combined = np.zeros(0)
            envelopes = []
            for spec in specs:
                stem_y, _ = load_audio(spec["path"], max(0, audio_start - args.midi_offset) or None,
                                       None if args.end is None else max(0, args.end - args.midi_offset))
                _, envelope = detect_onsets(stem_y)
                envelopes.append(envelope)
            if envelopes:
                length = min(map(len, envelopes))
                combined = sum((envelope[:length] for envelope in envelopes), np.zeros(length))
            source_start = max(0.0, audio_start - args.midi_offset)
            beats = estimate_beat_grid(
                combined, sr, GROOVE_HOP, audio_start, audio_start + len(y) / sr,
                source_start + args.midi_offset,
            )
        groove, groove_curve = analyze_groove(
            specs, beats, sr, audio_start, args.end, args.midi_offset,
            args.subdivisions,
            "MIDI time signature and tempo map" if midi_info else (
                "user-supplied BPM grid" if args.bpm else "audio beat estimate"
            ),
            midi_info,
        )
    summary = {
        "schema_version": 1,
        "audio": str(Path(args.audio)),
        "analysis": {
            "start_seconds": audio_start,
            "end_seconds": args.end,
            "duration_seconds": len(y) / sr,
            "sample_rate": sr,
            "midi_offset_seconds": args.midi_offset,
            "quick": args.quick,
        },
        "roughness": rough_summary,
        "harmony": harmony,
        "tune_masking": masking,
        "groove": groove,
    }
    curves = {
        "rough_times": rough_times,
        "roughness": roughness,
        "harmony_rows": harmony_rows,
        "chroma": chroma,
        "frame_hop": frame_hop,
        "masking": masking_curve,
        "groove": groove_curve,
    }
    if temporary_score_midi:
        temporary_score_midi.unlink(missing_ok=True)
    return summary, {"audio": y, "sr": sr, "audio_start": audio_start, **curves}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio")
    midi_group = parser.add_mutually_exclusive_group()
    midi_group.add_argument("--midi", help="performed or score MIDI")
    midi_group.add_argument("--score", help="LilyPond score; compiled to temporary MIDI")
    parser.add_argument("--stems", nargs="*", metavar="[NAME=]WAV",
                        help="stem WAVs; no values loads WAVs from qa.json")
    parser.add_argument("--qa", help="QA JSON containing stem paths")
    parser.add_argument("--melody", help="stem that carries the melody")
    parser.add_argument("--melody-section", action="append", default=[], metavar="START:END:STEM",
                        help="melody stem for an audio-time section; repeat for multiple sections")
    parser.add_argument("--harmony", help="JSON object keyed by bar:beat with chord symbols or pitch classes")
    parser.add_argument("--harmony-tracks", nargs="+", help="MIDI track-name filters for intended harmony")
    parser.add_argument("--midi-offset", type=float, default=0.0,
                        help="seconds to add to MIDI/stem times to align them with the mix")
    parser.add_argument("--bpm", type=float, help="beat grid for audio with no MIDI")
    parser.add_argument("--start", type=float)
    parser.add_argument("--end", type=float)
    parser.add_argument("--out", help="PNG path (default: MIX.listen.png)")
    parser.add_argument("--json", dest="json_path", help="summary path (default: MIX.listen.json)")
    parser.add_argument("--quick", action="store_true",
                        help="coarser roughness/harmony only; skip plot, masking, and groove")
    parser.add_argument("--harmony-threshold", type=float, default=0.18,
                        help="flag a beat when this fraction of chroma energy is outside harmony")
    parser.add_argument("--masking-margin-db", type=float, default=3.0,
                        help="melody bands must exceed all other stems by at least this margin")
    parser.add_argument("--audibility-threshold", type=float, default=0.5,
                        help="a frame is masked when fewer loud melody bands exceed the margin")
    parser.add_argument("--masked-min-seconds", type=float, default=1.0)
    parser.add_argument("--subdivisions", type=int, default=2,
                        help="metric grid subdivisions per beat for groove analysis")
    args = parser.parse_args(argv)
    try:
        for name in ("start", "end", "midi_offset", "bpm", "harmony_threshold",
                     "masking_margin_db", "audibility_threshold", "masked_min_seconds"):
            value = getattr(args, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"--{name.replace('_', '-')} must be finite")
        if args.start is not None and args.start < 0:
            raise ValueError("--start cannot be negative")
        start = args.start if args.start is not None else 0.0
        if args.end is not None and args.end <= start:
            raise ValueError("--end must be greater than --start")
        if args.bpm is not None and args.bpm <= 0:
            raise ValueError("--bpm must be positive")
        if not 0 <= args.harmony_threshold <= 1:
            raise ValueError("--harmony-threshold must be between 0 and 1")
        if not 0 <= args.audibility_threshold <= 1:
            raise ValueError("--audibility-threshold must be between 0 and 1")
        if args.masked_min_seconds < 0:
            raise ValueError("--masked-min-seconds cannot be negative")
        if args.subdivisions < 1:
            raise ValueError("--subdivisions must be at least 1")
        summary, curves = analyze(args)
        json_path = Path(args.json_path or Path(args.audio).with_suffix(".listen.json"))
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(summary, indent=2) + "\n")
        if not args.quick:
            out = args.out or str(Path(args.audio).with_suffix(".listen.png"))
            plot_report(args.audio, curves["audio"], curves["sr"], curves["audio_start"],
                        curves["rough_times"] - curves["audio_start"], curves["roughness"],
                        curves["harmony_rows"], curves["chroma"], curves["frame_hop"],
                        curves["masking"], curves["groove"], out)
            print(f"wrote {out}")
        print(verdict(summary))
        print(f"wrote {json_path}")
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"listen.py: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
