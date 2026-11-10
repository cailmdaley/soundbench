#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["mido>=1.3,<2", "numpy>=1.26", "numba>=0.61", "scipy>=1.11"]
# ///
"""Render a drum-machine manifest (.tr808, .tr808-bass, .tr909 or .linn) to stereo stems."""
from __future__ import annotations
import argparse
from collections import defaultdict, deque
import hashlib
import json
import math
from pathlib import Path
import time
import wave

import mido
import numpy as np

from soundbench.scripts.paths import parameters_label, parameters_path
from soundbench.scripts.machine_time import TIMING_KEYS
from soundbench.scripts.models.drum_machine import ROOT, load_config, render, voice_of
from soundbench.scripts.perform import BarMap, TempoMap

ENGINE_OPTIONS = frozenset(("seed", "output_gain", "unit", "accent", "velocity", "toms"))


def gestures(midi, index):
    """Note rows with explicit releases; a machine's decay is a knob, so CCs are ignored."""
    tempo = TempoMap(midi)
    active = defaultdict(deque)
    rows, tick = [], 0
    for msg in midi.tracks[index]:
        tick += msg.time
        t = tempo.seconds_at(tick)
        if msg.type == "note_on" and msg.velocity:
            row = dict(time=t, off=None, pitch=msg.note, velocity=msg.velocity)
            rows.append(row)
            active[msg.channel, msg.note].append(row)
        elif msg.type in ("note_off", "note_on"):
            if not active[msg.channel, msg.note]:
                raise ValueError("drum-machine release without matching onset")
            active[msg.channel, msg.note].popleft()["off"] = t
    if any(active.values()):
        raise ValueError("drum-machine notes need explicit releases")
    return rows, {}


def selected_config(settings, machine):
    """The machine preset plus the sidecar's [<machine>_model] table; knobs merge into voices."""
    options = settings.get(f"{machine}_model", {})
    unknown = set(options)-ENGINE_OPTIONS-TIMING_KEYS-{"parameters", "knobs", "rides"}
    if unknown:
        raise ValueError(f"unknown {machine}_model options: {sorted(unknown)}")
    path = None
    if "parameters" in options:
        path = parameters_path(options["parameters"])
    overrides = {"engine": {k: options[k] for k in ENGINE_OPTIONS if k in options}, "voices": options.get("knobs", {})}
    c = load_config(machine, path, overrides)
    digest = hashlib.sha256((path.read_bytes() if path else b"")+json.dumps(options, sort_keys=True).encode()).hexdigest()
    return c, path, digest


def apply_rides(rows, rides, score_midi, index, voice_of, voices):
    """Knob rides by score bar: [{voice, knob, bars = [first, last], value}], later rides winning.

    Machine onsets keep the score's order, so the n-th performed hit is the n-th score onset.
    """
    if not rides:
        return rows
    tick, onsets = 0, []
    for msg in score_midi.tracks[index]:
        tick += msg.time
        if msg.type == "note_on" and msg.velocity:
            onsets.append((tick, msg.note))
    onsets.sort()
    rows = sorted((dict(r) for r in rows), key=lambda r: (r["time"], r["pitch"]))
    if [n for _, n in onsets] != [r["pitch"] for r in rows]:
        raise ValueError("drum-machine rides need the performed hits to follow the score's order")
    bars = BarMap(score_midi)
    for ride in rides:
        if set(ride) != {"voice", "knob", "bars", "value"} or len(ride["bars"]) != 2 or ride["voice"] not in voices:
            raise ValueError("each ride needs a voice of this machine, knob, bars = [first, last] and value")
    for row, (t, _) in zip(rows, onsets):
        bar = bars.bar_at(t)
        for ride in rides:
            if ride["voice"] == voice_of(row["pitch"]) and ride["bars"][0] <= bar <= ride["bars"][1]:
                row.setdefault("knobs", {})[ride["knob"]] = float(ride["value"])
    return rows


def write_stereo_wav(path, audio, sr=48000):
    if audio.ndim != 2 or audio.shape[1] != 2:
        raise ValueError("drum-machine stems are stereo")
    if not np.all(np.isfinite(audio)) or np.max(np.abs(audio), initial=0) >= 1:
        raise ValueError("drum-machine stem clips or is nonfinite; lower output_gain")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(4)
        wav.setframerate(sr)
        wav.writeframes(np.rint(audio*2147483647).astype("<i4").tobytes())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path)
    a = p.parse_args()
    model = next((m for m in ("tr808-bass", "tr808", "tr909", "linn") if a.manifest.stem.endswith("."+m)), None)
    if model is None:
        raise ValueError("manifest suffix must be .tr808, .tr808-bass, .tr909 or .linn")
    machine = model.removesuffix("-bass")
    data = json.loads(a.manifest.read_text())
    midi = mido.MidiFile(data["performed_midi"])
    c, path, digest = selected_config(data.get("settings", {}), machine)
    end = TempoMap(midi).seconds_at(max(sum(m.time for m in t) for t in midi.tracks))
    seconds = end+c["engine"]["tail_seconds"]
    stems, played, peaks = {}, {}, {}
    start = time.perf_counter()
    rides = data.get("settings", {}).get(f"{machine}_model", {}).get("rides", [])
    score = mido.MidiFile(data["score_midi"]) if rides else None
    for job in data["jobs"]:
        rows, _ = gestures(midi, job["index"])
        rows = apply_rides(rows, rides, score, job["index"], lambda p: voice_of(machine, p, c, model == "tr808-bass"), c["voices"])
        audio = render(rows, seconds, machine, c, bass=model == "tr808-bass")
        write_stereo_wav(Path(job["output"]), audio, c["engine"]["sample_rate"])
        index = str(job["index"])
        stems[index] = dict(engine="model", model=model, parameters=parameters_label(path) if path else f"instruments/models/{machine}.toml",
                            parameters_sha256=digest, overrides=data.get("settings", {}).get(f"{machine}_model", {}))
        played[index] = dict(notes=rows)
        peaks[index] = float(np.max(abs(audio), initial=0))
    metrics = dict(stems=stems, gestures=played, peaks=peaks, render_seconds=time.perf_counter()-start, audio_seconds=seconds)
    a.manifest.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
    print(f"{model}: {len(stems)} stem(s), {metrics['render_seconds']:.2f} s, peaks {peaks}")


if __name__ == "__main__":
    main()
