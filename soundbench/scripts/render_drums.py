#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["mido>=1.3,<2", "numpy>=1.26", "numba>=0.61", "scipy>=1.11"]
# ///
"""Render a physical drum kit manifest, preserving channel-local MIDI CC4."""
from __future__ import annotations
import argparse
from collections import defaultdict, deque
import hashlib
import json
import math
from pathlib import Path
import time

import mido
import numpy as np

from soundbench.scripts.paths import preset_bytes, preset_label, preset_path
from soundbench.scripts.models.drums import load_config, render
from soundbench.scripts.perform import TempoMap
from soundbench.scripts.render_model import write_wav


def gestures(midi, index):
    tempo = TempoMap(midi)
    active = defaultdict(deque)
    rows, pedals, tick = [], defaultdict(list), 0
    for msg in midi.tracks[index]:
        tick += msg.time
        t = tempo.seconds_at(tick)
        if msg.type == "control_change" and msg.control == 4:
            pedals[msg.channel].append((t, msg.value/127))
        elif msg.type == "note_on" and msg.velocity:
            row = dict(time=t, off=None, pitch=msg.note, velocity=msg.velocity, channel=msg.channel)
            rows.append(row)
            active[msg.channel, msg.note].append(row)
        elif msg.type in ("note_off", "note_on"):
            if not active[msg.channel, msg.note]:
                raise ValueError("drum release without matching onset")
            active[msg.channel, msg.note].popleft()["off"] = t
    if any(active.values()):
        raise ValueError("drum notes need explicit releases")
    return rows, dict(pedals)


def selected_config(settings):
    options = settings.get("drums_model", {})
    if set(options)-{"preset", "output_gain", "seed"}:
        raise ValueError("unknown drums_model option")
    path = preset_path("drums", options.get("preset"))
    c = load_config(path)
    for k in ("seed", "output_gain"):
        if k in options:
            v = options[k]
            if not isinstance(v, (int, float)) or not math.isfinite(v) or (k == "seed" and (type(v) is not int or v < 0)) or (k == "output_gain" and not 0 < v <= 4):
                raise ValueError(f"invalid drums_model.{k}")
            c["engine"][k] = v
    digest = hashlib.sha256(preset_bytes(path)+json.dumps(options, sort_keys=True).encode()).hexdigest()
    return c, path, digest


def render_channels(rows, seconds, c, pedals):
    # Independent MIDI channels have independent pedals, but share mic geometry.
    channels = sorted({r["channel"] for r in rows})
    y = np.zeros((round(seconds*c["engine"]["sample_rate"]), 2))
    for channel in channels:
        y += render([r for r in rows if r["channel"] == channel], seconds, c, pedals.get(channel, []))
    return y


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path)
    a = p.parse_args()
    data = json.loads(a.manifest.read_text())
    midi = mido.MidiFile(data["performed_midi"])
    c, path, digest = selected_config(data.get("settings", {}))
    base = TempoMap(midi).seconds_at(max(sum(m.time for m in t) for t in midi.tracks))
    seconds = base+c["engine"]["tail_seconds"]
    stems, played, peaks = {}, {}, {}
    start = time.perf_counter()
    for job in data["jobs"]:
        rows, pedals = gestures(midi, job["index"])
        audio = render_channels(rows, seconds, c, pedals)
        peak = float(np.max(abs(audio), initial=0))
        if peak >= 1 or not np.isfinite(audio).all():
            raise ValueError(f"drum stem clips or is nonfinite ({peak}); reduce output_gain")
        write_wav(Path(job["output"]), audio, c["engine"]["sample_rate"])
        index = str(job["index"])
        stems[index] = dict(engine="model", model="drums", parameters=preset_label(path), parameters_sha256=digest, overrides=data.get("settings", {}).get("drums_model", {}))
        played[index] = dict(notes=rows, cc4=pedals)
        peaks[index] = peak
    metrics = dict(stems=stems, gestures=played, peaks=peaks, render_seconds=time.perf_counter()-start, audio_seconds=seconds)
    a.manifest.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
    print(f"Drums: {len(stems)} stem(s), {metrics['render_seconds']:.3f} s, peaks {peaks}")


if __name__ == "__main__":
    main()
