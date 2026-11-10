#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["mido>=1.3,<2", "numpy>=1.26", "numba>=0.61", "scipy>=1.11"]
# ///
"""Render tambori, castanyoles or ximbomba jobs from the studio model manifest."""
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

from soundbench.scripts.paths import parameters_label, parameters_path
from soundbench.scripts.instrumentation import resolve_instruments
from soundbench.scripts.models.mallorca_percussion import ROOT, load_config, render
from soundbench.scripts.perform import TempoMap
from soundbench.scripts.render_model import write_wav


def gestures(midi, index):
    """Pair repeated notes FIFO; preserve channel-local, continuous CC2 pressure."""
    tempo = TempoMap(midi)
    active = defaultdict(deque)
    pressure = defaultdict(lambda: .7)
    rows, tick = [], 0
    for msg in midi.tracks[index]:
        tick += msg.time
        t = tempo.seconds_at(tick)
        if msg.type == "control_change" and msg.control == 2:
            pressure[msg.channel] = msg.value/127
            for (channel, _), pending in active.items():
                if channel == msg.channel:
                    for row in pending:
                        row["pressure_events"].append((t, msg.value/127))
        elif msg.type == "note_on" and msg.velocity:
            row = dict(time=t, off=None, pitch=msg.note, velocity=msg.velocity,
                       pressure=pressure[msg.channel], pressure_events=[])
            rows.append(row)
            active[msg.channel, msg.note].append(row)
        elif msg.type in ("note_off", "note_on"):
            if not active[msg.channel, msg.note]:
                raise ValueError("percussion note-off has no matching onset")
            row = active[msg.channel, msg.note].popleft()
            row["off"] = t
    if any(active.values()):
        raise ValueError("percussion notes need explicit releases")
    return rows


def selected_config(model, settings):
    options = settings.get(f"{model}_model", {})
    allowed = {"parameters", "output_gain", "seed"}
    if model == "castanyoles":
        allowed |= {"dancers", "timing_ms", "tuning_cents"}
    if set(options)-allowed:
        raise ValueError(f"unknown {model}_model option")
    path = parameters_path(options.get("parameters", f"instruments/models/{model}.toml"))
    c = load_config(model, path)
    for k, v in options.items():
        if k == "parameters":
            continue
        if not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"invalid {model}_model.{k}")
        if k in ("dancers", "seed") and (type(v) is not int or v < (1 if k == "dancers" else 0)):
            raise ValueError(f"invalid integer {k}")
        if k == "dancers" and v > 16 or k == "timing_ms" and not 0 <= v <= 30 or k == "tuning_cents" and not 0 <= v <= 100 or k == "output_gain" and v <= 0:
            raise ValueError(f"invalid {k}")
        c["engine" if k in ("seed", "output_gain") else "body"][k] = v
    digest = hashlib.sha256(path.read_bytes()+json.dumps(options, sort_keys=True).encode()).hexdigest()
    return c, path, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path)
    args = p.parse_args()
    data = json.loads(args.manifest.read_text())
    midi = mido.MidiFile(data["performed_midi"])
    settings = data.get("settings", {})
    bindings = resolve_instruments(midi, settings)
    end = max(sum(m.time for m in track) for track in midi.tracks)
    base = TempoMap(midi).seconds_at(end)
    stems, played, peaks = {}, {}, {}
    start = time.perf_counter()
    for job in data["jobs"]:
        model = bindings[job["index"]].model
        c, path, digest = selected_config(model, settings)
        rows = gestures(midi, job["index"])
        audio = render(rows, base+c["engine"]["tail_seconds"], c, track_id=job['index'])
        peak = float(np.max(np.abs(audio), initial=0))
        if peak >= 1 or not np.all(np.isfinite(audio)):
            raise ValueError(f"{model} stem clips or is nonfinite (peak {peak}); reduce output_gain")
        write_wav(Path(job["output"]), audio, c["engine"]["sample_rate"])
        index = str(job["index"])
        stems[index] = dict(engine="model", model=model, parameters=parameters_label(path),
                            parameters_sha256=digest, overrides=settings.get(f"{model}_model", {}))
        played[index], peaks[index] = rows, peak
    metrics = dict(stems=stems, gestures=played, peaks=peaks, render_seconds=time.perf_counter()-start,
                   audio_seconds=base+c["engine"]["tail_seconds"])
    args.manifest.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
    print(f"Mallorcan percussion: {len(stems)} stem(s), {metrics['render_seconds']:.3f} s, peaks {peaks}")


if __name__ == "__main__":
    main()
