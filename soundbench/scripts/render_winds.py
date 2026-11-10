"""Manifest and MIDI adapter shared by the two Mallorcan wind renderers."""
from __future__ import annotations
import argparse
from collections import defaultdict, deque
import hashlib
import json
from pathlib import Path
import time
from scipy.io import wavfile
import mido
import numpy as np
from soundbench.scripts.paths import preset_bytes, preset_label, preset_path
from soundbench.scripts.models.mallorca_winds import load_config, render
from soundbench.scripts.perform import TempoMap


def midi_controls(midi, index):
    """Read only this track; CC64 is intentionally irrelevant to breath instruments."""
    tempo = TempoMap(midi)
    active = defaultdict(deque)
    rows, controls, tick = [], [], 0
    for msg in midi.tracks[index]:
        tick += msg.time
        at = tempo.seconds_at(tick)
        if msg.type == "control_change":
            controls.append(dict(time=at, control=msg.control, value=msg.value))
        elif msg.type == "note_on" and msg.velocity:
            row = dict(time=at, off=None, pitch=msg.note, velocity=msg.velocity)
            rows.append(row)
            active[msg.channel, msg.note].append(row)
        elif msg.type in ("note_off", "note_on"):
            if not active[msg.channel, msg.note]:
                raise ValueError("wind MIDI has unmatched note-off")
            active[msg.channel, msg.note].popleft()["off"] = at
    if any(active.values()):
        raise ValueError("wind MIDI has notes without release")
    return rows, controls


def selected_config(settings, instrument):
    options = settings.get(f"{instrument}_model", {})
    if set(options)-{"preset", "output_gain"}:
        raise ValueError(f"{instrument}_model accepts preset and output_gain only")
    path = preset_path(instrument, options.get("preset"))
    config = load_config(instrument, path)
    if "output_gain" in options:
        value = options["output_gain"]
        if not isinstance(value, (float, int)) or not np.isfinite(value) or not 0 < value <= 2:
            raise ValueError("output_gain must be finite and in (0,2]")
        config["output_gain"] = value
    digest = hashlib.sha256(preset_bytes(path)+json.dumps(options, sort_keys=True).encode()).hexdigest()
    return config, path, digest, options


def write_wav(path, audio, sr):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not np.all(np.isfinite(audio)) or np.max(np.abs(audio), initial=0) >= 1:
        raise ValueError("non-finite or clipping wind output")
    # Mono duplication preserves level; the mixer applies equal-power pan once.
    stereo = np.repeat(np.asarray(audio, dtype=np.float32)[:, None], 2, axis=1)
    wavfile.write(path, sr, stereo)


def main(instrument):
    parser = argparse.ArgumentParser(description=f"Render physical {instrument} stems")
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    data = json.loads(args.manifest.read_text())
    config, path, digest, options = selected_config(data.get("settings", {}), instrument)
    midi = mido.MidiFile(data["performed_midi"])
    tempo = TempoMap(midi)
    ticks = max(sum(m.time for m in t) for t in midi.tracks)
    seconds = tempo.seconds_at(ticks)+config["tail_seconds"]
    stems, assignments, peak = {}, {}, 0.0
    start = time.perf_counter()
    for job in data["jobs"]:
        rows, controls = midi_controls(midi, job["index"])
        audio = render(rows, seconds, config, instrument, controls, track_id=job['index'])
        write_wav(Path(job["output"]), audio, config["sample_rate"])
        peak = max(peak, float(np.max(np.abs(audio), initial=0)))
        stems[str(job["index"])] = dict(engine="model", model=instrument,
            parameters=preset_label(path), parameters_sha256=digest, overrides=options)
        assignments[str(job["index"])] = dict(notes=rows, controls=controls)
    elapsed = time.perf_counter()-start
    metrics = dict(stems=stems, audio_seconds=seconds, render_seconds=elapsed, peak=peak,
                   schedule=assignments, includes_compile_or_cache=True)
    args.manifest.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
    print(f"{instrument}: {len(stems)} stem(s), {seconds:.2f}s, peak {peak:.3f}; render/cache {elapsed:.3f}s")
