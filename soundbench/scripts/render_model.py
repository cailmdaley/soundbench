#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["mido>=1.3,<2", "numpy>=1.26", "numba>=0.61", "scipy>=1.11"]
# ///
"""Render shared banjo strings into aligned, source-attributed stereo WAV stems."""
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
from soundbench.scripts.instrumentation import aliases, matching_tracks
from soundbench.scripts.models.banjo import CONFIG, assign_string, load_config, render
from soundbench.scripts.perform import BarMap, TempoMap


def notes(midi, index):
    active = defaultdict(deque)
    held = defaultdict(list)
    pedal = defaultdict(bool)
    rows = []
    tick = 0
    for msg in midi.tracks[index]:
        tick += msg.time
        if msg.type == "note_on" and msg.velocity:
            row = {"tick": tick, "off": tick, "pitch": msg.note, "velocity": msg.velocity,
                   "token": len(rows)}
            rows.append(row)
            active[msg.channel, msg.note].append(row)
        elif msg.type == "control_change" and msg.control == 64:
            pedal[msg.channel] = msg.value >= 64
            if not pedal[msg.channel]:
                for row in held.pop(msg.channel, []):
                    row["off"] = tick
        elif msg.type in ("note_off", "note_on") and active[msg.channel, msg.note]:
            row = active[msg.channel, msg.note].popleft()
            row["off"] = tick
            if pedal[msg.channel]:
                held[msg.channel].append(row)
    for pending in [*active.values(), *held.values()]:
        for row in pending:
            row["off"] = max(tick, row["tick"] + midi.ticks_per_beat)
    return rows


def schedule(performed, score, settings, indices, config):
    """Anchor score-addressed gestures to the corresponding performed note.

    Durations and between-note gestures interpolate the score/performance note
    correspondence, so groove, rubato and onset jitter remain the timing owner.
    """
    sr = config["engine"]["sample_rate"]
    tempo, bars = TempoMap(performed), BarMap(score)
    source_tempo = TempoMap(score)
    names = {i: performed.tracks[i].name for i in indices}
    pinned = {}
    for selector, string in settings.get("banjo_strings", {}).items():
        matches = matching_tracks(selector, indices, names)
        if not matches or not isinstance(string, int) or not 1 <= string <= 5:
            raise ValueError(f"invalid banjo_strings entry {selector!r} = {string!r}")
        for i in matches:
            if i in pinned:
                raise ValueError("duplicate banjo string mapping")
            pinned[i] = string
    all_notes, maps = {}, {}
    for owner, index in enumerate(indices):
        played, scored = notes(performed, index), notes(score, index)
        by_pitch = defaultdict(deque)
        for row in scored:
            by_pitch[row["pitch"]].append(row)
        anchors = []
        for row in played:
            if not by_pitch[row["pitch"]]:
                raise ValueError("score/performed note correspondence differs")
            src = by_pitch[row["pitch"]].popleft()
            row["score_tick"] = src["tick"]
            row["owner"] = owner
            row["explicit"] = pinned.get(index)
            row["thumb"] = "thumb" in aliases(names[index])
            row["index"] = index
            row["kind"] = 0
            row["time"] = tempo.seconds_at(row["tick"])
            row["off_time"] = tempo.seconds_at(row["off"])
            anchors.append((source_tempo.seconds_at(src["tick"]), row["time"]))
        if any(by_pitch.values()):
            raise ValueError("unmatched scored notes")
        # Simultaneous score attacks share their mean performed time for interpolation.
        grouped = defaultdict(list)
        for x, y in anchors:
            grouped[x].append(y)
        x = np.array(sorted(grouped))
        y = np.array([np.mean(grouped[t]) for t in x])
        maps[index] = (x, y)
        all_notes[index] = played
    def at(index, score_tick):
        seconds = source_tempo.seconds_at(score_tick)
        x, y = maps[index]
        if seconds < x[0]:
            return max(0, seconds + y[0] - x[0])
        if seconds > x[-1]:
            return seconds + y[-1] - x[-1]
        return float(np.interp(seconds, x, y))
    gestures, chokes = [], []
    allowed = {"track", "bar", "beat", "kind", "to", "duration_beats", "shape", "string", "pitch"}
    occupied = set()
    for art in settings.get("articulation", []):
        if set(art) - allowed or not {"track", "bar", "kind"} <= set(art):
            raise ValueError(f"invalid articulation keys: {art}")
        matches = matching_tracks(art["track"], indices, names)
        if len(matches) != 1:
            raise ValueError(f"articulation track {art['track']!r} must select one model track")
        index = matches[0]
        if not isinstance(art["bar"], int):
            raise ValueError("articulation bar must be an integer")
        tick = bars.position(art["bar"], float(art.get("beat", 1)))
        kind = art["kind"]
        if kind not in ('pluck', 'bend', 'slide', 'hammer', 'pull', 'choke'):
            raise ValueError(f"unknown articulation {kind!r}")
        candidates = [r for r in all_notes[index] if r["score_tick"] <= tick]
        if not candidates:
            raise ValueError("articulation has no preceding note")
        exact = [r for r in candidates if r["score_tick"] == tick]
        if 'pitch' in art:
            if kind=='choke' or type(art['pitch']) is not int or not 0<=art['pitch']<=127:
                raise ValueError('pitch must be a MIDI note selector on a non-choke articulation')
            exact = [r for r in exact if r['pitch']==art['pitch']]
        if kind != "choke" and len(exact) != 1:
            raise ValueError("pitch articulations must address a single scored note onset")
        preceding = [r for r in candidates if r["score_tick"] < tick]
        if kind == "choke" and not preceding:
            raise ValueError("choke needs a preceding note")
        row = exact[0] if kind != "choke" else None
        key = (index, tick, row['pitch'] if row else None)
        if key in occupied or (index,tick,None) in occupied or (kind=='choke' and any(k[:2]==(index,tick) for k in occupied)):
            raise ValueError("only one articulation per scored note; choke occupies the track/position")
        occupied.add(key)
        string = art.get("string", pinned.get(index))
        if string is not None and (not isinstance(string, int) or not 1 <= string <= 5):
            raise ValueError("articulation string must be 1..5")
        if kind == "choke":
            if set(art) & {"to", "duration_beats", "shape"}:
                raise ValueError("choke has no pitch target, duration or shape")
            chokes.append((index, tick, string))
            continue
        row["explicit"] = string
        row["articulation"] = art
        if kind == 'pluck':
            if set(art) & {'to', 'duration_beats', 'shape'}:
                raise ValueError('pluck has a scored pitch and optional string, without a target or duration')
            continue
        if kind in ("hammer", "pull"):
            if set(art) & {"to", "duration_beats", "shape"}:
                raise ValueError("hammer/pull use the scored destination pitch, without to/duration/shape")
            row["kind"] = 3 if kind == "hammer" else 4
        else:
            if not {"to", "duration_beats"} <= set(art):
                raise ValueError("bend/slide need to (relative semitones) and duration_beats")
            to, duration = float(art["to"]), float(art["duration_beats"])
            shape = art.get("shape", "linear")
            if not math.isfinite(to) or not math.isfinite(duration) or duration <= 0 or shape not in ("linear", "ease"):
                raise ValueError("invalid articulation target/duration/shape")
            row["target"] = row["pitch"] + to
            end_tick = tick + duration * bars.beat_length_at(tick)
            row["gesture_length"] = max(1, round((at(index, end_tick) - row["time"]) * sr))

    # Plan a single-string legato move before plucking its source. A D4 -> C4
    # pull must start on the B string, not the open D string below which it
    # cannot descend. Explicit score/sidecar string choices remain authoritative.
    for index, rows in all_notes.items():
        for row in rows:
            if row["kind"] not in (3, 4):
                continue
            preceding = [r for r in rows if r["score_tick"] < row["score_tick"]]
            if not preceding:
                continue
            source = max(preceding, key=lambda r: r["score_tick"])
            rising = row["pitch"] > source["pitch"]
            if rising != (row["kind"] == 3) or row["pitch"] == source["pitch"]:
                continue
            default = assign_string(source["pitch"], source["thumb"], config, source["explicit"])
            if 0 <= row["pitch"] - config["strings"]["open_midi"][default] <= config["strings"]["max_fret"]:
                continue
            choices = [s for s, pitch in enumerate(config["strings"]["open_midi"], 1)
                       if s != 5 and all(0 <= p - pitch <= config["strings"]["max_fret"]
                                         for p in (source["pitch"], row["pitch"]))
                       and (source["explicit"] is None or source["explicit"] == s)
                       and (row["explicit"] is None or row["explicit"] == s)]
            if choices:
                string = min(choices, key=lambda s: source["pitch"] - config["strings"]["open_midi"][s - 1])
                source["explicit"] = row["explicit"] = string

    # All tracks share one instrument; allocate in performed-time order.
    current, last_frets = [None] * 5, [0] * 5
    ordered = sorted((r for rows in all_notes.values() for r in rows),
                     key=lambda r: (round(r["time"] * sr), r["owner"], r["token"]))
    for row in ordered:
        t, index = row["time"], row["index"]
        sources = []
        if row["kind"] in (3, 4):
            sources = [r for r in current if r is not None and r["index"] == index
                       and r["score_tick"] < row["score_tick"]
                       and (row["pitch"] > r["pitch"] if row["kind"] == 3 else row["pitch"] < r["pitch"])
                       and (row["explicit"] is None or r["string"] == row["explicit"] - 1)
                       and 0 <= row["pitch"] - config["strings"]["open_midi"][r["string"]]
                       <= config["strings"]["max_fret"]]
            if not sources:
                # another voice took the source string first: the note is plucked instead
                row["dropped"] = "hammer" if row["kind"] == 3 else "pull"
                row["kind"] = 0
        if sources:
            preferred = assign_string(row["pitch"], row["thumb"], config, row["explicit"])
            source = min(sources, key=lambda r: (r["string"] != preferred,
                         abs(row["pitch"] - r["pitch"]), -r["time"], r["token"]))
            row["string"] = source["string"]
            source["off_time"] = max(source["off_time"], t)
        else:
            row["string"] = assign_string(row["pitch"], row["thumb"], config, row["explicit"],
                sounding=[r is not None and round(r["off_time"] * sr) > round(t * sr) for r in current],
                started=[r["time"] if r is not None else -math.inf for r in current],
                last_frets=last_frets)
        s = row["string"]
        current[s] = row
        last_frets[s] = row["pitch"] - config["strings"]["open_midi"][s]
        if "target" in row:
            assign_string(row["target"], False, config, s + 1)
            gestures.append([round(t * sr), 2, s, row["owner"], row["target"],
                             0, row["gesture_length"], row["articulation"].get("shape", "linear") == "ease",
                             0, 0, index * 1000000 + row["token"]])
    for index, tick, string in chokes:
        sources = [r for r in all_notes[index] if r["score_tick"] < tick
                   and (string is None or r["string"] == string - 1)]
        if not sources:
            raise ValueError("choke has no preceding note on the selected string")
        row = max(sources, key=lambda r: (r["score_tick"], r["time"]))
        gestures.append([round(at(index, tick) * sr), 5, row["string"], row["owner"],
                         row["pitch"], 0, 0, 0, 0, 0, index * 1000000 + row["token"]])
    events, assignment = [], []
    for index in indices:
        for row in all_notes[index]:
            t = round(row["time"] * sr)
            events.append([t, row["kind"], row["string"], row["owner"], row["pitch"], row["velocity"],
                           sr * config["strings"]["pitch_smoothing_ms"] / 1000, 1, 0, 0,
                           index * 1000000 + row["token"]])
            events.append([round(row["off_time"] * sr), 1, row["string"], row["owner"],
                           0, 0, 0, 0, 0, 0, index * 1000000 + row["token"]])
            assignment.append({"track": names[index], "pitch": row["pitch"], "time_s": row["time"],
                               "off_time_s": row["off_time"], "token": index * 1000000 + row["token"],
                               "string": row["string"] + 1, "kind": row["kind"],
                               **({"dropped": row["dropped"]} if "dropped" in row else {})})
    # At a retrigger: release the old token, pluck/change, then start the gesture.
    events.sort(key=lambda r: (r[0], 0 if r[1] == 1 else 1))
    events.extend(gestures)
    end = max((sum(m.time for m in track) for track in performed.tracks), default=0)
    seconds = max(tempo.seconds_at(end), max((r[0] + r[6] for r in events), default=0) / sr)
    return events, seconds + config["engine"]["tail_seconds"], assignment


def assignment_statistics(events, sr=48000):
    """Count plucks that replace a held/sustained note; legato is a handoff."""
    offs = {int(r[10]): r[0] for r in events if r[1] == 1}
    current, lost, handoffs = {}, [], 0
    for row in sorted(events, key=lambda r: (r[0], 0 if r[1] == 1 else 1)):
        sample, kind, string = row[:3]
        if kind not in (0, 3, 4):
            continue
        previous = current.get(string)
        if previous is not None and offs.get(previous, 0) > sample:
            if kind == 0:
                lost.append((offs[previous] - sample) / sr)
            else:
                handoffs += 1
        current[string] = int(row[10])
    return {"notes": sum(r[1] in (0, 3, 4) for r in events),
            "cut_before_note_off": len(lost),
            "median_lost_seconds": float(np.median(lost)) if lost else 0.0,
            "legato_transfers_before_note_off": handoffs}


def write_wav(path, audio, sr=48000):
    if not np.all(np.isfinite(audio)):
        raise ValueError("non-finite model output")
    if np.max(np.abs(audio), initial=0) >= 1:
        raise ValueError("model clips; lower engine.output_gain")
    pcm = np.rint(audio * 2147483647).astype("<i4")
    if pcm.ndim == 2 and pcm.shape[1] == 2:
        stereo = pcm
    elif pcm.ndim == 1:
        stereo = np.repeat(pcm[:, None], 2, axis=1)
    else:
        raise ValueError(f"model output must be mono (N,) or stereo (N, 2), got {pcm.shape}")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(4)
        wav.setframerate(sr)
        wav.writeframes(stereo.tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--config", type=Path, help="Banjo parameter TOML (overrides manifest selection)")
    args = parser.parse_args()
    data = json.loads(args.manifest.read_text())
    selection = data['settings'].get('banjo_model', {})
    if set(selection) - {'output_gain', 'parameters', 'tuning'}:
        raise ValueError('banjo_model accepts output_gain, parameters and tuning')
    parameter_path = (args.config or parameters_path(selection.get('parameters', str(CONFIG))))
    config = load_config(parameter_path)
    parameter_name = parameters_label(parameter_path)
    options = {k:v for k,v in selection.items() if k != 'parameters'}
    if "output_gain" in options:
        gain = float(options["output_gain"])
        if not math.isfinite(gain) or gain <= 0:
            raise ValueError("invalid banjo_model.output_gain")
        config["engine"]["output_gain"] = gain
    if 'tuning' in options:
        tuning = options['tuning']
        if (not isinstance(tuning, list) or len(tuning) != 5
                or any(type(p) is not int or not 21 <= p <= 84 for p in tuning)):
            raise ValueError('banjo tuning needs five MIDI open pitches, strings 1..5')
        config['strings']['open_midi'] = tuning
    parameter_hash = hashlib.sha256(parameter_path.read_bytes() + (json.dumps(options, sort_keys=True).encode() if options else b"")).hexdigest()
    performed, score = mido.MidiFile(data["performed_midi"]), mido.MidiFile(data["score_midi"])
    indices = [job["index"] for job in data["jobs"]]
    events, seconds, assignment = schedule(performed, score, data["settings"], indices, config)
    # Separate compilation from the actual timed render.
    warm = time.perf_counter()
    render([], 0.001, len(indices), config)
    compile_seconds = time.perf_counter() - warm
    start = time.perf_counter()
    audio = render(events, seconds, len(indices), config)
    elapsed = time.perf_counter() - start
    stems = {}
    for owner, job in enumerate(data["jobs"]):
        path = Path(job["output"])
        write_wav(path, audio[owner])
        stems[str(job["index"])] = {"engine": "model", "model": "banjo",
                                   "parameters": parameter_name,
                                   "parameters_sha256": parameter_hash,
                                   "overrides": options}
    metrics = {"stems": stems, "audio_seconds": seconds, "render_seconds": elapsed,
               "compile_or_cache_seconds": compile_seconds, "realtime_factor": seconds / elapsed,
               "assignment": assignment, "assignment_statistics": assignment_statistics(events),
               "dropped_articulations": [a for a in assignment if "dropped" in a],
               "peak": float(np.max(np.abs(audio)))}
    args.manifest.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Banjo model: {seconds:.2f} s audio in {elapsed:.3f} s ({seconds/elapsed:.1f}x realtime); compile/cache {compile_seconds:.3f} s")


if __name__ == "__main__":
    main()
