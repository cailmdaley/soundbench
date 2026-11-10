#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["mido>=1.3,<2", "numpy>=1.26", "numba>=0.61", "scipy>=1.11"]
# ///
"""Sing the voice's probe phrases, and windows of real pieces, through a voice model.

Probe phrases (sung notes in the renderer's row format):
  sustain-{hm,ah}-{soft,loud}-{D4,A4}  one 4 s note
  melisma-{hm,ah}                     a held note turning to its neighbours and back
  slide-{hm,ah}                       a bent third, F to F sharp, then down to D
  breaths-{hm,ah}                     three short phrases with breaths between

Piece windows re-render the voice track of a finished build from its manifest
(`build/.../<piece>.voice.json`), so a v1 and a v2 render sing identical notes:
  --piece MANIFEST --window START-END --name NAME

Usage:
  uv run scripts/voice_phrases.py --model voice --out DIR [--only NAME ...]
  uv run scripts/voice_phrases.py --model voice --out DIR --piece M.json --window 250-275 --name daisy-coda
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
import wave

import mido
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

SR = 48000


def row(t, dur, pitch, vel, syl):
    return {"time": t, "off": t + dur, "pitch": pitch, "velocity": vel, "syllable": syl}


def phrases():
    out = {}
    for syl in ("hm", "ah"):
        for name, vel in (("soft", 40), ("loud", 100)):
            for label, pitch in (("D4", 62), ("A4", 69)):
                out[f"sustain-{syl}-{name}-{label}"] = [row(0.2, 4.0, pitch, vel, syl)]
        turn = [(0.2, 1.2, 69), (1.4, 0.14, 71), (1.54, 0.14, 69), (1.68, 0.14, 67), (1.82, 1.6, 69)]
        out[f"melisma-{syl}"] = [row(t, d, p, 70, syl if i == 0 else None) for i, (t, d, p) in enumerate(turn)]
        slide = [(0.2, 0.9, 69), (1.1, 0.5, 65), (1.6, 1.2, 66), (2.8, 1.6, 62)]
        out[f"slide-{syl}"] = [row(t, d, p, 72, syl if i == 0 else None) for i, (t, d, p) in enumerate(slide)]
        notes, t = [], 0.2
        for phrase in ([67, 69, 70, 69], [69, 67, 65, 64], [65, 64, 62]):
            for p in phrase:
                notes.append(row(t, 0.55, p, 64, syl))
                t += 0.55
            notes[-1]["off"] += 0.5
            t += 1.1          # a breath
        out[f"breaths-{syl}"] = notes
    return out


def piece_rows(manifest, start, end, model="voice"):
    """The window's sung notes and CC11 expression, re-timed to start at 0, and the piece settings."""
    from soundbench.scripts.perform import TempoMap
    if model == "voice2":
        from soundbench.scripts.render_voice2 import sung_notes
    else:
        from soundbench.scripts.render_voice import sung_notes
    data = json.loads(Path(manifest).read_text())
    performed, score = mido.MidiFile(data["performed_midi"]), mido.MidiFile(data["score_midi"])
    index = data["jobs"][0]["index"]
    rows = sung_notes(performed, score, index)
    tempo, tick, controls = TempoMap(performed), 0, []
    for m in performed.tracks[index]:
        tick += m.time
        if m.type == "control_change" and m.control == 11:
            controls.append({"time": tempo.seconds_at(tick) - start, "control": 11, "value": m.value})
    before = [c for c in controls if c["time"] <= 0]
    controls = ([{**before[-1], "time": 0.0}] if before else []) + [c for c in controls if c["time"] > 0]
    # Carry the syllable of a melisma's first note into the window when it starts mid-melisma.
    last = None
    leading = None
    picked = []
    for r in rows:
        if r["syllable"] is not None:
            last = r["syllable"]
        if start <= r["time"] < end:
            if not picked:
                leading = last
            picked.append(dict(r))
    if picked and picked[0]["syllable"] is None:
        picked[0]["syllable"] = leading if model == "voice2" else last
    for r in picked:
        r["time"] -= start
        r["off"] = min(r["off"], end) - start
    return picked, controls, data.get("settings", {})


def write(path, audio):
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="voice", help="module under scripts/models with render(notes, seconds, config)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preset", choices=("tenor", "alto"), default="tenor", help="voice2 preset")
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--piece")
    parser.add_argument("--window")
    parser.add_argument("--name")
    args = parser.parse_args()
    model = importlib.import_module(f"models.{args.model}")
    if args.piece:
        start, end = (float(v) for v in args.window.split("-"))
        rows, controls, settings = piece_rows(args.piece, start, end, args.model)
        jobs = {args.name: rows}
        tail = 3.0
        if args.model == "voice":
            # v1 sings with the piece's own [voice_model] settings, as in the build.
            from soundbench.scripts.render_voice import selected_config
            from soundbench.scripts.models.voice import syllable_key
            config, options, _ = selected_config(settings)
            if "ah_gain_db" in options:
                profile, syllable = np.array(options["ah_gain_db"]), None
                for r in rows:
                    syllable = syllable_key(r["syllable"]) if r["syllable"] is not None else syllable
                    r["source_gain_db"] = float(np.interp(r["pitch"], profile[:, 0], profile[:, 1])) if syllable == "ah" else 0.0
        else:
            config = None
    else:
        jobs = {k: v for k, v in phrases().items() if not args.only or k in args.only}
        tail, controls, config = 1.5, (), None
    if args.model == "voice2":
        config = model.load_config(args.preset)
    for name, rows in jobs.items():
        seconds = max(r["off"] for r in rows) + tail
        audio = model.render(rows, seconds, config, midi_controls=controls)
        peak = float(np.max(np.abs(audio)))
        write(args.out / f"{name}.wav", audio / max(peak, 1e-9) * 0.5)
        print(f"{name}: {len(rows)} notes, {seconds:.1f} s, raw peak {peak:.3f}")


if __name__ == "__main__":
    main()
