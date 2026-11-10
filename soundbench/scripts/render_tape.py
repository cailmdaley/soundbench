#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=1.26", "scipy>=1.11"]
# ///
"""Record a piece's raw stems on the Nagra (scripts/models/nagra.py).

Reads a JSON manifest written by build.py:
  {"stems": [{"name", "wav", "gain_db"}], "rolls": {roll: {...}},
   "changes": [{"at_s", "roll", "gap_s"}], "seed", "ir" (null: no room), "machine", "output"}
and writes the output as stereo 32-bit PCM WAV (the same mono signal in both
channels unless the optional transfer layer is enabled) plus <output>.json with the roll placements and segments.
"""
from __future__ import annotations
import json
import copy
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from soundbench.scripts.models import nagra  # noqa: E402


def mono(path):
    rate, data = wavfile.read(path)
    if rate != nagra.SAMPLE_RATE:
        raise ValueError(f"{path}: {rate} Hz, expected {nagra.SAMPLE_RATE}")
    scale = {"int16": 2 ** 15, "int32": 2 ** 31}.get(str(data.dtype), 1)
    # float32 holds a 24-bit stem exactly enough (-144 dB) and halves the night's resident stems.
    data = np.asarray(data, dtype=np.float32) / np.float32(scale)
    return data.mean(axis=1, dtype=np.float32) if data.ndim > 1 else data


def write24(path, y):
    peak = float(np.max(np.abs(y), initial=0))
    if not np.all(np.isfinite(y)):
        raise ValueError("non-finite tape output")
    if peak >= 1:
        y = y * (0.98 / peak)
    pcm = np.rint(y * (2 ** 31 - 1)).astype("<i4")
    wavfile.write(path, nagra.SAMPLE_RATE, np.repeat(pcm[:, None], 2, axis=1) if pcm.ndim == 1 else pcm)
    return peak


def finish_program(noisy, music, sr, settings):
    """Keep the musical release; fade only noise after measured program decay."""
    if set(settings) - {"below_median_db", "tail_s", "hiss_fade_s"}:
        raise ValueError("unknown tape ending setting")
    relative = float(settings.get("below_median_db", -45))
    tail = float(settings.get("tail_s", 3))
    fade = float(settings.get("hiss_fade_s", 2))
    if not all(np.isfinite(v) for v in (relative, tail, fade)) or relative >= 0 or not 0 < fade <= tail:
        raise ValueError("ending needs a negative threshold and 0 < hiss_fade_s <= tail_s")
    win, hop = round(.4 * sr), round(.01 * sr)
    power = music ** 2 if music.ndim == 1 else np.mean(music ** 2, axis=1)
    starts = np.arange(0, len(power) - win + 1, hop)
    cumulative = np.r_[0., np.cumsum(power)]
    db = 10 * np.log10(np.maximum((cumulative[starts+win] - cumulative[starts]) / win, 1e-30))
    median = float(np.median(db))
    active = starts[db >= median + relative]
    if not len(active):
        raise ValueError("no musical energy for tape ending")
    decay = (active[-1] + win) / sr
    end = round((decay + tail) * sr)
    if end > len(music):
        raise ValueError("tape program buffer is too short for the requested ending")
    envelope = np.ones(end)
    length = round(fade * sr)
    envelope[-length:] = np.linspace(1, 0, length) ** 2
    if noisy.ndim > 1:
        envelope = envelope[:, None]
    output = music[:end] + (noisy[:end] - music[:end]) * envelope
    return output, {"method": "noise-free tape music, 400 ms RMS, 10 ms hop",
                    "median_dbfs": median, "threshold_dbfs": median + relative,
                    "music_decayed_s": decay, "end_s": end / sr,
                    "hiss_fade_start_s": end / sr - fade, "hiss_fade_s": fade}


def source_gain(signal, points):
    """Player projection before both the direct path and the room field."""
    points = np.asarray(points)
    gain = np.interp(np.arange(len(signal))/nagra.SAMPLE_RATE, points[:,0], points[:,1])
    return signal * (10**(gain/20)).astype(np.float32)


def main(argv=None):
    manifest = json.loads(Path((argv or sys.argv[1:])[0]).read_text())
    cfg = nagra.preset_config(nagra.load_config(manifest.get("machine") or nagra.CONFIG), manifest.get("preset"))
    stems = []
    for s in manifest['stems']:
        signal = mono(s['wav']) * np.float32(10 ** (s['gain_db'] / 20))
        if 'source_gain_points' in s:
            signal = source_gain(signal, s['source_gain_points'])
        stems.append((s['name'], signal))
    ir = nagra.room_ir(Path(manifest["ir"]), cfg["room"]["ir_skip_ms"]) if manifest["ir"] else np.zeros(1)
    rolls = manifest["rolls"] or {"A": {}}
    y, report = nagra.render(stems, rolls, manifest["changes"], cfg, ir, manifest["seed"])
    if manifest.get("ending"):
        if any(change['roll'] == 'stop' for change in manifest['changes']):
            raise ValueError("a natural tape ending cannot include a terminal transport stop")
        quiet_cfg, quiet_rolls = copy.deepcopy(cfg), copy.deepcopy(rolls)
        quiet_cfg["room"]["tone_dbfs"] = -300.0
        quiet_cfg["tape"]["hiss_dbfs"] = -300.0
        for roll in quiet_rolls.values():
            roll.setdefault("room", {})["tone_dbfs"] = -300.0
            roll.setdefault("tape", {})["hiss_dbfs"] = -300.0
        music, _ = nagra.render(stems, quiet_rolls, manifest["changes"], quiet_cfg, ir, manifest["seed"])
        y, report["ending"] = finish_program(y, music, nagra.SAMPLE_RATE, manifest["ending"])
    if manifest.get("transfer", False):
        y = nagra.transfer_stereo(y, cfg["transfer"])
        report["transfer"] = cfg["transfer"]
    report["preset"] = manifest.get("preset")
    out = Path(manifest["output"])
    report["pre_write_peak"] = round(write24(out, y), 4)
    report["seconds"] = round(len(y) / nagra.SAMPLE_RATE, 3)
    out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"tape: {len(report['segments'])} segment(s), {report['seconds']} s")


if __name__ == "__main__":
    main()
