#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["mido>=1.3,<2", "numpy>=1.26", "numba>=0.61", "scipy>=1.11"]
# ///
"""Render physical-model voice tracks, singing the score's lyrics, into stereo WAV stems.

Lyrics come from the LilyPond score MIDI: a Lyrics context attached with
\\lyricsto writes its syllables as lyric meta events on a track of their own,
at the onsets of the notes they are sung to. A voice track takes the lyric
track whose events all fall on its note onsets (preferring one that shares a
name with it); a note with no syllable continues the previous one (melisma).
Performed notes are matched to score notes in pitch order, as for the banjo.
"""
from __future__ import annotations
import argparse
from collections import defaultdict, deque
import hashlib
import json
import math
from pathlib import Path
import re
import time

import mido
import numpy as np

from soundbench.scripts.instrumentation import aliases
from soundbench.scripts.models.voice import CONFIG, LEXICON, load_config, render, syllable_key
from soundbench.scripts.perform import TempoMap
from soundbench.scripts.render_model import notes as track_notes, write_wav


def lyric_tracks(midi):
    found = {}
    for i, track in enumerate(midi.tracks):
        if any(m.type == "note_on" and m.velocity for m in track):
            continue
        tick, events = 0, {}
        for m in track:
            tick += m.time
            if m.type == "lyrics":
                events[tick] = m.text.strip()
        if events:
            found[i] = events
    return found


def lyrics_for(score, index):
    onsets = {r["tick"] for r in track_notes(score, index)}
    name = aliases(score.tracks[index].name)
    matches = [(bool(name & aliases(score.tracks[i].name)), i, events)
               for i, events in lyric_tracks(score).items() if set(events) <= onsets]
    if not matches:
        raise ValueError(f"voice track {score.tracks[index].name!r} has no Lyrics aligned to its notes; "
                         "attach one with \\new Lyrics \\lyricsto")
    return max(matches, key=lambda m: (m[0], len(m[2]), -m[1]))[2]


def sung_notes(performed, score, index, unknown_syllable=None):
    """Scored notes at performed times with their lyric syllables.

    With `unknown_syllable`, a lyric missing from the lexicon is sung as that
    syllable instead (a hum or a vowel), so words can be scored before the
    voice can pronounce them; the row keeps the scored text as `word`.
    """
    tempo = TempoMap(performed)
    lyrics = lyrics_for(score, index)
    by_pitch = defaultdict(deque)
    for row in track_notes(score, index):
        by_pitch[row["pitch"]].append(row)
    rows = []
    for row in track_notes(performed, index):
        if not by_pitch[row["pitch"]]:
            raise ValueError("score/performed note correspondence differs")
        source = by_pitch[row["pitch"]].popleft()
        text = word = lyrics.get(source["tick"])
        if text is not None and unknown_syllable is not None and re.sub(r"[^a-z]", "", text.casefold()) not in LEXICON:
            text = unknown_syllable
        if text is not None:
            syllable_key(text)
        rows.append({"time": tempo.seconds_at(row["tick"]), "off": tempo.seconds_at(row["off"]),
                     "pitch": row["pitch"], "velocity": row["velocity"], "syllable": text,
                     **({"word": word} if word != text else {})})
    return rows


def selected_config(settings):
    """Piece-scoped source/release controls with hashed provenance."""
    options = settings.get("voice_model", {})
    keys = {"portamento_s": ("expression", 0.01, 0.5),
            "vibrato_cents": ("expression", 0.0, 100.0),
            "release_s": ("expression", 0.01, 5.0),
            "tail_seconds": ("engine", 0.1, 10.0),
            "output_lowpass_hz": ("engine", 100.0, 11000.0)}
    if set(options) - (set(keys)|{'ah_gain_db', 'unknown_syllable'}):
        raise ValueError("unknown voice_model setting")
    config = load_config()
    for key, value in options.items():
        if key=='unknown_syllable':
            if not isinstance(value,str) or re.sub(r"[^a-z]","",value.casefold()) not in LEXICON:
                raise ValueError('voice_model.unknown_syllable must be a lexicon syllable')
            continue
        if key=='ah_gain_db':
            if (not isinstance(value,list) or not value or any(len(row)!=2 or not isinstance(row[0],int)
                or not 0<=row[0]<=127 or not math.isfinite(row[1]) or not -40<=row[1]<=0 for row in value)
                or any(a[0]>=b[0] for a,b in zip(value,value[1:]))):
                raise ValueError('ah_gain_db requires ordered [MIDI pitch, cut dB] points')
            continue
        section, low, high = keys[key]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"invalid voice_model.{key}")
        config[section][key] = float(value)
    if config["engine"]["tail_seconds"] < config["expression"]["release_s"]:
        raise ValueError("voice tail must contain its release")
    digest = hashlib.sha256(CONFIG.read_bytes() + (json.dumps(options, sort_keys=True).encode() if options else b"")).hexdigest()
    return config, options, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    data = json.loads(args.manifest.read_text())
    config, options, parameter_hash = selected_config(data.get("settings", {}))
    performed, score = mido.MidiFile(data["performed_midi"]), mido.MidiFile(data["score_midi"])
    tempo = TempoMap(performed)
    end = max(sum(m.time for m in track) for track in performed.tracks)
    seconds = tempo.seconds_at(end) + config["engine"]["tail_seconds"]
    warm = time.perf_counter()
    render([{"time": 0, "off": 0.05, "pitch": 60, "velocity": 64, "syllable": "a"}], 0.1, config)
    compile_seconds = time.perf_counter() - warm
    stems, sung, peak, elapsed = {}, {}, 0.0, 0.0
    for job in data["jobs"]:
        rows = sung_notes(performed, score, job["index"], options.get("unknown_syllable"))
        if 'ah_gain_db' in options:
            syllable=None; profile=np.array(options['ah_gain_db'])
            for row in rows:
                if row['syllable'] is not None: syllable=syllable_key(row['syllable'])
                row['source_gain_db']=float(np.interp(row['pitch'],profile[:,0],profile[:,1])) if syllable=='ah' else 0
        start = time.perf_counter()
        tick=0; expression=[]
        for message in performed.tracks[job['index']]:
            tick+=message.time
            if message.type=='control_change' and message.control==11:
                expression.append(dict(time=tempo.seconds_at(tick),control=11,value=message.value))
        audio = render(rows, seconds, config, midi_controls=expression)
        elapsed += time.perf_counter() - start
        write_wav(Path(job["output"]), audio, config["engine"]["sample_rate"])
        peak = max(peak, float(np.max(np.abs(audio), initial=0)))
        stems[str(job["index"])] = {"engine": "model", "model": "voice",
                                    "parameters": str(CONFIG.relative_to(CONFIG.parents[2])),
                                    "parameters_sha256": parameter_hash, "overrides": options}
        sung[str(job["index"])] = [{"time_s": r["time"], "pitch": r["pitch"], "syllable": r["syllable"]} for r in rows]
    metrics = {"stems": stems, "audio_seconds": seconds, "render_seconds": elapsed,
               "compile_or_cache_seconds": compile_seconds,
               "realtime_factor": seconds * len(data["jobs"]) / max(elapsed, 1e-9),
               "sung": sung, "peak": peak}
    args.manifest.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Voice model: {len(data['jobs'])} track(s), {seconds:.2f} s audio in {elapsed:.3f} s; "
          f"compile/cache {compile_seconds:.3f} s; peak {peak:.3f}")


if __name__ == "__main__":
    main()
