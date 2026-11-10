#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["mido>=1.3,<2", "numpy>=1.26", "numba>=0.61", "scipy>=1.11", "cmudict>=1.0"]
# ///
"""Render voice3 parts: the score's lyrics sung by one singer or a section per part.

Each bound track is a part. Its words come from the Lyrics context attached to
it with \\lyricsto (the hyphens that join a word's syllables survive in the
MIDI through scripts/lilypond/sung-lyrics.ily), or from another part's lyrics
(`lyrics_from`): a syllable that falls inside one of this part's notes splits
the note, so a held harmony note re-sings the melody's words.

Sidecar:

    [instruments]
    lead = { model = "voice3" }
    alto = { model = "voice3", pan = -0.2 }

    [voice3]
    seed = 7                                   # the night: vibrato, flutter, scatter
    lexicon = { wayfaring = "W EY1 F EH2 R IH0 NG" }   # words CMUdict lacks, in ARPAbet
    [voice3.parts]
    lead = { singer = "tenor", character = "plain" }
    alto = { singers = ["alto", "alto", "soprano"], lyrics_from = "lead", vibrato = 0.6, lag_ms = 15 }
    robot = { singer = "machine", machine = "vocoder" }   # or { preset = "talkbox", lowpass_hz = 3000 }

Character can also be a table, e.g. {base = "intimate", vibrato_rate = 5.2}.
Sparse [[voice3.marks]] tables select part and bar, or inclusive bars = [first, last],
with optional beat (the starting beat in the first bar). Marks set character or
individual character fields, including vibrato = "straight" or "emerging".
Later overlapping marks win; every span restores the part defaults outside it.
Marks act on note onsets in score coordinates, including through performed rubato.

A part without an entry takes the preset named like its track (bass,
baritone, tenor, alto, soprano), else the one whose range fits its notes.
`range = [low, high]` lets a part sing outside its singers' gated ranges on
purpose (out-of-range notes otherwise raise).
`machine` sends every singer of a part through the machine voice
(scripts/models/machine_voice.py: corrected, vocoder, talkbox, speaking-piano),
pitch-corrected to the score's key unless the table gives a scale.
"""
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

from soundbench.scripts.instrumentation import aliases, matching_tracks
from soundbench.scripts.models import phonetics, voice3
from soundbench.scripts.perform import BarMap, TempoMap
from soundbench.scripts.render_model import notes as track_notes, write_wav
from soundbench.scripts.render_voice import lyric_tracks
from soundbench.scripts.render_machine_voice import selected_config as selected_machine

PART_KEYS = {"singer", "singers", "lyrics_from", "vibrato", "breathiness", "tension", "level_db",
             "consonant_scale", "timing_sd_s", "scatter", "spread", "machine", "lag_ms", "range", "character"}
LIMITS = {"vibrato": (0, 3), "breathiness": (0, 4), "tension": (-1, 1), "level_db": (-24, 12),
          "consonant_scale": (0.5, 2), "timing_sd_s": (0, 0.06), "scatter": (0, 3), "spread": (0, 1),
          "lag_ms": (-60, 60)}


def settings_for(settings):
    options = settings.get("voice3", {})
    if not isinstance(options, dict) or set(options) - {"seed", "lexicon", "parts", "spread", "marks"}:
        raise ValueError("[voice3] takes seed, lexicon, spread, parts and marks")
    seed = options.get("seed", voice3._table()["engine"]["seed"])
    if type(seed) is not int or seed < 0:
        raise ValueError("voice3 seed must be a nonnegative integer")
    lexicon = options.get("lexicon", {})
    if not isinstance(lexicon, dict):
        raise ValueError("voice3 lexicon maps words to ARPAbet strings")
    for word, spec in lexicon.items():
        phonetics._parse(spec if isinstance(spec, str) else " ".join(spec))
    parts = options.get("parts", {})
    if not isinstance(parts, dict):
        raise ValueError("voice3 parts must be a table")
    for name, part in parts.items():
        if not isinstance(part, dict) or set(part) - PART_KEYS:
            raise ValueError(f"[voice3.parts.{name}] takes {sorted(PART_KEYS)}")
        names = part.get("singers", [part["singer"]] if "singer" in part else [])
        if any(n not in voice3.singers() for n in names):
            raise ValueError(f"voice3 singers must be among {voice3.singers()}")
        for key, (lo, hi) in LIMITS.items():
            if key in part and (type(part[key]) not in (int, float) or not lo <= part[key] <= hi):
                raise ValueError(f"[voice3.parts.{name}].{key} must be {lo}..{hi}")
        voice3.character_settings(part.get("character", "plain"))
        if "machine" in part:
            machine_settings(part["machine"])
        if "range" in part and (not isinstance(part["range"], list) or len(part["range"]) != 2
                                or not all(type(v) is int and 0 <= v <= 127 for v in part["range"])):
            raise ValueError(f"[voice3.parts.{name}].range is [low, high] in MIDI")
    validate_marks(options.get("marks", []))
    return seed, lexicon, parts, float(options.get("spread", 0.5))


def validate_marks(marks):
    """Validate syntax without loading MIDI or rendering audio."""
    if not isinstance(marks, list):
        raise ValueError("voice3 marks must be an array of tables")
    fields = set(voice3.CHARACTER_LIMITS) | set(voice3.CHARACTER_CHOICES)
    for mark in marks:
        if not isinstance(mark, dict) or set(mark) - fields - {"part", "bar", "bars", "beat", "character"}:
            raise ValueError("unknown voice3 mark keys")
        if not isinstance(mark.get("part"), str) or ("bar" in mark) == ("bars" in mark):
            raise ValueError("voice3 mark needs part and exactly one of bar or bars")
        bars = [mark["bar"], mark["bar"]] if "bar" in mark else mark["bars"]
        if (not isinstance(bars, list) or len(bars) != 2
                or any(type(b) is not int or b < 1 for b in bars) or bars[0] > bars[1]):
            raise ValueError("voice3 mark bars must be an inclusive [first, last] range")
        if "beat" in mark and (type(mark["beat"]) not in (int, float) or not math.isfinite(mark["beat"])):
            raise ValueError("voice3 mark beat must be finite")
        parent = voice3.character_settings(mark.get("character", "plain"))
        voice3.character_settings({k: v for k, v in mark.items() if k in fields}, parent)


def marks_for(marks, score, names):
    """Resolve score positions, rejecting absent parts and bars before rendering.

    Selection uses score ticks, the same coordinate carried by score_rows;
    tempo changes and performed rubato therefore cannot move a mark to another note.
    Score seconds are retained for inspection using the score's own tempo map.
    A beat starts the span in its first bar; its last bar is included in full.
    """
    validate_marks(marks)
    bars, tempo = BarMap(score), TempoMap(score)
    last_tick = max((r["off"] for i in range(len(score.tracks)) for r in track_notes(score, i)), default=1)
    last_bar = bars.bar_at(max(0, last_tick - 1))
    result = defaultdict(list)
    for mark in marks:
        matches = matching_tracks(mark["part"], list(names), names)
        if len(matches) != 1:
            raise ValueError(f"voice3 mark part {mark['part']!r} must select one sung part")
        first, last = [mark["bar"]] * 2 if "bar" in mark else mark["bars"]
        if last > last_bar:
            raise ValueError(f"voice3 mark bar {last} is outside score (last bar {last_bar})")
        a, b = bars.position(first, mark.get("beat", 1)), bars.bar_start(last + 1)
        patch = {k: v for k, v in mark.items() if k in voice3.CHARACTER_LIMITS or k in voice3.CHARACTER_CHOICES}
        result[matches[0]].append({"start_tick": a, "end_tick": b,
                                  "start_seconds": tempo.seconds_at(a), "end_seconds": tempo.seconds_at(b),
                                  "character": mark.get("character"), "patch": patch})
    return result


def apply_marks(rows, part, marks):
    """Attach resolved intent to note onsets; later overlapping marks win."""
    default = voice3.character_settings(part.get("character", "plain"))
    out = []
    for row in rows:
        ch = default.copy()
        passage = None
        explicit = set()
        for mark in marks:
            if mark["start_tick"] <= row["tick"] < mark["end_tick"]:
                if mark["character"] is not None:
                    ch = voice3.character_settings(mark["character"])
                ch = voice3.character_settings(mark["patch"], ch)
                # Only a mode or character establishes a vibrato passage;
                # field-only marks retain the owning passage's progress.
                if mark["character"] is not None or "vibrato" in mark["patch"]:
                    passage = mark
                explicit.update(mark["patch"])
        r = {**row, "intent": ch, "marked_onset": bool(explicit & {"onset", "attack_ms"}),
             "marked_release": bool(explicit & {"release", "breath_tail_ms"})}
        if ch["vibrato"] == "emerging":
            selected = [x for x in rows if passage is None or passage["start_tick"] <= x["tick"] < passage["end_tick"]]
            start, end = selected[0]["time"], selected[-1]["time"]
            r["emerging_progress"] = float(np.clip((row["time"] - start) / max(end - start, 0.001), 0, 1))
        out.append(r)
    return out


def machine_settings(spec, score=None):
    """A part's machine colour: a machine-voice preset name, or a table with
    `preset` and any machine-voice overrides, over voice3.toml's [machine]
    defaults. The scale defaults to the score's key."""
    options = {"preset": spec} if isinstance(spec, str) else spec
    if not isinstance(options, dict):
        raise ValueError("voice3 machine is a machine-voice preset name or a table with preset and overrides")
    return selected_machine({"machine_voice": {**voice3._table()["machine"], **options}}, score)[0]


def track_syllables(score, lyric_index, lexicon):
    """tick -> phonetics.Syllable for one lyric track."""
    events = lyric_tracks(score)[lyric_index]
    ticks = sorted(events)
    parsed = [phonetics.lyric_event(events[t]) for t in ticks]
    return dict(zip(ticks, phonetics.syllabify(parsed, lexicon)))


def aligned_lyrics(score, index):
    """The lyric track sung by a note track: one whose syllables all fall on the
    track's onsets, preferring one that shares the track's name, then the one
    with the most syllables. None when there is none; an error on a tie
    (stanzas stacked on one Voice: write the song out in order, one Lyrics per Voice)."""
    onsets = {r["tick"] for r in track_notes(score, index)}
    name = aliases(score.tracks[index].name)
    fits = [((bool(name & aliases(score.tracks[i].name)), len(events)), i) for i, events in lyric_tracks(score).items()
            if set(events) <= onsets]
    if not fits:
        return None
    # The named one, else the one that sings the most of this track's notes.
    best = [i for rank, i in fits if rank == max(r for r, _ in fits)]
    if len(best) > 1:
        raise ValueError(f"track {score.tracks[index].name!r} has {len(best)} Lyrics that fit its notes equally; "
                         "voice3 sings one: write the verses out in order in one Lyrics per Voice")
    return best[0]


def score_rows(performed, score, index):
    """Performed notes with the score tick each one plays (matched in pitch order)."""
    tempo = TempoMap(performed)
    scored = defaultdict(deque)
    for r in track_notes(score, index):
        scored[r["pitch"]].append(r)
    rows = []
    for r in track_notes(performed, index):
        if not scored[r["pitch"]]:
            raise ValueError("score/performed note correspondence differs")
        s = scored[r["pitch"]].popleft()
        rows.append({"time": tempo.seconds_at(r["tick"]), "off": tempo.seconds_at(r["off"]),
                     "pitch": r["pitch"], "velocity": r["velocity"], "tick": s["tick"], "off_tick": s["off"]})
    by_tick = sorted(rows, key=lambda r: (r["tick"], r["off_tick"]))
    for a, b in zip(by_tick, by_tick[1:]):
        if b["tick"] < a["off_tick"]:
            raise ValueError(f"voice3 sings one note at a time: track {score.tracks[index].name!r} has notes "
                             f"sounding together at tick {b['tick']}; give each sung line its own Voice")
    return sorted(rows, key=lambda r: r["time"])


def borrow(rows, syllables, tempo, gap_s=None):
    """Give a part another part's syllables: on a shared onset the note takes the
    syllable; inside a held note the note is split and re-sung. A note that
    begins a phrase (after a rest longer than gap_s) off the melody's onsets
    takes the syllable the melody last sang."""
    gap_s = voice3._table()["timing"]["legato_gap_s"] if gap_s is None else gap_s
    out = []
    ticks = sorted(syllables)
    for r in rows:
        inside = [t for t in ticks if r["tick"] < t < r["off_tick"]]
        cuts = [r["tick"], *inside]
        for k, tick in enumerate(cuts):
            start = r["time"] if k == 0 else tempo.seconds_at(tick)
            if start >= r["off"]:
                raise ValueError(f"a borrowed syllable at tick {tick} falls after its note ends (tempo maps differ)")
            end = r["off"] if k == len(cuts) - 1 else tempo.seconds_at(cuts[k + 1]) - 0.01
            syllable = syllables.get(tick)
            if syllable is None and (not out or start - out[-1]["off"] > gap_s):
                earlier = [t for t in ticks if t <= tick]
                if not earlier:
                    raise ValueError("a borrowing part starts before the first borrowed syllable")
                syllable = syllables[earlier[-1]]
            out.append({**r, "time": start, "off": max(end, start + 0.02), "tick": tick, "syllable": syllable})
    return out


def default_singers(name, rows):
    named = [s for s in voice3.singers() if s in aliases(name)]
    if named:
        return [named[0]]
    median = float(np.median([r["pitch"] for r in rows]))
    ranges = {s: voice3._table()["singers"][s]["range"] for s in voice3.singers() if s != "machine"}
    return [min(ranges, key=lambda s: abs(sum(ranges[s]) / 2 - median))]


def controls_for(performed, index):
    tempo, tick, out = TempoMap(performed), 0, []
    for m in performed.tracks[index]:
        tick += m.time
        if m.type == "control_change" and m.control in (2, 11):
            out.append({"time": tempo.seconds_at(tick), "control": m.control, "value": m.value})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    data = json.loads(args.manifest.read_text())
    settings = data.get("settings", {})
    seed, lexicon, parts, spread = settings_for(settings)
    performed, score = mido.MidiFile(data["performed_midi"]), mido.MidiFile(data["score_midi"])
    tempo = TempoMap(performed)
    note_tracks = [i for i, t in enumerate(performed.tracks) if any(m.type == "note_on" and m.velocity for m in t)]
    names = {i: performed.tracks[i].name for i in note_tracks}
    end = max(sum(m.time for m in track) for track in performed.tracks)
    seconds = tempo.seconds_at(end) + voice3._table()["engine"]["tail_seconds"]
    found = {}

    def lyrics_of(i):
        # Only for sung tracks: an accompaniment's onsets may cover several Lyrics.
        if i not in found:
            found[i] = aligned_lyrics(score, i)
        return found[i]
    part_of = {}
    for selector, part in parts.items():
        matches = matching_tracks(selector, note_tracks, names)
        if len(matches) != 1:
            raise ValueError(f"[voice3.parts.{selector}] must select one note track; names: {names}")
        part_of[matches[0]] = part
    marked = marks_for(settings.get("voice3", {}).get("marks", []), score,
                       {job["index"]: names[job["index"]] for job in data["jobs"]})
    warm = time.perf_counter()
    voice3.render([{"time": 0.4, "off": 0.5, "pitch": 60, "velocity": 64,
                    "syllable": phonetics.Syllable("a", [], ["AA1"], [])}], 0.7)
    compile_seconds = time.perf_counter() - warm
    stems, sung, peak, elapsed = {}, {}, 0.0, 0.0
    digest = hashlib.sha256(voice3.CONFIG.read_bytes() + voice3.v1.CONFIG.read_bytes()
                            + json.dumps(settings.get("voice3", {}), sort_keys=True).encode()).hexdigest()
    for job in data["jobs"]:
        index = job["index"]
        part = part_of.get(index, {})
        rows = score_rows(performed, score, index)
        if "lyrics_from" in part:
            source = matching_tracks(part["lyrics_from"], note_tracks, names)
            if len(source) != 1 or lyrics_of(source[0]) is None:
                raise ValueError(f"lyrics_from {part['lyrics_from']!r} must name one track with lyrics")
            rows = borrow(rows, track_syllables(score, lyrics_of(source[0]), lexicon), tempo)
        elif lyrics_of(index) is not None:
            syllables = track_syllables(score, lyrics_of(index), lexicon)
            for r in rows:
                r["syllable"] = syllables.get(r["tick"])
        else:
            raise ValueError(f"voice3 part {names[index]!r} has no Lyrics aligned to its notes: attach one with "
                             "\\new Lyrics \\lyricsto, or borrow another part's with lyrics_from")
        rows = apply_marks(rows, part, marked.get(index, []))
        if part.get("lag_ms"):
            # A harmony singer a little behind (or ahead of) the lead, every note.
            lag = part["lag_ms"] / 1000
            rows = [{**r, "time": max(0.0, r["time"] + lag), "off": max(0.0, r["off"] + lag)} for r in rows]
        singers = part.get("singers", [part["singer"]] if "singer" in part else default_singers(names[index], rows))
        overrides = {k: part[k] for k in ("tension", "consonant_scale", "timing_sd_s", "level_db", "character") if k in part}
        if "range" in part:
            # Outside the gated range on purpose: allowed, and said.
            print(f"Voice3: part {names[index]!r} sings outside its singers' gated range: {part['range']}")
            overrides["range"] = part["range"]
        if "vibrato" in part:
            overrides["vibrato_scale"] = part["vibrato"]
        if "breathiness" in part:
            overrides["breathiness"] = part["breathiness"]
        start = time.perf_counter()
        part_seed = int(np.random.SeedSequence([seed, index]).generate_state(1)[0] % 2**31)
        colour = machine_settings(part["machine"], score) if "machine" in part else None
        audio = voice3.sing(rows, singers, seconds, seed=part_seed, midi_controls=controls_for(performed, index),
                            spread=part.get("spread", spread), overrides=overrides,
                            scatter_scale=part.get("scatter", 1.0), machine_settings=colour)
        elapsed += time.perf_counter() - start
        write_wav(Path(job["output"]), audio, voice3.SR)
        peak = max(peak, float(np.max(np.abs(audio), initial=0)))
        stems[str(index)] = {"engine": "model", "model": "voice3", "singers": singers,
                             "parameters": str(voice3.CONFIG.relative_to(voice3.CONFIG.parents[2])),
                             "parameters_sha256": digest, "overrides": part}
        sung[str(index)] = [{"time_s": r["time"], "pitch": r["pitch"],
                             "phones": r["syllable"].phones if r.get("syllable") else None} for r in rows]
    metrics = {"stems": stems, "audio_seconds": seconds, "render_seconds": elapsed,
               "compile_or_cache_seconds": compile_seconds,
               "realtime_factor": seconds * len(data["jobs"]) / max(elapsed, 1e-9), "peak": peak, "sung": sung}
    args.manifest.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Voice3: {len(data['jobs'])} part(s), {seconds:.2f} s audio in {elapsed:.2f} s; peak {peak:.3f}")


if __name__ == "__main__":
    main()
