"""The score side of singing: which words each sung part sings, and how.

Each bound track is a part. Its words come from the Lyrics context attached to
it with \\lyricsto (the hyphens that join a word's syllables survive in the
MIDI), or from another part's lyrics (`lyrics_from`): a syllable that falls
inside one of this part's notes splits the note, so a held harmony note
re-sings the melody's words.

Sidecar:

    [singing]
    seed = 7                                   # the night: vibrato and timing draws
    lexicon = { wayfaring = "W EY1 F EH2 R IH0 NG" }   # words CMUdict lacks, in ARPAbet
    [singing.parts]
    lead = { singer = "tenor", character = "plain" }
    alto = { singer = "alto", lyrics_from = "lead", vibrato = 0.6, lag_ms = 15 }

Character can also be a table, e.g. {base = "intimate", vibrato_rate = 5.2}.
Sparse [[singing.marks]] tables select part and bar, or inclusive bars = [first, last],
with optional beat (the starting beat in the first bar). Marks set character or
individual character fields, including vibrato = "straight" or "emerging".
Later overlapping marks win; every span restores the part defaults outside it.
Marks act on note onsets in score coordinates, including through performed rubato.

A part without an entry takes the singer named like its track (bass,
baritone, tenor, alto, soprano), else the one whose range fits its notes.
`range = [low, high]` lets a part sing outside its singer's gated range on
purpose. Ranges, characters and phrase timing are in singing.toml.
"""
from __future__ import annotations
from collections import defaultdict, deque
from functools import lru_cache
import math
from pathlib import Path
import tomllib

import numpy as np

from soundbench.scripts.instrumentation import aliases, matching_tracks
from soundbench.scripts.models import phonetics
from soundbench.scripts.perform import BarMap, TempoMap
from soundbench.scripts.render_model import notes as track_notes

TABLE = Path(__file__).with_name("singing.toml")
PART_KEYS = {"singer", "singers", "lyrics_from", "vibrato", "breathiness", "tension", "level_db",
             "consonant_scale", "lag_ms", "range", "character"}
LIMITS = {"vibrato": (0, 3), "breathiness": (0, 4), "tension": (-1, 1), "level_db": (-24, 12),
          "consonant_scale": (0.5, 2), "lag_ms": (-60, 60)}
CHARACTER_LIMITS = {
    "effort_center": (0, 1), "effort_span": (0, 1.5),
    "vibrato_rate": (5, 8), "vibrato_extent": (0, 50), "vibrato_delay": (0.1, 0.5),
    "attack_ms": (20, 150), "breath_tail_ms": (40, 200), "breath_gain": (0, 4),
    "roughness": (0, 2), "sag_cents": (0, 12), "vowel_opening_hz": (0, 40),
}
CHARACTER_CHOICES = {"onset": {"soft", "breathy", "hard", "scooped"},
                     "release": {"clean", "fall", "breath", "held"},
                     "vibrato": {"natural", "straight", "emerging"}}


@lru_cache(maxsize=1)
def table():
    return tomllib.loads(TABLE.read_text())


def singers():
    return sorted(table()["singers"])


def singer_range(singer):
    return table()["singers"][singer]["range"]


def character_settings(spec="plain", parent=None):
    """Resolve a named character or a strictly bounded table on top of a base."""
    if isinstance(spec, str):
        spec = {"base": spec}
    if not isinstance(spec, dict):
        raise ValueError("a character is a name or a table with base and overrides")
    unknown = set(spec) - {"base"} - CHARACTER_LIMITS.keys() - CHARACTER_CHOICES.keys()
    if unknown:
        raise ValueError(f"unknown character fields {sorted(unknown)}")
    base = spec.get("base", "plain")
    if not isinstance(base, str) or base not in table()["characters"]:
        raise ValueError(f"unknown character {base!r}")
    result = {**table()["characters"][base], **(parent or {}), **spec}
    for key, (lo, hi) in CHARACTER_LIMITS.items():
        if type(result[key]) not in (int, float) or not lo <= result[key] <= hi:
            raise ValueError(f"character {key} must be {lo}..{hi}")
    for key, choices in CHARACTER_CHOICES.items():
        if not isinstance(result[key], str) or result[key] not in choices:
            raise ValueError(f"character {key} must be one of {sorted(choices)}")
    return result


def settings_for(settings):
    """The [singing] block's seed, lexicon and parts, validated."""
    options = settings.get("singing", {})
    if not isinstance(options, dict) or set(options) - {"seed", "lexicon", "parts", "marks"}:
        raise ValueError("[singing] takes seed, lexicon, parts and marks")
    seed = options.get("seed", table()["seed"])
    if type(seed) is not int or seed < 0:
        raise ValueError("singing seed must be a nonnegative integer")
    lexicon = options.get("lexicon", {})
    if not isinstance(lexicon, dict):
        raise ValueError("singing lexicon maps words to ARPAbet strings")
    for word, spec in lexicon.items():
        phonetics._parse(spec if isinstance(spec, str) else " ".join(spec))
    parts = options.get("parts", {})
    if not isinstance(parts, dict):
        raise ValueError("singing parts must be a table")
    for name, part in parts.items():
        if not isinstance(part, dict) or set(part) - PART_KEYS:
            raise ValueError(f"[singing.parts.{name}] takes {sorted(PART_KEYS)}")
        names = part.get("singers", [part["singer"]] if "singer" in part else [])
        if any(n not in singers() for n in names):
            raise ValueError(f"singers must be among {singers()}")
        for key, (lo, hi) in LIMITS.items():
            if key in part and (type(part[key]) not in (int, float) or not lo <= part[key] <= hi):
                raise ValueError(f"[singing.parts.{name}].{key} must be {lo}..{hi}")
        character_settings(part.get("character", "plain"))
        if "range" in part and (not isinstance(part["range"], list) or len(part["range"]) != 2
                                or not all(type(v) is int and 0 <= v <= 127 for v in part["range"])):
            raise ValueError(f"[singing.parts.{name}].range is [low, high] in MIDI")
    validate_marks(options.get("marks", []))
    return seed, lexicon, parts


def validate_marks(marks):
    """Validate syntax without loading MIDI or rendering audio."""
    if not isinstance(marks, list):
        raise ValueError("singing marks must be an array of tables")
    fields = set(CHARACTER_LIMITS) | set(CHARACTER_CHOICES)
    for mark in marks:
        if not isinstance(mark, dict) or set(mark) - fields - {"part", "bar", "bars", "beat", "character"}:
            raise ValueError("unknown singing mark keys")
        if not isinstance(mark.get("part"), str) or ("bar" in mark) == ("bars" in mark):
            raise ValueError("a singing mark needs part and exactly one of bar or bars")
        bars = [mark["bar"], mark["bar"]] if "bar" in mark else mark["bars"]
        if (not isinstance(bars, list) or len(bars) != 2
                or any(type(b) is not int or b < 1 for b in bars) or bars[0] > bars[1]):
            raise ValueError("singing mark bars must be an inclusive [first, last] range")
        if "beat" in mark and (type(mark["beat"]) not in (int, float) or not math.isfinite(mark["beat"])):
            raise ValueError("singing mark beat must be finite")
        parent = character_settings(mark.get("character", "plain"))
        character_settings({k: v for k, v in mark.items() if k in fields}, parent)


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
            raise ValueError(f"singing mark part {mark['part']!r} must select one sung part")
        first, last = [mark["bar"]] * 2 if "bar" in mark else mark["bars"]
        if last > last_bar:
            raise ValueError(f"singing mark bar {last} is outside score (last bar {last_bar})")
        a, b = bars.position(first, mark.get("beat", 1)), bars.bar_start(last + 1)
        patch = {k: v for k, v in mark.items() if k in CHARACTER_LIMITS or k in CHARACTER_CHOICES}
        result[matches[0]].append({"start_tick": a, "end_tick": b,
                                  "start_seconds": tempo.seconds_at(a), "end_seconds": tempo.seconds_at(b),
                                  "character": mark.get("character"), "patch": patch})
    return result


def apply_marks(rows, part, marks):
    """Attach resolved intent to note onsets; later overlapping marks win."""
    default = character_settings(part.get("character", "plain"))
    out = []
    for row in rows:
        ch = default.copy()
        passage = None
        explicit = set()
        for mark in marks:
            if mark["start_tick"] <= row["tick"] < mark["end_tick"]:
                if mark["character"] is not None:
                    ch = character_settings(mark["character"])
                ch = character_settings(mark["patch"], ch)
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


def lyric_tracks(midi):
    """Track index -> {tick: lyric text} for every track that carries lyrics and no notes."""
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
                         "a part sings one: write the verses out in order in one Lyrics per Voice")
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
            raise ValueError(f"a part sings one note at a time: track {score.tracks[index].name!r} has notes "
                             f"sounding together at tick {b['tick']}; give each sung line its own Voice")
    return sorted(rows, key=lambda r: r["time"])


def borrow(rows, syllables, tempo, gap_s=None):
    """Give a part another part's syllables: on a shared onset the note takes the
    syllable; inside a held note the note is split and re-sung. A note that
    begins a phrase (after a rest longer than gap_s) off the melody's onsets
    takes the syllable the melody last sang."""
    gap_s = table()["legato_gap_s"] if gap_s is None else gap_s
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
    named = [s for s in singers() if s in aliases(name)]
    if named:
        return [named[0]]
    median = float(np.median([r["pitch"] for r in rows]))
    ranges = {s: singer_range(s) for s in singers()}
    return [min(ranges, key=lambda s: abs(sum(ranges[s]) / 2 - median))]


def controls_for(performed, index):
    """A track's breath (CC2) and expression (CC11) lanes in performed seconds."""
    tempo, tick, out = TempoMap(performed), 0, []
    for m in performed.tracks[index]:
        tick += m.time
        if m.type == "control_change" and m.control in (2, 11):
            out.append({"time": tempo.seconds_at(tick), "control": m.control, "value": m.value})
    return out
