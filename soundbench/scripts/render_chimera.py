#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["mido>=1.3,<2", "numpy>=1.26", "numba>=0.61", "scipy>=1.11", "cmudict>=1.0"]
# ///
"""Render CHIMERA from the studio's score/performed-MIDI manifest.

The voice3 lyric alignment and score marks are shared, not reinterpreted by
CHIMERA's text front end. --prepare-only writes native inputs without rendering.
This experimental adapter does not replace voice3 or any album master.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import mido
import numpy as np
from scipy.io import wavfile

from soundbench.scripts.instrumentation import matching_tracks
from soundbench.scripts.models import phonetics, voice3
from soundbench.scripts.perform import TempoMap
from soundbench.scripts.render_voice3 import (aligned_lyrics, apply_marks, borrow, controls_for, default_singers,
                           marks_for, score_rows, settings_for, track_syllables)

# The engine's source fork sits at the repository root, beside the package.
fork = Path(__file__).resolve().parents[2] / "vendor/chimera"
binary = fork / "target/release/chimera-studio"
UPSTREAM = "d361a82ce1b18d0ac19001aab19169ff2c1eb9e8"
ARPA = dict(zip(
    "AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW B CH D DH F G HH JH K L M N NG P R S SH T TH V W Y Z ZH".split(),
    "ɑ æ ʌ ɔ aʊ aɪ ɛ ɝ eɪ ɪ i oʊ ɔɪ ʊ u b tʃ d ð f g h dʒ k l m n ŋ p ɹ s ʃ t θ v w j z ʒ".split(), strict=True))


def segment(phone):
    ipa = ARPA[phonetics.base(phone)]
    if phone == "AH0":
        ipa = "ə"
    elif phone == "ER0":
        ipa = "ɚ"
    return {"ipa": ipa}


# Sung diction: a reduced vowel held on a long note takes its strong form
# ("and" sung on a beat is /ænd/, not a held schwa heard as "earned").
STRONG = {"ə": "ʌ", "ɚ": "ɝ"}
HELD = 0.45


def strong_form(syllable, parts, lexicon=None):
    """The held syllable's reduced nucleus, given the word's stressed reading if it has one."""
    vowel = None
    word = syllable.text.strip(" ,;.:?!\"'()").casefold()
    # Folk singers hold "a" as /ʌ/, not its citation /eɪ/.
    if syllable.word_start and syllable.word_end and len(parts) == 1 and word != "a":
        ours = [p for p in syllable.phones if not phonetics.is_vowel(p)]
        for entry in phonetics.pronunciations(word, lexicon):
            vowels = [p for p in entry if phonetics.is_vowel(p)]
            if [p for p in entry if not phonetics.is_vowel(p)] == ours and len(vowels) == 1 and vowels[0][-1] == "1":
                vowel = segment(vowels[0])["ipa"]
                break
    return [{**part, "nucleus": {"ipa": vowel or STRONG[part["nucleus"]["ipa"]]}}
            if part["nucleus"]["ipa"] in STRONG else part for part in parts]


def syllable_parts(syllable):
    """Compound nuclei become internal same-pitch subnotes; no phone is dropped."""
    phones = syllable.nucleus
    # A syllabic nasal/hiss is itself the nucleus.
    centers = [i for i, p in enumerate(phones) if phonetics.is_vowel(p)] or [0]
    result = []
    for k, center in enumerate(centers):
        following = centers[k + 1] if k + 1 < len(centers) else len(phones)
        result.append({"text": syllable.text,
                       "onset": [segment(p) for p in syllable.onset] if k == 0 else [],
                       "nucleus": segment(phones[center]),
                       "coda": [segment(p) for p in phones[center + 1:following]] +
                               ([segment(p) for p in syllable.coda] if k == len(centers) - 1 else []),
                       "wordEnd": syllable.word_end and k == len(centers) - 1,
                       "phraseEnd": syllable.text.rstrip().endswith((",", ";", ".", "?", "!")) and k == len(centers) - 1})
    return result


def timeline(rows, body, seconds, *, baseline=False, part=None, controls=(), seed=7, singer="tenor", lexicon=None):
    """Keep the shared lyric timeline; native expression never realigns its vowels.

    Baseline emits the frozen diagnostic protocol. Expressive inputs carry the
    shared resolved character/marks separately from upstream SungNote structures.
    """
    part = part or {}
    phrases = []
    previous = None
    phrase = None
    source = [{"time": 0.0, "pressure": 9.0, "tension": 0.55, "breathiness": 0.15}]
    # A syllable is held for its own note and every melisma note after it.
    held = [0.0] * len(rows)
    for k in reversed(range(len(rows))):
        held[k] = rows[k]["off"] - rows[k]["time"]
        if k + 1 < len(rows) and rows[k + 1].get("syllable") is None:
            held[k] += held[k + 1]
    last_parts = None
    for k, r in enumerate(rows):
        new_phrase = previous is None or r["time"] - previous["off"] > 0.08
        if new_phrase:
            phrase = {"t0": r["time"], "notes": [], "vibrato": 35.0 if baseline else 0.0}
            phrases.append(phrase)
        elif phrase["notes"]:
            # plan_song measures vowel-to-vowel, not note-off-to-note-on.
            phrase["notes"][-1]["beats"] += r["time"] - previous["off"]
        syllable = r.get("syllable")
        if syllable is not None:
            parts = syllable_parts(syllable)
            if not baseline and held[k] - (r["off"] - r["time"]) * (len(parts) - 1) / len(parts) >= HELD:
                parts = strong_form(syllable, parts, lexicon)
            last_parts = parts
        elif last_parts is not None:
            parts = [{**last_parts[-1], "onset": [], "coda": [], "melisma": True}]
        else:
            raise ValueError("first sung note needs a syllable")
        duration = r["off"] - r["time"]
        for syl in parts:
            phrase["notes"].append({"midi": r["pitch"], "beats": duration / len(parts), "syl": syl})
        previous = r
    for phrase in phrases:
        notes = phrase["notes"]
        for current, following in zip(notes, notes[1:]):
            if following["syl"].get("melisma"):
                # Punctuation belongs after the whole held word, not between
                # its notes: plan_song splits utterances at phraseEnd.
                current["syl"]["phraseEnd"] = False
    native = {"seconds": seconds, "body": body, "phrases": phrases,
              "controls": sorted(source, key=lambda c: c["time"])}
    if not baseline:
        native.update(baseline=False, singer=singer, seed=seed,
                      consonant_scale=part.get("consonant_scale", 1.0),
                      level_db=part.get("level_db", 0.0), tension=part.get("tension", 0.0),
                      breathiness=part.get("breathiness", 1.0), vibrato_scale=part.get("vibrato", 1.0),
                      midi_controls=sorted(controls, key=lambda c: c["time"]),
                      expression=[{"time": r["time"], "off": r["off"], "velocity": r["velocity"],
                                   "intent": {k: v for k, v in r.get("intent", voice3.character_settings(part.get("character", "plain"))).items() if k != "base"},
                                   "emerging_progress": r.get("emerging_progress", 0.0),
                                   "marked_onset": r.get("marked_onset", False),
                                   "marked_release": r.get("marked_release", False)} for r in rows])
    return native


def check_supported(options, part, rows, singer):
    """Reject voice3-only effects rather than silently swallowing their intent."""
    unsupported = set(part) & {"machine", "timing_sd_s", "scatter", "spread"}
    if "spread" in options:
        unsupported.add("voice3.spread")
    if unsupported:
        raise ValueError(f"CHIMERA one-singer native mode does not support {sorted(unsupported)}")
    if singer == "machine":
        raise ValueError("CHIMERA needs a physical singer, not singer='machine'")
    for r in rows:
        if r["intent"]["vowel_opening_hz"] != 0:
            raise ValueError("CHIMERA does not support nonzero vowel_opening_hz (no calibrated Hz-to-jaw mapping)")
    lo, hi = part.get("range", voice3._table()["singers"][singer]["range"])
    if any(not lo <= r["pitch"] <= hi for r in rows):
        raise ValueError(f"CHIMERA {singer} note outside range {lo}..{hi}; set voice3.parts range explicitly")


def prepare(data, *, baseline=False):
    settings = data.get("settings", {})
    seed, lexicon, parts, _ = settings_for(settings)
    performed, score = mido.MidiFile(data["performed_midi"]), mido.MidiFile(data["score_midi"])
    tempo = TempoMap(performed)
    tracks = [i for i, t in enumerate(performed.tracks) if any(m.type == "note_on" and m.velocity for m in t)]
    names = {i: performed.tracks[i].name for i in tracks}
    by_index = {}
    for selector, part in parts.items():
        matches = matching_tracks(selector, tracks, names)
        if len(matches) != 1:
            raise ValueError(f"part {selector!r} must select one track")
        by_index[matches[0]] = part
    seconds = tempo.seconds_at(max(sum(m.time for m in t) for t in performed.tracks)) + 1.0
    marked = {} if baseline else marks_for(settings.get("voice3", {}).get("marks", []), score,
                                           {job["index"]: names[job["index"]] for job in data["jobs"]})
    result = []
    for job in data["jobs"]:
        index = job["index"]
        part = by_index.get(index, {})
        if part.get("machine"):
            raise ValueError("CHIMERA comparison requires a physical sung part, not a machine effect")
        rows = score_rows(performed, score, index)
        source = index
        if "lyrics_from" in part:
            matches = matching_tracks(part["lyrics_from"], tracks, names)
            if len(matches) != 1:
                raise ValueError("lyrics_from must select one track")
            source = matches[0]
        lyric = aligned_lyrics(score, source)
        if lyric is None:
            raise ValueError(f"no lyrics for {names[index]}")
        syllables = track_syllables(score, lyric, lexicon)
        if source != index:
            rows = borrow(rows, syllables, tempo)
        else:
            rows = [{**r, "syllable": syllables.get(r["tick"])} for r in rows]
        if not baseline:
            rows = apply_marks(rows, part, marked.get(index, []))
        lag = part.get("lag_ms", 0) / 1000
        rows = [{**r, "time": max(0, r["time"] + lag), "off": max(0, r["off"] + lag)} for r in rows]
        singers = part.get("singers", [part["singer"]] if "singer" in part else default_singers(names[index], rows))
        if len(singers) != 1:
            raise ValueError("CHIMERA renders exactly one singer per part; sections are unsupported")
        if not baseline:
            check_supported(settings.get("voice3", {}), part, rows, singers[0])
        body = "female" if singers[0] in ("alto", "soprano") else "male"
        part_seed = int(np.random.SeedSequence([seed, index]).generate_state(1)[0])
        native = timeline(rows, body, seconds, baseline=baseline, part=part, singer=singers[0], lexicon=lexicon,
                          controls=controls_for(performed, index), seed=part_seed)
        result.append((job, native, {"name": names[index], "singer": singers[0], "notes": len(rows),
                                     "words": " ".join(r["syllable"].text for r in rows if r.get("syllable"))}))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--baseline", action="store_true", help="frozen steady-source diagnostic; ignores expression")
    args = parser.parse_args()
    data = json.loads(args.manifest.read_text())
    stems = {}
    started = time.monotonic()
    for job, native, details in prepare(data, baseline=args.baseline):
        output = Path(job["output"])
        output.parent.mkdir(parents=True, exist_ok=True)
        plan = output.with_suffix(".chimera.json")
        plan.write_text(json.dumps(native, indent=2) + "\n")
        if not args.prepare_only:
            subprocess.run([str(binary), str(plan), str(output)], check=True)
            sr, audio = wavfile.read(output)
            if sr != 48000 or not np.all(np.isfinite(audio)):
                raise ValueError("invalid CHIMERA output")
        stems[str(job["index"])] = {"engine": "model", "model": "chimera", "upstream": UPSTREAM,
                                    "performance": "steady-source diagnostic baseline" if args.baseline else "native expression v1",
                                    "native_input_sha256": hashlib.sha256(plan.read_bytes()).hexdigest(), **details}
        if not args.prepare_only:
            stems[str(job["index"])].update(
                binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                audio_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    args.manifest.with_suffix(".metrics.json").write_text(json.dumps({"stems": stems,
        "render_seconds": time.monotonic() - started, "prepared_only": args.prepare_only}, indent=2) + "\n")


if __name__ == "__main__":
    main()
