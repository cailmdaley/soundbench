#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["mido>=1.3,<2", "numpy>=1.26", "numba>=0.61", "scipy>=1.11", "cmudict>=1.0"]
# ///
"""Phonetics, sung timing, determinism, ensembles, machine colour and sidecar contracts of voice3."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from soundbench.scripts.models import machine_voice, phonetics, voice3
from soundbench.scripts.render_voice3 import borrow, default_singers, machine_settings, score_rows, settings_for

SR = voice3.SR


def syl(text, onset, nucleus, coda=()):
    return phonetics.Syllable(text, list(onset), list(nucleus), list(coda))


class PhoneticsTests(unittest.TestCase):
    def test_hyphenated_word_split_by_maximal_onset(self):
        out = phonetics.syllabify([("Dai", True), ("sy", False)])
        self.assertEqual([(s.onset, s.coda) for s in out], [(["D"], []), (["Z"], [])])
        self.assertTrue(out[0].word_start and not out[0].word_end and out[1].word_end)
        out = phonetics.syllabify([("an", True), ("swer", False)])
        self.assertEqual(out[0].coda, ["N"])
        self.assertEqual(out[1].onset, ["S"])

    def test_elisions_and_hums(self):
        trav = phonetics.syllabify([("trav'", True), ("ling", False)])
        self.assertEqual(len(trav), 2)
        self.assertEqual(trav[0].onset, ["T", "R"])
        hm = phonetics.syllabify([("hm", False)])[0]
        self.assertEqual(hm.nucleus, ["M"])
        # Two words on one note sing as one syllable through both vowels.
        tied = phonetics.syllabify([("to‿the", False)])[0]
        self.assertEqual(sum(map(phonetics.is_vowel, tied.nucleus)), 2)

    def test_more_vowels_than_notes_merge(self):
        flower = phonetics.syllabify([("flow'r", False)])
        self.assertEqual(len(flower), 1)
        self.assertGreaterEqual(sum(map(phonetics.is_vowel, flower[0].nucleus)), 1)

    def test_lexicon_and_unknown_words(self):
        out = phonetics.syllabify([("zorp", False)], {"zorp": "Z AO1 R P"})
        self.assertEqual(out[0].coda, ["R", "P"])
        with self.assertRaises(ValueError):
            phonetics.syllabify([("zxqvb", False)])

    def test_regular_endings_take_the_stem_voicing(self):
        self.assertEqual(phonetics._ending("ed", "K"), ["T"])
        self.assertEqual(phonetics._ending("ed", "T"), ["IH0", "D"])
        self.assertEqual(phonetics._ending("ed", "N"), ["D"])
        self.assertEqual(phonetics._ending("s", "T"), ["S"])
        self.assertEqual(phonetics._ending("s", "SH"), ["IH0", "Z"])
        self.assertEqual(phonetics._ending("s", "M"), ["Z"])

    def test_no_guessed_compounds(self):
        # A name or misspelling is not split into two dictionary words.
        with self.assertRaises(ValueError):
            phonetics.pronunciations("catdogfish")

    def test_lyric_event_marker(self):
        self.assertEqual(phonetics.lyric_event("Dai --"), ("Dai", True))
        self.assertEqual(phonetics.lyric_event("sy,"), ("sy,", False))


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.c = voice3.load_config("tenor")

    def test_vowel_on_the_beat_consonants_before(self):
        rows = voice3.lyric_rows("Dai-sy give me", [(60, 0.5), (62, 0.5), (64, 0.5), (65, 0.5)])
        p = voice3.plan(rows, self.c)
        lead = self.c["v3"]["timing"]["vowel_lead_s"]
        for u in p["units"]:
            self.assertAlmostEqual(u["vowel_on"], u["start"] - lead)
            onset = [s for s in p["segments"] if s["unit"] is u and s.get("role") == "onset"]
            for s in onset:
                self.assertLessEqual(s["after_end"], u["vowel_on"] + 1e-9)
        # "give": G before the vowel, V after it and before "me"'s M.
        give = p["units"][2]
        coda = [s for s in p["segments"] if s["unit"] is give and s.get("role") == "coda"]
        self.assertEqual([s["phone"] for s in coda], ["V"])
        self.assertLess(coda[0]["start"], p["units"][3]["vowel_on"])

    def test_short_notes_compress_consonants_to_the_floor(self):
        rows = voice3.lyric_rows("strengths strengths", [(60, 0.12), (62, 0.12)])
        p = voice3.plan(rows, self.c)
        a, b = p["units"]
        self.assertGreater(b["vowel_on"] - a["vowel_on"], 0)
        self.assertGreater(a["vowel_off"], a["vowel_on"])

    def test_melisma_holds_the_vowel(self):
        rows = voice3.lyric_rows("do _ _", [(60, 0.3), (62, 0.3), (64, 0.3)])
        p = voice3.plan(rows, self.c)
        self.assertEqual(len(p["units"]), 1)
        self.assertEqual(len(p["units"][0]["notes"]), 3)

    def test_an_untexted_note_after_a_rest_is_an_error(self):
        rows = voice3.lyric_rows("hel-lo", [(60, 0.5), (62, 0.5)])
        rows.append({"time": 4.0, "off": 4.5, "pitch": 64, "velocity": 70, "syllable": None})
        with self.assertRaises(ValueError):
            voice3.plan(rows, self.c)

    def test_a_detached_melisma_is_not_a_rest(self):
        rows = voice3.lyric_rows("ah", [(60, 0.4)])
        rows.append({"time": rows[-1]["off"] + 0.2, "off": rows[-1]["off"] + 0.6, "pitch": 62, "velocity": 70,
                     "syllable": None})
        self.assertEqual(len(voice3.plan(rows, self.c)["units"]), 1)

    def test_first_note_needs_a_syllable(self):
        with self.assertRaises(ValueError):
            voice3.plan([{"time": 0.4, "off": 0.8, "pitch": 60, "velocity": 64, "syllable": None}], self.c)


class RenderTests(unittest.TestCase):
    rows = voice3.lyric_rows("Dai-sy give me your an-swer", [(57, .35), (55, .35), (52, .35), (54, .35), (55, .35),
                                                            (57, .35), (55, .7)])

    def test_deterministic_and_finite(self):
        a = voice3.render(self.rows)
        b = voice3.render(self.rows)
        self.assertTrue(np.array_equal(a, b))
        self.assertTrue(np.isfinite(a).all())
        self.assertGreater(np.sqrt(np.mean(a ** 2)), 1e-3)
        self.assertLess(np.abs(a).max(), 1.0)

    def test_every_singer_and_vowel_is_finite(self):
        vowels = ["AA1", "AE1", "AH1", "AO1", "EH1", "ER1", "IH1", "IY1", "UH1", "UW1", "EY1", "OW1", "AY1"]
        for name in voice3.singers():
            lo, hi = voice3._table()["singers"][name]["range"]
            rows = [{"time": 0.3 + 0.25 * i, "off": 0.5 + 0.25 * i, "pitch": lo + (hi - lo) * i // len(vowels),
                     "velocity": 90, "syllable": syl(v, ["M"], [v])} for i, v in enumerate(vowels)]
            out = voice3.render(rows, config=voice3.load_config(name))
            self.assertTrue(np.isfinite(out).all(), name)
            self.assertLess(np.abs(out).max(), 1.0, name)

    def test_pitch_lands_on_the_note(self):
        rows = [{"time": 0.4, "off": 1.6, "pitch": 57, "velocity": 70, "syllable": syl("ah", [], ["AA1"])}]
        c = voice3.load_config("machine")
        _, k = voice3.render(rows, config=c, return_controls=True)
        lead = c["v3"]["engine"]["lead_seconds"]
        held = (k["t"] > lead + 0.6) & (k["t"] < lead + 1.4)
        cents = 1200 * np.log2(k["f0"][held] / 220.0)
        self.assertLess(np.abs(cents).max(), 3.0)

    def test_notes_outside_the_range_raise_unless_widened(self):
        rows = [{"time": 0.4, "off": 1.0, "pitch": 76, "velocity": 70, "syllable": syl("ah", [], ["AA1"])}]
        with self.assertRaises(ValueError):
            voice3.render(rows, config=voice3.load_config("alto"))
        out = voice3.render(rows, config=voice3.load_config("alto", range=[53, 76]))
        self.assertTrue(np.isfinite(out).all())

    def test_singer_overrides(self):
        c = voice3.load_config("tenor", vibrato_scale=0.5, tension=0.2, vowel_evenness=0.0)
        self.assertEqual(c["singer"]["vibrato_scale"], 0.5)
        with self.assertRaises(ValueError):
            voice3.load_config("tenor", nonsense=1)
        with self.assertRaises(ValueError):
            voice3.load_config("countertenor")


class EnsembleTests(unittest.TestCase):
    rows = voice3.lyric_rows("love of you", [(60, .4), (62, .4), (64, .8)])

    def test_sing_is_stereo_and_a_section_is_as_loud_as_a_soloist(self):
        solo = voice3.sing(self.rows, ["tenor"], seed=3)
        section = voice3.sing(self.rows, ["tenor", "tenor", "baritone"], seed=3)
        self.assertEqual(solo.shape[1], 2)
        self.assertEqual(solo.shape, section.shape)
        level = lambda x: 10 * np.log10(np.mean(x ** 2))
        self.assertLess(abs(level(solo) - level(section)), 3.0)
        # A section is spread: its channels differ.
        self.assertGreater(np.mean((section[:, 0] - section[:, 1]) ** 2), 0)

    def test_scatter_keeps_legato_and_order(self):
        rng = np.random.default_rng(1)
        out = voice3.scatter(self.rows, rng, 0.02)
        for a, b in zip(out, out[1:]):
            self.assertLessEqual(a["off"], b["time"] + 1e-9)
            self.assertGreater(a["off"], a["time"])

    def test_a_controller_set_once_reaches_every_block(self):
        # CC11 set at the start must hold in a later block (blocks split at rests).
        rows = voice3.lyric_rows("love of", [(60, .4), (62, .4)])
        late = voice3.lyric_rows("love of", [(60, .4), (62, .4)], start=30.0)
        quiet = voice3.sing(rows + late, ["tenor"], 32.0, seed=3, midi_controls=[{"time": 0.0, "control": 11, "value": 40}])
        loud = voice3.sing(rows + late, ["tenor"], 32.0, seed=3)
        tail = slice(int(29.5 * voice3.SR), None)
        level = lambda x: 10 * np.log10(np.mean(x[tail] ** 2))
        self.assertLess(level(quiet) - level(loud), -6.0)

    def test_machine_colour(self):
        settings = machine_voice.load_config(preset="vocoder", overrides={"scale": [0, 2, 4, 5, 7, 9, 11]})
        out = voice3.sing(self.rows, ["machine"], seed=3, machine_settings=settings)
        self.assertTrue(np.isfinite(out).all())
        self.assertGreater(np.sqrt(np.mean(out ** 2)), 1e-3)
        plain = voice3.sing(self.rows, ["machine"], seed=3)
        self.assertFalse(np.allclose(out, plain))


class SidecarTests(unittest.TestCase):
    def test_settings_validation(self):
        seed, lexicon, parts, spread = settings_for({"voice3": {"seed": 4, "parts": {
            "lead": {"singer": "tenor", "vibrato": 0.5, "lag_ms": -12}, "robot": {"singer": "machine", "machine": "talkbox"}}}})
        self.assertEqual(seed, 4)
        for bad in ({"voice3": {"tempo": 1}}, {"voice3": {"parts": {"a": {"singer": "x"}}}},
                    {"voice3": {"parts": {"a": {"vibrato": 9}}}}, {"voice3": {"parts": {"a": {"machine": "kazoo"}}}},
                    {"voice3": {"parts": {"a": {"color": 1}}}}, {"voice3": {"lexicon": {"w": "QQ1"}}},
                    {"voice3": {"parts": {"a": {"lag_ms": 200}}}}):
            with self.assertRaises(ValueError, msg=str(bad)):
                settings_for(bad)

    def test_machine_settings_take_overrides(self):
        c = machine_settings({"preset": "vocoder", "lowpass_hz": 3000.0})
        self.assertEqual((c["mode"], c["lowpass_hz"]), ("channel", 3000.0))

    def test_borrow_splits_held_notes(self):
        class Tempo:
            def seconds_at(self, tick):
                return tick / 1000
        syllables = {0: syl("Dai", ["D"], ["AY1"]), 500: syl("sy", ["Z"], ["IY0"]), 1000: syl("give", ["G"], ["IH1"], ["V"])}
        rows = [{"time": 0.0, "off": 0.99, "pitch": 55, "velocity": 70, "tick": 0, "off_tick": 990},
                {"time": 1.0, "off": 1.5, "pitch": 57, "velocity": 70, "tick": 1000, "off_tick": 1500}]
        out = borrow(rows, syllables, Tempo())
        self.assertEqual([r["syllable"].text for r in out], ["Dai", "sy", "give"])
        self.assertAlmostEqual(out[1]["time"], 0.5)
        self.assertLess(out[0]["off"], out[1]["time"])

    def test_borrow_after_a_rest_takes_the_last_syllable(self):
        class Tempo:
            def seconds_at(self, tick):
                return tick / 1000
        syllables = {0: syl("hel", ["HH"], ["EH1"]), 500: syl("lo", ["L"], ["OW1"])}
        rows = [{"time": 0.0, "off": 0.9, "pitch": 55, "velocity": 70, "tick": 0, "off_tick": 900},
                {"time": 3.0, "off": 3.5, "pitch": 57, "velocity": 70, "tick": 3000, "off_tick": 3500}]
        out = borrow(rows, syllables, Tempo())
        self.assertEqual([r["syllable"].text for r in out], ["hel", "lo", "lo"])
        self.assertEqual(len(voice3.plan(out, voice3.load_config("tenor"))["units"]), 3)

    def test_borrow_refuses_a_split_after_its_note(self):
        class Skewed:
            def seconds_at(self, tick):
                return tick / 500
        rows = [{"time": 0.0, "off": 0.9, "pitch": 55, "velocity": 70, "tick": 0, "off_tick": 900}]
        with self.assertRaises(ValueError):
            borrow(rows, {0: syl("a", [], ["AA1"]), 600: syl("b", ["B"], ["IY1"])}, Skewed())

    def test_a_chord_in_a_sung_part_is_an_error(self):
        import mido
        midi = mido.MidiFile(ticks_per_beat=480)
        track = mido.MidiTrack([mido.MetaMessage("track_name", name="lead", time=0)])
        for pitch in (60, 64):
            track.append(mido.Message("note_on", note=pitch, velocity=70, time=0))
        track += [mido.Message("note_off", note=60, velocity=0, time=480), mido.Message("note_off", note=64, velocity=0, time=0)]
        midi.tracks.append(track)
        with self.assertRaises(ValueError):
            score_rows(midi, midi, 0)

    def test_two_sung_voices_over_a_piano_render(self):
        # The piano's onsets cover both Lyrics lines; only sung tracks look for lyrics.
        import json
        import subprocess
        import tempfile
        import mido
        def track(name, events):
            t = mido.MidiTrack([mido.MetaMessage("track_name", name=name, time=0)])
            last = 0
            for tick, kind, value in sorted(events):
                t.append(mido.MetaMessage("lyrics", text=value, time=tick - last) if kind == "lyric" else
                         mido.Message(kind, note=value, velocity=70 if kind == "note_on" else 0, time=tick - last))
                last = tick
            return t
        def notes(pitches, ticks, length):
            return [e for p, t in zip(pitches, ticks) for e in ((t, "note_on", p), (t + length, "note_off", p))]
        midi = mido.MidiFile(ticks_per_beat=480)
        midi.tracks += [track("lead", notes([67, 69], [0, 960], 900)), track("", [(0, "lyric", "Dai --"), (960, "lyric", "sy")]),
                        track("tenor", notes([60, 62], [480, 1440], 450)),
                        track("", [(480, "lyric", "love"), (1440, "lyric", "you")]),
                        track("piano", notes([48, 50, 52, 53], [0, 480, 960, 1440], 450))]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "score.mid"
            midi.save(path)
            jobs = [{"index": 0, "output": str(Path(d) / "lead.wav")}, {"index": 2, "output": str(Path(d) / "tenor.wav")}]
            manifest = Path(d) / "voice3.json"
            manifest.write_text(json.dumps({"performed_midi": str(path), "score_midi": str(path), "jobs": jobs,
                                            "settings": {"voice3": {"parts": {"lead": {"singer": "alto"}}}}}))
            done = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "soundbench/scripts/render_voice3.py"),
                                   str(manifest)], capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stderr[-800:])
            self.assertTrue(all(Path(j["output"]).exists() for j in jobs))

    def test_overlapping_notes_in_a_sung_part_are_an_error(self):
        import mido
        midi = mido.MidiFile(ticks_per_beat=480)
        track = mido.MidiTrack([mido.MetaMessage("track_name", name="lead", time=0),
                                mido.Message("note_on", note=60, velocity=70, time=0),
                                mido.Message("note_on", note=64, velocity=70, time=240),
                                mido.Message("note_off", note=60, velocity=0, time=240),
                                mido.Message("note_off", note=64, velocity=0, time=240)])
        midi.tracks.append(track)
        with self.assertRaises(ValueError):
            score_rows(midi, midi, 0)

    def test_default_singers(self):
        self.assertEqual(default_singers("alto", [{"pitch": 40}]), ["alto"])
        self.assertEqual(default_singers("lead", [{"pitch": 43}, {"pitch": 50}]), ["bass"])


if __name__ == '__main__':
    unittest.main()
