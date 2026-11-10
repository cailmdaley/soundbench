"""Score-to-native timeline invariants for the experimental CHIMERA baseline."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from soundbench.scripts.models.phonetics import Syllable
from soundbench.scripts.render_chimera import segment, syllable_parts, timeline


def row(t, off, pitch=60, syllable=None):
    return {"time": t, "off": off, "pitch": pitch, "syllable": syllable, "velocity": 64}


class ChimeraTimeline(unittest.TestCase):
    def test_strong_form_counts_the_melisma_and_carries_through_it(self):
        def nuclei(first_off):
            rows = [row(0, first_off, syllable=Syllable("and", [], ["AH0"], ["N", "D"])), row(first_off, 2.0)]
            native = timeline(rows, "male", 2.5, singer="tenor")
            return [n["syl"]["nucleus"]["ipa"] for n in native["phrases"][0]["notes"]]
        self.assertEqual(nuclei(0.3), ["æ", "æ"])
        self.assertEqual(nuclei(1.0), ["æ", "æ"])

    def test_held_reduced_vowel_takes_strong_form(self):
        def nucleus(word, phones, off):
            s = Syllable(word, *phones)
            native = timeline([row(0, off, syllable=s)], "male", 2.0, singer="tenor")
            return native["phrases"][0]["notes"][0]["syl"]["nucleus"]["ipa"]
        self.assertEqual(nucleus("and", ([], ["AH0"], ["N", "D"]), 1.0), "æ")
        self.assertEqual(nucleus("and", ([], ["AH0"], ["N", "D"]), 0.3), "ə")
        self.assertEqual(nucleus("the", (["DH"], ["AH0"], []), 1.0), "ʌ")
        baseline = timeline([row(0, 1.0, syllable=Syllable("and", [], ["AH0"], ["N", "D"]))], "male", 2.0, baseline=True)
        self.assertEqual(baseline["phrases"][0]["notes"][0]["syl"]["nucleus"]["ipa"], "ə")

    def test_phone_mapping_and_reduction(self):
        self.assertEqual(segment("AH0"), {"ipa": "ə"})
        self.assertEqual(segment("AH1"), {"ipa": "ʌ"})
        self.assertEqual(segment("ER0"), {"ipa": "ɚ"})
        self.assertEqual(segment("CH"), {"ipa": "tʃ"})

    def test_compound_syllable_keeps_every_phone(self):
        s = Syllable("to the", ["T"], ["UW1", "DH", "AH0"], [])
        parts = syllable_parts(s)
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0]["onset"], [{"ipa": "t"}])
        self.assertEqual(parts[0]["coda"], [{"ipa": "ð"}])
        self.assertEqual(parts[1]["nucleus"], {"ipa": "ə"})

    def test_performed_times_gaps_and_total_duration(self):
        s = Syllable("song", ["S"], ["AO1"], ["NG"])
        rows = [row(1.0, 1.9, syllable=s), row(1.95, 2.5, pitch=62),
                row(3, 4, syllable=s)]
        t = timeline(rows, "male", 5)
        self.assertEqual(t["seconds"], 5)
        self.assertEqual([p["t0"] for p in t["phrases"]], [1, 3])
        a = t["phrases"][0]["notes"]
        self.assertAlmostEqual(sum(n["beats"] for n in a), 1.5)
        self.assertEqual(a[1]["midi"], 62)
        self.assertTrue(a[1]["syl"]["melisma"])
        self.assertEqual(a[1]["syl"]["onset"], [])

    def test_punctuation_waits_for_last_melisma_note(self):
        s = Syllable("here.", ["HH"], ["IH1"], ["R"])
        rows = [row(1, 2, syllable=s), row(2, 3), row(3, 4)]
        notes = timeline(rows, "male", 5)["phrases"][0]["notes"]
        self.assertEqual([n["syl"]["phraseEnd"] for n in notes], [False, False, True])
        self.assertEqual(notes[0]["syl"]["coda"], [{"ipa": "ɹ"}])
        self.assertTrue(notes[2]["syl"]["melisma"])

    def test_nasal_hum_and_punctuation(self):
        self.assertEqual(syllable_parts(Syllable("hm", [], ["M"], []))[0]["nucleus"], {"ipa": "m"})
        self.assertTrue(syllable_parts(Syllable("here?", [], ["IH1"], []))[0]["phraseEnd"])

    def test_missing_first_syllable_fails(self):
        with self.assertRaisesRegex(ValueError, "first sung note"):
            timeline([row(1, 2)], "male", 3)



class ChimeraExpression(unittest.TestCase):
    def test_shared_mark_resolution_passage_progress_and_defaults(self):
        from soundbench.scripts.render_voice3 import apply_marks
        s = Syllable("a", [], ["AA1"], [])
        rows = [{**row(.4 + i, 1.4 + i, syllable=s), "tick": i * 480} for i in range(4)]
        marks = [{"start_tick": 480, "end_tick": 1920, "character": None,
                  "patch": {"vibrato": "emerging"}},
                 {"start_tick": 960, "end_tick": 1440, "character": None,
                  "patch": {"attack_ms": 100, "release": "breath"}}]
        rows = apply_marks(rows, {"character": "young"}, marks)
        t = timeline(rows, "male", 5, singer="tenor", part={"level_db": -3, "consonant_scale": 1.5},
                     controls=[{"time": 2, "control": 11, "value": 55}])
        e = t["expression"]
        self.assertEqual([n["intent"]["vibrato"] for n in e], ["natural", "emerging", "emerging", "emerging"])
        self.assertEqual([n["emerging_progress"] for n in e], [0, 0, .5, 1])
        self.assertEqual([n["marked_onset"] for n in e], [False, False, True, False])
        self.assertEqual([n["marked_release"] for n in e], [False, False, True, False])
        self.assertEqual(e[-1]["intent"]["attack_ms"], 40)
        self.assertEqual(t["level_db"], -3)
        self.assertEqual(t["consonant_scale"], 1.5)
        self.assertEqual(t["midi_controls"][0]["value"], 55)
        self.assertFalse(t["baseline"])
        self.assertEqual(t["phrases"][0]["vibrato"], 0)

    def test_baseline_protocol_is_frozen_and_expression_free(self):
        s = Syllable("a", [], ["AA1"], [])
        rows = [row(.4, 1.4, syllable=s)]
        t = timeline(rows, "male", 2, baseline=True, part={"level_db": 12, "character": "worn"},
                     controls=[{"time": 0, "control": 11, "value": 0}])
        self.assertEqual(set(t), {"seconds", "body", "phrases", "controls"})
        self.assertEqual(t["controls"], [{"time": 0., "pressure": 9., "tension": .55, "breathiness": .15}])
        self.assertEqual(t["phrases"][0]["vibrato"], 35)

    def test_unsupported_options_fail_instead_of_disappearing(self):
        from soundbench.scripts.render_chimera import check_supported
        from soundbench.scripts.models import voice3
        rows = [{**row(.4, 1.4), "intent": voice3.character_settings()}]
        for option in ["machine", "scatter", "spread", "timing_sd_s"]:
            with self.subTest(option=option), self.assertRaisesRegex(ValueError, "does not support"):
                check_supported({}, {option: 0}, rows, "tenor")
        with self.assertRaisesRegex(ValueError, "does not support"):
            check_supported({"spread": .5}, {}, rows, "tenor")
        with self.assertRaisesRegex(ValueError, "physical singer"):
            check_supported({}, {}, rows, "machine")
        rows[0]["intent"]["vowel_opening_hz"] = 20
        with self.assertRaisesRegex(ValueError, "vowel_opening_hz"):
            check_supported({}, {}, rows, "tenor")
        rows[0]["intent"]["vowel_opening_hz"] = 0
        rows[0]["pitch"] = 47
        with self.assertRaisesRegex(ValueError, "outside range"):
            check_supported({}, {}, rows, "tenor")
        check_supported({}, {"range": [47, 72]}, rows, "tenor")


class ChimeraManifest(unittest.TestCase):
    def setUp(self):
        import tempfile
        import mido
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        score = mido.MidiFile(ticks_per_beat=480)
        score.tracks.append(mido.MidiTrack([mido.MetaMessage("set_tempo", tempo=1000000)]))
        for name in ["tenor", "alto"]:
            track = mido.MidiTrack([mido.MetaMessage("track_name", name=name)])
            for i in range(4):
                track.extend([mido.Message("note_on", note=60, velocity=64),
                              mido.Message("note_off", note=60, time=480)])
            score.tracks.append(track)
        score.tracks.append(mido.MidiTrack([mido.MetaMessage("track_name", name="tenor"),
            *[mido.MetaMessage("lyrics", text="a" if i != 3 else "a.", time=480 if i else 0) for i in range(4)]]))
        score.save(self.root / "score.mid")
        # Rubato, velocity and continuous controls are from the performed MIDI.
        performed = mido.MidiFile(self.root / "score.mid")
        performed.tracks[0][0] = mido.MetaMessage("set_tempo", tempo=800000)
        performed.tracks[1].insert(1, mido.Message("control_change", control=11, value=85))
        performed.tracks[1].insert(2, mido.Message("control_change", control=2, value=24))
        performed.save(self.root / "performed.mid")
        self.data = {"performed_midi": str(self.root / "performed.mid"),
                     "score_midi": str(self.root / "score.mid"),
                     "settings": {"voice3": {"seed": 1979, "parts": {
                         "tenor": {"singer": "tenor", "character": "worn", "consonant_scale": 1.5},
                         "alto": {"singer": "alto", "lyrics_from": "tenor", "lag_ms": 15}},
                         "marks": [{"part": "tenor", "bar": 1, "beat": 2, "vibrato": "straight"}]}},
                     "jobs": [{"index": i, "output": str(self.root / f"{i}.wav")} for i in [1, 2]]}

    def test_prepare_consumes_same_voice3_manifest_without_score_edits(self):
        from soundbench.scripts.render_chimera import prepare
        jobs = prepare(self.data)
        self.assertEqual(len(jobs), 2)
        a, b = jobs[0][1], jobs[1][1]
        self.assertEqual(a["singer"], "tenor")
        self.assertEqual(a["expression"][0]["intent"]["vibrato"], "natural")
        self.assertEqual(a["expression"][1]["intent"]["vibrato"], "straight")
        self.assertEqual(a["expression"][0]["intent"]["roughness"], 1)
        self.assertEqual(a["expression"][1]["time"], .8)
        self.assertEqual(a["midi_controls"][0]["value"], 85)
        self.assertEqual([n["syl"]["nucleus"] for n in b["phrases"][0]["notes"]], [{"ipa": "ʌ"}] * 4)
        self.assertAlmostEqual(b["expression"][0]["time"], .015)
        self.assertNotEqual(a["seed"], b["seed"])

    def test_section_rejected_and_baseline_only_unsupported_settings_documented(self):
        from soundbench.scripts.render_chimera import prepare
        p = self.data["settings"]["voice3"]["parts"]["tenor"]
        p["singers"] = ["tenor", "tenor"]
        with self.assertRaisesRegex(ValueError, "sections are unsupported"):
            prepare(self.data)
        del p["singers"]
        p["timing_sd_s"] = .01
        with self.assertRaisesRegex(ValueError, "does not support"):
            prepare(self.data)
        self.assertNotIn("expression", prepare(self.data, baseline=True)[0][1])

    def test_selectable_binding_voice3_unmodified_and_shared_marks_are_routed(self):
        import copy
        import mido
        from soundbench.scripts.instrumentation import resolve_instruments, STEREO_INSTRUMENTS
        from soundbench.scripts.model_dispatch import RENDERERS, vocal_marks
        midi = mido.MidiFile(self.data["performed_midi"])
        settings = {"instruments": {"tenor": {"model": "chimera"}, "alto": {"model": "voice3"}}}
        bindings = resolve_instruments(midi, settings)
        self.assertEqual(bindings[1].model, "chimera")
        self.assertEqual(bindings[2].model, "voice3")
        self.assertEqual(bindings[2].instrument, "voice3")
        self.assertEqual(bindings[2].gain_db, 0)
        self.assertIn("voice3", STEREO_INSTRUMENTS)
        self.assertNotIn("chimera", STEREO_INSTRUMENTS)
        self.assertEqual(RENDERERS["voice3"][0].name, "render_voice3.py")
        self.assertEqual(RENDERERS["chimera"][0].name, "render_chimera.py")
        marks = [{"part": "tenor", "bar": 1, "vibrato": "straight"},
                 {"part": "2", "bar": 1, "vibrato": "emerging"}]
        original = copy.deepcopy(marks)
        names = {1: "tenor", 2: "alto"}
        self.assertEqual(vocal_marks(marks, bindings, names, "chimera"), [{**marks[0], "part": "1"}])
        self.assertEqual(vocal_marks(marks, bindings, names, "voice3"), [{**marks[1], "part": "1"}])
        self.assertEqual(marks, original)
        with self.assertRaisesRegex(ValueError, "voice3 or chimera"):
            vocal_marks([{"part": "absent"}], bindings, names, "chimera")
        # A numeric part counts singers, not every track.
        from types import SimpleNamespace as B
        band = {0: B(model="guitar"), 1: B(model="voice3"), 2: B(model="voice3")}
        named = {0: "guitar", 1: "lead", 2: "harmony"}
        self.assertEqual(vocal_marks([{"part": "2"}], band, named, "voice3"), [{"part": "2"}])

    def test_dispatch_manifests_keep_lexicon_parts_and_filter_marks_without_mutation(self):
        import copy
        import json
        import mido
        from types import SimpleNamespace
        from unittest.mock import patch
        from soundbench.scripts.instrumentation import resolve_instruments
        from soundbench.scripts.model_dispatch import render_model_stems
        settings = copy.deepcopy(self.data["settings"])
        settings["instruments"] = {"tenor": {"model": "chimera"}, "alto": {"model": "voice3"}}
        settings["voice3"]["lexicon"] = {"a": "AA1"}
        settings["voice3"]["marks"].append({"part": "alto", "bar": 1, "vibrato": "emerging"})
        original = copy.deepcopy(settings)
        midi = mido.MidiFile(self.data["performed_midi"])
        bindings = resolve_instruments(midi, settings)
        seen = []
        def process(command, **kwargs):
            manifest = Path(command[-1])
            data = json.loads(manifest.read_text())
            seen.append(data)
            self.assertEqual(data["settings"]["voice3"]["parts"], settings["voice3"]["parts"])
            self.assertEqual(data["settings"]["voice3"]["lexicon"], {"a": "AA1"})
            self.assertEqual(len(data["settings"]["voice3"]["marks"]), 1)
            self.assertEqual(data["settings"]["voice3"]["marks"][0]["part"], "1")
            manifest.with_suffix(".metrics.json").write_text('{"stems": {}}')
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        with patch("soundbench.scripts.model_dispatch.subprocess.run", side_effect=process):
            render_model_stems(self.data["performed_midi"], self.data["score_midi"], settings,
                               bindings, self.root, "test")
        self.assertEqual(len(seen), 2)
        self.assertEqual(settings, original)


class ChimeraNativeAudio(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from soundbench.scripts.render_chimera import binary
        if not binary.exists():
            raise unittest.SkipTest("build pinned chimera-studio to run native audio integration")

    def test_velocity_changes_level_and_waveform_and_pitch_stays_written(self):
        import json
        import subprocess
        import tempfile
        import numpy as np
        from scipy.io import wavfile
        from scipy.signal import correlate
        from soundbench.scripts.render_chimera import binary
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = []
            for velocity in [40, 100]:
                rows = [{**row(.4, 1.8, syllable=Syllable("a", [], ["AA1"], [])), "velocity": velocity}]
                native = timeline(rows, "male", 2.2, singer="tenor",
                                  part={"character": {"base": "plain", "vibrato": "straight"}})
                plan, output = root / f"{velocity}.json", root / f"{velocity}.wav"
                plan.write_text(json.dumps(native))
                subprocess.run([str(binary), str(plan), str(output)], check=True, capture_output=True)
                sr, x = wavfile.read(output)
                self.assertEqual(sr, 48000)
                self.assertEqual(x.ndim, 1)
                self.assertTrue(np.all(np.isfinite(x)))
                self.assertGreater(np.max(np.abs(x)), 1e-4)
                audio.append(x)
                y = x[int(.8 * sr):int(1.5 * sr)].astype(float)
                y -= y.mean()
                ac = correlate(y, y, mode="full", method="fft")[len(y)-1:]
                expected = sr / (440 * 2**((60 - 69)/12))
                lag = max(range(int(expected*.96), int(expected*1.04)), key=lambda lag: ac[lag])
                offset = .5 * (ac[lag-1] - ac[lag+1]) / (ac[lag-1] - 2*ac[lag] + ac[lag+1])
                cents = 1200 * np.log2(expected / (lag + offset))
                self.assertLess(abs(cents), 5, f"audio intonation: {cents:.2f} cents")
            a, b = (x[int(.8*sr):int(1.5*sr)].astype(float) for x in audio)
            scalar = np.dot(a,b) / np.dot(a,a)
            self.assertGreater(np.linalg.norm(b - scalar*a)/np.linalg.norm(b), .01)
            # Louder singing is louder: the engine limiter must not level it.
            self.assertGreater(np.sqrt(np.mean(b*b)), 1.5 * np.sqrt(np.mean(a*a)))


if __name__ == "__main__":
    unittest.main()
