#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["mido>=1.3,<2", "numpy>=1.26", "numba>=0.61", "scipy>=1.11"]
# ///
"""Audio-domain tests of the Kelly-Lochbaum voice: pitch, formants, nasals, stability, lyrics."""
import copy
import sys
import unittest
from pathlib import Path

import mido
import numpy as np
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1] / "soundbench"
sys.path.insert(0, str(ROOT / "scripts"))
from soundbench.scripts.models.voice import STORY_FORMANTS_C, chain_formants, constrict, impulse_response, load_config, plan, render, section_length_cm, sections, story_vtl, tract_parameters, vowel_shapes
from soundbench.scripts.render_voice import lyrics_for, sung_notes, selected_config
from soundbench.scripts.voice_analysis import f0_track, lpc_formants, vibrato

SR = 48000


def note(syllable, pitch=59, start=0.0, end=3.0, velocity=80):
    return {"time": start, "off": end, "pitch": pitch, "velocity": velocity, "syllable": syllable}


def formant_targets(config):
    """Story (2008) computed formants scaled to the model's tract length;
    transmission-line formants for the constructed schwa and r-coloured schwa."""
    length = config["tract"]["length_cm"]
    shapes = vowel_shapes(config)
    targets = {name: [f * story_vtl(vowel) / length for f in STORY_FORMANTS_C[vowel]]
               for name, vowel in (("ee", "iy"), ("ih", "ih"), ("ei", "ey"), ("ae", "ae"), ("oo", "uw"))}
    targets["uh"] = chain_formants(shapes["ax"], length, config)
    targets["er"] = chain_formants(shapes["er"], length, config)
    return targets


class VoiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()

    def test_deterministic_and_seeded(self):
        notes = [note("dai", 71, 0, 1), note("sy", 68, 1, 2), note("give", 61, 2, 2.5), note("me", 63, 2.5, 3)]
        a, b = render(notes, 3.5, self.config), render(notes, 3.5, self.config)
        self.assertTrue(np.array_equal(a, b))
        c = copy.deepcopy(self.config)
        c["engine"]["seed"] += 1
        self.assertFalse(np.array_equal(a, render(notes, 3.5, c)))

    def test_closed_lip_hum_release_and_darkening(self):
        settings = {"voice_model": {"release_s": 2.0, "tail_seconds": 3.0,
                                     "output_lowpass_hz": 2000.0}}
        config, options, digest = selected_config(settings)
        _, _, default_digest = selected_config({})
        self.assertNotEqual(digest, default_digest)
        self.assertEqual(options, settings["voice_model"])
        notes = [note("hm", 59, 0, 1)]
        audio, controls = render(notes, 4.0, config, return_controls=True)
        closure = next(e for e in controls["plan"]["events"] if e["name"] == "m")
        self.assertGreaterEqual(closure["end"], 3.0 + config["engine"]["lead_seconds"])
        rms = lambda a: np.sqrt(np.mean(a*a))
        self.assertGreater(rms(audio[SR:2*SR]), rms(audio[2*SR:3*SR]))
        self.assertGreater(rms(audio[2*SR:3*SR]), 1e-7)
        self.assertLess(rms(audio[int(3.2*SR):]), 1e-7)
        bright = copy.deepcopy(config)
        bright["engine"]["output_lowpass_hz"] = 10000.0
        original = render(notes, 4.0, bright)
        from scipy.signal import butter, sosfilt
        high = butter(4, 3000, "highpass", fs=SR, output="sos")
        self.assertLess(rms(sosfilt(high, audio)), .35*rms(sosfilt(high, original)))

    def test_piece_scoped_portamento(self):
        config, options, digest = selected_config({"voice_model": {"portamento_s": .12}})
        default, _, default_digest = selected_config({})
        self.assertEqual(config["expression"]["portamento_s"], .12)
        self.assertEqual(default["expression"]["portamento_s"], .09)
        self.assertNotEqual(digest, default_digest)
        self.assertEqual(options, {"portamento_s": .12})
        for value in (0, .6, float("nan"), "fast"):
            with self.assertRaises(ValueError):
                selected_config({"voice_model": {"portamento_s": value}})

    def test_voice_override_validation(self):
        for settings in ({"oops": 1}, {"release_s": float("nan")},
                         {"release_s": 2, "tail_seconds": 1}):
            with self.assertRaises(ValueError):
                selected_config({"voice_model": settings})

    def test_pitch_within_five_cents_and_vibrato(self):
        x = self.config["expression"]
        for pitch in (59, 64, 71, 76, 83):
            y = render([note("ah", pitch)], 3.5, self.config)
            times, cents = f0_track(y[round(1.0 * SR):round(2.9 * SR)], pitch)
            mean, rate, depth = vibrato(times, cents)
            with self.subTest(pitch=pitch, cents=mean, rate=rate, depth=depth):
                self.assertLess(abs(mean), 5)
                self.assertLess(abs(rate - x["vibrato_hz"]), 0.6)
                self.assertLess(abs(depth - x["vibrato_cents"]), 10)

    def test_vowel_formants_within_fifteen_percent(self):
        # A low fundamental (G2) resolves the tract's formants, not its harmonics.
        for vowel, target in formant_targets(self.config).items():
            y = render([note(vowel, 43, 0, 2.2)], 2.5, self.config)
            measured = lpc_formants(y[round(0.6 * SR):round(2.0 * SR)])
            for i, (m, t) in enumerate(zip(measured, target), 1):
                with self.subTest(vowel=vowel, formant=i, measured=round(m), target=round(t)):
                    self.assertLess(abs(m / t - 1), 0.15)

    def test_nasal_antiformants(self):
        """m and n close the mouth with the velum open: the oral cavity becomes a
        side branch whose quarter-wave resonance is a deep spectral zero."""
        c = self.config
        shapes, count, dx = vowel_shapes(c), sections(c), section_length_cm(c)
        velum_cm = tract_parameters(c)["velum_at"] * dx
        f = np.fft.rfftfreq(1 << 15, 1 / SR)
        band = (f > 150) & (f < 4000)

        def dips(h):
            spec = 20 * np.log10(np.abs(np.fft.rfft(h, 1 << 15)) + 1e-12)[band]
            i, props = find_peaks(-spec, prominence=6)
            return f[band][i], props["prominences"]

        vowel_f, vowel_depth = dips(impulse_response(shapes["ax"], 0.0, 0.5, c))
        for name in "mn":
            cons = c["consonants"][name]
            area = constrict(shapes["ax"], count, [[cons["place_cm"], cons["width_cm"], cons["area_cm2"]]], dx)
            predicted = c["tract"]["speed_of_sound_cm_s"] / (4 * (c["tract"]["length_cm"] - velum_cm - cons["place_cm"]))
            freq, depth = dips(impulse_response(area, c["nasal"]["coupling_cm2"], 0.5, c))
            near = np.abs(freq / predicted - 1) < 0.3
            with self.subTest(nasal=name, predicted=round(predicted), dips=list(zip(freq.round(), depth.round(1)))):
                self.assertTrue(np.any(near & (depth > 20)))
                self.assertFalse(np.any((np.abs(vowel_f / predicted - 1) < 0.3) & (vowel_depth > 20)))

    def test_finite_unclipped_across_range(self):
        for syllable in ("ah", "ee", "oo", "er"):
            for pitch in (59, 66, 71, 78, 83):
                for velocity in (30, 127):
                    y = render([note(syllable, pitch, 0, 1.5, velocity)], 2.0, self.config)
                    with self.subTest(syllable=syllable, pitch=pitch, velocity=velocity):
                        self.assertTrue(np.isfinite(y).all())
                        self.assertLess(np.max(np.abs(y)), 1.0)
                        self.assertGreater(np.max(np.abs(y)), 1e-3)

    def test_consonants_finite_and_timed_inside_their_notes(self):
        words = [("dai", 71), ("sy", 68), ("give", 61), ("me", 63), ("your", 64), ("an", 61), ("swer", 64), ("do", 59)]
        notes = [note(w, p, i * 0.8, (i + 1) * 0.8) for i, (w, p) in enumerate(words)]
        y = render(notes, 7.5, self.config)
        self.assertTrue(np.isfinite(y).all())
        self.assertLess(np.max(np.abs(y)), 1.0)
        p = plan(notes, self.config)
        units = p["units"]
        for e in p["events"]:
            u = e["unit"]
            i = units.index(u)
            with self.subTest(consonant=e["name"], syllable=u["text"]):
                if e in [ev for ev in p["events"] if ev["unit"] is u][:len(u["onset"])]:
                    # Onsets straddle the beat and end where the vowel begins.
                    self.assertLess(e["start"], u["start"])
                    self.assertLessEqual(e["end"], u["vowel_start"] + 1e-9)
                    self.assertGreaterEqual(e["end"], u["start"])
                else:
                    # Codas follow the vowel and finish by the next syllable's onset.
                    self.assertGreaterEqual(e["start"], u["vowel_end"] - 1e-9)
                    limit = units[i + 1]["start"] if i + 1 < len(units) else u["end"]
                    self.assertLessEqual(e["end"], limit + 1e-9)
        for u in units:
            self.assertGreater(u["vowel_end"], u["vowel_start"])

    def test_lyrics_from_midi(self):
        midi = mido.MidiFile(ticks_per_beat=480)
        midi.tracks.append(mido.MidiTrack([mido.MetaMessage("set_tempo", tempo=500000)]))
        notes = mido.MidiTrack([mido.MetaMessage("track_name", name="voice:voice")])
        for pitch in (71, 68, 64):
            notes.extend([mido.Message("note_on", note=pitch, velocity=80, time=0),
                          mido.Message("note_off", note=pitch, velocity=0, time=480)])
        lyrics = mido.MidiTrack([mido.MetaMessage("track_name", name="voice:"),
                                 mido.MetaMessage("lyrics", text="Dai", time=0),
                                 mido.MetaMessage("lyrics", text="sy,", time=480)])
        midi.tracks.extend([notes, lyrics])
        self.assertEqual(lyrics_for(midi, 1), {0: "Dai", 480: "sy,"})
        rows = sung_notes(midi, midi, 1)
        self.assertEqual([r["syllable"] for r in rows], ["Dai", "sy,", None])
        self.assertAlmostEqual(rows[1]["time"], 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
