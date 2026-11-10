#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["mido>=1.3,<2", "numpy>=1.26", "numba>=0.61", "scipy>=1.11"]
# ///
"""Kit audio, contact, pedal and model-dispatch invariants."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import mido
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "soundbench"
sys.path.insert(0, str(ROOT / "scripts"))
from soundbench.scripts.models.drums import KEYS, _cascade, _plate, drum_modes, hit, load_config, plate_modes, render
from soundbench.scripts.models.mallorca_percussion import modal_signal
from soundbench.scripts.instrumentation import resolve_instruments, STEREO_INSTRUMENTS
from soundbench.scripts.model_dispatch import render_model_stems
from soundbench.scripts.render_drums import gestures, render_channels, selected_config


def note(p=38, v=80, time=0., off=.3):
    return dict(pitch=p, velocity=v, time=time, off=off)


class KitTests(unittest.TestCase):
    def test_every_key_finite_audible_released(self):
        c = load_config()
        for p in KEYS:
            with self.subTest(p=p):
                y = render([note(p)], 5.4, c)
                self.assertTrue(np.isfinite(y).all())
                self.assertLess(np.max(abs(y)), 1)
                self.assertGreater(np.max(abs(y)), 1e-5)
                self.assertLess(np.max(abs(y[-2400:])), 1e-6)

    def test_determinism_and_seed(self):
        rows = [note(), note(46, time=.5, off=.7)]
        c = load_config()
        y = render(rows, 5.8, c)
        np.testing.assert_array_equal(y, render(rows, 5.8, c))
        c["engine"]["seed"] += 1
        self.assertGreater(np.max(abs(y-render(rows, 5.8, c))), 1e-6)

    def test_velocity_ramp_continuous_and_monotone(self):
        c = load_config()
        c["engine"]["tail_seconds"] = .5
        for p in (36, 38, 42):
            rms = np.array([np.sqrt(np.mean(hit(note(p, v), c)**2)) for v in range(1, 128)])
            self.assertTrue(np.all(np.diff(rms) > 0), (p, rms))
            self.assertLess(np.max(rms[31:]/rms[30:-1]), 1.15)

    def test_modal_poles_tuned_within_one_cent(self):
        c = load_config()
        f, t, w, _ = drum_modes(c["snare"])
        self.assertTrue(np.all(f > 0))
        self.assertTrue(np.all(t > 0))
        impulse = np.zeros(96000)
        impulse[0] = 1
        for freq in (f[np.argmax(abs(w))], plate_modes(c, "ride")[0][4]):
            y = modal_signal(impulse, [freq], [.6], [1])
            measured = np.argmax(abs(np.fft.rfft(y, n=2**20)))*48000/2**20
            self.assertLess(abs(1200*np.log2(measured/freq)), 1)

    def test_upward_cascade_conserves_power_and_phase(self):
        z = np.array([1+2j, .1+.2j, .01+.01j, 0j])
        original = z.copy()
        _cascade(z, .2, False)
        self.assertAlmostEqual(float(np.sum(abs(z)**2)), float(np.sum(abs(original)**2)), places=12)
        self.assertLess(abs(z[0]), abs(original[0]))
        self.assertGreater(abs(z[-1]), abs(original[-1]))
        np.testing.assert_allclose(np.angle(z[:3]), np.angle(original[:3]), atol=1e-12)

    def test_plate_contracts_without_excitation(self):
        force = np.zeros(24000)
        force[0] = 1
        poles = np.exp(-.001+2j*np.pi*np.array([300, 730, 1230, 2010])/48000)
        y = _plate(force, poles, np.ones(4)*.5, .2, np.zeros(len(force)), .001, False)
        self.assertLess(np.max(abs(y[-1000:])), 1e-8)
        self.assertLess(np.max(abs(y)), 1.01)

    def test_cc4_closes_ringing_hat_after_note_off(self):
        c = load_config()
        c["engine"]["tail_seconds"] = 1.5
        a = hit(note(46, off=.01), c)
        b = hit(note(46, off=.01), c, pedal=[(.2, 1.)])
        np.testing.assert_array_equal(a[:9600], b[:9600])
        self.assertLess(np.sum(b[24000:]**2), np.sum(a[24000:]**2)*.01)

    def test_gm_closed_and_foot_choke_open_hat(self):
        c = load_config()
        a = render([note(46, off=.1)], 5.8, c)
        for p in (42, 44):
            b = render([note(46, off=.1), note(p, time=.2, off=.3)], 5.8, c)
            self.assertLess(np.sum(b[24000:48000]**2), np.sum(a[24000:48000]**2)*.01)

    def test_brush_noteoff_stops_friction_not_head(self):
        c = load_config()
        a, b = hit(note(60, off=.2), c), hit(note(60, off=.8), c)
        self.assertGreater(np.sum(b[24000:36000]**2), np.sum(a[24000:36000]**2)*100)
        self.assertGreater(np.sum(a[10000:11000]**2), 0)

    def test_rim_bell_and_stereo_differ(self):
        c = load_config()
        for a, b in ((38, 40), (38, 37), (51, 53), (42, 46)):
            self.assertGreater(np.max(abs(hit(note(a), c)-hit(note(b), c))), 1e-5)
        y = render([note(42)], 5.4, c)
        self.assertGreater(np.max(abs(y[:, 0]-y[:, 1])), 1e-5)
        self.assertIn("drums", STEREO_INSTRUMENTS)

    def test_all_presets_forte_no_clipping(self):
        for name in ("drums", "drums-open", "drums-rock-fitted"):
            path = ROOT/f"instruments/models/{name}.toml"
            c = load_config(path)
            for p in (36, 38, 42, 46, 60):
                y = render([note(p, 127)], 5.4, c)
                self.assertTrue(np.isfinite(y).all())
                self.assertLess(np.max(abs(y)), 1, (name, p))

    def test_invalid_gestures_and_preset(self):
        for row in (note(70), note(v=0), note(time=-1), note(off=0), note(off=float("nan"))):
            with self.assertRaises(ValueError):
                hit(row)
        with self.assertRaises(ValueError):
            render([note()], 1)
        for options in ({"wat": 3}, {"seed": 1.2}):
            with self.assertRaises(ValueError):
                selected_config({"drums_model": options})

    def test_midi_fifo_channel_pedal_and_dispatch(self):
        midi = mido.MidiFile()
        midi.tracks.append(mido.MidiTrack([mido.MetaMessage("set_tempo", tempo=600000)]))
        track = mido.MidiTrack([
            mido.MetaMessage("track_name", name="kit"),
            mido.Message("control_change", control=4, value=30, channel=9),
            mido.Message("note_on", note=46, velocity=70, channel=9),
            mido.Message("note_on", note=46, velocity=35, time=24, channel=9),
            mido.Message("note_off", note=46, time=24, channel=9),
            mido.Message("note_off", note=46, time=24, channel=9),
            mido.Message("control_change", control=4, value=127, time=48, channel=9),
            mido.MetaMessage("end_of_track", time=480)])
        midi.tracks.append(track)
        rows, pedals = gestures(midi, 1)
        self.assertLess(rows[0]["off"], rows[1]["off"])
        self.assertEqual(pedals[9][-1][1], 1)
        settings = {"instruments": {"kit": {"model": "drums"}}}
        bindings = resolve_instruments(midi, settings)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            path = out/"kit.mid"
            midi.save(path)
            stems = render_model_stems(path, path, settings, bindings, out, "kit")
            self.assertEqual(stems["1"]["model"], "drums")
            self.assertEqual(len(list(out.glob("*.raw.wav"))), 1)
            self.assertLess(json.loads((out/"kit.drums.metrics.json").read_text())["peaks"]["1"], 1)


if __name__ == "__main__":
    unittest.main()
