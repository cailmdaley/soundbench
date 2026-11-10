#!/usr/bin/env python3
# /// script
# dependencies = ["numpy>=1.26", "scipy>=1.11"]
# ///
"""Tape, transport, room-tail and signal-domain regression tests."""
import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.signal import hilbert

ROOT = Path(__file__).resolve().parents[1] / "soundbench"
sys.path.insert(0, str(ROOT / "scripts"))
from soundbench.scripts.models import nagra

SR = nagra.SAMPLE_RATE


def steady():
    cfg = nagra.load_config()
    cfg["wow"]["cents"] = cfg["flutter"]["cents"] = 0
    cfg["transport"].update(start_s=0, stop_s=0, stop_fade_s=0)
    cfg["room"]["tone_dbfs"] = None
    return cfg


class NagraTests(unittest.TestCase):
    def test_mic_movement_endpoints_preserve_distance_and_room_paths(self):
        cfg=steady();sr=8000
        x=.01*np.sin(2*np.pi*220*np.arange(sr)/sr)
        ir=np.array([.2, .1, .05])
        near=2/10**(5/20)
        for gain,distance in [(-18,2*10**(18/20)),(0,2),(5,near)]:
            moving={'far_m':2,'near_m':near,'gain_db':[[0,gain],[1,gain]]}
            a=nagra.mic([x],[moving],cfg,ir,np.random.default_rng(0),sr)
            b=nagra.mic([x],[distance],cfg,ir,np.random.default_rng(0),sr)
            # The moving path reserves the far-position delay and its filter tail.
            np.testing.assert_allclose(a[:len(b)],b,atol=1e-12)

    def test_player_projection_precedes_direct_and_room_fields(self):
        from soundbench.scripts.render_tape import source_gain
        sr=8000;cfg=steady()
        x=np.sin(2*np.pi*220*np.arange(sr)/sr)*.01
        quiet=source_gain(x,[[0,-12],[1,-12]])
        ratio=10**(-12/20)
        np.testing.assert_allclose(quiet,x*ratio,atol=1e-9)
        a=nagra.mic([quiet],[2],cfg,np.array([.2,.1]),np.random.default_rng(0),sr)
        b=nagra.mic([x],[2],cfg,np.array([.2,.1]),np.random.default_rng(0),sr)
        np.testing.assert_allclose(a,b*ratio,atol=1e-9)

    def test_moving_mic_has_one_delay_without_two_path_cancellation(self):
        cfg=steady();cfg['room']['wet_db']=-300;sr=8000
        near=2/10**(5/20)
        delta=round(2/343*sr)-round(near/343*sr)
        x=.01*np.sin(2*np.pi*(sr/(2*delta))*np.arange(sr)/sr)
        gain=20*np.log10((1+10**(5/20))/2)
        moving={'far_m':2,'near_m':near,'gain_db':[[0,gain],[1,gain]]}
        a=nagra.mic([x],[moving],cfg,np.array([1.]),np.random.default_rng(0),sr)
        b=nagra.mic([x],[2],cfg,np.array([1.]),np.random.default_rng(0),sr)
        measured=20*np.log10(np.linalg.norm(a[2000:7000])/np.linalg.norm(b[2000:7000]))
        self.assertAlmostEqual(measured,gain,delta=.15)

    def test_transfer_ratio_mid_delay_and_zero_gaps(self):
        from scipy.signal import correlate
        cfg = nagra.load_config()["transfer"]
        x = np.random.default_rng(7).normal(0, 0.1, 3 * SR)
        x[-SR:] = 0
        y = nagra.transfer_stereo(x, cfg)
        mid, side = y.mean(axis=1), (y[:, 0] - y[:, 1]) / 2
        np.testing.assert_allclose(mid, x, atol=1e-16)
        db = 10 * np.log10(np.dot(side, side) / np.dot(mid, mid))
        self.assertAlmostEqual(db, -4.7, places=10)
        lag = np.argmax(correlate(side, x, mode="full")) - len(x) + 1
        self.assertAlmostEqual(lag / SR * 1000, cfg["delay_ms"], delta=0.2)
        self.assertEqual(np.count_nonzero(y[-SR:]), 0)
        self.assertTrue(np.isfinite(nagra.transfer_stereo(np.zeros(SR), cfg)).all())
        np.testing.assert_array_equal(y, nagra.transfer_stereo(x, cfg))

    def test_room_presets_shape_effective_decay(self):
        ir = np.random.default_rng(5).normal(size=3 * SR) * np.exp(-np.arange(3 * SR) / SR * np.log(1000))
        cfg = nagra.load_config()
        for name, target in (("stage", 1.1), ("basement", 0.5)):
            selected = nagra.preset_config(cfg, name)
            shaped = nagra.shape_room_ir(ir, selected["room"]["rt60_s"])
            energy = np.cumsum((shaped ** 2)[::-1])[::-1]
            db = 10 * np.log10(np.maximum(energy / energy[0], 1e-30))
            mask = (db < -5) & (db > -25)
            measured = -60 / np.polyfit(np.flatnonzero(mask) / SR, db[mask], 1)[0]
            self.assertAlmostEqual(measured, target, delta=0.03)
        self.assertIs(nagra.preset_config(cfg, None), cfg)
        self.assertNotIn("rt60_s", cfg["room"])

    def test_stereo_writer_preserves_channels(self):
        import tempfile
        from scipy.io import wavfile
        from soundbench.scripts.render_tape import write24
        x = np.random.default_rng(2).normal(0, 0.01, SR)
        y = nagra.transfer_stereo(x, nagra.load_config()["transfer"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "transfer.wav"
            write24(path, y)
            _, pcm = wavfile.read(path)
            np.testing.assert_allclose(pcm / (2 ** 31 - 1), y, atol=1e-9)
            write24(path, x)
            _, pcm = wavfile.read(path)
            np.testing.assert_array_equal(pcm[:, 0], pcm[:, 1])

    def test_seed_reproducibility_and_variation(self):
        cfg = nagra.load_config()
        x = 0.1 * np.sin(2 * np.pi * 440 * np.arange(SR) / SR)
        def run(seed):
            return nagra.render([("guitar", x)], {"A": {}}, [], cfg, np.zeros(1), seed)[0].tobytes()
        self.assertEqual(run(1969), run(1969))
        self.assertNotEqual(run(1969), run(1970))

    def test_sparse_recording_uses_active_level_and_silence_is_safe(self):
        tone = 0.1 * np.sin(2 * np.pi * 1000 * np.arange(round(0.4 * SR)) / SR)
        sparse = np.pad(tone, (0, 40 * SR - len(tone)))
        self.assertAlmostEqual(nagra._program_rms(sparse), 0.1 / np.sqrt(2), places=6)
        ending = np.r_[np.zeros(40 * SR), tone[:SR // 5]]
        self.assertAlmostEqual(nagra._program_rms(ending), 0.1 / np.sqrt(2), places=6)
        a = nagra.record(tone, steady())
        b = nagra.record(sparse, steady())[:len(tone)]
        np.testing.assert_array_equal(a, b)
        self.assertTrue(np.isfinite(nagra.record(np.zeros(SR), steady())).all())
        self.assertEqual(np.count_nonzero(nagra.record(np.zeros(SR), steady())), 0)

    def test_changes_keep_performance_boundaries_and_exact_zero_gaps(self):
        cfg = nagra.load_config()  # transport slows, but the next source starts on schedule
        sources = {n: np.arange(8 * SR, dtype=float) / SR for n in "ABC"}
        changes = [dict(at_s=2, roll="B", gap_s=1), dict(at_s=5, roll="C", gap_s=1)]
        with patch.object(nagra, "playback_colour", side_effect=lambda x, *args: x):
            y, segments = nagra.splice(sources, changes, {n: cfg for n in sources}, np.random.default_rng(1))
        self.assertEqual([s["output_start_s"] for s in segments], [0, 3, 6])
        self.assertEqual([s["program_start_s"] for s in segments], [0, 3, 6])
        self.assertEqual(segments[1]["seconds"], 2)
        self.assertEqual(np.count_nonzero(y[2 * SR:3 * SR]), 0)
        self.assertEqual(np.count_nonzero(y[5 * SR:6 * SR]), 0)
        self.assertAlmostEqual(y[3 * SR + 192], 3 + 0.003 * 0.75, delta=0.002)
        with self.assertRaisesRegex(ValueError, "gap consumes"):
            nagra.splice(sources, [dict(at_s=2, roll="B", gap_s=4), dict(at_s=5, roll="C")],
                         {n: cfg for n in sources}, np.random.default_rng(1))

    def test_speed_integration_uses_exclusive_sum(self):
        cfg = steady()
        ratio = np.linspace(0.5, 1.5, SR)
        source = np.arange(3 * SR, dtype=float)
        with (patch.object(nagra, "speed_curve", return_value=(ratio, np.arange(SR) / SR)),
              patch.object(nagra, "playback_colour", side_effect=lambda x, *args: x)):
            y, _ = nagra.splice({"A": source}, [dict(at_s=1, roll="stop")], {"A": cfg}, np.random.default_rng(1))
        np.testing.assert_allclose(y[:SR], np.r_[0, np.cumsum(ratio[:-1])])

    def test_emphasis_and_deemphasis_are_reciprocal(self):
        x = np.random.default_rng(1).normal(size=SR)
        y = nagra.shelf(nagra.shelf(x, 3000, 6), 3000, 6, inverse=True)
        np.testing.assert_allclose(x, y, atol=1e-12)

    def test_room_reflection_after_stem_end_survives(self):
        cfg = steady()
        cfg["mic"]["direct_exponent"] = 0
        x = np.zeros(SR); x[-1] = 1
        ir = np.zeros(round(0.3 * SR)); ir[round(0.2 * SR)] = 1
        y = nagra.mic([x], [1], cfg, ir, np.random.default_rng(1))
        self.assertEqual(len(y), len(x) + len(ir) - 1)
        self.assertGreater(abs(y[SR - 1 + round(0.2 * SR)]), 0.1)

    def test_final_segment_plays_all_source_at_nonunit_speed(self):
        for cents, seconds in ((-1200, 2), (1200, 0.5)):
            cfg = steady(); cfg["speed_cents"] = cents
            source = np.arange(SR, dtype=float)
            with patch.object(nagra, "playback_colour", side_effect=lambda x, *args: x):
                y, _ = nagra.splice({"A": source}, [], {"A": cfg}, np.random.default_rng(1))
            self.assertAlmostEqual(len(y) / SR, seconds, delta=1 / SR)
            self.assertGreaterEqual(y[-1], SR - 2)

    def test_stop_truncates_then_has_exactly_one_point_five_seconds_silence(self):
        cfg = nagra.load_config()
        y, records = nagra.splice({"A": np.ones(5 * SR) * 0.1}, [dict(at_s=2, roll="stop")],
                                  {"A": cfg}, np.random.default_rng(1))
        self.assertEqual(len(y), round(3.5 * SR))
        self.assertEqual(np.count_nonzero(y[2 * SR:]), 0)
        self.assertGreater(np.count_nonzero(y[:2 * SR]), SR)
        self.assertEqual(records[-1], {"roll": "stop", "output_start_s": 2, "seconds": 1.5})
        for changes in ([dict(at_s=1, roll="stop"), dict(at_s=2, roll="A")],
                        [dict(at_s=1, roll="stop"), dict(at_s=2, roll="stop")]):
            with self.assertRaisesRegex(ValueError, "at most once and be last"):
                nagra.splice({"A": np.ones(SR)}, changes, {"A": cfg}, np.random.default_rng(1))

    def test_silent_roll_hiss_matches_configured_rms(self):
        cfg = steady()
        y, _ = nagra.render([("guitar", np.zeros(3 * SR))], {"A": {}}, [], cfg, np.zeros(1), 1)
        db = 20 * np.log10(np.sqrt(np.mean(y[SR:-SR] ** 2)))
        self.assertLess(abs(db - cfg["tape"]["hiss_dbfs"]), 1)

    def test_wow_on_one_kilohertz_tone_matches_cents(self):
        cfg = steady()
        cfg["wow"].update(cents=6, wander=0)
        x = np.sin(2 * np.pi * 1000 * np.arange(12 * SR) / SR)
        with patch.object(nagra, "playback_colour", side_effect=lambda x, *args: x):
            y, _ = nagra.splice({"A": x}, [], {"A": cfg}, np.random.default_rng(1))
        phase = np.unwrap(np.angle(hilbert(y)))
        hz = np.diff(phase) * SR / (2 * np.pi)
        cents = 1200 * np.log2(hz[SR:-SR] / 1000)
        measured = np.std(cents) * np.sqrt(2)
        self.assertGreater(measured, 3)
        self.assertLess(measured, 12)


if __name__ == "__main__":
    unittest.main()
