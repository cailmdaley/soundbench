#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["mido>=1.3,<2", "numpy", "numba>=0.61", "scipy", "soundfile"]
# ///
"""Fast invariants for the pipe organ: determinism, compass, no velocity, bounded memory, routing."""
import copy
from pathlib import Path
import sys
import subprocess
import tempfile
import tracemalloc
import unittest
import numpy as np
ROOT = Path(__file__).resolve().parents[1] / "soundbench"
sys.path.insert(0, str(ROOT/'scripts'))
from soundbench.scripts.models import pipe_organ
from soundbench.scripts.models.keys_common import SR
from soundbench.scripts.instrumentation import MODELS, STEREO_INSTRUMENTS
from soundbench.scripts.model_dispatch import RENDERERS


def note(p=60, v=80, t=0.05, off=.4):
    return {'pitch': p, 'velocity': v, 'time': t, 'off': off}


def render(notes, registration='principal', seconds=1.2, controls=()):
    cfg = copy.deepcopy(pipe_organ.load_config()); cfg['active'] = registration
    return pipe_organ.render(notes, seconds, cfg, controls)


def peak_hz(y, lo, hi):
    y = y[4800:19200].mean(axis=1); size = 262144
    spec = abs(np.fft.rfft(y*np.hanning(len(y)), size)); freq = np.fft.rfftfreq(size, 1/SR)
    band = (freq >= lo) & (freq <= hi)
    return freq[band][spec[band].argmax()]


class PipeOrgan(unittest.TestCase):
    def test_deterministic_and_order_independent(self):
        a = render([note(60), note(64, t=.1)]); b = render([note(64, t=.1), note(60)])
        self.assertTrue(np.array_equal(a, render([note(60), note(64, t=.1)])))
        np.testing.assert_allclose(a, b, atol=1e-6)

    def test_stereo_float32_and_silent_before_onset(self):
        y = render([note(60, t=.5, off=.8)])
        self.assertEqual((y.dtype, y.shape[1]), (np.float32, 2))
        self.assertLess(float(abs(y[:int(.49*SR)]).max()), 1e-3)

    def test_no_velocity_dependence(self):
        self.assertTrue(np.array_equal(render([note(60, v=1)]), render([note(60, v=127)])))

    def test_compass(self):
        for registration, low, high in (('principal', 36, 96), ('pedal', 36, 67)):
            render([note(low)], registration, .6); render([note(high)], registration, .6)
            for bad in (low-1, high+1):
                with self.assertRaises(ValueError):
                    render([note(bad)], registration, .6)

    def test_footage_sets_pitch(self):
        f_c3 = 440*2**((48-69)/12)
        self.assertAlmostEqual(peak_hz(render([note(48)], 'pedal', .6), f_c3/2*.97, f_c3/2*1.03)/(f_c3/2), 1, delta=.02)
        flutes = render([note(60)], 'flutes', .6)
        self.assertAlmostEqual(peak_hz(flutes, 255, 270)/261.63, 1, delta=.02)

    def test_stopped_pipe_is_odd_harmonic(self):
        cfg = copy.deepcopy(pipe_organ.load_config()); cfg['active'] = 'gedackt-alone'
        cfg['registration']['gedackt-alone'] = {'range': [36, 96], 'stop': [{'family': 'gedackt', 'footage': 8, 'db': 0.0}]}
        y = pipe_organ.render([note(48)], .8, cfg, [])[4800:30000].mean(axis=1)
        spec = abs(np.fft.rfft(y*np.hanning(len(y)), 1 << 18)); freq = np.fft.rfftfreq(1 << 18, 1/SR)
        f0 = 440*2**((48-69)/12)
        level = lambda h: spec[(freq > h*f0*.97) & (freq < h*f0*1.03)].max()
        self.assertGreater(level(3), 3*level(2))

    def test_swell_box_closes_gain_and_top_end(self):
        n = [note(72, t=.05, off=1.0)]
        open_ = render(n, 'principal', 1.5); shut = render(n, 'principal', 1.5, [(0., 11, 0)])
        self.assertLess(float(np.sqrt(np.mean(shut**2)))*10, float(np.sqrt(np.mean(open_**2))))

    def test_memory_bounded(self):
        tracemalloc.start()
        render([note(p, t=.1*i, off=6.0) for i, p in enumerate((36, 48, 60, 72))], 'plenum', 8.0)
        peak = tracemalloc.get_traced_memory()[1]; tracemalloc.stop()
        self.assertLess(peak, 200e6)

    def test_registered_in_dispatch(self):
        self.assertIn('pipe-organ', MODELS); self.assertIn('pipe-organ', STEREO_INSTRUMENTS)
        self.assertEqual(RENDERERS['pipe-organ'][1], 'pipe-organ')

    def test_attack_ramp_settles_before_it_ends(self):
        for tau0 in (.02, .04, .115):
            for h in (1, 10, 80):
                m = pipe_organ.ramp_samples(tau0, 80)
                step = np.exp(-m/SR/pipe_organ.attack_tau(tau0, h))
                self.assertLess(step, 1e-5)

    def test_restruck_pipe_does_not_double(self):
        single = render([note(60, t=.05, off=1.0)], 'plenum', 1.8)
        again = render([note(60, t=.05, off=.4), note(60, t=.3, off=1.0)], 'plenum', 1.8)
        self.assertLess(float(abs(again).max()), 1.1*float(abs(single).max()))


if __name__ == '__main__':
    unittest.main()
