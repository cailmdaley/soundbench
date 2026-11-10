#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2", "scipy>=1.14", "numba>=0.61"]
# ///
"""Continuous contraction, loop phase, polarization and default regression."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy.signal import freqz
ROOT = Path(__file__).resolve().parents[1] / "soundbench"
sys.path.insert(0, str(ROOT/'scripts'))
from soundbench.scripts.models.banjo import load_config, render, _rich_coefficients, _output_guard
from soundbench.scripts.models.string_loss import design_loss, loss_bank


def rich_config():
    c = load_config()
    c['strings']['decay_hz'] = [80., 300., 1000., 2000., 3000., 8000., 24000.]
    c['strings']['decay_t60'] = [[2.5, 2.5, 1., .4, .2, .1, .03]]*5
    c['bridge']['coupling'] = c['bridge']['head_feedback'] = 0.
    return c


class LossTests(unittest.TestCase):
    def test_continuous_contraction_certificate(self):
        for f0 in (150, 440, 1200):
            h = design_loss([80, 1000, 3000, 24000], [3, 8, 40, 250], f0)
            _, response = freqz(h, worN=32768, whole=True)
            bound = np.max(abs(response))+np.pi/32768*np.sum(np.arange(len(h))*abs(h))
            self.assertLess(bound, 1)
            self.assertLess(np.max(abs(freqz(h, worN=131072)[1])), 1)

    def test_exact_fundamental_phase_across_fractional_pitches(self):
        c = rich_config()
        bank = loss_bank(c['strings'])
        for pitch in np.linspace(50, 96, 70):
            integer, a, h = _rich_coefficients(pitch, 48000, -.1, bank, 0)
            w = 2*np.pi*440*2**((pitch-69)/12)/48000
            z = np.exp(-1j*w)
            loop = z**(integer+3)*(a+z)/(1+a*z)*(-.1+z)/(1-.1*z)*np.sum(h*z**np.arange(len(h)))
            self.assertLess(abs(np.angle(loop)), 1e-10)
            self.assertLess(abs(loop), 1)

    def test_curve_tracks_partial_decay(self):
        f0 = 195.
        hz = np.array([80, 300, 1000, 2000, 3000, 8000, 24000])
        rates = np.array([3, 3, 8, 20, 40, 80, 230])
        h = design_loss(hz, rates, f0)
        partials = f0*np.arange(1, 41)
        response = freqz(h, worN=2*np.pi*partials/48000)[1]
        expected = np.interp(np.log(partials), np.log(hz), rates)
        error = np.sqrt(np.mean((np.log(-f0*np.log(abs(response)))-np.log(expected))**2))
        self.assertLess(error, .2)

    def test_polarization_zero_mix_is_identity_and_release_resets(self):
        c = rich_config()
        event = [0, 0, 2, 0, 55, 90, 0, 0, 0, 0, 0]
        a = render([event], .7, config=c)
        c['strings'].update(polarization_mix=[0.]*5, polarization_rate=[1.5]*5, polarization_cents=[4.]*5)
        b = render([event], .7, config=c)
        np.testing.assert_array_equal(a, b)
        c['strings']['polarization_mix'] = [.3]*5
        d = render([event], .7, config=c)
        self.assertTrue(np.isfinite(d).all())
        self.assertGreater(np.max(abs(d-a)), 1e-4)
        choke = [4800, 5, 2, 0, 55, 90, 0, 0, 0, 0, 0]
        e = render([event, choke], .7, config=c)
        self.assertLess(np.linalg.norm(e[:, -4800:]), np.linalg.norm(d[:, -4800:])*.1)

    def test_output_guard_is_odd_monotone_and_linear_below_knee(self):
        samples = np.linspace(-4., 4., 10001)
        guarded = _output_guard(samples[None,:].copy(), .90, .98)[0]
        region = abs(samples) <= .9
        np.testing.assert_array_equal(guarded[region], samples[region])
        np.testing.assert_allclose(guarded, -guarded[::-1], atol=1e-15)
        self.assertTrue(np.all(np.diff(guarded) >= 0))
        self.assertLessEqual(np.max(abs(guarded)), .98)
        epsilon = 1e-5
        near = _output_guard(np.array([[.9-epsilon, .9+epsilon]]), .9, .98)[0]
        self.assertAlmostEqual((near[1]-near[0])/(2*epsilon), 1., places=6)

    def test_invalid_output_guard_schema_rejected(self):
        for fields in ({'peak_knee': .9}, {'peak_knee': .9, 'peak_ceiling': .8}, {'peak_knee': .9, 'peak_ceiling': 1.}):
            c = rich_config()
            c['engine'].update(fields)
            text = ''.join('\n['+name+']\n'+''.join(k+' = '+json.dumps(v)+'\n' for k,v in group.items()) for name,group in c.items())
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)/'bad.toml'
                path.write_text(text)
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_invalid_curve_and_partial_polarization_rejected(self):
        c = rich_config()
        for key, value in [('decay_t60', [[1., 2.]]*5), ('decay_hz', [80., 80.]), ('polarization_mix', [.2]*5)]:
            invalid = copy.deepcopy(c)
            invalid['strings'][key] = value
            text = ''.join('\n['+name+']\n'+''.join(k+' = '+json.dumps(v)+'\n' for k,v in group.items()) for name,group in invalid.items())
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)/'bad.toml'
                path.write_text(text)
                with self.assertRaises(ValueError):
                    load_config(path)


if __name__ == '__main__':
    unittest.main()
