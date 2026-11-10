# /// script
# dependencies = ["jax==0.11.2", "numpy>=2"]
# ///
"""Measured-noise scaling, phase separation and diagnostics for the all-way loss."""
import sys
from pathlib import Path
import unittest

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from soundbench.scripts.models.banjo_fit import allway_loss

jax.config.update('jax_enable_x64', True)


class AllWayLossTests(unittest.TestCase):
    def setUp(self):
        self.note = dict(pitch=69, frequency=440., background_psd=[2e-8]*24,
                         background_scale=1.)

    def test_fft_noise_power_uses_hann_energy_and_endpoint_factor(self):
        window = 256
        sigma = allway_loss._noise_sigma([self.note], window)[0]
        expected = 2e-8 * 48000 * np.sum(np.hanning(window)**2) / 2
        self.assertAlmostEqual(sigma[1]**2, expected, delta=expected*1e-6)
        self.assertAlmostEqual(sigma[0]**2, expected*2, delta=expected*2e-6)
        self.assertAlmostEqual(sigma[-1]**2, expected*2, delta=expected*2e-6)

    def test_noise_scale_and_note_scale_are_applied_quadratically_to_power(self):
        a = dict(self.note, background_scale=.25)
        b = dict(self.note, background_scale=2.)
        scale = allway_loss._noise_sigma([a, b], 128)
        self.assertAlmostEqual(scale[0, 10] / scale[1, 10], .125, places=6)

    def test_objective_is_finite_differentiable_and_zero_at_identity(self):
        n, samples = 1, 8192
        t = jnp.arange(samples)/48000
        y = (.1*jnp.sin(2*jnp.pi*440*t)+.025*jnp.sin(2*jnp.pi*1320*t))[None]
        prepared = allway_loss.prepare(y, [self.note])
        value = allway_loss.objective(y, prepared)
        grad = jax.grad(lambda z: allway_loss.objective(z, prepared))(y)
        self.assertTrue(np.isfinite(float(value)))
        self.assertTrue(np.isfinite(np.asarray(grad)).all())
        self.assertAlmostEqual(float(value), 0., places=6)
        changed = y*.8
        self.assertGreater(float(allway_loss.objective(changed, prepared)), float(value))

    def test_phase_sensitive_residual_is_separate_from_magnitude_residual(self):
        samples = 8192
        t = np.arange(samples)/48000
        target = (.1*np.sin(2*np.pi*440*t))[None]
        shifted = (.1*np.cos(2*np.pi*440*t))[None]
        result = allway_loss.noise_units(target, shifted, [self.note], window=512)
        # A quarter-cycle shift preserves the steady sinusoid's magnitude while
        # changing complex phase. Edge frames are ignored for this comparison.
        mag = result['magnitude_map'][0, 4:-4]
        phase = result['complex_residual_map'][0, 4:-4]
        self.assertLess(float(np.mean(np.abs(mag[:, 4:8]))), .1)
        self.assertGreater(float(np.mean(np.abs(phase[:, 4:8]))), 10.)
        self.assertIn('phase-sensitive', result['map_labels']['complex'])
        self.assertIn('not chi-square', result['interpretation'])

    def test_independent_background_draws_are_order_one_in_noise_units(self):
        # Independently drawn white backgrounds should land on the measured
        # FFT scale, while finite Welch/STFT sampling leaves visible scatter.
        rng = np.random.default_rng(7)
        n, samples = 1, 240000
        psd = 2e-8
        variance = psd * 48000 / 2
        target = rng.normal(0., np.sqrt(variance), (n, samples))
        prediction = rng.normal(0., np.sqrt(variance), (n, samples))
        note = dict(self.note, background_psd=[psd]*24)
        result = allway_loss.noise_units(target, prediction, [note], window=1024)
        summary = result['notes'][0]['regions']['sustain/mid']['complex_abs_noise_phase_sensitive']
        self.assertGreater(summary['mean'], .5)
        self.assertLess(summary['mean'], 2.5)
        self.assertGreater(summary['samples'], 10)
        self.assertGreaterEqual(summary['sem'], 0.)

    def test_partial_centers_are_resolved_when_the_peak_clears_measured_noise(self):
        samples = 48000
        t = np.arange(samples)/48000
        y = (.1*np.sin(2*np.pi*443*t)+.03*np.sin(2*np.pi*1338*t))[None]
        centers = allway_loss.partial_centers(y, [self.note])
        self.assertEqual(len(centers), 1)
        self.assertGreaterEqual(len(centers[0]), 3)
        # The first ridge follows the measured 443-Hz peak rather than forcing
        # the nominal 440-Hz harmonic.
        self.assertLess(abs(centers[0][0]-443), 443*.005)
        prepared = allway_loss.prepare(y, [self.note])
        self.assertEqual(prepared['ridge_mask'].shape[-1], 2049)
        self.assertTrue(prepared['partial_resolved'][0][0])


if __name__ == '__main__':
    unittest.main()
