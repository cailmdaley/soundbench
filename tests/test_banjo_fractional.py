#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["jax==0.11.2", "numpy>=2", "scipy>=1.14", "numba>=0.61", "soundfile>=0.13"]
# ///
"""Fractional pluck interpolation, deployable gradients and engine parity."""
from pathlib import Path
import sys
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "soundbench"
sys.path.insert(0, str(ROOT / 'scripts'))
from soundbench.scripts.models.banjo import CONFIG, load_config, render as engine_render
from soundbench.scripts.models.banjo_fit import twin


def normalized(audio):
    audio = np.asarray(audio, dtype=float)
    return audio / max(np.sqrt(np.mean(audio * audio)), 1e-20)


class FractionalPluckTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.config['excitation']['fractional_pluck'] = True
        self.config['excitation']['position_jitter'] = 0.
        self.note = dict(string=1, pitch=62, frequency=440 * 2 ** ((62 - 69) / 12))

    def test_flag_is_strictly_boolean(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'banjo.toml'
            original = CONFIG.read_text()
            path.write_text(original.replace('pluck_position =', 'fractional_pluck = 1\npluck_position =', 1))
            with self.assertRaisesRegex(ValueError, 'fractional_pluck'):
                load_config(path)

    def test_deployable_position_gradient_matches_finite_difference(self):
        p, base = twin.initial_parameters(self.config, [self.note], 8)
        p = dict(p)
        p['position'] = jnp.array([.37])
        target = twin.render(p, base, self.config, [self.note], 2048,
                             deployable=True, two_stage=True)
        weights = jnp.sin(jnp.arange(target.size, dtype=jnp.float64) * .017).reshape(target.shape)

        def objective(position):
            candidate = dict(p, position=jnp.array([position]))
            audio = twin.render(candidate, base, self.config, [self.note], 2048,
                                deployable=True, two_stage=True)
            return jnp.sum(audio * weights)

        point = p['position'][0]
        analytic = jax.grad(objective)(point)
        epsilon = 1e-3
        numeric = (objective(point + epsilon) - objective(point - epsilon)) / (2 * epsilon)
        self.assertTrue(np.isfinite(float(analytic)))
        self.assertGreater(abs(float(analytic)), 1e-8)
        np.testing.assert_allclose(float(analytic), float(numeric), rtol=2e-2, atol=2e-5)

    def test_reseed_moves_only_latents_outside_the_interior(self):
        p, base = twin.initial_parameters(self.config, [self.note], 8)
        p = dict(p)
        p['loss'] = p['loss'].at[0, 0].set(-40.)
        p['mode_gain'] = p['mode_gain'].at[0].set(20.)
        p['width'] = p['width'].at[0].set(.5)
        original_physical = twin.physical(p, base, self.config)
        result = twin.reseed_bounded_latents(p, base)
        after_physical = twin.physical(result, base, self.config)
        self.assertEqual(float(result['width'][0]), float(p['width'][0]))
        self.assertGreater(float(result['loss'][0, 0]), -12.)
        self.assertLess(float(result['mode_gain'][0]), 6.)
        np.testing.assert_allclose(float(after_physical['width'][0]), float(original_physical['width'][0]))
        self.assertTrue(np.isfinite(np.asarray(result['mode_time'])).all())

    def test_fractional_engine_and_twin_match_with_detuned_second_lane(self):
        f0 = 148.53684
        event_pitch = 69 + 12 * np.log2(f0 / 440)
        note = dict(self.note, pitch=event_pitch, frequency=f0)
        self.config['pickups'] = dict(head=0., string=1.)
        self.config['bridge'] = dict(coupling=0., head_feedback=0.)
        self.config['excitation'].update(position_jitter=0., exponential_pick=1,
                                         velocity_corner_exponent=0.,
                                         pluck_position=23.5*f0/48000)
        self.assertAlmostEqual(self.config['excitation']['pluck_position']*48000/f0 % 1, .5)
        p, base = twin.initial_parameters(self.config, [note], 8)
        p = dict(p)
        p['second_mix'] = jnp.array([np.log(.8 / .2)])
        p['second_detune'] = jnp.full(5, np.arctanh(7.9 / 8))
        p['second_rate'] = jnp.zeros(5)
        q = twin.physical(p, base, self.config)
        self.config['excitation']['pluck_position'] = float(q['position'][0])
        self.config['strings'].update(
            decay_hz=twin.KNOTS.tolist(),
            decay_t60=(np.log(1000) / np.asarray(q['rates'])).tolist(),
            dispersion_stages=np.asarray(q['stages']).tolist(),
            polarization_mix=[.4] * 5,
            polarization_rate=np.asarray(q['second_rate']).tolist(),
            polarization_cents=[7.9] * 5,
        )

        predicted = np.asarray(twin.render(p, base, self.config, [note], 24000,
                                           deployable=True, two_stage=True))[0]
        actual = engine_render([[0, 0, 0, 0, event_pitch, 90, 0, 0, 0, 0, 0]], .5,
                               config=self.config)[0]
        actual *= .05 / np.sqrt(np.mean(actual ** 2))
        np.testing.assert_allclose(actual, predicted, atol=2e-5)

    def test_historical_integer_path_is_unchanged_without_flag(self):
        legacy = load_config()
        explicit = load_config()
        explicit['excitation']['fractional_pluck'] = False
        events = [[0, 0, 0, 0, 62, 90, 0, 0, .173, 0, 1]]
        a = engine_render(events, .1, config=legacy)
        b = engine_render(events, .1, config=explicit)
        np.testing.assert_array_equal(a, b)


if __name__ == '__main__':
    unittest.main()
