#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["jax==0.11.2", "numpy>=2"]
# ///
"""Exact zero-start recording-chain semantics and boundary gradients."""
import sys
from pathlib import Path
import unittest

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from soundbench.scripts.models.banjo_fit import chain

jax.config.update('jax_enable_x64', True)


class ZeroStartChainTests(unittest.TestCase):
    def setUp(self):
        self.p = chain.initial_parameters(zero_start=True)
        self.notes = [dict(background_psd=[2e-8]*24, background_scale=1.)]
        t = jnp.arange(12000)/chain.SR
        self.y = (jnp.sin(2*jnp.pi*440*t) + .3*jnp.sin(2*jnp.pi*22000*t))[None, :]

    def test_seed_is_exact_identity_through_every_recording_stage(self):
        q = chain.physical(self.p)
        np.testing.assert_array_equal(np.asarray(chain.room_ir(self.p)), np.eye(1, chain.IR_SAMPLES)[0])
        np.testing.assert_allclose(np.asarray(chain.mic_filter(self.p)), np.r_[1., np.zeros(128)], atol=1e-14)
        self.assertEqual(float(q['room_wet']), 0.)
        np.testing.assert_array_equal(np.asarray(q['room_early']), np.zeros(4))
        np.testing.assert_array_equal(np.asarray(q['mic_db']), np.zeros(7))
        self.assertEqual(float(q['background_amplitude']), 0.)
        self.assertEqual(float(q['codec_fraction']), 0.)
        self.assertEqual(float(q['codec_rolloff']), 0.)
        for stages in (
            dict(room=True), dict(mic=True), dict(background=True), dict(codec=True),
            dict(room=True, mic=True, background=True, codec=True),
        ):
            np.testing.assert_allclose(np.asarray(chain.apply(self.y, self.p, self.notes, **stages)),
                                       np.asarray(self.y), atol=1e-12, rtol=1e-12)

    def test_zero_start_keeps_historical_seed_opt_in(self):
        historical = chain.initial_parameters()
        self.assertNotIn('chain_zero_start', historical)
        self.assertEqual(float(historical['chain_room'][0]), -3.)
        self.assertEqual(float(historical['chain_codec']), -2.)
        self.assertIn('chain_zero_start', self.p)
        self.assertEqual(self.p['chain_zero_start'].shape, (0,))

    def test_chain_controls_have_finite_useful_gradients_at_zero(self):
        target = jnp.cos(2*jnp.pi*21000*jnp.arange(12000)/chain.SR)[None, :]

        def objective(p):
            prediction = chain.apply(self.y, p, self.notes, room=True, mic=True,
                                     background=True, codec=True)
            return jnp.mean(prediction*target) + chain.penalty(
                p, room=True, mic=True, background=True, codec=True)

        grads = jax.grad(objective)(self.p)
        for name, grad in grads.items():
            self.assertTrue(np.isfinite(np.asarray(grad)).all(), name)
        for name in ('chain_room', 'chain_mic', 'chain_background', 'chain_codec', 'chain_codec_rolloff'):
            self.assertGreater(float(jnp.linalg.norm(grads[name])), 1e-9, name)


if __name__ == '__main__':
    unittest.main()
