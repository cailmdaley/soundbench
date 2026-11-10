#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["jax==0.11.2", "numpy>=2", "scipy>=1.14", "numba>=0.61", "soundfile>=0.13"]
# ///
"""Exact discrete twin parity, physical constraints and differentiable loss tests."""
import copy
from pathlib import Path
import sys
import unittest
import tempfile
import soundfile as sf
import json
import hashlib
import subprocess
import numpy as np
import jax
import jax.numpy as jnp
from scipy.signal import butter, sosfiltfilt, hilbert
ROOT = Path(__file__).resolve().parents[1] / "soundbench"
sys.path.insert(0, str(ROOT/'scripts'))
from soundbench.scripts.models.banjo import load_config, _render, modal_coefficients, render as numba_render
from soundbench.scripts.paths import preset_path

CONFIG = preset_path('banjo', 'membrane')
from soundbench.scripts.models.banjo_fit import twin, loss, chain
from soundbench.scripts.models.banjo_fit.audio import load as load_audio, restore_scales
from soundbench.scripts.audio_pitch import pitch_cents


class FitTests(unittest.TestCase):
    def test_string_impulse_pitch_and_partial_t60_match_numba(self):
        config = load_config(CONFIG)
        sr = 48000
        samples = 72000
        for string in range(5):
            pitch = config['strings']['open_midi'][string]
            s = config['strings']
            c, r2, b, weights = modal_coefficients(config)
            row = np.array([[0, 0, string, 0, pitch, 90, 0, 0, .125, 0, 0]], dtype=float)
            pulse = np.ones((1, 1))
            actual = _render(row, pulse, samples, 1, sr, np.array(s['open_midi'], dtype=float),
                             np.array(s['t60']), np.array(s['loss_mix']), np.array(s['dispersion']),
                             s['release_seconds'], s['choke_seconds'], 1., 0., 0., c, r2, b, weights,
                             0., 1., 1., 1e-18, np.empty((5, 0, 65)), np.zeros((5, 3)))[0]
            nfft = twin.fft_size(samples/sr)
            f = jnp.fft.rfftfreq(nfft, 1/sr)
            comb = int(.125*sr/(440*2**((pitch-69)/12)))
            transfer = twin.string_response(f, pitch, string, config)*(1-jnp.exp(-2j*jnp.pi*f*comb/sr))
            predicted = np.asarray(jnp.fft.irfft(transfer, nfft))[:samples]
            cents = abs(pitch_cents(actual[4800:24000], pitch)-pitch_cents(predicted[4800:24000], pitch))
            self.assertLess(cents, 1.)
            fundamental = 440*2**((pitch-69)/12)
            errors = []
            for partial in range(1, 7):
                center = fundamental*partial
                sos = butter(3, [center-fundamental*.18, center+fundamental*.18], btype='bandpass', fs=sr, output='sos')
                times = np.arange(samples)/sr
                region = (times>.15)&(times<.8)
                decays = []
                for y in (actual, predicted):
                    envelope = abs(hilbert(sosfiltfilt(sos, y)))
                    slope = np.polyfit(times[region], np.log(envelope[region]+1e-10), 1)[0]
                    decays.append(-np.log(1000)/slope)
                error = abs(decays[1]/decays[0]-1)
                errors.append(error)
                self.assertLess(error, .1)
            print(f'string {string+1}: pitch difference {cents:.4f} cents; max partial T60 difference {max(errors)*100:.3f}%')

    def test_constraints_and_gradients(self):
        config = load_config(CONFIG)
        notes = [dict(string=1, pitch=62, frequency=293.664768)]
        p, base = twin.initial_parameters(config, notes, 8)
        q = twin.physical(p, base, config)
        self.assertTrue(np.all(np.diff(q['rates'], axis=-1)>0))
        self.assertTrue(np.all(q['rates']>=twin.MIN_RATE))
        rendered = twin.render(p, base, config, notes, 4800)
        self.assertTrue(np.isfinite(rendered).all())
        prepared = loss.prepare(rendered, notes)
        gradient = jax.grad(lambda x: loss.objective(twin.render(x, base, config, notes, 4800), prepared, x, base))(p)
        for name, value in gradient.items():
            self.assertTrue(np.isfinite(value).all(), name)
        self.assertGreater(float(np.linalg.norm(gradient['mode_gain'])), 0)
        second = twin.decay_rate(jnp.array([0., 80., 1000., 24000.]), q['rates'][0], .001)
        self.assertTrue(np.all(second>=twin.MIN_RATE))
        self.assertTrue(np.all(np.diff(second)>=0))
        both = twin.render(p, base, config, notes, 4800, two_stage=True)
        self.assertTrue(np.isfinite(both).all())

    def test_deployable_fir_and_loop_match_engine(self):
        from soundbench.scripts.models.string_loss import design_loss, loss_bank
        from soundbench.scripts.models.banjo import _rich_coefficients
        from scipy.signal import freqz
        rates = np.array([3., 3.1, 5., 8., 12., 20., 40., 70., 120., 250.])
        h = np.asarray(twin.fir_loss(jnp.array(rates), 195.))
        expected = design_loss(twin.KNOTS, rates, 195.)
        np.testing.assert_allclose(h, expected, atol=2e-5)
        config = load_config(CONFIG)
        config['strings'].update(decay_hz=twin.KNOTS.tolist(), decay_t60=(np.log(1000)/np.tile(rates, (5, 1))).tolist())
        bank = loss_bank(config['strings'])
        pitch = 55.17
        stages = np.array([-.6, -.4, -.2])
        integer, a, h = _rich_coefficients(pitch, 48000, -.1, bank, 0, stages)
        f = np.fft.rfftfreq(262144, 1/48000)
        z = np.exp(-2j*np.pi*f/48000)
        expected = z**integer*(a+z)/(1+a*z)*np.fft.rfft(h, 262144)
        for d in np.r_[-.1, stages]:
            expected *= (d+z)/(1+d*z)
        actual = np.asarray(twin.discrete_loop(jnp.array(f), pitch, jnp.array(rates), -.1, jnp.array(stages)))
        np.testing.assert_allclose(actual, expected, atol=2e-4)

    def test_deployable_audio_parity_and_note_specific_nuisance_invariance(self):
        config = load_config(CONFIG)
        config['pickups'] = dict(head=0., string=1.)
        config['bridge'] = dict(coupling=0., head_feedback=0.)
        config['excitation'].update(position_jitter=0., exponential_pick=1, velocity_corner_exponent=0.)
        notes = [dict(string=1, pitch=62, frequency=440*2**((62-69)/12))]
        p, base = twin.initial_parameters(config, notes, 8)
        p['onset'] = jnp.ones(1)
        p['detune'] = jnp.ones(1)
        p['glide'] = jnp.ones(1)
        p['noise'] = jnp.ones(1)
        q = twin.physical(p, base, config)
        config['strings'].update(decay_hz=twin.KNOTS.tolist(), decay_t60=(np.log(1000)/np.asarray(q['rates'])).tolist(), dispersion_stages=np.asarray(q['stages']).tolist())
        config['excitation']['pluck_position'] = float(q['position'][0])
        actual = numba_render([[0, 0, 0, 0, 62, 90, 0, 0, 0, 0, 0]], .5, config=config)[0]
        actual *= .05/np.sqrt(np.mean(actual**2))
        predicted = np.asarray(twin.render(p, base, config, notes, 24000, deployable=True))[0]
        np.testing.assert_allclose(actual, predicted, atol=2e-5)
        p['onset'] = p['detune'] = p['glide'] = p['noise'] = jnp.zeros(1)
        clean = np.asarray(twin.render(p, base, config, notes, 24000, deployable=True))[0]
        np.testing.assert_array_equal(predicted, clean)

    def test_second_polarization_comb_boundary_matches_engine(self):
        config=load_config(CONFIG)
        f0=440*2**((62-69)/12)
        config['pickups']=dict(head=0.,string=1.)
        config['bridge']=dict(coupling=0.,head_feedback=0.)
        config['excitation'].update(position_jitter=0.,exponential_pick=1,velocity_corner_exponent=0.,pluck_position=23.04*f0/48000)
        notes=[dict(string=1,pitch=62,frequency=f0)]
        p,base=twin.initial_parameters(config,notes,8)
        p['second_mix']=jnp.full(1,np.log(.8/.2))
        p['second_detune']=jnp.full(5,np.arctanh(7.9/8))
        q=twin.physical(p,base,config)
        primary=int(float(q['position'][0])*48000/f0)
        secondary=int(float(q['position'][0])*48000/(f0*2**(7.9/1200)))
        self.assertEqual((primary,secondary),(23,22))
        config['strings'].update(decay_hz=twin.KNOTS.tolist(),decay_t60=(np.log(1000)/np.asarray(q['rates'])).tolist(),
                                 dispersion_stages=np.asarray(q['stages']).tolist(),polarization_mix=[.4]*5,
                                 polarization_rate=np.asarray(q['second_rate']).tolist(),polarization_cents=[7.9]*5)
        actual=numba_render([[0,0,0,0,62,90,0,0,0,0,0]],.5,config=config)[0]
        actual*=.05/np.sqrt(np.mean(actual**2))
        predicted=np.asarray(twin.render(p,base,config,notes,24000,deployable=True,two_stage=True))[0]
        np.testing.assert_allclose(actual,predicted,atol=2e-5)

    def test_recording_latents_cannot_change_the_dry_twin(self):
        config=load_config(CONFIG)
        notes=[dict(string=1,pitch=62,frequency=293.664768)]
        p,base=twin.initial_parameters(config,notes,8)
        p.update(chain.initial_parameters())
        expected=np.asarray(twin.render(p,base,config,notes,4800,deployable=True,two_stage=True))
        p.update({k:jnp.full_like(v,5.) for k,v in chain.initial_parameters().items()})
        actual=np.asarray(twin.render(p,base,config,notes,4800,deployable=True,two_stage=True))
        np.testing.assert_array_equal(expected,actual)

    def test_shared_excitation_canonicalization_preserves_audio_and_releases_clipped_latents(self):
        config = load_config(CONFIG)
        notes = [dict(string=1,pitch=p,frequency=440*2**((p-69)/12)) for p in (62,71)]
        p, base = twin.initial_parameters(config, notes, 8)
        p['width'] = jnp.array([-4.,4.])
        p['corner'] = jnp.array([-4.,2.])
        p['position'] = jnp.array([-.5,1.])
        p['second_mix'] = jnp.array([-4.,2.])
        shared = twin.canonical_excitation(p,base,config,notes)
        original = twin.render(p,base,config,notes,24000,deployable=True,two_stage=True)
        canonical = twin.render(shared,base,config,notes,24000,deployable=True,two_stage=True)
        np.testing.assert_allclose(original,canonical,atol=1e-12)
        for key in ('width','corner','position','second_mix'):
            self.assertEqual(float(shared[key][0]),float(shared[key][1]))
        self.assertTrue(np.all(abs(shared['width'])<2))
        derivative = jax.grad(lambda x: twin.physical(dict(shared,width=x),base,config)['width'].sum())(shared['width'])
        self.assertTrue(np.all(derivative>0))

    def test_native_ingestion_hashes_and_constrains_pre_onset_psd(self):
        sr = 48000
        t = np.arange(sr*2)/sr
        age = np.maximum(t-.4,0.)
        y = np.random.default_rng(8).normal(size=len(t))*1e-5
        y += .1*np.sin(2*np.pi*293.664768*age)*np.exp(-4*age)*(t>=.4)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            sf.write(path/'note.wav',y,sr,subtype='FLOAT')
            subprocess.run(['ffmpeg','-v','error','-y','-i',str(path/'note.wav'),'-ar','44100',
                            '-c:a','libmp3lame','-b:a','96k',str(path/'note.mp3')],check=True)
            digest=hashlib.sha256((path/'note.mp3').read_bytes()).hexdigest()
            entry=dict(path='note.wav',pitch=62,string=1,sha256=digest)
            (path/'notes.json').write_text(json.dumps([entry]))
            audio,notes=load_audio(path/'notes.json',.8,native=True)
            self.assertEqual(notes[0]['native_sha256'],digest)
            self.assertEqual(len(notes[0]['background_psd']),24)
            self.assertTrue(np.all(np.array(notes[0]['background_psd'])>0))
            self.assertAlmostEqual(notes[0]['background_scale']*notes[0]['rms_before_alignment'],.05)
            self.assertAlmostEqual(float(np.sqrt(np.mean(audio**2))),.05,places=6)
            entry['sha256']='not the source hash'
            (path/'notes.json').write_text(json.dumps([entry]))
            with self.assertRaises(ValueError):
                load_audio(path/'notes.json',.8,native=True)

    def test_alignment_and_missing_fundamental(self):
        sr = 48000
        time = np.arange(sr)/sr
        rng = np.random.default_rng(1)
        y = rng.normal(size=sr)*.0001
        age = np.maximum(time-.1, 0)
        y += (time>=.1)*np.exp(-age*4)*(.08*np.sin(2*np.pi*148.4*2*age)+.04*np.sin(2*np.pi*148.4*3*age))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            sf.write(path/'note.wav', y, sr, subtype='FLOAT')
            (path/'notes.json').write_text(json.dumps([dict(path='note.wav', pitch=50, string=4)]))
            audio, notes = load_audio(path/'notes.json', .8)
        self.assertAlmostEqual(notes[0]['frequency'], 148.4, delta=.1)
        self.assertAlmostEqual(notes[0]['onset_seconds'], .095, delta=.005)
        self.assertNotIn(1, notes[0]['tuning_partials'])
        self.assertAlmostEqual(float(np.sqrt(np.mean(audio**2))), .05, places=5)

    def test_head_gain_initialization_uses_engine_normalization(self):
        config = load_config(CONFIG)
        changed = copy.deepcopy(config)
        changed['head']['weights'] = [3*x for x in config['head']['weights']]
        notes = [dict(string=1, pitch=62, frequency=293.664768)]
        original, base = twin.initial_parameters(config, notes, 8)
        scaled, scaled_base = twin.initial_parameters(changed, notes, 8)
        np.testing.assert_allclose(base[2], scaled_base[2])
        np.testing.assert_allclose(original['mode_gain'], scaled['mode_gain'])

    def test_exported_preset_fortissimo_headroom(self):
        config = load_config(preset_path('banjo'))
        for pitches in (config['strings']['open_midi'], [67]*5):
            rows = [[0, 0, s, 0, pitch, 127, 0, 0, 0, 0, s] for s, pitch in enumerate(pitches)]
            y = numba_render(rows, 1.5, config=config)
            self.assertTrue(np.isfinite(y).all())
            self.assertLessEqual(float(np.max(np.abs(y))), config['engine']['peak_ceiling'])

    def test_resume_preserves_loudness_reference(self):
        notes = [dict(pitch=62, string=1, decoded_sha256='abc')]
        saved = [dict(pitch=62, string=1, decoded_sha256='abc', scale=12.3)]
        restore_scales(notes, saved)
        self.assertEqual(notes[0]['scale'], 12.3)
        notes[0]['decoded_sha256'] = 'changed'
        with self.assertRaises(ValueError):
            restore_scales(notes, saved)

    def test_metallic_loss_detects_quiet_upper_partial_and_tail(self):
        time = jnp.arange(24000)/48000
        fundamental = jnp.sin(2*jnp.pi*293*time)*jnp.exp(-3*time)
        upper = .012*jnp.sin(2*jnp.pi*293*20*time)*jnp.exp(-8*time)
        target = (fundamental+upper)[None,:]
        prepared = loss.metallic_prepare(target, [dict(frequency=293.)])
        exact = float(loss.metallic_term(target, prepared))
        missing = float(loss.metallic_term(fundamental[None,:], prepared))
        fast_tail = (fundamental+upper*jnp.exp(-40*time))[None,:]
        self.assertLess(exact, missing)
        self.assertLess(exact, float(loss.metallic_term(fast_tail, prepared)))
        gradient = jax.grad(lambda y: loss.metallic_term(y, prepared))(target*.9)
        self.assertTrue(np.isfinite(gradient).all())

    def test_spectral_loss_prefers_identical_signal(self):
        y = jnp.sin(jnp.arange(4800)[None, :]*.04)*.1
        notes = [dict(frequency=.04*48000/(2*np.pi))]
        prepared = loss.prepare(y, notes)
        exact = sum(float(x) for x in loss.terms(y, prepared))
        wrong = sum(float(x) for x in loss.terms(y*.5, prepared))
        self.assertLess(exact, wrong)


if __name__ == '__main__':
    unittest.main()
