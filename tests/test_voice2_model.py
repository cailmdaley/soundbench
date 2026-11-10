#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["mido>=1.3,<2", "numpy>=1.26", "numba>=0.61", "scipy>=1.11"]
# ///
"""Deterministic source, tract compatibility, performer and sidecar contracts."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import tempfile
import mido
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from soundbench.scripts.models import voice, voice2
from soundbench.scripts.render_voice2 import selected_config, schedule
from soundbench.scripts.voice_phrases import phrases, row
from scipy.signal import welch


class Voice2Tests(unittest.TestCase):
    def test_rd(self):
        np.testing.assert_allclose(voice2.rd_timing(1), [.4843636363636364, .650016, .038], atol=.0001)
        for rd in (.3, .5, 1, 1.5, 2.7):
            tp, te, ta = voice2.rd_timing(rd)
            self.assertTrue(0 < tp < te < 1 and 0 < ta < 1 - te)
        table, flow = voice2.rd_tables()
        self.assertTrue(np.isfinite(table).all() and np.isfinite(flow).all())

    def test_v1_bit_identity(self):
        expected = {'sustain-hm-soft-D4': 'b039e8755e54753a89d6f25056ecd13255f8ae33c41a8e711928ace376c8377f',
                    'sustain-ah-loud-D4': '5a4fe2de97c6a13088da6c82ae794a7f18c96c85e3f757abd6ee15c062251ff3'}
        for name, digest in expected.items():
            self.assertEqual(hashlib.sha256(voice.render(phrases()[name], 5.7).tobytes()).hexdigest(), digest)

    def test_empty_render(self):
        self.assertEqual(voice2.render([]).shape, (0,))
        y, k = voice2.render([], .2, return_controls=True)
        self.assertEqual(y.shape, (9600,))
        self.assertTrue(np.all(y == 0))
        self.assertIsNone(k)

    def test_long_rests_render_span_by_span(self):
        c = voice2.load_config()
        a = [dict(time=.5, off=1.5, pitch=62, velocity=70, syllable='hm')]
        b = [dict(time=6.5, off=7.5, pitch=66, velocity=70, syllable='hm')]
        y, k = voice2.render(a + b, 9, c, return_controls=True)
        spans = voice2.sounding_spans(k, len(y), 48)
        self.assertEqual(len(spans), 2)
        self.assertTrue(np.all(y[spans[0][1]:spans[1][0]] == 0))
        # A span starts from rest: it carries the same energy as one synthesis of the whole timeline.
        q, sr = c['voice2'], 48000
        rng = np.random.default_rng(np.random.SeedSequence([q['seed'], 2]))
        jitter, shimmer = rng.normal(0, q['jitter'], 32768), rng.normal(0, q['shimmer'], 32768)
        params = voice.tract_parameters(c)
        params.update(sinus_hz=q['sinus_hz'], sinus_area=q['sinus_area'], sinus_q=q['sinus_q'], nasal_wall_hz=q['nasal_wall_hz'])
        filters = [voice2.butter(4, 10000, fs=sr, output='sos'), voice2.butter(1, 30, 'highpass', fs=sr, output='sos')]
        whole = voice2._synthesize(k, q, 0, len(y), sr, 48, rng, jitter, shimmer, *voice2.rd_tables(), params, filters)
        whole *= c['engine']['output_gain']
        for x0, x1 in spans:
            ratio = np.sqrt(np.mean(y[x0:x1] ** 2) / np.mean(whole[x0:x1] ** 2))
            self.assertAlmostEqual(20 * np.log10(ratio), 0, delta=.2)
        self.assertGreater(np.max(np.abs(y[spans[1][0]:spans[1][1]])), 1e-3)  # the second span sounds

    def test_determinism(self):
        for preset in ('tenor', 'alto'):
            c = voice2.load_config(preset)
            rows = phrases()['breaths-hm']
            a = voice2.render(rows, 10.5, c)
            self.assertTrue(np.array_equal(a, voice2.render(rows, 10.5, c)))
            c['voice2']['seed'] += 1
            self.assertFalse(np.array_equal(a, voice2.render(rows, 10.5, c)))

    def test_stability_grid(self):
        for preset in ('tenor', 'alto'):
            c = voice2.load_config(preset)
            for syl in ('hm', 'ah', 'ee', 'oo'):
                for pitch in range(59, 84):
                    for vel in (30, 80, 127):
                        y = voice2.render([row(.02, .16, pitch, vel, syl)], .3, c)
                        self.assertTrue(np.isfinite(y).all(), (preset, syl, pitch, vel))
                        self.assertLess(np.max(np.abs(y)), 1, (preset, syl, pitch, vel))

    def test_controls(self):
        c = voice2.load_config()
        rows = [row(.2, 1, 62, 80, 'hm'), row(1.2, 1, 67, 80, None), row(2.2, .14, 67, 80, None), row(3, 2, 62, 80, 'hm')]
        k = voice2.controls(rows, c, 5.5)
        mask = (k['t'] > 2.2) & (k['t'] < 2.34)
        self.assertLess(np.max(np.abs(k['vibrato'][mask])), 15)
        mask = (k['t'] > 1.2) & (k['t'] < 1.45)
        overshoot = 100 * (k['pitch'][mask].max() - 67)
        self.assertTrue(10 <= overshoot <= 40, overshoot)
        self.assertTrue(all(2 <= d['depth_db'] <= 8 for d in k['dips']))
        self.assertEqual(len(k['breaths']), 1)
        self.assertTrue(np.any(k['inhale'] > 0))
        self.assertGreater(len(set(d for d in k['delays'] if d is not None)), 1)
        self.assertTrue(3 <= np.std(k['flutter']) <= 10)
        breaths = voice2.controls(phrases()['breaths-hm'], c, 10.5)
        inhale = breaths['inhale'] > 0
        self.assertTrue(np.all(breaths['amplitude'][inhale] == 0), 'inhalation must not open a still-voiced release')

    def test_ff_swell_headroom(self):
        for preset in ('tenor', 'alto'):
            c = voice2.load_config(preset)
            for pitch in (69, 83):
                rows = [row(.2, 4, pitch, 127, 'ah')]
                for arts in ((), ({'note': 0, 'art': {'kind': 'cry', 'vowel': 'ah'}},)):
                    y = voice2.render(rows, 4.7, c, articulations=arts)
                    self.assertLess(np.max(np.abs(y)), 1, (preset, pitch, arts))

    def test_effort_spectrum(self):
        for preset in ('tenor', 'alto'):
            c = voice2.load_config(preset)
            ratios = []
            for velocity in (40, 100):
                y = voice2.render([row(.2, 4, 62, velocity, 'ah')], 4.7, c)
                f, p = welch(y[24000:192000], 48000, nperseg=1920, noverlap=1440)
                ratio = 10 * np.log10(p[(f >= 1000) & (f < 5000)].sum() / p[(f >= 50) & (f < 1000)].sum())
                ratios.append(ratio)
            self.assertGreaterEqual(ratios[1] - ratios[0], 5)
            self.assertTrue(-14 <= ratios[1] <= -3, (preset, ratios))

    def test_cc11_and_cry_context(self):
        c = voice2.load_config()
        rows = [row(.2, 1, 74, 15, 'ah'), row(1.2, 1, 72, 60, 'hm')]
        k = voice2.controls(rows, c, 2.7)
        mask = (k['t'] > .3) & (k['t'] < 1.0)
        self.assertGreaterEqual(k['effort'][mask].min(), .85)
        self.assertLess(k['velum'][mask].max(), .01)
        rows[1]['syllable'] = 'HM,'
        p = voice2.controls(rows, c, 2.7)
        self.assertGreaterEqual(p['effort'][mask].min(), .85)
        self.assertEqual(voice2.syllable_key('NG!'), 'ng')
        # A separate vowel phrase is not a cry merely because a hum occurs later.
        rows[1]['time'], rows[1]['off'] = 2, 3
        k = voice2.controls(rows, c, 3.5)
        mask = (k['t'] > .3) & (k['t'] < 1.0)
        self.assertLess(k['effort'][mask].max(), .5)
        control = [dict(time=0, control=11, value=100), dict(time=1, control=11, value=0)]
        k = voice2.controls([row(.2, 2, 62, 80, 'ah')], c, 2.7, control)
        self.assertTrue(np.all(k['amplitude'][k['t'] > 1.1] == 0))

    def test_config_validation(self):
        for opt in ({'oops': 1}, {'preset': 'bass'}, {'seed': -1}, {'vibrato': float('nan')},
                    {'tension': 2}, {'release_s': 2, 'tail_seconds': 1}, {'seed': True}):
            with self.assertRaises(ValueError):
                selected_config({'voice2_model': opt})
        self.assertNotEqual(selected_config({})[2], selected_config({'voice2_model': {'preset': 'alto'}})[2])

    def test_legato_bridges_short_midi_gaps(self):
        rows = [row(.2, 1, 62, 80, 'hm'), row(1.25, 1, 65, 80, None)]
        k = voice2.controls(rows, voice2.load_config(), 2.7)
        at = np.searchsorted(k['t'], 1.22)
        self.assertGreater(k['amplitude'][at], 0)
        self.assertEqual(k['hum'][at], 1)
        self.assertEqual(k['velum'][at], 1)

    def test_cry_override_does_not_replace_lyric(self):
        rows = [row(.2, 1, 62, 80, 'hm'), row(1.2, 1, 65, 80, None)]
        arts = [{'note': 0, 'art': {'kind': 'cry', 'vowel': 'ah'}}]
        k = voice2.controls(rows, voice2.load_config(), 2.7, articulations=arts)
        at = np.searchsorted(k['t'], 1.5)
        self.assertEqual(k['hum'][at], 1)
        self.assertEqual(k['velum'][at], 1)

    def test_diphthong_spans_melisma(self):
        c = voice2.load_config()
        rows = [row(.2, 1, 62, 80, 'ai'), row(1.2, 1, 65, 80, None)]
        k = voice2.controls(rows, c, 2.7)
        at = np.searchsorted(k['t'], .95)
        np.testing.assert_allclose(k['areas'][at], voice.vowel_shapes(c)['aa'])

    def test_slide_destination_continuity(self):
        rows = [row(.2, 1, 62, 80, 'hm'), row(1.2, 1, 65, 80, None)]
        arts = [{'note': 0, 'art': {'kind': 'slide', 'to': 2, 'duration_s': .2}}]
        k = voice2.controls(rows, voice2.load_config(), 2.7, articulations=arts)
        at = np.searchsorted(k['t'], 1.2)
        self.assertAlmostEqual(k['pitch'][at], 64., places=6)

    def test_articulations(self):
        c = voice2.load_config()
        rows = [row(.5, 1, 62, 80, 'hm')]
        for kind in voice2.ART_KEYS:
            a = {'kind': kind}
            if kind == 'slide': a.update(to=2, duration_s=.2)
            if kind == 'onset': a.update(style='scoop', depth=8)
            y = voice2.render(rows, 2, c, articulations=[{'note': 0, 'art': a}])
            self.assertTrue(np.isfinite(y).all())
        for a in ({'kind': 'oops'}, {'kind': 'cry', 'oops': 1}, {'kind': 'slide'}, {'kind': 'onset', 'style': 'oops'}):
            with self.assertRaises(ValueError): voice2.validate_articulation(a)

    def test_window_carries_its_own_syllable(self):
        from soundbench.scripts.voice_phrases import piece_rows
        build = Path(__file__).resolve().parents[1] / 'build'
        with tempfile.TemporaryDirectory(dir=build) as tmp:
            p = Path(tmp)
            midi = mido.MidiFile()
            tr = mido.MidiTrack([mido.MetaMessage('track_name', name='voice')])
            for pitch in (62, 65, 67):
                tr.extend([mido.Message('note_on', note=pitch, velocity=80), mido.Message('note_off', note=pitch, time=480)])
            midi.tracks.append(tr)
            midi.tracks.append(mido.MidiTrack([mido.MetaMessage('lyrics', text='ng'), mido.MetaMessage('lyrics', text='ah', time=960)]))
            midi.save(p / 'score.mid')
            manifest = p / 'voice.json'
            manifest.write_text(json.dumps(dict(performed_midi=str(p / 'score.mid'), score_midi=str(p / 'score.mid'), jobs=[dict(index=0)])))
            rows, _, _ = piece_rows(manifest, .5, .9, 'voice2')
            self.assertEqual(rows[0]['syllable'], 'ng')
            self.assertEqual(rows[0]['pitch'], 65)

    def test_manifest_and_registry(self):
        from soundbench.scripts import render_voice2
        from soundbench.scripts.instrumentation import resolve_instruments
        from soundbench.scripts.model_dispatch import RENDERERS
        build = Path(__file__).resolve().parents[1] / 'build'
        with tempfile.TemporaryDirectory(dir=build) as tmp:
            p = Path(tmp)
            midi = mido.MidiFile()
            midi.tracks.append(mido.MidiTrack([mido.MetaMessage('track_name', name='voice'), mido.Message('note_on', note=62, velocity=80), mido.Message('note_off', note=62, time=480)]))
            midi.tracks.append(mido.MidiTrack([mido.MetaMessage('lyrics', text='ng')]))
            midi.save(p / 'score.mid')
            settings = {'instruments': {'voice': {'model': 'voice2'}}, 'voice2_model': {'preset': 'alto'},
                        'articulation': [dict(track='voice', bar=1, kind='quiver', depth=.6)]}
            self.assertEqual(resolve_instruments(midi, settings)[0].model, 'voice2')
            self.assertEqual(RENDERERS['voice2'][1], 'voice2')
            manifest = p / 'test.voice2.json'
            manifest.write_text(json.dumps(dict(performed_midi=str(p / 'score.mid'), score_midi=str(p / 'score.mid'), settings=settings, jobs=[dict(index=0, output=str(p / 'stem.wav'))])))
            with patch.object(sys, 'argv', ['render_voice2', str(manifest)]):
                render_voice2.main()
            self.assertGreater((p / 'stem.wav').stat().st_size, 100)
            metrics = json.loads(manifest.with_suffix('.metrics.json').read_text())
            self.assertEqual(metrics['stems']['0']['model'], 'voice2')
            self.assertEqual(metrics['stems']['0']['overrides']['preset'], 'alto')

    def test_articulation_ordinals_follow_midi_order(self):
        from soundbench.scripts.instrumentation import Binding
        from soundbench.scripts.model_dispatch import model_articulations
        # Sidecar order binds track 2 before track 1; ordinals still count MIDI tracks.
        bindings = {2: Binding('voice2', 0, 0, 'voice2'), 1: Binding('voice', 0, 0, 'voice')}
        names = {1: 'lead', 2: 'harmony'}
        arts = [dict(track=1, bar=1, kind='quiver'), dict(track=2, bar=1, kind='quiver'),
                dict(track='harmony', bar=2, kind='quiver')]
        self.assertEqual(model_articulations({'articulation': arts}, bindings, names, 'voice2'), arts[1:])
        self.assertEqual(model_articulations({'articulation': arts}, bindings, names, 'voice'), arts[:1])

    def test_score_addressing(self):
        score = mido.MidiFile()
        score.tracks.append(mido.MidiTrack([mido.MetaMessage('track_name', name='voice'), mido.Message('note_on', note=62, velocity=80), mido.Message('note_off', note=62, time=480)]))
        rows = {0: [dict(row(.2, 1, 62, 80, 'hm'), score_tick=0)]}
        settings = {'articulation': [dict(track='voice', bar=1, beat=1, kind='slide', to=2, duration_beats=.5)]}
        out = schedule(score, score, settings, [0], rows)
        self.assertAlmostEqual(out[0][0]['art']['duration_s'], .25)
        for change in ({'kind': 'unknown'}, {'oops': 1}, {'bar': 2}, {'duration_beats': -1}):
            s = copy.deepcopy(settings)
            s['articulation'][0].update(change)
            with self.assertRaises(ValueError): schedule(score, score, s, [0], rows)

    def test_cry_sits_lift_above_its_hum(self):
        # An open-vowel cry in a hummed phrase is radiation-balanced to cry_lift_db
        # above the hum on either side, at any pitch, whatever its own velocity.
        for preset in ('tenor', 'alto'):
            c = voice2.load_config(preset)
            for pitch in (62, 74, 83):
                notes = [row(0, 2, pitch, 60, 'hm'), row(2, 2, pitch, 20, 'ah'), row(4, 2, pitch, 60, 'hm')]
                out = voice2.render(notes, 6.5, c)
                db = lambda a, b: voice2._a_weighted_db(out[int(a * 48000):int(b * 48000)])
                lift = db(2.5, 3.7) - 0.5 * (db(0.5, 1.7) + db(4.5, 5.7))
                self.assertLess(abs(lift - c['voice2']['cry_lift_db']), 2.5, (preset, pitch, lift))


if __name__ == '__main__':
    unittest.main()
