#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy", "numba", "mido"]
# ///
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import mido
import numpy as np
from soundbench.scripts.models.fiddle import load_config, render, assign_string, controls
from soundbench.scripts.fiddle_check import estimate_pitch
from soundbench.scripts.render_fiddle import schedule


class FiddleTest(unittest.TestCase):
    def setUp(self):
        self.c = load_config()
        self.row = dict(time=0, off=.8, pitch=69, velocity=90)

    def test_determinism(self):
        row = dict(self.row, pitch=71, vibrato=12)
        np.testing.assert_array_equal(render([row], 1.5, config=self.c), render([row], 1.5, config=self.c))
        c = copy.deepcopy(self.c)
        c['engine']['seed'] += 1
        self.assertFalse(np.array_equal(render([row], 1.5, config=c), render([row], 1.5, config=self.c)))

    def test_tuning(self):
        for pitch in (55, 57, 62, 64, 69, 71, 76, 81, 88, 96):
            with self.subTest(pitch=pitch):
                y = render([dict(self.row, pitch=pitch, vibrato=0)], 1.2, config=self.c)[0, 12000:35000, 0]
                target = 440*2**((pitch-69)/12)
                cents = 1200*np.log2(estimate_pitch(y, target)/target)
                self.assertLess(abs(cents), 8)
                self.assertGreater(np.sqrt(np.mean(y*y)), 1e-4)

    def test_release(self):
        y = render([self.row], 2.8, config=self.c)[0, :, 0]
        held = np.sqrt(np.mean(y[18000:30000]**2))
        tail = np.sqrt(np.mean(y[-12000:]**2))
        self.assertGreater(held, .001)
        self.assertLess(tail, held*.005)
        self.assertGreater(np.max(abs(y[38400:40000])), 1e-6)

    def test_velocity_ramp(self):
        levels = []
        for velocity in [*range(20, 128, 9), 127]:
            y = render([dict(self.row, velocity=velocity)], 1.2, config=self.c)
            self.assertTrue(np.all(np.isfinite(y)))
            self.assertLess(abs(y).max(), .98)
            levels.append(np.sqrt(np.mean(y[:, 12000:30000]**2)))
        jumps = abs(20*np.log10(np.array(levels[1:])/levels[:-1]))
        self.assertLess(max(jumps), 7)

    def test_silence_and_owner_sum(self):
        self.assertEqual(abs(render([], .2, config=self.c)).max(), 0)
        rows = [dict(self.row, owner=0), dict(self.row, pitch=76, owner=1)]
        stereo = render(rows, 1.5, 2, self.c)
        mono = render([dict(r, owner=0) for r in rows], 1.5, 1, self.c)
        np.testing.assert_allclose(stereo.sum(axis=0), mono[0], atol=1e-12)

    def test_open_string_bow_lift_is_not_finger_damping(self):
        base = dict(time=0, off=.4, pitch=69, velocity=90, string=2)
        opened = render([base], 1.4, config=self.c)[0, 26400:33600, 0]
        stopped = render([dict(base, pitch=71)], 1.4, config=self.c)[0, 26400:33600, 0]
        self.assertGreater(np.sqrt(np.mean(opened**2)), 5*np.sqrt(np.mean(stopped**2)))
        np.testing.assert_array_equal(render([dict(base, vibrato=12)], 1.4, config=self.c), render([base], 1.4, config=self.c))

    def test_sympathetic(self):
        _, d = render([self.row], 1.2, config=self.c, diagnostics=True)
        self.assertGreater(np.sqrt(np.mean(d['bridge'][0, 10000:30000]**2)), 1e-7)
        c = copy.deepcopy(self.c)
        c['strings']['coupling'] = 0
        _, uncoupled = render([self.row], 1.2, config=c, diagnostics=True)
        self.assertEqual(abs(uncoupled['bridge'][0]).max(), 0)

    def test_gestures(self):
        rows = [dict(time=0, off=.3, pitch=71, velocity=100, string=2, slide=-1),
                dict(time=.3, off=.6, pitch=73, velocity=95, string=2, slur=True, vibrato=8),
                dict(time=.7, off=.85, pitch=69, velocity=90, chop=True)]
        y, d = render(rows, 1.5, config=self.c, diagnostics=True)
        self.assertTrue(d['assignment'][1]['slur'])
        self.assertTrue(np.all(np.isfinite(y)))

    def test_regions(self):
        ratios = []
        for force in (.004, .12, .8):
            _, d = render([dict(self.row, force=force)], 1.2, config=self.c, diagnostics=True)
            ratios.append(np.mean(d['stick'][1, 12000:30000]))
        self.assertGreater(max(ratios)-min(ratios), .1)

    def test_validation(self):
        for update in (dict(off=-1), dict(pitch=30), dict(velocity=128), dict(force=np.nan), dict(position=.7), dict(slide=-4), dict(string=0)):
            with self.subTest(update=update), self.assertRaises(ValueError):
                render([dict(self.row, **update)], 1.2, config=self.c)
        with self.assertRaises(ValueError):
            render([dict(self.row, string=2), dict(self.row, string=2)], 1.2, config=self.c)
        with self.assertRaises(ValueError):
            load_config(tuning='anything')

    def test_bow_change_configuration(self):
        c = copy.deepcopy(self.c)
        c['bow']['change_seconds'] = .025
        self.assertFalse(np.array_equal(render([self.row], 1.2, config=c, player=False), render([self.row], 1.2, config=self.c, player=False)))
        _, d = render([dict(self.row, slur=True)], 1.2, config=self.c, diagnostics=True)
        self.assertFalse(d['assignment'][0]['slur'])

    def test_extreme_double_stop(self):
        rows = [dict(self.row, pitch=69, string=2, force=.8, speed=.4, velocity=127, grit=1),
                dict(self.row, pitch=76, string=1, force=.8, speed=.4, velocity=127, grit=1)]
        y = render(rows, 2, config=self.c)
        self.assertTrue(np.all(np.isfinite(y)))
        self.assertLess(abs(y).max(), .98)

    def test_cross_tuning(self):
        c = load_config(tuning='AEAE')
        self.assertEqual(c['strings']['open_midi'], [76, 69, 64, 57])
        self.assertEqual(assign_string(57, c), 3)
        self.assertEqual(load_config(tuning='GDGD')['strings']['open_midi'], [74, 67, 62, 55])
        adae = load_config(tuning='ADAE')
        self.assertEqual(adae['strings']['open_midi'], [76, 69, 62, 57])
        self.assertEqual(assign_string(62, adae), 2)

    def test_midi_controls_and_release(self):
        midi = mido.MidiFile()
        track = mido.MidiTrack()
        midi.tracks.append(track)
        track.append(mido.MetaMessage('track_name', name='fiddle'))
        track.extend([mido.Message('control_change', control=20, value=2),
                      mido.Message('note_on', note=71, velocity=90),
                      mido.Message('control_change', control=2, value=16, time=240),
                      mido.Message('note_off', note=71, time=240)])
        rows, seconds = schedule(midi, {}, [0], self.c)
        self.assertEqual(rows[0]['string'], 2)
        self.assertEqual(rows[0]['controls'][0][1], 'force')
        self.assertAlmostEqual(rows[0]['off'], .5)
        track.pop()
        with self.assertRaises(ValueError):
            schedule(midi, {}, [0], self.c)

    def test_performed_legato_overlap(self):
        # The room moves an onset a few ms early; the voice still has one finger down.
        midi = mido.MidiFile()
        track = mido.MidiTrack([mido.MetaMessage('track_name', name='fiddle'),
                                mido.Message('control_change', control=20, value=1),
                                mido.Message('note_on', note=76, velocity=90),
                                mido.Message('note_on', note=78, velocity=90, time=476),
                                mido.Message('note_off', note=76, time=4),
                                mido.Message('note_off', note=78, time=480)])
        midi.tracks.append(track)
        rows, seconds = schedule(midi, {}, [0], self.c)
        self.assertAlmostEqual(rows[0]['off'], rows[1]['time'])
        self.assertTrue(np.all(np.isfinite(render(rows, seconds, config=self.c))))
        # Across tracks: a drone pinned to string 1 hands over to the melody.
        drone = mido.MidiTrack([mido.MetaMessage('track_name', name='fiddledrone'),
                                mido.Message('control_change', control=20, value=1),
                                mido.Message('note_on', note=76, velocity=60, time=960),
                                mido.Message('note_off', note=76, time=484)])
        midi.tracks.append(drone)
        track[-1].time = 960
        rows, seconds = schedule(midi, {}, [0, 1], self.c)
        render(rows, seconds, owners=2, config=self.c)
        pinned = sorted((r for r in rows if r['string'] == 1), key=lambda r: r['time'])
        self.assertTrue(all(a['off'] <= b['time']+1e-9 for a, b in zip(pinned, pinned[1:])))

    def test_sidecar_binding(self):
        # A piece's sidecar can map a track to the fiddle model by name.
        from soundbench.scripts.instrumentation import resolve_instruments
        midi = mido.MidiFile()
        track = mido.MidiTrack([mido.MetaMessage('track_name', name='fiddle'),
                                mido.Message('note_on', note=69, velocity=90), mido.Message('note_off', note=69, time=480)])
        midi.tracks.append(track)
        binding = resolve_instruments(midi, {'instruments': {'fiddle': {'model': 'fiddle', 'gain_db': 6}}})[0]
        self.assertEqual(binding.instrument, 'fiddle')
        self.assertEqual(binding.gain_db, 6)

    def test_manifest_dispatch(self):
        from soundbench.scripts.model_dispatch import render_model_stems
        from soundbench.scripts.instrumentation import MODELS
        from types import SimpleNamespace
        self.assertIn('fiddle', MODELS)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            midi = mido.MidiFile()
            track = mido.MidiTrack()
            midi.tracks.append(track)
            track.append(mido.MetaMessage('track_name', name='fiddle'))
            track.extend([mido.Message('note_on', note=69, velocity=90), mido.Message('note_off', note=69, time=480)])
            midi.save(root/'note.mid')
            result = render_model_stems(root/'note.mid', root/'note.mid', {}, {0: SimpleNamespace(model='fiddle')}, root, 'probe')
            self.assertEqual(result['0']['model'], 'fiddle')
            # One stem frame per rendered sample: the stem plays at pitch and in time.
            import wave
            metrics = json.loads((root/'probe.fiddle.metrics.json').read_text())
            with wave.open(str(root/'probe.stem.01-fiddle.raw.wav')) as stem:
                self.assertEqual(stem.getnchannels(), 2)
                self.assertEqual(stem.getnframes(), round(metrics['audio_seconds']*48000))

    def test_blocks_do_not_change_the_sound(self):
        # Rendering in blocks carries every state across: noise, slew, kernel,
        # decimator and body. Short blocks split notes, slurs and crossings.
        rows = [dict(time=0, off=.5, pitch=69, velocity=90, string=2),
                dict(time=.5, off=.9, pitch=71, velocity=80, string=2, slur=True),
                dict(time=.9, off=1.3, pitch=76, velocity=85, string=1, slur=True),
                dict(time=.2, off=1.4, pitch=62, velocity=60, string=3, controls=[(.4, 'force', .2)])]
        whole, d = render(rows, 2.5, config=self.c, diagnostics=True)
        split, e = render(rows, 2.5, config=self.c, diagnostics=True, block_seconds=.13)
        np.testing.assert_allclose(split, whole, atol=1e-13)
        np.testing.assert_array_equal(d['stick'], e['stick'])
        self.assertEqual([x['factor'] for x in d['player_decisions']], [x['factor'] for x in e['player_decisions']])
        mono = render(rows, 2.5, config=self.c, channels=1)
        np.testing.assert_array_equal(mono, whole[:, :, 0])
        # Block sizes in samples, down to one, and below the decimator FIR and
        # body tail; two owners, with string 2 changing hands at a block edge.
        short = [dict(time=0, off=.0525, pitch=69, velocity=90, string=2, owner=0),
                 dict(time=.0525, off=.12, pitch=71, velocity=80, string=2, owner=1, slur=True),
                 dict(time=.03, off=.12, pitch=76, velocity=70, string=1, owner=1)]
        reference = render(short, .2, 2, self.c)
        for samples in (1, 7, 19, 20, 21, 97, 1000, 2520):
            with self.subTest(block=samples):
                np.testing.assert_allclose(render(short, .2, 2, self.c, block_seconds=samples/48000), reference, atol=1e-13)

    def test_held_note_previews_its_start(self):
        # A held note's force is chosen over its first two seconds and kept.
        row = dict(time=0, off=5, pitch=69, velocity=90, string=2)
        _, d = render([row], 5.5, config=self.c, diagnostics=True, block_seconds=.5)
        self.assertEqual(len(d['player_decisions']), 1)
        self.assertAlmostEqual(d['player_decisions'][0]['off'], 5)


if __name__ == '__main__':
    unittest.main()
