#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["mido>=1.3,<2", "numpy>=1.26", "numba>=0.61", "scipy>=1.11", "cmudict>=1.0"]
# ///
"""Character and score-coordinate mark contracts; no audio renders."""
from pathlib import Path
import sys
import unittest

import mido
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from soundbench.scripts.models import voice3, phonetics
from soundbench.scripts.render_voice3 import settings_for, marks_for, apply_marks
from soundbench.scripts.perform import BarMap


def score():
    midi = mido.MidiFile(ticks_per_beat=480)
    midi.tracks.append(mido.MidiTrack([
        mido.MetaMessage('set_tempo', tempo=500000),
        mido.MetaMessage('set_tempo', tempo=1000000, time=1920),
        mido.MetaMessage('time_signature', numerator=3, denominator=4, time=1920)]))
    track = mido.MidiTrack([mido.MetaMessage('track_name', name='tenor')])
    for _ in range(11):
        track.extend([mido.Message('note_on', note=60, velocity=70),
                      mido.Message('note_off', note=60, time=480)])
    midi.tracks.append(track)
    return midi


class MarksTests(unittest.TestCase):
    def test_marks_use_score_tempo_and_meter_not_seconds_at_a_fixed_tempo(self):
        marks = marks_for([{'part': 'tenor', 'bars': [2, 3], 'beat': 2, 'character': 'intimate'},
                           {'part': 'tenor', 'bar': 3, 'vibrato': 'straight', 'attack_ms': 90}], score(), {1: 'tenor'})[1]
        self.assertEqual(marks[0]['start_tick'], 2400)
        self.assertEqual(marks[0]['end_tick'], 5280)
        self.assertEqual(marks[0]['start_seconds'], 3)
        self.assertEqual(marks[0]['end_seconds'], 9)
        rows = [dict(tick=tick, time=tick / 300, off=tick / 300 + 1) for tick in [0, 1920, 2400, 3840, 5279, 5280]]
        out = apply_marks(rows, {'character': {'base': 'plain', 'vibrato_rate': 6.2}}, marks)
        self.assertEqual([r['intent']['attack_ms'] for r in out], [70, 70, 150, 90, 90, 70])
        self.assertEqual(out[3]['intent']['vibrato'], 'straight')
        self.assertEqual(out[3]['intent']['vibrato_rate'], 5.5)
        self.assertEqual(out[-1]['intent']['vibrato_rate'], 6.2)

    def test_unknown_keys_names_bars_and_bad_values_raise(self):
        invalid = [
            {'part': 'tenor', 'bar': 0}, {'part': 'tenor', 'bar': True},
            {'part': 'tenor', 'bar': 1, 'bars': [1, 2]},
            {'part': 'tenor', 'bars': [3, 2]}, {'part': 'tenor', 'bar': 1, 'typo': 2},
            {'part': 'tenor', 'bar': 1, 'vibrato_rate': 9},
            {'part': 'tenor', 'bar': 1, 'vibrato_extent': float('nan')},
            {'part': 'tenor', 'bar': 1, 'vibrato_delay': True},
            {'part': 'tenor', 'bar': 1, 'character': 'unknown'},
            {'part': 'tenor', 'bar': 1, 'onset': 'random'},
            {'part': 'tenor', 'bar': 1, 'release': 'random'},
            {'part': 'tenor', 'bar': 1, 'beat': 5},
            {'part': 'tenor', 'bar': 1, 'beat': float('inf')},
            {'part': 'absent', 'bar': 1}, {'part': 'tenor', 'bar': 4},
        ]
        for mark in invalid:
            with self.subTest(mark=mark), self.assertRaises(ValueError):
                marks_for([mark], score(), {1: 'tenor'})

    def test_character_overlays_singer_and_explicit_part_scales(self):
        for singer in voice3.singers():
            c = voice3.load_config(singer, character={'base': 'intimate', 'vibrato_rate': 5.2}, vibrato_scale=.6)
            self.assertEqual(c['character']['vibrato_rate'], 5.2)
            self.assertEqual(c['singer']['vibrato_scale'], .6)
            self.assertEqual(c['singer']['length_cm'], voice3._table()['singers'][singer]['length_cm'])
        settings_for({'voice3': {'parts': {'tenor': {'character': 'young'}}, 'marks': []}})
        with self.assertRaises(ValueError):
            settings_for({'voice3': {'parts': {'tenor': {'character': {'base': 'plain', 'typo': 1}}}}})

    def test_straight_emerging_and_marked_note_controls(self):
        syl = phonetics.Syllable('a', [], ['AA1'], [])
        rows = [dict(time=.5 + i * 1.6, off=2 + i * 1.6, pitch=60, velocity=70, syllable=syl, tick=i*480)
                for i in range(4)]
        marks = [{'start_tick': 0, 'end_tick': 1920, 'character': None,
                  'patch': {'vibrato': 'emerging'}}]
        emerged = apply_marks(rows, {}, marks)
        c = voice3.load_config()
        k = voice3.controls(emerged, c, 8)
        def extent(k, i):
            select = (k['t'] > rows[i]['time'] + .6) & (k['t'] < rows[i]['off'] - .15)
            return np.ptp(k['vibrato'][select]) / 2
        self.assertEqual(extent(k, 0), 0)
        self.assertGreater(extent(k, 3), 20)
        self.assertGreater(extent(k, 2), extent(k, 1))
        c = voice3.load_config(character={'base': 'plain', 'vibrato': 'straight'})
        k = voice3.controls(rows, c, 8)
        np.testing.assert_array_equal(k['vibrato'], 0)
        self.assertEqual(voice3.character_settings()['vowel_opening_hz'], 0)

    def test_overlapping_marks_do_not_restart_emerging_vibrato(self):
        rows = [dict(time=.5 + i * 1.6, off=2 + i * 1.6, tick=i * 480) for i in range(8)]
        emerging = {'start_tick': 0, 'end_tick': 8 * 480, 'character': None,
                    'patch': {'vibrato': 'emerging'}}
        unrelated = {'start_tick': 3 * 480, 'end_tick': 4 * 480, 'character': None,
                     'patch': {'onset': 'hard', 'vibrato_rate': 6}}
        # A mode mark, a character establishing the mode, and the part default
        # each own their span independently of later field-only overrides.
        for part, marks in [({}, [emerging, unrelated]),
                            ({}, [{**emerging, 'character': {'base': 'plain', 'vibrato': 'emerging'},
                                   'patch': {}}, unrelated]),
                            ({'character': {'base': 'plain', 'vibrato': 'emerging'}}, [unrelated])]:
            with self.subTest(part=part, marks=marks):
                out = apply_marks(rows, part, marks)
                np.testing.assert_allclose([r['emerging_progress'] for r in out], np.linspace(0, 1, 8))
        # An explicit new emerging-mode mark does establish a new passage.
        restart = {**unrelated, 'end_tick': 8 * 480, 'patch': {'vibrato': 'emerging'}}
        out = apply_marks(rows, {}, [emerging, restart])
        np.testing.assert_allclose([r['emerging_progress'] for r in out[3:]], np.linspace(0, 1, 5))

    def test_every_character_keeps_the_intonation_limits(self):
        from test_voice3_intonation import deviations, HUMAN
        for character in voice3._table()['characters']:
            for singer in HUMAN:
                errors = np.concatenate([deviations(singer, seed, character=character) for seed in range(6)])
                self.assertLess(np.percentile(np.abs(errors), 95), 15)
                self.assertLess(np.percentile(np.abs(errors), 50), 6)

    def test_release_and_onset_marks_act_inside_a_phrase(self):
        syl = phonetics.Syllable('a', [], ['AA1'], [])
        rows = [dict(time=.5 + i*1.6, off=2.05 + i*1.6, pitch=60, velocity=70, syllable=syl, tick=i*480)
                for i in range(3)]
        m = [{'start_tick': 480, 'end_tick': 960, 'character': None,
              'patch': {'onset': 'breathy', 'release': 'breath'}}]
        marked = apply_marks(rows, {}, m)
        self.assertTrue(marked[1]['marked_onset'])
        self.assertTrue(marked[1]['marked_release'])
        k = voice3.controls(marked, voice3.load_config(), 6)
        tail = (k['t'] > 3.66) & (k['t'] < 3.69)
        self.assertGreater(k['tail'][tail].max(), 0)
        np.testing.assert_array_equal(k['envelope'][tail], 0)

    def test_phrase_breath_release_is_an_unvoiced_tail(self):
        syl = phonetics.Syllable('a', [], ['AA1'], [])
        rows = [dict(time=.5, off=2, pitch=60, velocity=70, syllable=syl)]
        for ms in (40, 200):
            c = voice3.load_config(character={'base': 'intimate', 'breath_tail_ms': ms})
            k = voice3.controls(rows, c, 3)
            tail = (k['tail'] > 0) & (k['t'] >= rows[0]['off'])
            self.assertGreater(tail.sum(), ms - 3)
            self.assertLessEqual(tail.sum(), ms + 1)
            np.testing.assert_array_equal(k['envelope'][tail], 0)


if __name__ == '__main__':
    unittest.main()
