#!/usr/bin/env python3
# /// script
# dependencies = ["mido>=1.3,<2"]
# ///
"""Tests for the structured expression layer."""
import sys
import unittest
from pathlib import Path

import mido

ROOT = Path(__file__).resolve().parent.parent / "soundbench"
sys.path.insert(0, str(ROOT / "scripts"))
from soundbench.scripts import expression  # noqa: E402
from soundbench.scripts import perform  # noqa: E402


def two_staff_midi(path):
    """Upper: a two-note chord then a melody note over a held inner note; lower: a bass note."""
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    midi.tracks.append(mido.MidiTrack([
        mido.MetaMessage("time_signature", numerator=3, denominator=4, time=0),
        mido.MetaMessage("set_tempo", tempo=1_000_000, time=0),
    ]))
    midi.tracks.append(mido.MidiTrack([
        mido.MetaMessage("track_name", name="upper:", time=0),
        mido.Message("note_on", note=72, velocity=64, time=0),
        mido.Message("note_on", note=64, velocity=64, time=0),
        mido.Message("note_off", note=72, velocity=0, time=480),
        mido.Message("note_on", note=74, velocity=64, time=0),
        mido.Message("note_off", note=74, velocity=0, time=480),
        mido.Message("note_off", note=64, velocity=0, time=0),
    ]))
    midi.tracks.append(mido.MidiTrack([
        mido.MetaMessage("track_name", name="lower:", time=0),
        mido.Message("note_on", note=43, velocity=64, time=0),
        mido.Message("note_on", note=55, velocity=64, time=0),
        mido.Message("note_off", note=43, velocity=0, time=960),
        mido.Message("note_off", note=55, velocity=0, time=0),
    ]))
    midi.save(path)
    return mido.MidiFile(path)


class ExpressionTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.midi = two_staff_midi(Path(self.tmp.name) / "s.mid")
        self.events = perform._event_positions(self.midi)
        self.pairs, self.onsets, _ = perform._pair_notes(self.events)
        self.bars = perform.BarMap(self.midi)
        tracks = perform._note_track_indices(self.events)
        names = perform._track_names(self.events)
        self.resolve = lambda sel: perform._resolve_track_index(sel, tracks, names)

    def tearDown(self):
        self.tmp.cleanup()

    def offsets(self, ex):
        return expression.expression_offsets(self.events, self.pairs, self.onsets,
                                             {"expression": ex}, self.bars, self.resolve)

    def test_voicing_by_role(self):
        vel, onset, _ = self.offsets({"melody": "upper", "melody_boost": 10, "inner_cut": 8,
                                      "accomp_cut": 6, "bass_boost": 2, "melody_lead_ms": 20,
                                      "high_loud": 0})
        by_note = {e.message.note: e for e in self.onsets}
        self.assertEqual(vel[id(by_note[72])], 10)   # top of the chord is the tune
        self.assertEqual(vel[id(by_note[74])], 10)   # the tune moves over a held inner note
        self.assertEqual(vel[id(by_note[64])], -8)   # inner voice of the tune's track
        self.assertEqual(vel[id(by_note[43])], 2)    # bass of the other track
        self.assertEqual(vel[id(by_note[55])], -6)   # accompaniment
        self.assertEqual(onset[id(by_note[72])], -20)
        self.assertEqual(onset[id(by_note[64])], 0)

    def test_arch_shape_ends_low_peaks_high(self):
        self.assertAlmostEqual(expression._shape(0.0, 0.6), -1.0)
        self.assertAlmostEqual(expression._shape(0.6, 0.6), 1.0)
        self.assertAlmostEqual(expression._shape(1.0, 0.6), -1.0)

    def test_asymmetric_arch_hits_its_three_points(self):
        triple = (-4.0, 8.0, -8.0)
        self.assertAlmostEqual(expression._arch_value(0.0, 0.6, triple), -4.0)
        self.assertAlmostEqual(expression._arch_value(0.6, 0.6, triple), 8.0)
        self.assertAlmostEqual(expression._arch_value(1.0, 0.6, triple), -8.0)

    def test_no_expression_changes_nothing(self):
        vel, onset, release = expression.expression_offsets(
            self.events, self.pairs, self.onsets, {}, self.bars, self.resolve)
        self.assertFalse(vel or onset or release)


if __name__ == "__main__":
    unittest.main()
