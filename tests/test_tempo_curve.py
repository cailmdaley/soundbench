#!/usr/bin/env python3
# /// script
# dependencies = ["mido>=1.3,<2"]
# ///
"""Tempo diagnostics follow the score's meter and tempo boundaries."""
import sys
import unittest
from pathlib import Path
import mido
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from soundbench.scripts import tempo_curve


def fixture(scale=1):
    midi = mido.MidiFile(ticks_per_beat=480)
    midi.tracks.append(mido.MidiTrack([
        mido.MetaMessage("set_tempo", tempo=500000 * scale),
        mido.MetaMessage("time_signature", numerator=4, denominator=4),
        mido.MetaMessage("set_tempo", tempo=1000000 * scale, time=1920),
        mido.MetaMessage("time_signature", numerator=3, denominator=4),
    ]))
    midi.tracks.append(mido.MidiTrack([
        mido.Message("note_on", note=60, velocity=80),
        mido.Message("note_off", note=60, time=1920),
        mido.Message("note_on", note=62, velocity=80),
        mido.Message("note_off", note=62, time=1440),
        mido.Message("note_on", note=64, velocity=80),
        mido.Message("note_off", note=64, time=1440),
    ]))
    return midi


class TempoTests(unittest.TestCase):
    def test_mixed_meter_and_tempo_have_three_correct_bars(self):
        self.assertEqual(tempo_curve.ratios(fixture(), fixture()), [1, 1, 1])
        self.assertEqual(tempo_curve.ratios(fixture(), fixture(2)), [2, 2, 2])

    def test_no_notes_produces_no_ratios(self):
        midi = mido.MidiFile()
        midi.tracks.append(mido.MidiTrack())
        self.assertEqual(tempo_curve.ratios(midi, midi), [])


if __name__ == "__main__":
    unittest.main()
