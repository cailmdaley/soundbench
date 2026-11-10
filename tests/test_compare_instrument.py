#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "librosa>=0.10.2,<0.12",
#   "matplotlib>=3.8",
#   "mido>=1.3,<2",
#   "numpy>=1.26",
#   "pyloudnorm>=0.1.1",
#   "scipy>=1.11",
# ]
# ///
from __future__ import annotations

import math
import re
import sys
import tempfile
import unittest
from pathlib import Path

import mido
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "soundbench"
sys.path.insert(0, str(ROOT / "scripts"))
from soundbench.scripts import compare_instrument as compare


class CompareInstrumentTests(unittest.TestCase):
    def test_one_third_octave_average_finds_tonal_band(self):
        sr = compare.SAMPLE_RATE
        time = np.arange(sr * 2) / sr
        audio = np.sin(2 * math.pi * 1000 * time)
        centers, levels = compare.third_octave_spectrum(audio, sr, 100, 5000)
        strongest = centers[int(np.argmax(levels))]
        self.assertAlmostEqual(strongest, 1000, delta=80)

    def test_decay_times_require_observed_isolated_decay(self):
        sr = compare.SAMPLE_RATE
        time = np.arange(sr * 3) / sr
        audio = np.sin(2 * math.pi * 440 * time) * np.exp(-5 * time)
        envelope_times, envelope = compare.rms_envelope(audio, sr)
        centroid_times, centroids = compare.spectral_centroid(audio, sr, 100, 5000)
        event = {"start": 0.0, "note_off": 0.1, "end": 3.0, "pitches": [69]}
        features = compare.event_measurements(
            audio, sr, [event], envelope_times, envelope, centroid_times, centroids,
        )
        self.assertAlmostEqual(features["t20_s"], 20 / (20 / math.log(10) * 5), delta=0.08)
        self.assertAlmostEqual(features["t30_s"], 30 / (20 / math.log(10) * 5), delta=0.1)
        self.assertGreater(features["t30_s"], features["t20_s"])
        self.assertEqual(features["t20_notes"], 1)

    def test_midi_note_events_use_rendered_timing(self):
        midi = mido.MidiFile(type=1, ticks_per_beat=480)
        conductor = mido.MidiTrack([mido.MetaMessage("set_tempo", tempo=500_000)])
        notes = mido.MidiTrack([
            mido.Message("note_on", note=55, velocity=80, time=0),
            mido.Message("note_off", note=55, time=240),
            mido.Message("note_on", note=59, velocity=80, time=240),
            mido.Message("note_off", note=59, time=240),
            mido.MetaMessage("end_of_track", time=960),
        ])
        midi.tracks.extend((conductor, notes))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "probe.mid"
            midi.save(path)
            events = compare.midi_note_events(path, 0.0, 2.0)
        self.assertEqual(len(events), 2)
        self.assertAlmostEqual(events[0]["start"], 0.0)
        self.assertAlmostEqual(events[1]["start"], 0.5)
        self.assertEqual(events[0]["pitches"], [55])


if __name__ == "__main__":
    unittest.main()
