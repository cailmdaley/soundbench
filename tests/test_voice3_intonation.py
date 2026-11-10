#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["mido>=1.3,<2", "numpy>=1.26", "numba>=0.61", "scipy>=1.11", "cmudict>=1.0"]
# ///
"""voice3 sings in tune: each held note's pitch, with vibrato averaged out, stays near its target.

Intonation regression fixtures use Day's Eye's plain answer and Harbor Song,
sung with a per-note wander of SD 30 cents and a 22-cent starting offset, were
"so out of tune". Their singers fail
this test; the singers in voice3.toml must pass it.
"""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from soundbench.scripts.models import phonetics, voice3

HUMAN = [s for s in voice3.singers() if s != "machine"]
# Negative control: pitch wander large enough to sound out of tune.
OUT_OF_TUNE = {"drift_cents": [22.0, 30.0], "flutter_cents": 10.0}


def deviations(singer, seed, **overrides):
    """Per-note pitch error in cents after a 170 ms moving average (vibrato removed)."""
    c = voice3.load_config(singer, seed=seed, **overrides)
    lo, hi = c["singer"]["range"]
    pitches = [lo + 3, lo + 7, lo + 5, lo + 10, lo + 8, lo + 12, lo + 10, lo + 7]
    syl = phonetics.Syllable("ah", [], ["AA1"], [])
    rows = [{"time": 0.5 + 1.6 * i, "off": 0.5 + 1.6 * i + 1.5, "pitch": p, "velocity": 70, "syllable": syl}
            for i, p in enumerate(pitches)]
    k = voice3.controls(rows, c, rows[-1]["off"] + 1.0)
    cents = 1200 * np.log2(k["f0"] / 440) + 6900
    window = np.ones(170) / 170
    out = []
    for r in rows:
        i0, i1 = int((r["time"] + 0.35) / k["dt"]), int((r["off"] - 0.15) / k["dt"])
        error = cents[i0:i1] - 100 * r["pitch"]
        out.append(np.convolve(error, window, mode="valid"))
    return np.concatenate(out)


class IntonationTests(unittest.TestCase):
    def test_every_singer_holds_its_notes_in_tune(self):
        for singer in HUMAN:
            errors = np.concatenate([deviations(singer, seed) for seed in range(6)])
            with self.subTest(singer=singer):
                self.assertLess(np.percentile(np.abs(errors), 95), 15.0)
                self.assertLess(np.percentile(np.abs(errors), 50), 6.0)

    def test_large_pitch_wander_fails(self):
        errors = np.concatenate([deviations("tenor", seed, **OUT_OF_TUNE) for seed in range(6)])
        self.assertGreater(np.percentile(np.abs(errors), 95), 15.0)


if __name__ == "__main__":
    unittest.main()
