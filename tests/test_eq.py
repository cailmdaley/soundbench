#!/usr/bin/env python3
# /// script
# dependencies = ["mido>=1.3,<2"]
# ///
"""An instrument's `eq` in the sidecar parses strictly and becomes one ffmpeg chain."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from soundbench.scripts.instrumentation import eq_filter, parse_eq  # noqa: E402


class Eq(unittest.TestCase):
    def test_bands_become_one_chain_in_order(self):
        eq = parse_eq("guitar", [{"type": "highshelf", "hz": 1500, "db": -10},
                                 {"type": "peak", "hz": 3000, "db": -4},
                                 {"type": "lowpass", "hz": 6000}])
        self.assertEqual(eq_filter(eq), "highshelf=f=1500:g=-10,equalizer=f=3000:t=o:w=1:g=-4,"
                                        "lowpass=f=6000:p=2")

    def test_no_eq_is_no_filter(self):
        self.assertEqual(eq_filter(parse_eq("guitar", [])), "")

    def test_bad_bands_are_refused(self):
        for band in ({"type": "shelf", "hz": 1500, "db": -10},        # unknown type
                     {"type": "highshelf", "hz": 1500},               # shelf without db
                     {"type": "lowpass", "hz": 3000, "db": -6},       # a cut takes no db
                     {"type": "peak", "hz": 5, "db": -3},             # below 20 Hz
                     {"type": "peak", "hz": 3000, "db": -3, "q": 2}): # unknown key
            with self.subTest(band=band), self.assertRaises(ValueError):
                parse_eq("guitar", [band])
        with self.assertRaises(ValueError):
            parse_eq("guitar", {"type": "highshelf", "hz": 1500, "db": -10})  # not a list


if __name__ == "__main__":
    unittest.main()
