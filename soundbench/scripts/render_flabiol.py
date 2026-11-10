#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["mido>=1.3,<2", "numpy>=1.26", "numba>=0.61", "scipy>=1.11"]
# ///
"""Render the flabiol jobs in a studio JSON manifest."""
from soundbench.scripts.render_winds import main

if __name__ == "__main__":
    main("flabiol")
