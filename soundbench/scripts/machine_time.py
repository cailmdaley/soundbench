"""Drum-machine clock: Linn/MPC swing and the machine's own timing table.

Pure Python, shared by the performer (score MIDI -> performed MIDI) and the
drum-machine lab, so both place a machine's hits identically.
"""
from __future__ import annotations
import math

MACHINES = frozenset(("tr808", "tr808-bass", "tr909", "linn"))
# The LM-1/LinnDrum's fixed swing settings.
LINN_SWING = {f"linn{p}": p/100 for p in (50, 54, 58, 62, 66, 71)}
TIMING_KEYS = frozenset(("swing", "swing_unit", "humanize_ms"))


def timing(settings: dict, model: str) -> tuple[float, int, float]:
    """(swing fraction, swing unit in notes per whole, onset sigma in ms) for one machine model."""
    model = model.removesuffix("-bass")  # the 808's bass mode is the same box
    table = settings.get(f"{model}_model", {})
    swing = table.get("swing", 0.5)
    if isinstance(swing, str):
        if swing not in LINN_SWING:
            raise ValueError(f"{model}_model.swing names one of {sorted(LINN_SWING)}")
        swing = LINN_SWING[swing]
    unit = table.get("swing_unit", 16)
    sigma = table.get("humanize_ms", 0.0)
    if (isinstance(swing, bool) or not isinstance(swing, (int, float)) or not math.isfinite(swing)
            or not 0.5 <= swing <= 0.75):
        raise ValueError(f"{model}_model.swing must be 0.5..0.75 or a linnNN name")
    if unit not in (8, 16, 32):
        raise ValueError(f"{model}_model.swing_unit must be 8, 16 or 32")
    if isinstance(sigma, bool) or not isinstance(sigma, (int, float)) or not 0 <= sigma <= 30:
        raise ValueError(f"{model}_model.humanize_ms must be 0..30")
    return float(swing), int(unit), float(sigma)


def swing_tick(tick: float, ticks_per_beat: int, swing: float, unit: int) -> float:
    """Move the second step of each pair to `swing` of the pair; the warp is piecewise linear.

    Pairs are aligned to the beat grid (the quarter note), so a pair of 16ths spans an 8th.
    Positions between gridlines move proportionally, which keeps note-offs after their onsets.
    """
    if swing == 0.5:
        return float(tick)
    pair = ticks_per_beat * 8 / unit
    start = math.floor(tick / pair) * pair
    p = (tick - start) / pair
    q = p * 2 * swing if p <= 0.5 else swing + (p - 0.5) * 2 * (1 - swing)
    return start + q * pair
