"""Independent track and onset-addressed streams for stochastic excitation."""
import numpy as np


def note_rng(seed, track_id, onset, sr, stream=0):
    """A note's draws depend on its onset sample, never preceding note count."""
    return np.random.default_rng(np.random.SeedSequence(
        [seed, track_id, round(onset*sr), stream]))


def wind_noise(rows, n, sr, seed, track_id, stream=0, normal=False):
    """Continuous noise with independent streams at each fingering onset.

    Waveguide and radiation-filter history remain continuous across onsets.
    Separate stream IDs keep noise layers independent of each other's draws.
    """
    starts = sorted({0, *(min(n, round(row['time']*sr)) for row in rows)})
    out = np.empty(n)
    for a, b in zip(starts, starts[1:]+[n]):
        rng = note_rng(seed, track_id, a/sr, sr, stream)
        out[a:b] = rng.standard_normal(b-a) if normal else rng.uniform(-1, 1, b-a)
    return out
