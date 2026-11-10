"""Explicit end-of-phrase damping and measurable sustain regression checks."""
import numpy as np
from scipy.signal import resample_poly


def finish(events, at, sr=48000):
    """Damp all still-owned note tokens at an intentional end of performance."""
    return list(events) + [[round(at*sr), 5, row[2], row[3], 0, 0, 0, 0, 0, 0, row[10]]
                          for row in events if row[1] in (0, 3, 4) and row[0] < at*sr]


def sustain_metrics(audio, sr=48000, end=24.):
    """House QA: 50 ms RMS, hops 256/22050 s, -45 dB, gaps >=120 ms.

    Fixed performance window excludes intentional end damping and A/B separator.
    The floor describes all performance hops, including attack/release valleys.
    """
    y = resample_poly(audio, 147, 320) if sr == 48000 else audio
    rate = 22050 if sr == 48000 else sr
    n, hop = round(.05*rate), 256
    frames = np.lib.stride_tricks.sliding_window_view(np.pad(y, (0,n)), n)[::hop]
    rms = np.sqrt(np.mean(frames*frames,axis=1))+1e-9
    db = 20*np.log10(rms/rms.max())
    times = np.arange(len(db))*hop/rate
    use = (times > .3) & (times < end)
    quiet = (db < -45) & use
    edges = np.diff(np.r_[False,quiet,False].astype(int))
    gaps = [(float(times[a]),float(times[min(b,len(times)-1)])) for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)) if (b-a)*hop/rate >= .12]
    return dict(dead_gaps=gaps, count=len(gaps), floor_db=float(db[use].min()),
                p10_db=float(np.percentile(db[use],10)),p50_db=float(np.percentile(db[use],50)),
                p90_db=float(np.percentile(db[use],90)))
