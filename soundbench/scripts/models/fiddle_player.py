"""A deterministic player around the physical fiddle's bow contact.

Force follows a numerically calibrated, neighbor-supported playable corridor.
This supplies control gestures, not an independent pitched sound source.
"""
from functools import lru_cache
import json
import numpy as np
from scipy.signal import periodogram
from soundbench.scripts.paths import data_path

CALIBRATION = data_path('fiddle', 'player.json')


@lru_cache(maxsize=1)
def force_ratios():
    data = json.loads(CALIBRATION.read_text())
    return np.array([n['pitch'] for n in data['notes']]), np.array([n['ratio'] for n in data['notes']])


@lru_cache(maxsize=1)
def intonation_table():
    data = json.loads(CALIBRATION.read_text())
    return np.array([n['pitch'] for n in data['notes']]), np.array([n.get('cents', 0.) for n in data['notes']])


@lru_cache(maxsize=1)
def corridor_table():
    data = json.loads(CALIBRATION.read_text())
    pitches = np.array([n['pitch'] for n in data['notes']])
    found = np.array([bool(n.get('corridor')) for n in data['notes']])
    widths = np.array([np.log(n['corridor'][1]/n['corridor'][0]) if n.get('corridor') else 0. for n in data['notes']])
    # A pitch with no clean corridor borrows its neighbours' width; a
    # single-point corridor is genuinely narrow and keeps width 0.
    widths[~found] = np.interp(pitches[~found], pitches[found], widths[found]) if found.any() else 0.
    return pitches, widths


def steadiness(pitch):
    """How freely the player lets the arm sway on this note, 0..1.

    A player feels how much room the string gives before the tone breaks:
    full sway where the calibrated Helmholtz corridor spans a factor of 1.8
    or more, none where it is a single point. Pitches without a clean
    corridor take their neighbours' width.
    """
    pitches, widths = corridor_table()
    return float(np.clip(np.interp(pitch, pitches, widths)/np.log(1.8), 0, 1))


def intonation(pitch):
    """Semitones the player adds to correct the bow's pitch flattening."""
    pitches, cents = intonation_table()
    return -float(np.interp(pitch, pitches, cents))/100


def normal_force(pitch, speed, position, impedance):
    pitches, ratios = force_ratios()
    ratio = np.interp(pitch, pitches, ratios)
    # Schelleng-like velocity/position scaling. The empirical corridor is
    # measured at beta=.12; the inverse-position extension is a design prior.
    return np.clip(2*impedance*speed*ratio*.12/position, .002, 1.)


def bow_shape(t, duration, phrase_end=False):
    u = np.clip(t/duration, 0, 1)
    shape = .84+.28*np.sin(np.pi*u)**1.2
    if phrase_end and duration > .6:
        fade = np.clip((t-(duration-.4))/.4, 0, 1)
        shape *= 1-.42*fade*fade*(3-2*fade)
    return shape


def tone_quality(y, pitch, threshold_db=-22):
    f0 = 440*2**((pitch-69)/12)
    f, power = periodogram(y, 48000, nfft=65536)
    bands = [abs(f-h*f0) < max(10, .02*h*f0) for h in range(1, 13)]
    partials = np.array([power[mask].sum() for mask in bands])
    db = 10*np.log10((partials+1e-25)/max(partials.max(), 1e-25))
    band = (f > f0*.8) & (f < min(12000, 12.5*f0))
    harmonic = np.any(bands, axis=0)
    indices = np.flatnonzero(bands[0])
    peak = indices[np.argmax(power[indices])]
    values = np.log(power[peak-1:peak+2]+1e-30)
    shift = .5*(values[0]-values[2])/(values[0]-2*values[1]+values[2])
    measured = (peak+shift)*48000/65536
    return dict(partials=int(sum(db > threshold_db)), tuning_cents=float(1200*np.log2(measured/f0)),
                harmonic_fraction=float(power[band & harmonic].sum()/max(power[band].sum(), 1e-25)),
                upper_fraction=float(partials[2:].sum()/max(partials.sum(), 1e-25)),
                rms=float(np.sqrt(np.mean(y*y))))


def runs(mask):
    """Start indices, lengths and values of the runs of a boolean array."""
    edges = np.flatnonzero(np.diff(mask.astype(np.int8)))+1
    starts = np.r_[0, edges]
    return starts, np.diff(np.r_[starts, len(mask)]), mask[starts]


def helmholtz(stick, pitch, rate=48000):
    """Slip episodes per fundamental period and the fraction of time stuck.

    Helmholtz motion slips once per period and sticks for most of it. Stick
    runs of one or two samples inside a slip are reattachment flicker, and
    one-sample slips are discarded.
    """
    slip = ~np.asarray(stick, bool)
    if not len(slip):
        return dict(slips=0., stuck=1.)
    starts, lengths, values = runs(slip)
    for a, n, v in zip(starts, lengths, values):
        if not v and n <= 2 and a > 0 and a+n < len(slip):
            slip[a:a+n] = True
    starts, lengths, values = runs(slip)
    episodes = int(np.sum(values & (lengths > 1)))
    f0 = 440*2**((np.mean(pitch)-69)/12)
    return dict(slips=float(episodes/(len(slip)/rate*f0)), stuck=float(1-slip.mean()))


def helmholtz_score(checks):
    """The weakest window's distance from one slip per period, mostly stuck."""
    return min(-4*abs(q['slips']-1)-4*max(0., .55-q['stuck']) for q in checks)


def grit_controls(grit):
    """Map the player's sweet-to-gritty choice onto rosin grain and contact point.

    Grit 0.25 is the house tone. Sweet (0) plays with 40% of the grain, 15%
    further from the bridge; full grit (1) doubles the grain at 75% of the
    house distance, where the force corridor rises with 1/position. Beyond
    that the bow leaves Helmholtz motion; bow_style 'crunch' is the way there.
    """
    if grit <= .25:
        return .4+2.4*grit, 1.15-.6*grit
    return 1+(grit-.25)*4/3, 1-(grit-.25)/3


def oldtime(melody, beat, open_midi, drone=True, rock=True, slide_into=(), grit=.35, push=.008):
    """Round Peak bowing around a melody of eighth notes, as fiddle note rows.

    `melody` is a list of (onset seconds, pitch, velocity), one per eighth,
    beat-aligned from the first note. Each 2/4 bar is bowed 1 | (& 2) | &: the
    slur starts on the off-beat and carries across the beat (rolling onto
    another string if the melody crosses), so bow changes
    and their grip land on the off-beats, which also sit `push` seconds early.
    Every stroke rides the adjacent lower string (the higher one for melody on
    the bottom string) as an open drone in the same bow. With `rock`, the drone
    swells to near the melody's level on the heads of 3+3+2 eighth groups.
    Pitch classes in `slide_into` are approached from 0.7 semitone below.
    """
    strings = sorted(range(4), key=lambda s: -open_midi[s])
    eighth = beat/2
    starts = []
    for index, (onset, _, _) in enumerate(melody):
        starts.append(onset-(push if index % 2 else 0))
    rows = []
    for index, (onset, pitch, velocity) in enumerate(melody):
        slot = index % 4
        string = next(s for s in strings if pitch >= open_midi[s])
        slurred = slot == 2
        off = starts[index+1] if index+1 < len(melody) else onset+eighth
        accent = 3 if slot in (1, 3) else 0
        row = dict(time=starts[index], off=off, pitch=pitch, velocity=min(127, velocity+accent),
                   string=string+1, slur=slurred, vibrato=0, grit=grit)
        if pitch % 12 in slide_into and pitch-.7 >= open_midi[string]:
            row.update(slide=-.7, slide_seconds=.05)
        rows.append(row)
        if drone:
            partner = string+1 if string < 3 else string-1
            level = .82 if rock and index % 8 in (0, 3, 6) else .55
            rows.append(dict(time=starts[index], off=off, pitch=open_midi[partner], string=partner+1,
                             velocity=max(1, round(velocity*level)), slur=slurred, vibrato=0, grit=grit))
    return rows
