"""Voice 3: a singer of words, and of several voices at once.

Sung syllables (scripts/models/phonetics.py) become timed phones, the phones
become articulatory gestures on a 1 ms grid, and the gestures drive voice2's
Fant-Rd glottal source through voice's Kelly-Lochbaum tract and nasal branch.

**Timing (the singer).** A syllable's vowel starts on its note (a few ms
early, since a syllable's perceptual centre lies after its onset) and holds
through any melisma. Consonants are sung in the time before: the coda of one
syllable and the onset of the next share the gap before the next vowel,
compressed when notes are short, but never below a floor, and never taking
more than a set share of the previous note. Phrase-final codas straddle the
note's end. Diphthongs hold their first element and glide late.

**Articulation (Ohman's model).** A vowel layer moves between vowel shapes
(Story 2008 MRI area functions, smoothstep transitions through the
consonants); consonants are constrictions laid over it. Each constriction has
a place, a width and an area, approached and released at a physical rate
(cm^2/s), so a stop's area grows linearly after release and its burst is as
short as a real one. Glides (w, y, r) are whole-tract shapes in the vowel
layer; nasals open the velum; voiceless consonants abduct the glottis.

**Aerodynamics and sources.** The glottis and the narrowest oral
constriction are two orifices in series, the velum a parallel outlet; oral
pressure relaxes to their steady state (slower in a voiced closure, whose
walls and larynx expand). Three sources follow from it, each with the
amplitude laws of Stevens (1998, ch. 2): turbulence at a constriction,
injected at its obstacle (the teeth for sibilants) and scaled by
sqrt(A) P^1.5; aspiration at an abducted glottis, scaled by the
transglottal pressure; and a release transient proportional to the oral
pressure at a stop's release. Phonation fades as the folds abduct or the
oral pressure rises.

**The singer.** Pitch moves between notes with a glide whose length grows
with the interval, phrase onsets scoop up, some phrase ends fall, vibrato
follows character and score marks, with flutter,
jitter and shimmer. The numbers come from references/voice-3/real-singing.
Several singers differ in tract length, source and habits; doubled singers
on one part scatter in time and pitch (ensemble()).
"""
from __future__ import annotations

from functools import lru_cache
import copy
import math
import warnings
from pathlib import Path
import tomllib

import numpy as np
from numba import njit
from scipy.signal import butter, resample_poly, sosfilt

from . import phonetics
from . import voice as v1
from . import voice2 as v2

CONFIG = v1.CONFIG.with_name("voice3.toml")
SR = 48000
REFERENCE_CM = 17.5
OBSTRUENTS = ("stop", "fricative", "affricate")


@lru_cache(maxsize=1)
def _table():
    return tomllib.loads(CONFIG.read_text())


# Singer keys a preset may leave out.
SINGER_DEFAULTS = {"vibrato_scale": 1.0, "tension": 0.0, "vowel_evenness": 1.0, "aperiodicity": 0.0,
                   "aperiodic_floor": 0.0, "f1_tuning": 1.0}


def singers():
    return sorted(_table()["singers"])


CHARACTER_LIMITS = {
    "effort_center": (0, 1), "effort_span": (0, 1.5),
    "vibrato_rate": (5, 8), "vibrato_extent": (0, 50), "vibrato_delay": (0.1, 0.5),
    "attack_ms": (20, 150), "breath_tail_ms": (40, 200), "breath_gain": (0, 4),
    "roughness": (0, 2), "sag_cents": (0, 12), "vowel_opening_hz": (0, 40),
}
CHARACTER_CHOICES = {"onset": {"soft", "breathy", "hard", "scooped"},
                     "release": {"clean", "fall", "breath", "held"},
                     "vibrato": {"natural", "straight", "emerging"}}


def character_settings(spec="plain", parent=None):
    """Resolve a named character or a strictly bounded table on top of a base."""
    if isinstance(spec, str):
        spec = {"base": spec}
    if not isinstance(spec, dict):
        raise ValueError("voice3 character is a name or a table with base and overrides")
    unknown = set(spec) - {"base"} - CHARACTER_LIMITS.keys() - CHARACTER_CHOICES.keys()
    if unknown:
        raise ValueError(f"unknown voice3 character fields {sorted(unknown)}")
    base = spec.get("base", "plain")
    if not isinstance(base, str) or base not in _table()["characters"]:
        raise ValueError(f"unknown voice3 character {base!r}")
    result = {**_table()["characters"][base], **(parent or {}), **spec}
    for key, (lo, hi) in CHARACTER_LIMITS.items():
        if type(result[key]) not in (int, float) or not lo <= result[key] <= hi:
            raise ValueError(f"voice3 character {key} must be {lo}..{hi}")
    for key, choices in CHARACTER_CHOICES.items():
        if not isinstance(result[key], str) or result[key] not in choices:
            raise ValueError(f"voice3 character {key} must be one of {sorted(choices)}")
    return result


def intent(note, c):
    return note.get("intent", c["character"])


def load_config(singer="tenor", character="plain", **overrides):
    """The tract of voice.toml and voice2.toml's nasal tuning, shaped for one singer."""
    table = copy.deepcopy(_table())
    if singer not in table["singers"]:
        raise ValueError(f"voice3 singer must be one of {sorted(table['singers'])}")
    s = {**SINGER_DEFAULTS, **table["singers"][singer]}
    unknown = set(overrides) - set(s) - {"seed"}
    if unknown:
        raise ValueError(f"unknown voice3 singer overrides {sorted(unknown)}")
    s.update({k: v for k, v in overrides.items() if k != "seed"})
    c = v2.load_config("tenor")
    q = c["voice2"]
    c["tract"]["length_cm"] = s["length_cm"]
    c["nasal"]["length_cm"] = v1.load_config()["nasal"]["length_cm"] * s["nasal_scale"]
    c["nasal"]["velum_from_glottis_cm"] = v1.load_config()["nasal"]["velum_from_glottis_cm"] * s["nasal_scale"]
    c["engine"]["output_gain"] = table["engine"]["output_gain"]
    c["engine"]["seed"] = overrides.get("seed", table["engine"]["seed"])
    c["v3"] = table
    table["articulation"].setdefault("blends", {})   # optional: [articulation] blends
    c["singer"] = s
    c["singer_name"] = singer
    c["character"] = character_settings(character)
    c["phones"] = table["phones"]
    c["tract_params"] = {**v1.tract_parameters(c), "sinus_hz": q["sinus_hz"], "sinus_area": q["sinus_area"],
                         "sinus_q": q["sinus_q"], "nasal_wall_hz": q["nasal_wall_hz"]}
    return c


# ---------------------------------------------------------------- geometry

def tract_shapes(c):
    """Vowel and glide shapes at this singer's section count.

    Story's shapes are a man's. A shorter tract keeps their shape but gives
    the pharynx (the glottal half) a smaller share than the mouth, as female
    tracts do (Fant 1966: pharynx about 0.8, mouth about 0.9 of a man's).
    """
    count = v1.sections(c)
    p = c["singer"]["pharynx_scale"]
    # Source coordinate (0..1, glottis to lips) for each target section centre:
    # the pharynx half of the source maps to a share p/(p+1) of the target.
    share = p / (p + 1)
    x = (np.arange(count) + 0.5) / count
    src = np.where(x < share, 0.5 * x / share, 0.5 + 0.5 * (x - share) / (1 - share))

    def warp(areas):
        areas = np.asarray(areas, float)
        fine = v1.resample_area(areas, 440)
        return np.interp(src, (np.arange(440) + 0.5) / 440, fine)

    shapes = {v: warp(a) for v, a in v1.STORY_AREAS.items()}
    shapes["ax"] = warp(np.mean([v1.STORY_AREAS[v] for v in v1.STORY_AREAS], axis=0))
    section = v1.section_length_cm(c)
    scale = c["tract"]["length_cm"] / REFERENCE_CM
    for name, spec in c["constructed"].items():
        places = [[pl * scale, w * scale, a] for pl, w, a in spec["constrictions"]]
        shapes[name] = v1.constrict(shapes[spec["base"]], count, places, section)
    # [articulation] blends: a shape retuned part of the way towards another.
    for name, (base, towards, weight) in c["v3"]["articulation"].get("blends", {}).items():
        shapes[name] = (1 - weight) * shapes[base] + weight * shapes[towards]
    return shapes


# Intrinsic vowel intensity in speech, dB re /a/ (Lehiste & Peterson 1959, rounded):
# open vowels are louder; a singer keeps about this much of the difference.
INTRINSIC_DB = {"aa": 0.0, "ae": -1.0, "ah": -1.0, "ao": -1.0, "eh": -2.0, "er": -3.0, "ax": -3.0,
                "ih": -3.0, "ey": -2.5, "uh": -3.0, "ow": -2.0, "iy": -4.0, "uw": -4.5}


_TRACT_CACHE = {}


def _tract_key(c, what):
    """A cache key for everything a tract response depends on."""
    return (what, c["tract"]["length_cm"], c["singer"]["pharynx_scale"], c["nasal"]["length_cm"],
            tuple(sorted((k, v if not isinstance(v, np.ndarray) else v.tobytes()) for k, v in c["tract_params"].items())),
            tuple(c["singer"]["range"]),
            repr(sorted(c["constructed"].items())), repr(sorted(c["v3"]["articulation"].get("blends", {}).items())))


def _cached(c, what, compute):
    key = _tract_key(c, what)
    if key not in _TRACT_CACHE:
        _TRACT_CACHE[key] = compute(c)
    return _TRACT_CACHE[key]


def _vowel_levels(c):
    return _cached(c, "levels", _compute_vowel_levels)


def _compute_vowel_levels(c):
    shapes = tract_shapes(c)
    lo, hi = c["singer"]["range"]
    n = 8192
    src = np.zeros(n)
    src[0] = 1.0
    f = np.fft.rfftfreq(n, 1 / SR)
    out = {}
    for name in INTRINSIC_DB:
        area = shapes[name]
        h = v1._tract(src, np.zeros(n), np.zeros(2, np.int64), np.repeat(area[None], 2, 0), np.zeros(2), n,
                      **c["tract_params"])
        H = np.abs(np.fft.rfft(h)) ** 2
        levels = []
        for pitch in np.linspace(lo + 2, hi - 2, 5):
            f0 = 440 * 2 ** ((pitch - 69) / 12)
            k = np.arange(1, int(5000 / f0))
            # Harmonics of a glottal flow-derivative falling 6 dB/oct above ~500 Hz.
            levels.append(np.sum(np.interp(k * f0, f, H) / (1 + (k * f0 / 500) ** 2)))
        out[name] = 10 * np.log10(np.mean(levels))
    return out


def _first_formants(c):
    """F1 (Hz) of each vowel and glide shape: the first strong peak of the tract's response."""
    return _cached(c, "f1", _compute_first_formants)


def _compute_first_formants(c):
    shapes = tract_shapes(c)
    n = 8192
    src = np.zeros(n)
    src[0] = 1.0
    f = np.fft.rfftfreq(n, 1 / SR)
    band = (f > 150) & (f < 1300)
    out = {}
    for name, area in shapes.items():
        h = v1._tract(src, np.zeros(n), np.zeros(2, np.int64), np.repeat(area[None], 2, 0), np.zeros(2), n,
                      **c["tract_params"])
        H = np.abs(np.fft.rfft(h))[band]
        peaks = np.flatnonzero((H[1:-1] > H[:-2]) & (H[1:-1] >= H[2:])) + 1
        peaks = peaks[H[peaks] > H.max() / 10]           # within 20 dB of the strongest: not a ripple
        out[name] = float(f[band][peaks[0] if len(peaks) else np.argmax(H)])
    return out


def _formant_pair(area, c):
    """Sub-bin first two resonance frequencies of a fixed oral tract."""
    from scipy.signal import find_peaks
    n = 32768
    src = np.zeros(n)
    src[0] = 1
    response = v1._tract(src, np.zeros(n), np.zeros(2, np.int64),
                         np.repeat(area[None], 2, 0), np.zeros(2), n, **c["tract_params"])
    db = 20 * np.log10(np.abs(np.fft.rfft(response)) + 1e-15)
    peaks, _ = find_peaks(db, prominence=4)
    peaks = [i for i in peaks if 150 < i * SR / n < 3500][:2]
    if len(peaks) != 2:
        raise ValueError("vowel opening needs two resolved tract resonances")
    result = []
    for i in peaks:
        shift = 0.5 * (db[i - 1] - db[i + 1]) / (db[i - 1] - 2 * db[i] + db[i + 1])
        result.append((i + shift) * SR / n)
    return np.array(result)


def _opening_shapes(c):
    """Area deformations per Hz F1, with F2 held at its resting resonance.

    Four broad tongue/jaw deformations form a small physical inverse problem.
    This changes tract geometry; no filter is applied to the rendered voice.
    """
    def compute(c):
        result = {}
        for name, area in tract_shapes(c).items():
            used = {v for shapes in c["v3"]["articulation"]["vowels"].values() for v in shapes}
            used.add(c["v3"]["articulation"]["schwa"])
            if name not in used:
                continue
            x = (np.arange(len(area)) + 0.5) / len(area)
            basis = np.array([area * np.exp(-0.5 * ((x - centre) / 0.18) ** 2)
                              for centre in (0.2, 0.4, 0.65, 0.85)])
            target = _formant_pair(area, c) + np.array([40., 0.])
            opened = area.copy()
            for _ in range(4):
                jac = np.array([(_formant_pair(opened + 0.005 * b, c) - _formant_pair(opened - 0.005 * b, c)) / 0.01
                                for b in basis]).T
                weights = np.linalg.pinv(jac) @ (target - _formant_pair(opened, c))
                opened += weights @ basis
            result[name] = (opened - area) / 40
        return result
    return _cached(c, "opening", compute)


def upper_register(pitch, c):
    """0 in the lower half of a singer's range, rising to 1 at its top note:
    where the folds stop closing completely (more noise between the partials)
    and the jaw opens to keep F1 under the pitch."""
    lo, hi = c["singer"]["range"]
    middle = 0.5 * (lo + hi)
    return np.clip((np.asarray(pitch, float) - middle) / (hi - middle), 0, 1)


def vowel_balance(c):
    """dB to add to each vowel's source so the singer's vowels differ only as
    much as real vowels do (INTRINSIC_DB), not as much as the bare tract makes them."""
    levels = _vowel_levels(c)
    keep = c["singer"]["vowel_evenness"]
    return {v: keep * (INTRINSIC_DB[v] - (levels[v] - levels["aa"])) for v in levels}


def section_index(c, cm_from_lips):
    """The section whose centre is nearest a distance from the lips."""
    count = v1.sections(c)
    k = count - 1 - int(round(cm_from_lips * c["tract"]["length_cm"] / REFERENCE_CM / v1.section_length_cm(c) - 0.5))
    return int(np.clip(k, 2, count - 1))


# ---------------------------------------------------------------- planning

def vowel_shapes_for(phone, c):
    """Tract shape names for an ARPAbet vowel (two for a diphthong)."""
    a = c["v3"]["articulation"]
    b = phonetics.base(phone)
    if b == "AH" and phone.endswith("0"):
        return [a["schwa"]]
    return list(a["vowels"][b])


def _durations(phone, role, c):
    """(main, after) nominal seconds of a consonant in a role: a stop's closure
    and its aspiration, an affricate's closure and its frication."""
    spec = c["phones"][phone]
    main = spec["onset_ms" if role == "onset" else "coda_ms"] / 1000 * c["singer"]["consonant_scale"]
    after = 0.0
    if spec["kind"] == "stop":
        after = spec["vot_ms"] / 1000 * (1.0 if role == "onset" else 0.35)
    elif spec["kind"] == "affricate":
        after = spec["frication_ms"] / 1000
    return main, after


def _cluster(phones, role, c):
    return [(p, *_durations(p, role, c)) for p in phones]


def _layout(cluster, start, scale):
    """Lay a consonant cluster end to end from `start`; returns segments and the end time."""
    out = []
    t = start
    for phone, main, after in cluster:
        out.append({"kind": "cons", "phone": phone, "start": t, "end": t + main * scale,
                    "after_end": t + (main + after) * scale})
        t += (main + after) * scale
    return out, t


def plan(rows, c, context_end=None):
    """Timed phones for sung notes.

    rows: dicts with time, off (s), pitch, velocity and syllable (a
    phonetics.Syllable, or None to continue the previous syllable). Returns
    units (syllables with their notes and vowel span), phrases and segments
    (vowels and consonants with start/end times). context_end: where the
    previous phrase (sung in an earlier render) ended, in the rows' time.
    """
    tm = c["v3"]["timing"]
    rows = sorted(rows, key=lambda r: r["time"])
    units = []
    for r in rows:
        if r.get("syllable") is None:
            if not units:
                raise ValueError("the first sung note needs a lyric syllable")
            gap = r["time"] - max(n["off"] for n in units[-1]["notes"])
            if gap > tm["rest_gap_s"]:
                # A melisma continues a vowel, even detached; after a rest a note needs its own word.
                raise ValueError(f"the note at {r['time']:.2f} s has no syllable but follows a {gap:.2f} s rest: "
                                 "give it a lyric (a Lyrics line shorter than its notes?)")
            units[-1]["notes"].append(r)
            continue
        units.append({"syl": r["syllable"], "notes": [r]})
    for u in units:
        u["start"] = u["notes"][0]["time"]
        u["end"] = max(n["off"] for n in u["notes"])
        syl = u["syl"]
        u["onset"] = _cluster([p for p in syl.onset], "onset", c)
        u["coda"] = _cluster([p for p in syl.coda], "coda", c)
        u["held"] = not any(phonetics.is_vowel(p) for p in syl.nucleus)
    phrases = []
    for i, u in enumerate(units):
        if not phrases or u["start"] - units[i - 1]["end"] > tm["legato_gap_s"]:
            phrases.append([])
        phrases[-1].append(u)
    segments = []
    previous_end = -math.inf if context_end is None else context_end
    for phrase in phrases:
        for i, u in enumerate(phrase):
            u["vowel_on"] = u["start"] - tm["vowel_lead_s"]
            onset_len = sum(m + a for _, m, a in u["onset"])
            if i == 0:
                # A phrase's first consonants take the rest before it.
                room = u["vowel_on"] - previous_end - 0.02
                scale = 1.0 if onset_len <= 0 else float(np.clip(room / onset_len, tm["consonant_floor"], 1.0))
                block, _ = _layout(u["onset"], u["vowel_on"] - onset_len * scale, scale)
                for s in block:
                    s["unit"], s["role"] = u, "onset"
                segments += block
            else:
                prev = phrase[i - 1]
                coda_len = sum(m + a for _, m, a in prev["coda"])
                need = coda_len + onset_len
                room = (1 - tm["min_vowel_fraction"]) * (u["vowel_on"] - prev["vowel_on"])
                scale = 1.0 if need <= 0 else float(np.clip(room / need, tm["consonant_floor"], 1.0))
                start = u["vowel_on"] - need * scale
                start = max(start, prev["vowel_on"] + 0.25 * (u["vowel_on"] - prev["vowel_on"]))
                scale = 1.0 if need <= 0 else min(scale, (u["vowel_on"] - start) / need)
                prev["vowel_off"] = start
                block, t = _layout(prev["coda"], start, scale)
                for s in block:
                    s["unit"], s["role"] = prev, "coda"
                segments += block
                block, _ = _layout(u["onset"], t, scale)
                for s in block:
                    s["unit"], s["role"] = u, "onset"
                segments += block
        last = phrase[-1]
        coda_len = sum(m + a for _, m, a in last["coda"])
        last["vowel_off"] = max(last["end"] - tm["final_coda_overlap"] * coda_len,
                                last["vowel_on"] + 0.5 * (last["end"] - last["vowel_on"]))
        block, previous_end = _layout(last["coda"], last["vowel_off"], 1.0)
        for s in block:
            s["unit"], s["role"] = last, "coda"
        # Where phonation stops: a vowel fades; a final sonorant is held while
        # the voice fades through it; an obstruent ends the voice at once.
        release = tm["release_s"]
        if not block:
            # A held final vowel tapers: the longer the note, the longer the
            # diminuendo, begun inside the note so it closes near the note-off.
            held = last["end"] - last["vowel_on"]
            fade = float(np.clip(tm["final_taper"] * held, release, tm["final_taper_max_s"]))
            last["voice_off"], last["fade"] = last["end"] - 0.5 * (fade - release), fade
        elif c["phones"][block[-1]["phone"]]["kind"] in ("nasal", "lateral", "glide"):
            final = block[-1]
            last["voice_off"], last["fade"] = final["start"] + 0.04, release
            final["end"] = final["after_end"] = max(final["end"], last["voice_off"] + release)
            previous_end = final["end"]
        else:
            voiced = c["phones"][block[-1]["phone"]]["voiced"]
            last["voice_off"], last["fade"] = block[-1]["start"], 0.08 if voiced else 0.03
        ending = intent(last["notes"][-1], c)["release"]
        if ending == "held":
            last["voice_off"], last["fade"] = last["end"], release
        elif ending == "breath":
            last["voice_off"], last["fade"] = last["end"] - 0.06, 0.06
        elif ending in ("clean", "fall"):
            last["voice_off"], last["fade"] = last["end"] - 0.02, 0.08
        previous_end = max(previous_end, last["end"])
        segments += block
    for u in units:
        segments += _nucleus(u, c)
    segments.sort(key=lambda s: s["start"])
    return {"units": units, "phrases": phrases, "segments": segments}


def _nucleus(u, c):
    """Vowel segments (and any consonants inside an elided nucleus) over the vowel span."""
    tm = c["v3"]["timing"]
    a, b = u["vowel_on"], u["vowel_off"]
    phones = u["syl"].nucleus
    if u["held"]:
        # A hum or a hiss: the consonants are held through the note.
        out = []
        span = (b - a) / len(phones)
        for k, p in enumerate(phones):
            out.append({"kind": "cons", "phone": p, "start": a + k * span, "end": a + (k + 1) * span,
                        "after_end": a + (k + 1) * span, "unit": u, "role": "held"})
        return out
    inner = [(p, *_durations(p, "onset", c)) for p in phones if not phonetics.is_vowel(p)]
    vowels = [p for p in phones if phonetics.is_vowel(p)]
    cons_time = sum(m + x for _, m, x in inner)
    cons_time = min(cons_time, 0.4 * (b - a))
    vowel_time = (b - a) - cons_time
    share = tm["second_vowel_share"]
    weights = [1.0] + [share / (1 - share)] * (len(vowels) - 1)
    weights = np.array(weights) / sum(weights)
    out = []
    t = a
    queue = list(phones)
    k = 0
    scale = 1.0 if not inner else cons_time / max(sum(m + x for _, m, x in inner), 1e-9)
    while queue:
        p = queue.pop(0)
        if phonetics.is_vowel(p):
            d = vowel_time * weights[k]
            out.append({"kind": "vowel", "phone": p, "shapes": vowel_shapes_for(p, c), "start": t, "end": t + d, "unit": u})
            t += d
            k += 1
        else:
            m, x = _durations(p, "onset", c)
            out.append({"kind": "cons", "phone": p, "start": t, "end": t + m * scale, "after_end": t + (m + x) * scale,
                        "unit": u, "role": "inner"})
            t += (m + x) * scale
    return out


# ---------------------------------------------------------------- lanes

def _span(t0, t1, dt, steps):
    return max(0, int(math.floor(t0 / dt))), min(steps, int(math.ceil(t1 / dt)) + 1)


def _ramp(x):
    return v1.smoothstep(x)


def _neighbour_vowel(seg, segments, direction):
    """The nearest vowel shape before (-1) or after (+1) a segment."""
    vowels = [s for s in segments if s["kind"] == "vowel"]
    if direction > 0:
        later = [s for s in vowels if s["start"] >= seg["start"]]
        return later[0]["shapes"][0] if later else (vowels[-1]["shapes"][-1] if vowels else "ax")
    earlier = [s for s in vowels if s["end"] <= seg["end"] + 1e-9]
    return earlier[-1]["shapes"][-1] if earlier else (vowels[0]["shapes"][0] if vowels else "ax")


def articulation(p, c, steps, dt, pitch=None, opening_hz=None, effort=None):
    """Area, velum, abduction and noise lanes from a plan.

    pitch (MIDI, per step) lets the singer tune the first formant: when a
    note in the upper half of the range rises past a vowel's F1 the jaw opens
    towards /a/ until F1 meets the pitch again, as singers do on high notes
    (Sundberg 1975; Joliveau, Smith & Wolfe 2004). Without it a high closed
    vowel loses its body and turns thin and shrill; lower down it would cost
    the vowel its identity.
    """
    tm, ar = c["v3"]["timing"], c["v3"]["articulation"]
    shapes = tract_shapes(c)
    count = v1.sections(c)
    section = v1.section_length_cm(c)
    scale = c["tract"]["length_cm"] / REFERENCE_CM
    t = np.arange(steps) * dt
    segments = p["segments"]
    # Vowel layer: keyframes, smoothstep between them.
    keys = []
    for s in segments:
        if s["kind"] == "vowel":
            a, b = s["start"], s["end"]
            n_in = min(tm["vowel_in_s"], 0.3 * (b - a))
            n_out = min(tm["vowel_out_s"], 0.2 * (b - a))
            if len(s["shapes"]) == 1:
                keys += [(a + n_in, s["shapes"][0]), (b - n_out, s["shapes"][0])]
            else:
                g = min(tm["diphthong_glide_s"], tm["diphthong_share"] * (b - a))
                keys += [(a + n_in, s["shapes"][0]), (b - g, s["shapes"][0]), (b - 0.3 * n_out, s["shapes"][1])]
        elif c["phones"][s["phone"]]["kind"] == "glide":
            a, b = s["start"], s["end"]
            mid, half = 0.5 * (a + b), 0.5 * tm["glide_hold"] * (b - a)
            shape = c["phones"][s["phone"]]["shape"]
            keys += [(mid - half, shape), (mid + half, shape)]
    keys.sort(key=lambda k: k[0])
    areas = np.empty((steps, count))
    if keys:
        times = np.array([k[0] for k in keys])
        vecs = np.array([shapes[k[1]] for k in keys])
        idx = np.searchsorted(times, t, side="right")
        lo, hi = np.clip(idx - 1, 0, len(times) - 1), np.clip(idx, 0, len(times) - 1)
        span = np.where(hi > lo, times[hi] - times[lo], 1.0)
        w = _ramp(np.where(hi > lo, (t - times[lo]) / np.maximum(span, 1e-9), 0.0))[:, None]
        areas[:] = vecs[lo] * (1 - w) + vecs[hi] * w
        tune = c["singer"]["f1_tuning"]
        if pitch is not None and tune > 0:
            f1 = _first_formants(c)
            keyf1 = np.array([f1[k[1]] for k in keys])
            here = keyf1[lo] * (1 - w[:, 0]) + keyf1[hi] * w[:, 0]
            f0 = 440 * 2 ** ((pitch - 69) / 12)
            opened = f1[ar["open_vowel"]]
            # (A vowel whose F1 is already at or above the open vowel's has nowhere to open.)
            room = opened - here
            need = np.where(room > 50.0, np.clip((f0 - here) / np.maximum(room, 50.0), 0, 1), 0.0)
            need = need * tune * upper_register(pitch, c)
            areas[:] = areas * (1 - need[:, None]) + shapes[ar["open_vowel"]][None, :] * need[:, None]
    else:
        areas[:] = shapes["ax"]
    if opening_hz is not None and np.any(opening_hz):
        opening = _opening_shapes(c)
        for seg in p["segments"]:
            if seg["kind"] != "vowel":
                continue
            k0, k1 = _span(seg["start"], seg["end"], dt, steps)
            shift = opening_hz[k0:k1] * effort[k0:k1]
            tt = t[k0:k1]
            win = _ramp((tt - seg["start"]) / 0.03) * _ramp((seg["end"] - tt) / 0.03)
            areas[k0:k1] += (shift * win)[:, None] * opening[seg["shapes"][0]][None, :]
        areas = np.maximum(areas, c["tract"]["min_area_cm2"])
    # A consonant also positions the tongue body (its locus): alveolars front
    # it, so F2 points to about 1.8 kHz at their release, labials leave it alone.
    for s in segments:
        if s["kind"] != "cons":
            continue
        spec = c["phones"][s["phone"]]
        if s["phone"] in ("K", "G", "NG"):
            # Velars have no single locus: the dorsum rises in front of a front
            # vowel, behind a rounded one.
            neighbour = _neighbour_vowel(s, segments, +1 if s["role"] in ("onset", "inner") else -1)
            name, weight = ar["velar_locus_round" if neighbour in ar["rounded"] else "velar_locus"]
        elif "locus" in spec:
            name, weight = spec["coda_locus"] if s["role"] == "coda" and "coda_locus" in spec else spec["locus"]
        else:
            continue
        # Full at the closure and the release; the CV transition takes ~50 ms.
        a, b = s["start"], s["end"]
        i0, i1 = _span(a - 0.06, b + 0.06, dt, steps)
        tt = t[i0:i1]
        win = weight * _ramp((tt - a + 0.05) / 0.05) * (1 - _ramp((tt - b) / 0.05))
        areas[i0:i1] = areas[i0:i1] * (1 - win[:, None]) + shapes[name][None, :] * win[:, None]
    first_oral = max(2, int(round(3.0 / section)))
    vowel_min = areas[:, first_oral:].min(axis=1)          # the vowel layer's own narrowest place
    xs = (count - 0.5 - np.arange(count)) * section       # section centres from the lips, cm
    velum = np.zeros(steps)
    abduction = np.zeros(steps)
    constriction = np.full(steps, np.inf)   # the narrowest consonant constriction, cm^2
    obstacle = np.zeros(steps)              # turbulence strength of that consonant's jet
    sibilance = np.zeros(steps)             # 1 sibilant source, 0 flat
    source_at = np.zeros(steps, np.int64)   # injection section; `count` radiates at the lips
    voiced_closure = np.zeros(steps)
    releases = []
    vr = ar["velar_range_cm"]
    a_hold = c["v3"]["aero"]["aspiration_hold"]
    for s in segments:
        if s["kind"] != "cons":
            continue
        spec = c["phones"][s["phone"]]
        kind = spec["kind"]
        a, b, after = s["start"], s["end"], s["after_end"]
        if kind in ("stop", "fricative", "affricate", "nasal", "lateral"):
            place, width, target = spec["place"]
            if s["phone"] in ("K", "G", "NG"):
                neighbour = _neighbour_vowel(s, segments, +1 if s["role"] in ("onset", "inner") else -1)
                front = ar["frontness"].get(neighbour, 0.5)
                place = vr[1] - front * (vr[1] - vr[0])
            rate = spec["release"]
            i0, i1 = _span(a - 0.12, after + 0.12, dt, steps)
            tt = t[i0:i1]
            if kind == "affricate":
                fric = c["phones"][spec["fricative"]]
                opening = np.where(tt <= b, target, np.minimum(fric["place"][2], target + rate * (tt - b)))
                opening = np.where(tt > after, fric["place"][2] + fric["release"] * (tt - after), opening)
            else:
                opening = np.where(tt <= b, target, target + rate * (tt - b))
            closing = np.where(tt < a, target + rate * (a - tt), np.inf)
            con = np.where(tt < a, closing, opening)
            weight = v1.spatial_weight(xs, place * scale, width * scale)
            block = areas[i0:i1]
            here = block[:, weight > 0.99].min(axis=1) if (weight > 0.99).any() else block[:, np.argmax(weight)]
            areas[i0:i1] = block * (1 - weight[None, :]) + weight[None, :] * np.minimum(block, con[:, None])
            effective = np.minimum(con, here)
            at = _source(spec, place, c)
            if kind in ("stop", "affricate"):
                releases.append({"time": b, "phone": s["phone"], "at": at, "gain": spec["burst"]})
            if "lips_cm2" in spec:
                act = _ramp((tt - a + 0.04) / 0.04) * _ramp((after + 0.03 - tt) / 0.03)
                lip_w = v1.spatial_weight(xs, 0.36 * scale, 0.8 * scale)[None, :] * act[:, None]
                block = areas[i0:i1]
                areas[i0:i1] = block * (1 - lip_w) + lip_w * np.minimum(block, spec["lips_cm2"])
            if kind == "nasal":
                ahead = tm["nasal_anticipation_s"]
                win = _ramp((tt - a + ahead) / ahead) * _ramp((b + tm["nasal_carryover_s"] - tt) / tm["nasal_carryover_s"])
                velum[i0:i1] = np.maximum(velum[i0:i1], spec["velum_cm2"] * (1.6 if s["role"] == "held" else 1.0) * win)
            if kind in OBSTRUENTS:
                if spec["voiced"]:
                    win = _ramp((tt - a + 0.01) / 0.01) * _ramp((b - tt) / 0.005)
                    voiced_closure[i0:i1] = np.maximum(voiced_closure[i0:i1], win)
                strength = spec["obstacle"] * _ramp((tt - a + 0.03) / 0.02) * _ramp((after + 0.04 - tt) / 0.02)
            else:
                strength = np.zeros_like(tt)
            # The narrowest consonant constriction makes the noise, from its own source.
            take = effective < constriction[i0:i1]
            constriction[i0:i1] = np.where(take, effective, constriction[i0:i1])
            obstacle[i0:i1] = np.where(take, strength, obstacle[i0:i1])
            sibilance[i0:i1] = np.where(take, float(spec.get("sibilant", False) or (kind == "affricate")), sibilance[i0:i1])
            source_at[i0:i1] = np.where(take, at, source_at[i0:i1])
        if spec["voiced"] and spec.get("abduction", 0) > 0:
            # Voiced fricatives are made with the folds a little apart: weaker,
            # breathier voicing and more air through the constriction.
            i0, i1 = _span(a - 0.05, after + 0.05, dt, steps)
            tt = t[i0:i1]
            g = spec["abduction"] * _ramp((tt - a + 0.02) / 0.03) * (1 - _ramp((tt - after + 0.005) / 0.03))
            abduction[i0:i1] = np.maximum(abduction[i0:i1], g)
        if not spec["voiced"]:
            i0, i1 = _span(a - 0.06, after + 0.08, dt, steps)
            tt = t[i0:i1]
            if kind == "stop":
                # Abduction peaks at the release and stays open through most of
                # the aspiration (aspiration_hold of the VOT); voicing resumes as
                # the folds come together.
                rise = _ramp((tt - a + 0.01) / max(b - a + 0.01, 1e-3))
                hold = b + a_hold * (after - b)
                fall = 1 - _ramp((tt - hold) / max(after - hold + 0.01, 0.01))
                g = np.where(tt <= b, rise, fall)
            elif kind == "aspirate":
                g = _ramp((tt - a + 0.02) / 0.03) * (1 - _ramp((tt - b + 0.01) / 0.03))
            else:
                g = _ramp((tt - a + 0.02) / 0.035) * (1 - _ramp((tt - after + 0.005) / 0.03))
            abduction[i0:i1] = np.maximum(abduction[i0:i1], g)
    return {"areas": areas, "velum": velum, "abduction": abduction, "obstacle": obstacle,
            "sibilance": sibilance, "source_at": source_at, "voiced_closure": voiced_closure,
            "constriction": constriction, "vowel_min": vowel_min, "releases": releases, "shapes": shapes}


def _source(spec, place, c):
    """Injection section for a consonant's turbulence; sources at the lips radiate directly."""
    count = v1.sections(c)
    cm = spec.get("source_cm", -1.0)
    if cm == 0:
        return count
    if cm > 0:
        return section_index(c, cm)
    return min(section_index(c, place) + 1, count - 1)


@njit(cache=True)
def _relax(target, tau_steps):
    out = np.empty_like(target)
    p = 0.0
    for n in range(len(target)):
        p += (target[n] - p) / max(tau_steps[n], 1.0)
        out[n] = p
    return out


def aerodynamics(lanes, breath, phonating, c, dt):
    """Pressures and the three noise-source amplitudes, per control step.

    Glottis (A_g), the vowel's narrowest place (A_v) and the consonant
    constriction (A_k, with the velum beside it) are orifices in series: each
    takes a share of the subglottal pressure proportional to 1/A^2.
    """
    a = c["v3"]["aero"]
    ag = a["glottal_area_cm2"] * phonating + a["abduction_cm2"] * lanes["abduction"] + 1e-4
    ak = np.minimum(lanes["constriction"], 50.0)
    out = ak + lanes["velum"] + c["tract"]["min_area_cm2"]
    resistance = ag ** -2 + lanes["vowel_min"] ** -2 + out ** -2
    behind = breath * out ** -2 / resistance               # pressure drop across the consonant
    glottal = breath * ag ** -2 / resistance               # transglottal pressure
    tau = np.where(lanes["voiced_closure"] > 0.5, a["voiced_tau_ms"], a["pressure_tau_ms"]) / 1000 / dt
    po = _relax(behind, tau)
    frication = a["frication_gain"] * lanes["obstacle"] * np.sqrt(ak) * np.maximum(po, 0) ** 1.5
    frication *= ak < a["turbulence_max_area_cm2"]
    aspiration = a["aspiration_gain"] * lanes["abduction"] * np.clip(breath - po, 0, None) ** 1.5
    voicing = (1 - _ramp(lanes["abduction"] / a["voicing_abduction"])) * \
        (1 - a["voicing_pressure_loss"] * np.clip(po / np.maximum(breath, 1e-9), 0, 1))
    bursts = []
    for r in lanes["releases"]:
        k = int(round(r["time"] / dt))
        if 0 <= k < len(po):
            ms = c["phones"][r["phone"]].get("burst_ms", a["burst_ms"])
            bursts.append((r["time"], r["at"], a["burst_gain"] * r["gain"] * po[k], ms))
    return {"oral_pressure": po, "transglottal": glottal, "frication": frication, "aspiration": aspiration,
            "voicing": voicing, "bursts": bursts}


def _step(x, rise, zeta=0.6):
    """A note change: the step response of a damped second-order system whose
    10-90 % rise takes `rise` seconds, landing with a small overshoot (about 10 %)."""
    omega = (1 - 0.4167 * zeta + 2.917 * zeta ** 2) / rise
    wd = omega * math.sqrt(1 - zeta ** 2)
    return 1 - np.exp(-zeta * omega * x) * (np.cos(wd * x) + zeta * omega / wd * np.sin(wd * x))


def boundary_of(notes, i, release):
    return notes[i + 1]["time"] if i + 1 < len(notes) else notes[i]["off"] + release


def _walk(rng, n, spacing_steps, scale):
    knots = rng.normal(0, scale, n // spacing_steps + 3)
    return np.interp(np.arange(n) / spacing_steps, np.arange(len(knots)), knots)


def performance(p, c, steps, dt, rng, midi_controls=(), context_end=None):
    """Pitch, level, effort and breath lanes: the singer's habits on the notes."""
    s = c["singer"]
    tm = c["v3"]["timing"]
    t = np.arange(steps) * dt
    units = p["units"]
    notes = [n for u in units for n in u["notes"]]
    pitch = np.full(steps, float(notes[0]["pitch"]) if notes else 60.0)
    level = np.full(steps, -60.0)
    vibrato = np.zeros(steps)
    intonation = np.zeros(steps)
    breath = np.zeros(steps)
    envelope = np.zeros(steps)
    onset_rd = np.zeros(steps)
    inhale = np.zeros(steps)
    fields = {key: np.full(steps, c["character"][key], dtype=float) for key in CHARACTER_LIMITS}
    tail = np.zeros(steps)
    previous_end = -math.inf if context_end is None else context_end
    for phrase in p["phrases"]:
        pnotes = [n for u in phrase for n in u["notes"]]
        segs = [x for x in p["segments"] if x["unit"] in phrase]
        start = min([phrase[0]["vowel_on"]] + [x["start"] for x in segs])
        end = max([phrase[-1]["end"]] + [x.get("after_end", x["end"]) for x in segs])
        release = max(tm["release_s"], phrase[-1]["fade"])
        # Breath pressure: up just before the first consonant, down after the last.
        i0, i1 = _span(start - 0.08, end + release + 0.1, dt, steps)
        tt = t[i0:i1]
        breath[i0:i1] = np.maximum(breath[i0:i1], _ramp((tt - start + 0.06) / 0.05) * (1 - _ramp((tt - end) / release)))
        # Phonation envelope: voiced from the first voiced sound to the end.
        v_on = phrase[0]["vowel_on"]
        lead = [x for x in segs if x["unit"] is phrase[0] and x["kind"] == "cons" and x["role"] == "onset"]
        if lead and c["phones"][lead[0]["phone"]]["voiced"]:
            v_on = lead[0]["start"]
        first_intent = intent(pnotes[0], c)
        last_intent = intent(pnotes[-1], c)
        onset_style = first_intent["onset"]
        dynamic = np.clip(pnotes[0]["velocity"] / 127, 0, 1)
        attack = np.clip(first_intent["attack_ms"] * (1.1 - 0.2 * dynamic), 20, 150) / 1000
        attack = min(attack, max(0.02, (pnotes[0]["off"] - pnotes[0]["time"]) * 0.4))
        voice_off, fade = phrase[-1]["voice_off"], phrase[-1]["fade"]
        x = np.clip((tt - voice_off) / fade, 0, 1)
        if fade > tm["release_s"]:
            # A taper is a diminuendo, even in dB, closed off at its very end.
            closing = 10 ** (-tm["final_taper_db"] * x / 20) * (1 - _ramp((x - 0.85) / 0.15))
        else:
            closing = 1 - _ramp(x)
        envelope[i0:i1] = np.maximum(envelope[i0:i1], _ramp((tt - v_on + attack) / attack) * closing)
        if onset_style in ("soft", "breathy", "hard"):
            k0, k1 = _span(v_on - 0.03, v_on + 0.12, dt, steps)
            colour = -0.15 if onset_style == "hard" else 0.8
            onset_rd[k0:k1] += colour * (1 - _ramp((t[k0:k1] - v_on + 0.03) / 0.12))
        # An audible inhalation in a long enough rest.
        gap = start - previous_end
        if gap >= tm["breath_gap_s"] and math.isfinite(previous_end):
            length = min(0.55, gap - 0.2)
            b1 = start - 0.06
            k0, k1 = _span(b1 - length, b1, dt, steps)
            inhale[k0:k1] = np.sin(np.pi * np.clip((t[k0:k1] - b1 + length) / length, 0, 1)) ** 2
        if onset_style == "breathy":
            k0, k1 = _span(v_on - 0.06, v_on + attack, dt, steps)
            tail[k0:k1] += 0.012 * _ramp((t[k0:k1] - v_on + 0.06) / 0.04) * (1 - _ramp((t[k0:k1] - v_on) / attack))
        if last_intent["release"] == "breath":
            length = last_intent["breath_tail_ms"] / 1000
            k0, k1 = _span(end, end + length, dt, steps)
            tail[k0:k1] += 0.015 * np.sin(np.pi * np.clip((t[k0:k1] - end) / length, 0, 1)) ** 2
        previous_end = end
        # Notes: target pitch, glides, scoop, vibrato, level.
        for i, n in enumerate(pnotes):
            nxt = pnotes[i + 1] if i + 1 < len(pnotes) else None
            boundary = nxt["time"] if nxt else n["off"] + release
            k0, k1 = _span(n["time"], boundary, dt, steps)
            pitch[k0:k1] = n["pitch"]
            if i == 0:
                k0, k1 = _span(start - 0.1, n["time"], dt, steps)
                pitch[k0:k1] = n["pitch"]
            db = 24 * (n["velocity"] / 127 - 1)
            k0, k1 = _span(n["time"] - (0.25 if i == 0 else 0.0), boundary + 0.3, dt, steps)
            level[k0:k1] = db
            duration = boundary - n["time"]
            age = t - n["time"]
            if duration > 0.9:
                k0, k1 = _span(n["time"], boundary, dt, steps)
                level[k0:k1] += rng.uniform(1, 3) * np.sin(np.pi * np.clip(age[k0:k1] / duration, 0, 1))
            ch = intent(n, c)
            k0, k1 = _span(n["time"] - (0.25 if i == 0 else 0), boundary + (0.3 if nxt is None else 0), dt, steps)
            for key in fields:
                fields[key][k0:k1] = ch[key]
            # Re-articulation is a two-sided dip in an already active voice.
            # Both edges have zero slope; hard entries are shorter and deeper.
            unit = next((u for u in phrase if u["notes"][0] is n), None)
            vowel_led = unit is not None and not unit["syl"].onset
            marked = n.get("marked_onset", False)
            if i and (vowel_led or marked):
                at = unit["vowel_on"] if unit is not None else n["time"]
                attack = min(ch["attack_ms"] / 1000, max(0.02, duration * 0.4))
                edge = max(0.02, attack * (0.5 if ch["onset"] == "hard" else 1.0))
                depth = {"soft": 0.2, "breathy": 0.5, "hard": 0.95, "scooped": 0.3}[ch["onset"]]
                depth *= 1.0 if marked else 0.5
                k0, k1 = _span(at - 2 * edge, at, dt, steps)
                tt = t[k0:k1]
                dip = _ramp((tt - at + 2 * edge) / edge) * (1 - _ramp((tt - at + edge) / edge))
                envelope[k0:k1] *= 1 - depth * dip
            if i and marked:
                if ch["onset"] == "scooped":
                    k0, k1 = _span(n["time"] - attack, n["time"] + 0.09, dt, steps)
                    tt = t[k0:k1]
                    pitch[k0:k1] -= 0.3 * _ramp((tt - n["time"] + attack) / attack) * (1 - _ramp((tt - n["time"]) / 0.09))
                elif ch["onset"] == "hard":
                    k0, k1 = _span(n["time"] - attack, n["time"] + 0.08, dt, steps)
                    tt = t[k0:k1]
                    onset_rd[k0:k1] -= 0.15 * _ramp((tt - n["time"] + attack) / attack) * (1 - _ramp((tt - n["time"]) / 0.08))
                elif ch["onset"] == "breathy":
                    k0, k1 = _span(n["time"] - attack, n["time"] + 0.04, dt, steps)
                    tail[k0:k1] += 0.012 * np.sin(np.pi * np.clip((t[k0:k1] - n["time"] + attack) / (attack + 0.04), 0, 1)) ** 2
            if nxt is not None and n.get("marked_release"):
                if ch["release"] == "fall":
                    k0, k1 = _span(n["off"] - 0.1, min(boundary, n["off"] + 0.08), dt, steps)
                    pitch[k0:k1] -= 0.5 * _ramp((t[k0:k1] - n["off"] + 0.1) / 0.18)
                elif ch["release"] == "breath":
                    length = ch["breath_tail_ms"] / 1000
                    k0, k1 = _span(n["off"], min(boundary, n["off"] + length), dt, steps)
                    tail[k0:k1] += 0.015 * np.sin(np.pi * np.clip((t[k0:k1] - n["off"]) / length, 0, 1)) ** 2
                if ch["release"] != "held":
                    fade = 0.06 if ch["release"] == "breath" else 0.1
                    start_fade = n["off"] - (fade if ch["release"] == "breath" else 0.02)
                    k0, k1 = _span(start_fade, min(boundary, start_fade + fade), dt, steps)
                    envelope[k0:k1] *= 1 - _ramp((t[k0:k1] - start_fade) / fade)
                    if ch["release"] == "breath":
                        k0, k1 = _span(n["off"], boundary - dt, dt, steps)
                        envelope[k0:k1] = 0
            if duration >= 0.5 and ch["vibrato"] != "straight":
                delay = ch["vibrato_delay"]
                k0, k1 = _span(n["time"], boundary, dt, steps)
                a = age[k0:k1]
                rate = ch["vibrato_rate"] * (1 + (0.025 + 0.04 * ch["roughness"]) * _walk(rng, k1 - k0, 150, 1.0))
                phase = 2 * np.pi * np.cumsum(rate) * dt + rng.uniform(0, 2 * np.pi)
                depth = ch["vibrato_extent"] * np.clip(1 + 0.08 * _walk(rng, k1 - k0, 220, 1.0), 0.7, 1.3)
                if ch["vibrato"] == "emerging":
                    # Absolute passage progress is attached before block rendering.
                    depth *= n.get("emerging_progress", i / max(len(pnotes) - 1, 1))
                vibrato[k0:k1] = np.sin(phase) * depth * _ramp((a - delay) / 0.25) * (1 - _ramp((a - duration + 0.08) / 0.08))
            # No note is held dead straight: each starts a little off, settles in
            # its first 0.3 s, and wanders a few cents. Larger wander is heard as
            # singing out of tune, above all over a drone or in parallel harmony.
            k0, k1 = _span(n["time"], boundary, dt, steps)
            if k1 > k0:
                drift = s["drift_cents"]
                intonation[k0:k1] += rng.normal(0, drift[0]) * (1 - _ramp(age[k0:k1] / 0.3)) \
                    + _walk(rng, k1 - k0, 250, drift[1])
        # Glides between notes: the 10-90 % rise grows with the interval (68 ms
        # for a step, 156 ms for a third or fourth, in the folk measurements).
        # Into a new word the pitch moves while the consonants are sung, so the
        # vowel starts near its note; within a melisma it moves on the beat.
        starts = {id(u["notes"][0]): u for u in phrase}
        for i in range(1, len(pnotes)):
            a, b = pnotes[i - 1], pnotes[i]
            interval = b["pitch"] - a["pitch"]
            if interval == 0:
                continue
            rise = min(0.3, 0.03 + 0.031 * abs(interval))
            u = starts.get(id(b))
            lead_in = [x for x in p["segments"] if u is not None and x["unit"] is u and x.get("role") == "onset"]
            middle = 0.5 * (lead_in[0]["start"] + u["vowel_on"]) if lead_in else b["time"] - 0.15 * rise
            g0 = middle - 0.5 * rise
            k0, k1 = _span(g0, min(g0 + 3 * rise, boundary_of(pnotes, i, release)), dt, steps)
            pitch[k0:k1] = a["pitch"] + interval * _step(np.clip(t[k0:k1] - g0, 0, None), rise)
        # A word after a consonant is sung up into from below (folk: about 23 cents).
        for u in phrase[1:]:
            if u["syl"].onset:
                depth = rng.uniform(*s["word_scoop_cents"]) / 100
                k0, k1 = _span(u["vowel_on"] - 0.03, u["vowel_on"] + 0.07, dt, steps)
                pitch[k0:k1] -= depth * (1 - _ramp((t[k0:k1] - u["vowel_on"] + 0.03) / 0.1))
        # Phrase-onset scoop (folk: about 60 cents over 70 ms), some falls at the end.
        n0 = pnotes[0]
        depth = (rng.uniform(*s["scoop_cents"]) / 100) if onset_style == "scooped" else 0.0
        k0, k1 = _span(start - 0.05, n0["time"] + 0.09, dt, steps)
        pitch[k0:k1] -= depth * (1 - _ramp((t[k0:k1] - n0["time"] + 0.02) / 0.09))
        last = pnotes[-1]
        fall = 0.5 if last_intent["release"] == "fall" else 0.0
        fall += last_intent["sag_cents"] / 100
        k0, k1 = _span(last["off"] - 0.1, last["off"] + release + 0.1, dt, steps)
        pitch[k0:k1] -= fall * _ramp((t[k0:k1] - last["off"] + 0.1) / 0.25)
    # A singer evens out the vowels' loudness.
    balance = vowel_balance(c)
    for x in p["segments"]:
        if x["kind"] == "vowel":
            names = [n for n in x["shapes"] if n in balance]
            if names:
                k0, k1 = _span(x["start"] - 0.03, x["end"] + 0.03, dt, steps)
                win = _ramp((t[k0:k1] - x["start"] + 0.03) / 0.03) * (1 - _ramp((t[k0:k1] - x["end"]) / 0.03))
                level[k0:k1] += balance[names[0]] * win
    cc = sorted((r for r in midi_controls if r["control"] == 11), key=lambda r: r["time"])
    if cc:
        if any(not math.isfinite(r["time"]) or not 0 <= r["value"] <= 127 for r in cc):
            raise ValueError("invalid voice3 CC11")
        expression = np.interp(t, [r["time"] for r in cc], [r["value"] / 127 for r in cc])
        level = level + 20 * np.log10(np.maximum(expression, 1e-4))
    effort = np.clip(fields["effort_center"] + fields["effort_span"] * ((level + 17) / 17 - 0.5), 0, 1)
    return {"pitch": pitch, "level": level, "vibrato": vibrato, "intonation": intonation, "breath": breath,
            "envelope": envelope, "onset_rd": onset_rd, "inhale": inhale, "effort": effort,
            "tail": tail, **fields}


def controls(rows, c, seconds, midi_controls=(), context_end=None):
    """Every lane on the 1 ms grid, from sung notes."""
    dt = 0.001
    steps = int(math.ceil(seconds / dt)) + 2
    p = plan(rows, c, context_end)
    rng = np.random.default_rng(np.random.SeedSequence([c["engine"]["seed"], 31]))
    perf = performance(p, c, steps, dt, rng, midi_controls, context_end)
    lanes = articulation(p, c, steps, dt, perf["pitch"], perf["vowel_opening_hz"], perf["effort"])
    phonating = (perf["envelope"] > 0).astype(float)
    aero = aerodynamics(lanes, perf["breath"], phonating, c, dt)
    s = c["singer"]
    t = np.arange(steps) * dt
    flutter = sum(np.sin(2 * np.pi * hz * t + rng.uniform(0, 2 * np.pi)) for hz in (.71, 1.13, 2.37)) \
        * s["flutter_cents"] * math.sqrt(2 / 3)
    # Lanes a performer layer may add (the machine voice's contract): a cents
    # offset before vibrato, and breath (CC2) trading phonation for aspiration.
    offset = _lane(t, midi_controls, "pitch_cents", 0.0, (-2400, 2400))
    breathy = _lane(t, midi_controls, 2, 0.0, (0, 127)) / 127
    cents = 100 * perf["pitch"] + offset + flutter + perf["intonation"] + s["vibrato_scale"] * perf["vibrato"]
    cents += perf["roughness"] * _walk(rng, steps, 5, 2.0)
    f0 = 440 * 2 ** ((cents / 100 - 69) / 12)
    # Rd controls the fold pulse's open quotient and return phase together.
    # Keep the singer's firm source, opening the folds at low effort.
    soft_rd = 1.60 + (s["rd"][0] - 0.9) * 0.15
    firm_rd = 0.65 + (s["rd"][1] - 0.85) * 0.15
    rd = soft_rd + perf["effort"] * (firm_rd - soft_rd) + perf["onset_rd"] + 0.5 * lanes["abduction"] \
        + 1.2 * breathy - s["tension"]
    # The top register retains the singer's gated source: effort cannot press
    # it beyond the closure used by the harshness reference.
    upper = upper_register(perf["pitch"], c)
    safe_effort = np.clip((perf["level"] + 17) / 17, 0, 1)
    safe_rd = s["rd"][0] + safe_effort * (s["rd"][1] - s["rd"][0]) + np.maximum(perf["onset_rd"], 0)
    rd = np.maximum(rd, safe_rd * upper + rd * (1 - upper))
    rd = np.clip(rd, 0.3, 2.7)
    amplitude = 10 ** ((perf["level"] + s["level_db"]) / 20) * perf["envelope"] * aero["voicing"] * (1 - 0.8 * breathy)
    amplitude *= np.maximum(0.8, 1 + perf["roughness"] * _walk(rng, steps, 7, 0.025))
    aero["aspiration"] = aero["aspiration"] + 0.06 * breathy * perf["envelope"] * perf["breath"] + perf["tail"]
    return {"t": t, "dt": dt, "steps": steps, "plan": p, "f0": f0, "amplitude": amplitude, "rd": rd,
            **{k: v for k, v in lanes.items() if k != "shapes"}, **perf, **aero, "flutter": flutter}


def _lane(t, midi_controls, control, default, limits):
    rows = sorted((r for r in midi_controls if r["control"] == control), key=lambda r: r["time"])
    if not rows:
        return np.full(len(t), default)
    if any(not math.isfinite(r["time"]) or not limits[0] <= r["value"] <= limits[1] for r in rows):
        raise ValueError(f"invalid voice3 control {control!r}")
    return np.interp(t, [r["time"] for r in rows], [float(r["value"]) for r in rows])


# ---------------------------------------------------------------- rendering

def _highpass(x, hz):
    return sosfilt(butter(2, hz, "highpass", fs=SR, output="sos"), x)


def _lowpass(x, hz):
    return sosfilt(butter(2, hz, fs=SR, output="sos"), x)


def render(rows, seconds=None, config=None, return_controls=False, midi_controls=(), f0_map=None, context_end=None):
    """Sing `rows` (see plan) and return mono audio at 48 kHz.

    f0_map(t, f0) -> f0 may replace the control-rate pitch before the source
    (t in seconds from the rows' zero), as pitch correction does. context_end:
    where an earlier render's last phrase ended (rows' time, may be negative),
    so the first phrase here still breathes and spaces its onset after it.
    """
    c = config or load_config()
    e = c["v3"]["engine"]
    if seconds is None:
        seconds = max((r["off"] for r in rows), default=0.0) + e["tail_seconds"]
    lo, hi = c["singer"]["range"]
    outside = sorted({round(r["pitch"]) for r in rows if not lo - 0.5 <= r["pitch"] <= hi + 0.5})
    if outside:
        raise ValueError(f"{c['singer_name']} sings MIDI {lo}-{hi} (above, held notes whistle; see VOICE-MODEL.md, "
                         f"the harshness gate); notes {outside} are outside: choose another singer or key, or "
                         "widen the part's range on purpose")
    late = [r for r in rows if r["off"] > seconds]
    if late:
        warnings.warn(f"{len(late)} sung note(s) end after the render's {seconds:.2f} s and are cut off")
    samples = round(seconds * SR)
    if not rows or samples == 0:
        out = np.zeros(samples)
        return (out, None) if return_controls else out
    lead = e["lead_seconds"]
    shifted = [{**r, "time": r["time"] + lead, "off": r["off"] + lead} for r in rows]
    cc = [{**r, "time": r["time"] + lead} for r in midi_controls]
    k = controls(shifted, c, seconds + lead, cc, None if context_end is None else context_end + lead)
    if f0_map is not None:
        mapped = np.asarray(f0_map(k["t"] - lead, k["f0"].copy()), dtype=float)
        if mapped.shape != k["f0"].shape or not np.all(np.isfinite(mapped)) or np.any(mapped <= 0):
            raise ValueError("f0_map must return positive finite Hz with the control track's shape")
        k["f0"] = mapped
    total = round((seconds + lead) * SR)
    n = np.arange(total) / SR
    lane = lambda key: np.interp(n, k["t"], k[key])
    s = c["singer"]
    seed = c["engine"]["seed"]
    rng = np.random.default_rng(np.random.SeedSequence([seed, 32]))
    jitter = rng.normal(0, s["jitter"], 32768)
    shimmer = rng.normal(0, s["shimmer"], 32768)
    rough = lane("roughness")
    table, flow = v2.rd_tables()
    f0, amp, rd = lane("f0"), lane("amplitude"), lane("rd")
    d, g = v1._glottis(f0, amp, (rd - .3) / 2.4, jitter, shimmer, table, flow, 4, SR)
    d, g = resample_poly(d, 1, 4)[:total], resample_poly(g, 1, 4)[:total]
    # Firmer fold contact sharpens closure as subglottal effort rises.
    effort = lane("effort")
    upper = upper_register(np.interp(n, k["t"], k["pitch"]), c)
    safe_effort = np.clip((lane("level") + 17) / 17, 0, 1)
    safe_sharp = s["closure"][0] + (s["closure"][1] - s["closure"][0]) * safe_effort
    sharp = s["closure"][1] * effort * (1 - upper) + safe_sharp * upper
    d = d + np.r_[0., np.diff(d)] * SR / (2 * np.pi * 1000.0) * sharp
    a = c["v3"]["aero"]
    # The return phase is never instantaneous: above this the source falls a
    # further 6 dB/oct, as the real voices' spectra do between 4 and 16 kHz.
    d = sosfilt(butter(1, a["source_corner_hz"], fs=SR, output="sos"), d)
    q = c["voice2"]
    white = rng.standard_normal((5, total))
    breathy = s["breathiness"] * q["breathiness"] * (rd / 2.7) ** 2 * (q["aspiration"] * g + q["leak"] * amp)
    breathy *= ((0.015 + 0.045 * effort) * (1 - upper) + upper) * lane("breath_gain")
    aspiration = _lowpass(_highpass(white[0], a["aspiration_hz"]), a["aspiration_lowpass_hz"])
    aspiration /= np.std(aspiration) + 1e-12
    src = d + aspiration * (breathy + lane("aspiration") + a["inhale_gain"] * lane("inhale") * lane("breath_gain"))
    # Band aperiodicity: near the top of the range the folds no longer close
    # completely, and turbulence pulsed by the flow fills the space between the
    # high partials, so no clean line stands alone in the ear's most sensitive
    # band. Kept above F3, where the words' cues are; scaled to the flow.
    if s["aperiodicity"] > 0:
        hiss = _lowpass(_highpass(white[4], a["aperiodic_hz"]), a["aperiodic_lowpass_hz"])
        hiss /= np.std(hiss) + 1e-12
        upper = np.maximum(upper_register(np.interp(n, k["t"], k["pitch"]), c), s["aperiodic_floor"])
        upper = np.maximum(upper, 0.25 * rough)
        src = src + s["aperiodicity"] * (1 + 2 * rough) * a["aperiodic_gain"] * upper * hiss * g * (amp > 0)
    # A jet striking the teeth: a broad hump; other turbulence is broadband.
    sib = _lowpass(_highpass(white[1], a["sibilant_hz"]), a["sibilant_lowpass_hz"])
    flat = _highpass(white[2], a["flat_hz"])
    sib /= np.std(sib) + 1e-12
    flat /= np.std(flat) + 1e-12
    sibilance = lane("sibilance")
    fric = lane("frication") * (sibilance * sib + (1 - sibilance) * flat)
    # The release transient: a few ms of broadband noise at the release point.
    burst_noise = _highpass(white[3], a["flat_hz"])
    burst_noise /= np.std(burst_noise) + 1e-12
    for time, at, size, ms in k["bursts"]:
        # A few decay times of broadband noise: short at the lips, longest
        # behind the tongue body, whose release opens slowest.
        width = int(4 * ms / 1000 * SR)
        shape = np.exp(-np.arange(width) / (ms / 1000 * SR))
        i = int(round(time * SR))
        if 0 <= i and i + width < total:
            fric[i:i + width] += size * shape * burst_noise[i:i + width]
    count = v1.sections(c)
    live = k["obstacle"] > 0
    radiated = live & (k["source_at"] >= count)
    fric_at = np.where(live & ~radiated, k["source_at"], 0).astype(np.int64)
    at_lips = np.interp(n, k["t"], radiated.astype(float))
    out = v1._tract(src, fric * (1 - at_lips), fric_at, k["areas"], k["velum"], 48, **c["tract_params"])
    # Turbulence at the lips has no front cavity to shape it: weak and falling (diffuse).
    out += a["radiated_gain"] * sosfilt(butter(1, a["radiated_lowpass_hz"], fs=SR, output="sos"), fric * at_lips)
    out = sosfilt(butter(4, 10000, fs=SR, output="sos"), out)
    out = sosfilt(butter(1, 30, "highpass", fs=SR, output="sos"), out) * c["engine"]["output_gain"]
    out = out[round(lead * SR):][:samples]
    out = np.pad(out, (0, samples - len(out)))
    return (out, k) if return_controls else out


# ---------------------------------------------------------------- helpers

def lyric_rows(text, melody, lexicon=None, start=0.4):
    """Rows for `render` from hyphenated text and (pitch, beats) notes.

    text: words with hyphens between sung syllables ("Dai-sy Dai-sy");
    "_" continues the previous syllable onto the next note (a melisma).
    melody: (midi pitch, duration s[, velocity]) per note, with None pitch for a rest.
    The first note starts at `start` s, leaving room for its onset consonants.
    """
    tokens = []
    for word in text.split():
        if word == "_":
            tokens.append(None)
            continue
        parts = word.split("-")
        tokens += [(part, k < len(parts) - 1) for k, part in enumerate(parts)]
    events = [tok for tok in tokens if tok is not None]
    syllables = iter(phonetics.syllabify(events, lexicon))
    rows, t, it = [], start, iter(tokens)
    for note in melody:
        pitch, dur, *vel = note
        if pitch is None:
            t += dur
            continue
        tok = next(it)
        rows.append({"time": t, "off": t + dur * 0.97, "pitch": pitch, "velocity": vel[0] if vel else 80,
                     "syllable": None if tok is None else next(syllables)})
        t += dur
    return rows


# ---------------------------------------------------------------- ensembles

def scatter(rows, rng, timing_sd_s, pitch_sd_cents=6.0, bias_sd_s=0.008, bias_sd_cents=4.0, velocity_sd=4.0):
    """One singer's reading of a shared part: every note a little early or late,
    sharp or flat, louder or softer, around a bias of the singer's own.

    Choir singers on one part spread by about 10-30 ms in onset and 5-15 cents
    in pitch (Ternstrom 2003, "Choir acoustics: an overview of scientific
    research published to date", Int. J. Res. Choral Singing 1(1)). Legato
    stays legato: a note ends where the next one now begins.
    """
    bias_t, bias_c = rng.normal(0, bias_sd_s), rng.normal(0, bias_sd_cents)
    bias_v = rng.normal(0, velocity_sd)
    rows = sorted(rows, key=lambda r: r["time"])
    shifted = []
    for r in rows:
        dt = bias_t + rng.normal(0, timing_sd_s)
        shifted.append({**r, "time": max(0.0, r["time"] + dt), "off": max(0.0, r["off"] + dt),
                        "pitch": r["pitch"] + (bias_c + rng.normal(0, pitch_sd_cents)) / 100,
                        "velocity": float(np.clip(r["velocity"] + bias_v + rng.normal(0, velocity_sd / 2), 1, 127))})
    for i in range(len(shifted) - 1):
        a, b = rows[i], rows[i + 1]
        if b["time"] - a["off"] < 0.03:
            shifted[i]["off"] = shifted[i + 1]["time"] - (b["time"] - a["off"])
        shifted[i]["off"] = max(shifted[i]["off"], shifted[i]["time"] + 0.02)
    return shifted


def machine(rows, seconds, c, settings, midi_controls=(), context_end=None):
    """A singer through the machine voice (machine-voice.toml): pitch corrected
    to the scale, then vocoded, talkboxed or coloured, still singing the words."""
    from . import machine_voice
    capture = {}

    def correct(t, f0):
        capture["t"], capture["f0"] = t, machine_voice.correct_f0(t, f0, settings)
        return capture["f0"]

    corrected = render(rows, seconds, c, midi_controls=midi_controls, f0_map=correct, context_end=context_end)
    f0 = np.interp(np.arange(len(corrected)) / SR, capture["t"], capture["f0"])
    out = settings["mix"] * machine_voice.process(corrected, f0, settings, rows)
    if settings["mix"] < 1:
        out += (1 - settings["mix"]) * render(rows, seconds, c, midi_controls=midi_controls, context_end=context_end)
    return out


def segments(rows, gap_s=0.45, block_s=20.0):
    """Rows grouped into blocks of whole phrases (split at rests longer than
    gap_s, each block at most block_s long unless one phrase is longer)."""
    rows = sorted(rows, key=lambda r: r["time"])
    phrases = []
    for r in rows:
        if phrases and r["time"] - max(x["off"] for x in phrases[-1]) <= gap_s:
            phrases[-1].append(r)
        else:
            phrases.append([r])
    blocks = []
    for ph in phrases:
        if blocks and max(x["off"] for x in ph) - blocks[-1][0]["time"] <= block_s:
            blocks[-1] += ph
        else:
            blocks.append(list(ph))
    return blocks


def _render_blocks(rows, seconds, c, midi_controls, settings):
    """One singer's part in blocks of phrases, so memory follows the longest
    block rather than the piece. Each block has its own seed and knows where
    the previous one's singing ended."""
    out = np.zeros(round(seconds * SR))
    blocks = segments(rows)
    previous_end = None
    for i, block in enumerate(blocks):
        cb = c if len(blocks) == 1 else {**c, "engine": {**c["engine"], "seed": int(
            np.random.SeedSequence([c["engine"]["seed"], 101, i]).generate_state(1)[0] % 2**31)}}
        # Room before the block for its onset consonants and an inhalation (up to
        # about 0.9 s before the first note).
        t0 = max(0.0, block[0]["time"] - 1.2)
        t1 = min(seconds, max(r["off"] for r in block) + c["v3"]["engine"]["tail_seconds"] + 0.6)
        local = [{**r, "time": r["time"] - t0, "off": r["off"] - t0} for r in block]
        cc = [{**r, "time": r["time"] - t0} for r in midi_controls if t0 - 1.0 <= r["time"] <= t1 + 1.0]
        # Each controller holds its last value from before the block.
        for control in {r["control"] for r in midi_controls}:
            before = [r for r in midi_controls if r["control"] == control and r["time"] < t0 - 1.0]
            if before:
                cc.append({**max(before, key=lambda r: r["time"]), "time": -1.0})
        cc.sort(key=lambda r: r["time"])
        context = None if previous_end is None else previous_end - t0
        if settings is None:
            audio = render(local, t1 - t0, cb, midi_controls=cc, context_end=context)
        else:
            audio = machine(local, t1 - t0, cb, {**settings, "seed": cb["engine"]["seed"]}, cc, context)
        i0 = round(t0 * SR)
        n = min(len(audio), len(out) - i0)
        out[i0:i0 + n] += audio[:n]
        previous_end = max(r["off"] for r in block)
    return out


def sing(rows, singers, seconds=None, seed=1892, midi_controls=(), spread=0.5, overrides=None, scatter_scale=1.0,
         machine_settings=None):
    """A part sung by one or more singers, as stereo (N, 2).

    singers: preset names; one name is a soloist, several a section singing in
    unison, each with its own seed (vibrato, flutter, jitter differ) and its
    own scatter of the shared notes, spread evenly across `spread` of the
    stereo field. Levels sum by power, so a section is as loud as a soloist.
    machine_settings: a validated machine-voice configuration every singer
    passes through (see machine()).
    """
    if not singers:
        raise ValueError("a voice3 part needs at least one singer")
    overrides = overrides or {}
    if seconds is None:
        seconds = max((r["off"] for r in rows), default=0.0) + _table()["engine"]["tail_seconds"]
    out = np.zeros((round(seconds * SR), 2), np.float32)
    count = len(singers)
    for i, name in enumerate(singers):
        c = load_config(name, seed=int(np.random.SeedSequence([seed, i]).generate_state(1)[0] % 2**31), **overrides)
        mine = rows
        if c["character"]["vibrato"] == "emerging" and rows:
            start, end = rows[0]["time"], rows[-1]["time"]
            mine = [{**r, "emerging_progress": r.get("emerging_progress",
                     float(np.clip((r["time"] - start) / max(end - start, 0.001), 0, 1)))} for r in rows]
        if count > 1:
            rng = np.random.default_rng(np.random.SeedSequence([seed, i, 7]))
            mine = scatter(mine, rng, c["singer"]["timing_sd_s"] * scatter_scale,
                           pitch_sd_cents=6.0 * scatter_scale, bias_sd_s=0.008 * scatter_scale,
                           bias_sd_cents=4.0 * scatter_scale)
        audio = _render_blocks(mine, seconds, c, midi_controls, machine_settings)
        pan = 0.0 if count == 1 else spread * (i / (count - 1) - 0.5)
        angle = (pan + 1) * math.pi / 4
        gain = math.sqrt(2) / math.sqrt(count)
        out[:, 0] += gain * math.cos(angle) * audio
        out[:, 1] += gain * math.sin(angle) * audio
    return out
