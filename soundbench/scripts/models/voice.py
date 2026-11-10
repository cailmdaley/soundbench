"""A Kelly-Lochbaum singing voice: an LF glottal source driving a lossy
scattering-junction vocal tract with a nasal branch at the velum.

Every sample comes from the model's equations. The pipeline is

    syllables + notes --plan--> control tracks (1 ms) --source--> glottal flow
    derivative --tract--> lip + nostril + wall radiation --> mono audio.

**Tract.** N cylindrical sections, N = round(length * sample_rate / c), each a
one-sample delay in each direction (24 sections for a 17.5 cm tract at 48 kHz).
The waveguide's response is therefore periodic in z^-2: the band 0-12 kHz is
the voice and the band above it is its mirror image, removed by the output
low-pass. Waves are pressure waves; a junction between areas A1 and A2 has
reflection r = (A1 - A2)/(A1 + A2) (one-multiply form). The glottis reflects
with a constant coefficient; the lips with a frequency-dependent coefficient
(-g times a one-pole low-pass), so low frequencies reflect as at an open end
and high frequencies radiate. Radiated output is the lip volume velocity,
A_lips * (p+ - p-); since the source is the flow *derivative*, the 6 dB/oct
radiation characteristic is already in the source.

**Nasal branch.** A fixed-area nasal tube joins the oral tract at the velum
through a three-port (parallel) junction whose port area is the velum opening:
P = 2 * sum(A_i p_i_in) / sum(A_i), p_i_out = P - p_i_in. With the velum
closed this reduces exactly to the two-port junction.

**Source.** A Liljencrants-Fant (LF) flow-derivative pulse, tabulated per vocal
effort (tp, te, ta as fractions of the period), read by a phase accumulator at
4x oversampling and decimated. Jitter and shimmer perturb each period;
aspiration is noise modulated by the glottal flow.

**Aerodynamics (consonants).** At each control step the narrowest oral section
A_c and the glottal area A_g form two orifices in series, giving the
steady oral pressure P_ss = P_s A_g^2 / (A_g^2 + A_c^2); the oral pressure P_o
relaxes towards it with a time constant. Turbulence noise of amplitude
proportional to sqrt(A_c) * P_o^1.5 is injected just downstream of the
constriction, while a consonant supplies an obstacle to the jet (its window
and noise_gain); a vowel's narrow lips or palate make no turbulence. A closure builds P_o with no noise; its release makes the
burst. A held narrow constriction makes a fricative. Phonation weakens as P_o
approaches P_s (the transglottal pressure falls), which gives voiced stops
their quiet voice bar, radiated by a low-passed wall term.

**Area functions.** Story (2008), J. Acoust. Soc. Am. 123(1):327-335, Table I:
44-section MRI area functions (cm^2, glottis to lips) of the same adult male
speaker as Story, Titze & Hoffman (1996), imaged in 2002. Each is resampled to
the model's N sections over its own length, so every vowel keeps its shape and
takes the configured tract length; formants therefore scale by
VTL_vowel / length. The schwa is the mean of the eleven shapes; the
r-coloured schwa and the glides are constructed in voice.toml (not measured).
"""
from __future__ import annotations

from functools import lru_cache
import math
from pathlib import Path
import re
import tomllib

import numpy as np
from numba import njit
from scipy.optimize import brentq
from scipy.signal import butter, lfilter, resample_poly, sosfilt

CONFIG = Path(__file__).resolve().parents[2] / "instruments/models/voice.toml"

# Story (2008) Table I, 2002 data. Section 1 is the glottis, 44 the lips.
STORY_DELTA_CM = {"iy": 0.384, "ih": 0.376, "ey": 0.386, "eh": 0.393, "ae": 0.366,
                  "ah": 0.390, "aa": 0.388, "ao": 0.395, "ow": 0.417, "uh": 0.440,
                  "uw": 0.445}
STORY_AREAS = {
    "iy": [0.51, 0.59, 0.62, 0.72, 1.24, 2.30, 3.30, 3.59, 3.22, 2.86, 3.00, 3.61, 4.39, 4.95, 5.17,
           5.16, 5.18, 5.26, 5.20, 5.02, 4.71, 4.13, 3.43, 2.83, 2.32, 1.83, 1.46, 1.23, 1.08, 0.94,
           0.80, 0.67, 0.55, 0.46, 0.40, 0.36, 0.35, 0.35, 0.38, 0.51, 0.74, 0.92, 0.96, 0.91],
    "ih": [0.28, 0.21, 0.21, 0.30, 0.47, 0.71, 1.12, 1.48, 1.35, 1.05, 0.92, 0.92, 1.19, 1.94, 2.83,
           3.31, 3.48, 3.60, 3.64, 3.49, 3.20, 2.90, 2.59, 2.21, 1.87, 1.54, 1.20, 0.92, 0.74, 0.59,
           0.52, 0.54, 0.59, 0.65, 0.71, 0.67, 0.61, 0.57, 0.50, 0.48, 0.54, 0.73, 0.93, 0.82],
    "ey": [0.29, 0.26, 0.30, 0.40, 0.55, 0.74, 0.99, 1.09, 0.90, 0.69, 0.77, 1.31, 2.13, 2.74, 3.03,
           3.23, 3.33, 3.27, 3.09, 2.84, 2.66, 2.46, 2.14, 1.79, 1.44, 1.17, 1.00, 0.88, 0.80, 0.81,
           0.85, 0.84, 0.86, 0.96, 1.18, 1.35, 1.48, 1.62, 1.49, 1.29, 1.24, 1.17, 1.04, 0.95],
    "eh": [0.37, 0.30, 0.24, 0.23, 0.29, 0.41, 0.58, 0.82, 0.97, 0.82, 0.62, 0.54, 0.54, 0.68, 1.09,
           1.62, 2.03, 2.35, 2.51, 2.39, 2.22, 2.13, 2.00, 1.78, 1.58, 1.43, 1.31, 1.23, 1.24, 1.38,
           1.61, 1.82, 1.96, 2.01, 2.00, 1.95, 1.77, 1.48, 1.30, 1.21, 1.10, 0.99, 0.91, 0.78],
    "ae": [0.31, 0.21, 0.18, 0.23, 0.33, 0.50, 0.78, 0.96, 0.85, 0.63, 0.46, 0.36, 0.33, 0.46, 0.73,
           1.00, 1.30, 1.66, 1.97, 2.06, 2.03, 2.01, 1.89, 1.66, 1.49, 1.42, 1.37, 1.34, 1.41, 1.58,
           1.82, 2.19, 2.63, 2.97, 3.17, 3.40, 3.56, 3.57, 3.58, 3.44, 3.15, 3.38, 3.99, 4.17],
    "ah": [0.23, 0.34, 0.47, 0.60, 0.77, 1.06, 1.26, 1.09, 0.80, 0.65, 0.56, 0.47, 0.37, 0.24, 0.17,
           0.18, 0.23, 0.27, 0.28, 0.29, 0.33, 0.52, 0.97, 1.50, 1.91, 2.23, 2.65, 3.29, 4.13, 5.00,
           5.77, 6.33, 6.61, 6.63, 6.45, 6.04, 5.39, 4.42, 3.29, 2.37, 1.74, 1.36, 1.17, 0.99],
    "aa": [0.56, 0.62, 0.66, 0.78, 0.97, 1.16, 1.12, 0.82, 0.55, 0.45, 0.37, 0.29, 0.21, 0.15, 0.16,
           0.25, 0.34, 0.43, 0.54, 0.61, 0.67, 0.98, 1.76, 2.75, 3.52, 4.08, 4.74, 5.61, 6.60, 7.61,
           8.48, 9.06, 9.29, 9.26, 9.06, 8.64, 7.91, 6.98, 6.02, 5.13, 4.55, 4.52, 4.71, 4.72],
    "ao": [0.27, 0.43, 0.54, 0.67, 0.83, 0.92, 0.89, 0.73, 0.55, 0.44, 0.37, 0.28, 0.18, 0.14, 0.15,
           0.14, 0.13, 0.14, 0.18, 0.20, 0.23, 0.49, 1.03, 1.58, 2.06, 2.61, 3.35, 4.34, 5.51, 6.70,
           7.75, 8.63, 9.29, 9.59, 9.42, 8.78, 7.82, 6.50, 4.95, 3.47, 2.15, 1.38, 1.11, 0.90],
    "ow": [0.38, 0.45, 0.57, 0.77, 1.31, 1.92, 1.74, 1.11, 0.75, 0.59, 0.57, 0.68, 0.73, 0.67, 0.58,
           0.49, 0.44, 0.42, 0.49, 0.53, 0.38, 0.30, 0.45, 0.61, 0.71, 0.79, 0.86, 1.01, 1.41, 2.09,
           3.00, 4.10, 5.16, 6.22, 7.34, 8.15, 8.61, 8.37, 6.76, 4.37, 2.30, 1.06, 0.58, 0.47],
    "uh": [0.37, 0.38, 0.49, 0.62, 0.85, 1.28, 1.62, 1.47, 1.04, 0.81, 1.03, 1.44, 1.49, 1.28, 1.06,
           0.85, 0.69, 0.56, 0.42, 0.27, 0.27, 0.41, 0.51, 0.49, 0.47, 0.50, 0.58, 0.80, 1.19, 1.62,
           2.27, 3.24, 4.16, 5.00, 5.70, 6.11, 6.21, 6.29, 6.24, 4.91, 2.61, 1.09, 0.63, 0.59],
    "uw": [0.54, 0.61, 0.66, 0.75, 1.13, 1.99, 2.83, 2.90, 2.52, 2.40, 2.83, 3.56, 3.99, 3.89, 3.50,
           3.04, 2.64, 2.44, 2.31, 2.07, 1.80, 1.52, 1.14, 0.74, 0.42, 0.22, 0.14, 0.20, 0.47, 0.89,
           1.15, 1.42, 2.17, 3.04, 3.69, 4.70, 5.74, 5.41, 3.82, 2.34, 1.35, 0.65, 0.29, 0.16],
}
# Story (2008) Table II: natural-speech means (N) and the formants Story
# computed from these area functions (C), in Hz, at each vowel's own length.
STORY_FORMANTS_N = {"iy": (295, 1923, 2797), "ih": (478, 1965, 2688), "ey": (509, 1917, 2617),
                    "eh": (659, 1740, 2485), "ae": (765, 1724, 2501), "ah": (752, 1300, 2698),
                    "aa": (715, 1180, 2691), "ao": (665, 932, 2814), "ow": (563, 835, 2637),
                    "uh": (518, 947, 2355), "uw": (430, 917, 2087)}
STORY_FORMANTS_C = {"iy": (325, 2139, 2976), "ih": (413, 2103, 2626), "ey": (484, 1887, 2404),
                    "eh": (565, 1595, 2200), "ae": (810, 1655, 2310), "ah": (593, 836, 3099),
                    "aa": (694, 942, 2948), "ao": (557, 709, 3176), "ow": (499, 768, 2421),
                    "uh": (477, 789, 2499), "uw": (314, 702, 2298)}
IPA = {"iy": "i", "ih": "ɪ", "ey": "e", "eh": "ɛ", "ae": "æ", "ah": "ʌ", "aa": "ɑ", "ao": "ɔ",
       "ow": "o", "uh": "ʊ", "uw": "u", "ax": "ə", "er": "ɚ", "y": "j", "w": "w"}

# Sung syllables (ARPAbet-like names; IPA above). Each entry is
# (onset, nucleus, coda): onset/coda list consonants and glides; the nucleus
# lists vowel:position keyframes across the vowel's span (0 start, 1 end).
LEXICON = {
    "dai": ("d", "ey:0 ey:.55 iy:1", ""), "sy": ("z", "iy:0", ""),
    "give": ("g", "ih:0", "v"), "me": ("m", "iy:0", ""),
    "your": ("y", "uh:0 uh:.45 er:1", ""), "an": ("", "ae:0", "n"),
    "swer": ("s", "er:0", ""), "do": ("d", "uw:0", ""),
    # A few common syllables for testing.
    "day": ("d", "ey:0 ey:.55 iy:1", ""), "see": ("s", "iy:0", ""), "zoo": ("z", "uw:0", ""),
    "my": ("m", "aa:0 aa:.55 ih:1", ""), "no": ("n", "ow:0 ow:.6 uh:1", ""),
    "we": ("w", "iy:0", ""), "go": ("g", "ow:0 ow:.6 uh:1", ""), "mama": ("m", "aa:0", ""),
    "dim": ("d", "ih:0", "m"), "dan": ("d", "ae:0", "n"), "news": ("n", "uw:0", "z"),
    "a": ("", "ax:0", ""), "the": ("d", "ax:0", ""),
    # Sustained vowels, by name.
    "ee": ("", "iy:0", ""), "ih": ("", "ih:0", ""), "ei": ("", "ey:0", ""),
    "eh": ("", "eh:0", ""), "ae": ("", "ae:0", ""), "uh": ("", "ax:0", ""),
    "er": ("", "er:0", ""), "oo": ("", "uw:0", ""), "ah": ("", "aa:0", ""),
    "aw": ("", "ao:0", ""), "oh": ("", "ow:0", ""), "ay": ("", "ey:0 ey:.55 iy:1", ""),
    "mm": ("", "ax:0", "m"), "nn": ("", "ax:0", "n"),
    # Humming: lips closed and the velum open for the whole syllable (see plan).
    "hm": ("", "ax:0", ""),
}
HUMS = frozenset(("hm",))
STOPS, FRICATIVES, NASALS, GLIDES = "dg", "zsv", "mn", "yw"


def syllable_key(text):
    """Lyric text to a lexicon key: case, punctuation and hyphens removed."""
    key = re.sub(r"[^a-z]", "", text.casefold())
    if key not in LEXICON:
        raise ValueError(f"no pronunciation for lyric syllable {text!r}; add it to LEXICON in scripts/models/voice.py")
    return key


def parse_syllable(text):
    onset, nucleus, coda = LEXICON[syllable_key(text)]
    frames = []
    for item in nucleus.split():
        vowel, at = item.split(":")
        frames.append((vowel, float(at)))
    return list(onset), frames, list(coda)


# ---------------------------------------------------------------- configuration

SCHEMA = {
    "engine": "sample_rate seed control_ms oversample tail_seconds lead_seconds output_gain output_lowpass_hz",
    "tract": "length_cm speed_of_sound_cm_s air_density_g_cm3 wall_mass_g_cm2 wall_resistance_cgs glottal_reflection lip_reflection lip_radiation_hz wall_loss min_area_cm2 wall_radiation wall_radiation_hz",
    "nasal": "length_cm velum_from_glottis_cm areas_cm2 loss nostril_reflection nostril_radiation_hz coupling_cm2 open_ms close_ms",
    "source": "soft_lf loud_lf effort_velocity velocity_db jitter shimmer breathiness aspiration_hz",
    "expression": "vibrato_hz vibrato_cents vibrato_delay_s vibrato_rise_s vibrato_wander vibrato_amplitude portamento_s legato_gap_s scoop_cents scoop_s attack_s release_s release_droop_cents note_arc_db onset_after_beat min_vowel_fraction",
    "aero": "subglottal_pressure glottal_area_cm2 abduction_cm2 pressure_tau_ms noise_gain aspiration_gain turbulence_max_area_cm2 voicing_pressure_loss breath_lead_s",
    "constructed": "",
    "consonants": "",
}
CONSONANT_KEYS = set("place_cm width_cm area_cm2 duration_ms in_ms out_ms voiced noise_gain noise_hz noise_q".split())


def load_config(path=CONFIG):
    config = tomllib.loads(Path(path).read_text())
    if set(config) != set(SCHEMA):
        raise ValueError(f"voice model tables must be {sorted(SCHEMA)}")
    for name, keys in SCHEMA.items():
        if keys and set(config[name]) != set(keys.split()):
            raise ValueError(f"[{name}] parameters must be {keys}")
    if set(config["consonants"]) != set(STOPS + FRICATIVES + NASALS):
        raise ValueError(f"[consonants] needs tables {sorted(STOPS + FRICATIVES + NASALS)}")
    for name, table in config["consonants"].items():
        if set(table) != CONSONANT_KEYS:
            raise ValueError(f"[consonants.{name}] parameters must be {sorted(CONSONANT_KEYS)}")
    if set(config["constructed"]) != {"er", "y", "w"}:
        raise ValueError("[constructed] needs er, y and w")
    for name, spec in config["constructed"].items():
        if set(spec) != {"base", "constrictions", "duration_ms"} or spec["base"] not in {*STORY_AREAS, "ax"}:
            raise ValueError(f"[constructed.{name}] needs base (a Story vowel or ax), constrictions, duration_ms")
    e, t, n, s, x, a = (config[k] for k in ("engine", "tract", "nasal", "source", "expression", "aero"))
    if e["sample_rate"] != 48000 or not isinstance(e["oversample"], int) or e["oversample"] < 1:
        raise ValueError("voice uses 48000 Hz and an integer oversample factor")
    if not isinstance(e["seed"], int) or e["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not 0 < e["output_lowpass_hz"] < e["sample_rate"] / 4:
        raise ValueError("output_lowpass_hz must lie below the tract's mirror frequency (sample_rate/4)")
    if not 10 <= t["length_cm"] <= 22:
        raise ValueError("tract length must be 10..22 cm")
    for key in ("glottal_reflection", "lip_reflection", "nostril_reflection"):
        value = t.get(key, n.get(key))
        if not 0 <= value < 1:
            raise ValueError(f"{key} must be 0..1")
    if not 0.9 < t["wall_loss"] <= 1 or not 0.9 < n["loss"] <= 1:
        raise ValueError("per-section losses must be 0.9..1")
    if not 2 <= n["velum_from_glottis_cm"] <= t["length_cm"] - 2:
        raise ValueError("velum must lie inside the tract")
    if min(n["areas_cm2"]) <= 0 or t["min_area_cm2"] <= 0:
        raise ValueError("areas must be positive")
    for key in ("soft_lf", "loud_lf"):
        tp, te, ta = s[key]
        if not 0 < tp < te < 1 or not 0 < ta < 1 - te:
            raise ValueError(f"{key} must satisfy 0 < tp < te < 1 and 0 < ta < 1 - te")
    if len(s["effort_velocity"]) != 2 or s["effort_velocity"][0] >= s["effort_velocity"][1]:
        raise ValueError("effort_velocity is [soft, loud] with soft < loud")
    if x["vibrato_cents"] < 0 or x["vibrato_hz"] <= 0 or x["legato_gap_s"] < 0:
        raise ValueError("invalid vibrato or legato parameters")
    if not 0 < x["min_vowel_fraction"] < 1 or not 0 <= x["onset_after_beat"] <= 1:
        raise ValueError("min_vowel_fraction and onset_after_beat are fractions")
    if a["pressure_tau_ms"] <= 0 or a["glottal_area_cm2"] <= 0:
        raise ValueError("invalid aerodynamic parameters")
    for table in [e, t, n, s, x, a, *config["consonants"].values(),
                  *({"c": v["constrictions"], "d": v["duration_ms"]} for v in config["constructed"].values())]:
        for value in table.values():
            values = [v for row in value for v in (row if isinstance(row, list) else [row])] if isinstance(value, list) else [value]
            if not all(isinstance(v, bool) or (type(v) in (int, float) and math.isfinite(v)) for v in values):
                raise ValueError("voice parameters must be finite numbers")
    return config


# ---------------------------------------------------------------- geometry

def sections(config):
    t = config["tract"]
    return max(8, round(t["length_cm"] * config["engine"]["sample_rate"] / t["speed_of_sound_cm_s"]))


def section_length_cm(config):
    return config["tract"]["speed_of_sound_cm_s"] / config["engine"]["sample_rate"]


def resample_area(areas, count):
    """Average a piecewise-constant area function into `count` equal sections."""
    areas = np.asarray(areas, float)
    edges = np.linspace(0, len(areas), count + 1)
    cumulative = np.concatenate([[0], np.cumsum(areas)])
    integral = np.interp(edges, np.arange(len(areas) + 1), cumulative)
    return np.diff(integral) / np.diff(edges)


def constrict(base, count, places, section_cm):
    """Apply [distance-from-lips cm, width cm, area cm^2] constrictions to a shape."""
    area = np.array(base, float)
    x = (count - 0.5 - np.arange(count)) * section_cm   # section centres, from the lips
    for place, width, target in places:
        weight = spatial_weight(x, place, width)
        area = area * (1 - weight) + target * weight
    return area


def spatial_weight(x, place, width):
    """1 within width/2 of place, cosine taper to 0 over a further width/2."""
    d = np.abs(x - place) - width / 2
    return np.where(d <= 0, 1.0, np.where(d < width / 2, 0.5 * (1 + np.cos(np.pi * d / (width / 2 + 1e-12))), 0.0))


def vowel_shapes(config):
    """Area vectors (cm^2, glottis to lips) for every vowel and glide, at N sections."""
    count = sections(config)
    shapes = {v: resample_area(a, count) for v, a in STORY_AREAS.items()}
    shapes["ax"] = resample_area(np.mean([STORY_AREAS[v] for v in STORY_AREAS], axis=0), count)
    section_cm = section_length_cm(config)
    for name, spec in config["constructed"].items():
        shapes[name] = constrict(shapes[spec["base"]], count, spec["constrictions"], section_cm)
    return shapes


def story_vtl(vowel):
    return 44 * STORY_DELTA_CM[vowel]


def chain_formants(area, length_cm, config=None, count=3, fmax=5000.0):
    """Formants of the area function as a transmission line, by chain matrices.

    Each section is a uniform line with series inertance rho/A and shunt
    admittance (air compliance plus the yielding wall of voice.toml, if a
    config is given). The glottis is closed. Without a config the lips are an
    ideal open end; with one, they are loaded by the impedance equivalent to
    the waveguide's lip reflection R(f): Z = Z_lips (1 + R)/(1 - R).
    Formants are the peaks of |U_lips / U_glottis| = 1/|K21 Z + K22|, refined to 1 Hz.
    """
    area = np.asarray(area, float)
    dx = length_cm / len(area)
    c = 35000.0 if config is None else config["tract"]["speed_of_sound_cm_s"]
    rho = 0.00114 if config is None else config["tract"]["air_density_g_cm3"]

    def gain(f):
        w = 2 * np.pi * f
        m = np.eye(2, dtype=complex)
        for a in area:
            z = 1j * w * rho / a
            y = 1j * w * a / (rho * c * c)
            if config is not None:
                t = config["tract"]
                y = y + 2 * math.sqrt(math.pi * a) / (t["wall_resistance_cgs"] + 1j * w * t["wall_mass_g_cm2"])
            gamma, z0 = np.sqrt(z * y), np.sqrt(z / y)
            ch, sh = np.cosh(gamma * dx), np.sinh(gamma * dx)
            m = m @ np.array([[ch, z0 * sh], [sh / z0, ch]])
        if config is None:
            return 1 / abs(m[1, 1])
        t = config["tract"]
        a = _one_pole(t["lip_radiation_hz"], config["engine"]["sample_rate"])
        zinv = np.exp(-1j * w / config["engine"]["sample_rate"])
        r = -t["lip_reflection"] * a / (1 - (1 - a) * zinv)
        load = rho * c / area[-1] * (1 + r) / (1 - r)
        return 1 / abs(m[1, 0] * load + m[1, 1])

    grid = np.arange(60.0, fmax, 4.0)
    values = np.array([gain(f) for f in grid])
    peaks = [i for i in range(1, len(grid) - 1) if values[i] > values[i - 1] and values[i] >= values[i + 1]]
    out = []
    for i in peaks[:count]:
        fine = np.arange(grid[i - 1], grid[i + 1], 1.0)
        out.append(float(fine[np.argmax([gain(f) for f in fine])]))
    return out


# ---------------------------------------------------------------- source

def _lf_pulse(tp, te, ta, points):
    """One LF flow-derivative period (T0 = 1, Ee = 1) and its normalized flow."""
    eps = brentq(lambda e: e * ta - 1 + math.exp(-e * (1 - te)), 1e-6 / ta + 1e-9, 1 / ta * (1 + 1e-9) + 50)
    wg = math.pi / tp
    tc = 1 - te
    returned = -(1 / (eps * ta)) * ((1 - math.exp(-eps * tc)) / eps - tc * math.exp(-eps * tc))
    fine = np.linspace(0, te, 20001)

    def opened(alpha):
        e0 = -1 / (math.exp(alpha * te) * math.sin(wg * te))
        y = e0 * np.exp(alpha * fine) * np.sin(wg * fine)
        return np.trapezoid(y, fine)

    alpha = brentq(lambda a: opened(a) + returned, -200, 200)
    t = np.arange(points) / points
    e0 = -1 / (math.exp(alpha * te) * math.sin(wg * te))
    out = np.where(t <= te, e0 * np.exp(alpha * t) * np.sin(wg * t),
                   -(1 / (eps * ta)) * (np.exp(-eps * (t - te)) - math.exp(-eps * tc)))
    flow = np.cumsum(out) / points
    flow -= flow.min()
    return out, flow / flow.max()


@lru_cache(maxsize=8)
def lf_tables(soft, loud, rows=9, points=4096):
    """LF pulses from soft (effort 0) to loud (effort 1), interpolating tp, te, ta."""
    derivative, flow = np.zeros((rows, points)), np.zeros((rows, points))
    for i, effort in enumerate(np.linspace(0, 1, rows)):
        tp, te, ta = (s + effort * (l - s) for s, l in zip(soft, loud))
        derivative[i], flow[i] = _lf_pulse(tp, te, ta, points)
    return derivative, flow


@njit(cache=True)
def _glottis(f0, amplitude, effort, jitter, shimmer, table, flow_table, oversample, sr):
    """Phase-accumulated LF pulses at oversample*sr; period-level jitter/shimmer/effort."""
    rows, points = table.shape
    samples = len(f0) * oversample
    out = np.zeros(samples, np.float32)
    flow = np.zeros(samples, np.float32)
    phase = 0.0
    period = 0
    j = jitter[0]
    s = shimmer[0]
    row = effort[0] * (rows - 1)
    for m in range(samples):
        n = m // oversample
        phase += f0[n] * (1.0 + j) / (sr * oversample)
        if phase >= 1.0:
            phase -= 1.0
            period += 1
            k = period % len(jitter)
            j, s = jitter[k], shimmer[k]
            row = min(max(effort[n], 0.0), 1.0) * (rows - 1)
        x = phase * points
        i = int(x)
        frac = x - i
        i1 = (i + 1) % points
        r0 = int(row)
        r1 = min(r0 + 1, rows - 1)
        w = row - r0
        d0 = table[r0, i] + frac * (table[r0, i1] - table[r0, i])
        d1 = table[r1, i] + frac * (table[r1, i1] - table[r1, i])
        g0 = flow_table[r0, i] + frac * (flow_table[r0, i1] - flow_table[r0, i])
        g1 = flow_table[r1, i] + frac * (flow_table[r1, i1] - flow_table[r1, i])
        a = amplitude[n] * (1.0 + s)
        out[m] = a * (d0 + w * (d1 - d0))
        flow[m] = a * (g0 + w * (g1 - g0))
    return out, flow


# ---------------------------------------------------------------- tract

@njit(cache=True)
def _tract(src, fric, fric_at, areas, velum, step, nasal_areas, velum_at,
           r_glottis, lip_gain, lip_a, nose_gain, nose_a, wall_loss, nasal_loss,
           wall_gain, wall_a, min_area, wall_mass, wall_resistance, rho_c, dx, sr,
           sinus_hz=500.0, sinus_area=0.0, sinus_q=2.0, sinus_at=3,
           nasal_wall_hz=0.0):
    count = areas.shape[1]
    nn = len(nasal_areas)
    controls = areas.shape[0]
    samples = len(src)
    R = np.zeros(count)
    L = np.zeros(count)
    inR = np.zeros(count)
    inL = np.zeros(count)
    Rn = np.zeros(nn)
    Ln = np.zeros(nn)
    inRn = np.zeros(nn)
    inLn = np.zeros(nn)
    A = np.zeros(count)
    An = nasal_areas.copy()
    nasal_R_lp = np.zeros(nn)
    nasal_L_lp = np.zeros(nn)
    nasal_alpha = 1 - math.exp(-2 * math.pi * nasal_wall_hz / sr)
    wall_u = np.zeros(count)
    wall_p = np.zeros(count)
    out = np.zeros(samples)
    lip_lp = 0.0
    nose_lp = 0.0
    wall_lp = 0.0
    sinus_u = 0.0
    sinus_p = 0.0
    sinus_last = 0.0
    # Series inertance/resistance feeding a cavity compliance, normalized
    # to the waveguide's area admittance. Trapezoidal companion circuit.
    omega = 2 * math.pi * sinus_hz
    sinus_mass = 1.0 / max(sinus_area * omega, 1e-12)
    sinus_compliance = max(sinus_area, 1e-12) / omega
    sinus_resistance = sinus_mass * omega / sinus_q
    sinus_den = sinus_mass * sr + sinus_resistance / 2 + 1 / (4 * sinus_compliance * sr)
    for n in range(samples):
        c = n // step
        frac = (n - c * step) / step
        c0 = min(c, controls - 1)
        c1 = min(c + 1, controls - 1)
        for k in range(count):
            A[k] = max(areas[c0, k] + frac * (areas[c1, k] - areas[c0, k]), min_area)
        port = velum[c0] + frac * (velum[c1] - velum[c0])
        An[0] = max(port, 0.0)
        # Glottis.
        inR[0] = r_glottis * L[0] + src[n]
        # Each oral junction is a parallel junction with a yielding-wall shunt
        # (mass and resistance per unit wall area, trapezoidal companion
        # model); the velum junction adds the nasal port as a third branch.
        for k in range(1, count):
            surface = (math.sqrt(math.pi * A[k - 1]) + math.sqrt(math.pi * A[k])) * dx
            mass = wall_mass / (surface * rho_c)
            resistance = wall_resistance / (surface * rho_c)
            den = mass * sr + 0.5 * resistance
            g = 0.5 / den
            h = (wall_u[k] * (mass * sr - 0.5 * resistance) + 0.5 * wall_p[k]) / den
            total = A[k - 1] + A[k] + g
            drive = A[k - 1] * R[k - 1] + A[k] * L[k]
            if k == velum_at:
                total += An[0]
                drive += An[0] * Ln[0]
            p = (2.0 * drive - h) / total
            wall_u[k] = g * p + h
            wall_p[k] = p
            inR[k] = p - L[k]
            inL[k - 1] = p - R[k - 1]
            if k == velum_at:
                inRn[0] = p - Ln[0]
        # Lips: open-end reflection whose magnitude falls with frequency.
        lip_lp += lip_a * (R[count - 1] - lip_lp)
        inL[count - 1] = -lip_gain * lip_lp
        lips = A[count - 1] * (R[count - 1] - inL[count - 1])
        # Nasal junctions and nostrils.
        for k in range(1, nn):
            a0 = max(An[k - 1], min_area)
            r = (a0 - An[k]) / (a0 + An[k])
            if sinus_area > 0 and k == sinus_at:
                g = 0.5 / sinus_den
                h = (sinus_u * (sinus_mass * sr - sinus_resistance / 2
                     - 1 / (4 * sinus_compliance * sr)) + sinus_last / 2 - sinus_p) / sinus_den
                p = (2 * (a0 * Rn[k - 1] + An[k] * Ln[k]) - h) / (a0 + An[k] + g)
                u = g * p + h
                sinus_p += (u + sinus_u) / (2 * sinus_compliance * sr)
                sinus_u, sinus_last = u, p
                inRn[k] = p - Ln[k]
                inLn[k - 1] = p - Rn[k - 1]
            else:
                w = r * (Rn[k - 1] - Ln[k])
                inRn[k] = Rn[k - 1] + w
                inLn[k - 1] = Ln[k] + w
        nose_lp += nose_a * (Rn[nn - 1] - nose_lp)
        inLn[nn - 1] = -nose_gain * nose_lp
        nose = An[nn - 1] * (Rn[nn - 1] - inLn[nn - 1])
        # Turbulence: a pressure dipole just downstream of the constriction.
        k = fric_at[c0]
        if k > 0:
            inR[k] += 0.5 * fric[n]
            inL[k - 1] -= 0.5 * fric[n]
        # Each section is a one-sample delay in each direction.
        for k in range(count):
            R[k] = inR[k] * wall_loss
            L[k] = inL[k] * wall_loss
        for k in range(nn):
            if nasal_wall_hz > 0:
                nasal_R_lp[k] += nasal_alpha * (inRn[k] - nasal_R_lp[k])
                nasal_L_lp[k] += nasal_alpha * (inLn[k] - nasal_L_lp[k])
                Rn[k] = nasal_R_lp[k] * nasal_loss
                Ln[k] = nasal_L_lp[k] * nasal_loss
            else:
                Rn[k] = inRn[k] * nasal_loss
                Ln[k] = inLn[k] * nasal_loss
        # Wall vibration radiates the low-passed pharyngeal pressure.
        wall_lp += wall_a * ((R[2] + L[2]) - wall_lp)
        out[n] = lips + nose + wall_gain * wall_lp
    return out


def _one_pole(hz, sr):
    return 1 - math.exp(-2 * math.pi * hz / sr)


def tract_parameters(config):
    sr = config["engine"]["sample_rate"]
    t, n = config["tract"], config["nasal"]
    count = sections(config)
    section_cm = section_length_cm(config)
    nn = max(4, round(n["length_cm"] / section_cm))
    nasal = resample_area(n["areas_cm2"], nn)
    velum_at = min(count - 2, max(2, round(n["velum_from_glottis_cm"] / section_cm)))
    return dict(nasal_areas=nasal, velum_at=velum_at, r_glottis=t["glottal_reflection"],
                lip_gain=t["lip_reflection"], lip_a=_one_pole(t["lip_radiation_hz"], sr),
                nose_gain=n["nostril_reflection"], nose_a=_one_pole(n["nostril_radiation_hz"], sr),
                wall_loss=t["wall_loss"], nasal_loss=n["loss"], wall_gain=t["wall_radiation"],
                wall_a=_one_pole(t["wall_radiation_hz"], sr), min_area=t["min_area_cm2"],
                wall_mass=t["wall_mass_g_cm2"], wall_resistance=t["wall_resistance_cgs"],
                rho_c=t["air_density_g_cm3"] * t["speed_of_sound_cm_s"], dx=section_cm, sr=float(sr))


def impulse_response(area, velum=0.0, seconds=0.25, config=None):
    """The static tract's response at the lips + nostrils to a glottal impulse."""
    c = config or load_config()
    sr = c["engine"]["sample_rate"]
    samples = round(seconds * sr)
    src = np.zeros(samples)
    src[0] = 1.0
    areas = np.asarray(area, float)[None, :].repeat(2, axis=0)
    p = tract_parameters(c)
    return _tract(src, np.zeros(samples), np.zeros(2, np.int64), areas,
                  np.full(2, float(velum)), samples, **p)


# ---------------------------------------------------------------- planning

def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def _window(t, start, end, rise, fall):
    """1 on [start, end], cosine ramps of `rise` before and `fall` after."""
    return smoothstep((t - start + rise) / max(rise, 1e-6)) * smoothstep((end + fall - t) / max(fall, 1e-6))


def plan(notes, config):
    """Turn sung notes into syllable timings.

    notes: dicts with time, off (seconds), pitch (MIDI, may be fractional),
    velocity and syllable (lyric text, or None to continue the previous
    syllable as a melisma). Returns units (one per syllable) grouped into
    phrases, with their consonant events and vowel keyframes.
    """
    x = config["expression"]
    rows = sorted(notes, key=lambda r: r["time"])
    units = []
    for row in rows:
        if row.get("syllable") is None:
            if not units:
                raise ValueError("the first sung note needs a lyric syllable")
            units[-1]["notes"].append(row)
            continue
        onset, frames, coda = parse_syllable(row["syllable"])
        units.append({"text": row["syllable"], "onset": onset, "frames": frames, "coda": coda,
                      "notes": [row]})
    for u in units:
        u["start"] = u["notes"][0]["time"]
        u["end"] = max(r["off"] for r in u["notes"])
    phrases = [[units[0]]] if units else []
    for prev, u in zip(units, units[1:]):
        if u["start"] - prev["end"] <= x["legato_gap_s"]:
            phrases[-1].append(u)
        else:
            phrases.append([u])
    cons = config["consonants"]

    def length(c):
        return (cons[c]["duration_ms"] if c in cons else config["constructed"][c]["duration_ms"]) / 1000

    events, keyframes = [], []
    for phrase in phrases:
        for i, u in enumerate(phrase):
            nxt = phrase[i + 1] if i + 1 < len(phrase) else None
            prev = phrase[i - 1] if i else None
            onset_d = sum(length(c) for c in u["onset"])
            # Onset consonants end a fraction of their length after the beat.
            onset_end = u["start"] + x["onset_after_beat"] * onset_d
            if prev is not None:
                room = (1 - x["min_vowel_fraction"]) * (u["start"] - prev["start"])
                onset_end = min(onset_end, u["start"] + room)
            t = onset_end - onset_d
            for c in u["onset"]:
                events.append({"name": c, "start": t, "end": t + length(c), "unit": u})
                t += length(c)
            u["vowel_start"] = onset_end
            boundary = nxt["start"] if nxt is not None else u["end"]
            coda_d = sum(length(c) for c in u["coda"])
            next_onset = sum(length(c) for c in nxt["onset"]) if nxt is not None else 0.0
            coda_end = boundary - (1 - x["onset_after_beat"]) * next_onset if nxt is not None else u["end"]
            span = boundary - u["start"]
            coda_end = max(coda_end, u["vowel_start"] + x["min_vowel_fraction"] * span + coda_d)
            t = coda_end - coda_d
            u["vowel_end"] = t
            for c in u["coda"]:
                events.append({"name": c, "start": t, "end": t + length(c), "unit": u})
                t += length(c)
        for u in phrase:
            for vowel, at in u["frames"]:
                keyframes.append((u["vowel_start"] + at * (u["vowel_end"] - u["vowel_start"]), vowel))
    # A run of hummed syllables in a phrase is one closed-lip nasal: "m" from
    # the first's start to the last's end, the vowel shape held behind the lips.
    for phrase in phrases:
        run = []
        for u in [*phrase, None]:
            if u is not None and syllable_key(u["text"]) in HUMS:
                run.append(u)
                continue
            if run:
                events.append({"name": "m", "start": run[0]["start"] - 0.02, "end": run[-1]["end"] + x["release_s"] + 0.05, "unit": run[0]})
                run = []
    for e in events:
        if e["name"] in GLIDES:
            keyframes.append((0.5 * (e["start"] + e["end"]), e["name"]))
    keyframes.sort(key=lambda k: k[0])
    return {"units": units, "phrases": phrases, "events": events, "keyframes": keyframes}


def controls(notes, config, seconds=None, midi_controls=(), timeline_offset=0.0):
    """Per-control-step tracks for the source and tract (1 ms by default)."""
    sr = config["engine"]["sample_rate"]
    dt = config["engine"]["control_ms"] / 1000
    x, a, s, n_cfg = config["expression"], config["aero"], config["source"], config["nasal"]
    p = plan(notes, config)
    if seconds is None:
        end = max((u["end"] for u in p["units"]), default=0.0)
        seconds = end + config["engine"]["tail_seconds"]
    steps = int(math.ceil(seconds / dt)) + 2
    t = np.arange(steps) * dt
    shapes = vowel_shapes(config)
    count = sections(config)
    section_cm = section_length_cm(config)
    notes_sorted = [r for u in p["units"] for r in u["notes"]]

    # Vowel layer: smoothstep between consecutive keyframes.
    frames = p["keyframes"]
    areas = np.empty((steps, count))
    if frames:
        times = np.array([f[0] for f in frames])
        vecs = np.array([shapes[f[1]] for f in frames])
        idx = np.searchsorted(times, t, side="right")
        lo, hi = np.clip(idx - 1, 0, len(times) - 1), np.clip(idx, 0, len(times) - 1)
        span = np.where(hi > lo, times[hi] - times[lo], 1.0)
        w = smoothstep(np.where(hi > lo, (t - times[lo]) / span, 0.0))[:, None]
        areas[:] = vecs[lo] * (1 - w) + vecs[hi] * w
    else:
        areas[:] = shapes["ax"]

    # Consonant layer and velum.
    xs = (count - 0.5 - np.arange(count)) * section_cm
    velum = np.zeros(steps)
    unvoiced = np.zeros(steps)
    abduction = np.zeros(steps)
    noise_colours = []
    obstacle = np.zeros(steps)   # turbulence source strength of the active consonant
    for e in p["events"]:
        if e["name"] in GLIDES:
            continue
        c = config["consonants"][e["name"]]
        act = _window(t, e["start"] + c["in_ms"] / 1000, e["end"] - c["out_ms"] / 1000,
                      c["in_ms"] / 1000, c["out_ms"] / 1000)
        e["activation"] = act
        weight = spatial_weight(xs, c["place_cm"], c["width_cm"])[None, :] * act[:, None]
        areas = areas * (1 - weight) + c["area_cm2"] * weight
        if e["name"] in NASALS:
            v = _window(t, e["start"], e["end"], n_cfg["open_ms"] / 1000, n_cfg["close_ms"] / 1000)
            velum = np.maximum(velum, n_cfg["coupling_cm2"] * v)
        if not c["voiced"]:
            # Glottal abduction spans the consonant and its ramps.
            g = _window(t, e["start"], e["end"], 0.03, 0.04)
            unvoiced = np.maximum(unvoiced, g)
            abduction = np.maximum(abduction, g)
        if c["noise_gain"] > 0:
            window = _window(t, e["start"], e["end"] + 0.04, 0.02, 0.03)
            noise_colours.append((e, window))
            obstacle = np.maximum(obstacle, c["noise_gain"] * window)

    # Breath (subglottal pressure) and phonation per phrase.
    breath = np.zeros(steps)
    phonation = np.zeros(steps)
    for phrase in p["phrases"]:
        first = phrase[0]
        lead = [e for e in p["events"] if e["unit"] is first]
        start = min([first["vowel_start"]] + [e["start"] for e in lead])
        end = phrase[-1]["end"]
        breath = np.maximum(breath, _window(t, start - a["breath_lead_s"] + 0.02, end, 0.02, x["release_s"]))
        voiced_start = start
        if lead and lead[0]["name"] not in GLIDES and not config["consonants"][lead[0]["name"]]["voiced"]:
            voiced_start = first["vowel_start"] - 0.02
        phonation = np.maximum(phonation, _window(t, voiced_start + x["attack_s"], end, x["attack_s"], x["release_s"]))

    # Loudness, effort and pitch from the notes.
    vel = np.zeros(steps)
    pitch = np.full(steps, float(notes_sorted[0]["pitch"]) if notes_sorted else 60.0)
    for i, r in enumerate(notes_sorted):
        nxt = notes_sorted[i + 1]["time"] if i + 1 < len(notes_sorted) else math.inf
        mask = t >= r["time"] - (0.2 if i == 0 else 0.0)
        vel[mask] = r["velocity"]
        pitch[mask] = r["pitch"]
    kernel = max(1, round(0.03 / dt))
    vel = np.convolve(np.pad(vel, (kernel, kernel), mode="edge"), np.ones(kernel) / kernel, "same")[kernel:-kernel]
    lo_v, hi_v = s["effort_velocity"]
    effort = np.clip((vel - lo_v) / (hi_v - lo_v), 0, 1)
    level_db = s["velocity_db"] * (vel / 127 - 1)
    vib_env = np.zeros(steps)
    for phrase in p["phrases"]:
        pnotes = [r for u in phrase for r in u["notes"]]
        for i, r in enumerate(pnotes):
            if i:
                prev = pnotes[i - 1]["pitch"]
                g0, g1 = r["time"] - 0.4 * x["portamento_s"], r["time"] + 0.6 * x["portamento_s"]
                m = (t >= g0 - dt) & (t <= g1 + dt)
                pitch[m] = prev + (r["pitch"] - prev) * smoothstep((t[m] - g0) / (g1 - g0))
            following = pnotes[i + 1]["time"] if i + 1 < len(pnotes) else r["off"] + x["release_s"]
            rise = smoothstep((t - r["time"] - x["vibrato_delay_s"]) / x["vibrato_rise_s"])
            fall = 1 - smoothstep((t - following) / 0.08)
            vib_env = np.maximum(vib_env, rise * fall * (t >= r["time"]))
            duration = r["off"] - r["time"]
            if duration > 0.8:
                inside = (t >= r["time"]) & (t <= r["off"])
                level_db[inside] += x["note_arc_db"] * np.sin(np.pi * (t[inside] - r["time"]) / duration)
        s0 = pnotes[0]["time"]
        scoop = (t >= s0 - 0.2) & (t <= s0 + x["scoop_s"])
        pitch[scoop] -= x["scoop_cents"] / 100 * (1 - smoothstep((t[scoop] - s0) / x["scoop_s"]))
        e0 = phrase[-1]["end"]
        droop = (t > e0) & (t <= e0 + 2 * x["release_s"])
        pitch[droop] -= x["release_droop_cents"] / 100 * smoothstep((t[droop] - e0) / x["release_s"])
    rng = np.random.default_rng(np.random.SeedSequence([config["engine"]["seed"], 1]))
    # Cycle-to-cycle rate wander (~0.1 s correlation), zero-mean within a note.
    wander = lfilter([0.01], [1, -0.99], rng.standard_normal(steps))
    wander *= x["vibrato_wander"] / (np.std(wander) + 1e-12)
    vib_phase = 2 * np.pi * np.cumsum(x["vibrato_hz"] * (1 + wander)) * dt
    vibrato = np.sin(vib_phase) * vib_env
    # External lanes use the caller's timeline, before engine pre-roll.
    lanes = {}
    for control in ("pitch_cents", 2):
        rows = sorted((r for r in midi_controls if r["control"] == control), key=lambda r: r["time"])
        if rows:
            if any(not math.isfinite(r["time"]) or r["time"] < 0 or
                   not math.isfinite(r["value"]) or
                   (control == 2 and not 0 <= r["value"] <= 127) for r in rows):
                raise ValueError(f"invalid vocal {control} control")
            lanes[control] = np.interp(t - timeline_offset, [r["time"] for r in rows],
                                       [r["value"] for r in rows])
    if "pitch_cents" in lanes:
        pitch += lanes["pitch_cents"] / 100
    cents = 100 * pitch + x["vibrato_cents"] * vibrato
    f0 = 440 * 2 ** ((cents / 100 - 69) / 12)
    amplitude = 10 ** (level_db / 20) * (1 + x["vibrato_amplitude"] * vibrato)

    # Aerodynamics: two orifices in series; turbulence downstream of the constriction.
    ps = a["subglottal_pressure"] * breath
    ag = a["glottal_area_cm2"] + a["abduction_cm2"] * abduction
    first_oral = max(2, round(3.0 / section_cm))   # above the larynx
    ac = areas[:, first_oral:].min(axis=1)
    where = areas[:, first_oral:].argmin(axis=1) + first_oral
    escape = ac + velum                            # the nose is a parallel outlet
    target = ps * ag ** 2 / (ag ** 2 + escape ** 2)
    alpha = 1 - math.exp(-dt / (a["pressure_tau_ms"] / 1000))
    po = lfilter([alpha], [1, -(1 - alpha)], target)
    # Turbulence needs an obstacle in the jet (teeth, lips, palate): it sounds
    # only within a consonant's window, scaled by that consonant's gain.
    turbulence = a["noise_gain"] * obstacle * np.sqrt(ac) * np.maximum(po, 0) ** 1.5
    turbulence *= ac < a["turbulence_max_area_cm2"]
    fric_at = np.where(turbulence > 0, np.minimum(where + 1, count - 1), 0).astype(np.int64)
    voicing = phonation * (1 - unvoiced) * (1 - a["voicing_pressure_loss"] * np.clip(po / np.maximum(ps, 1e-9), 0, 1))
    whisper = abduction * breath
    if 2 in lanes:
        intake = lanes[2] / 127
        voicing *= 1 - intake
        whisper = np.maximum(whisper, intake)
    return {"t": t, "dt": dt, "areas": areas, "velum": velum, "f0": f0, "amplitude": amplitude * voicing,
            "effort": effort, "whisper": whisper, "turbulence": turbulence, "fric_at": fric_at,
            "noise_colours": noise_colours, "plan": p, "oral_pressure": po, "vibrato": vibrato,
            "pitch": pitch, "steps": steps}


def _coloured(noise, hz, q, sr):
    """Two-pole band-pass (RBJ constant-peak) noise colouring."""
    w = 2 * math.pi * hz / sr
    alpha = math.sin(w) / (2 * q)
    b = np.array([alpha, 0, -alpha]) / (1 + alpha)
    a = np.array([1, -2 * math.cos(w) / (1 + alpha), (1 - alpha) / (1 + alpha)])
    y = lfilter(b, a, noise)
    return y / (np.std(y) + 1e-12)


def render(notes, seconds=None, config=None, return_controls=False, midi_controls=(), f0_map=None):
    """Sing `notes` (see plan) and return mono audio at 48 kHz."""
    c = config or load_config()
    sr = c["engine"]["sample_rate"]
    e, s = c["engine"], c["source"]
    if not notes:
        samples = round((seconds or 0.0) * sr)
        return (np.zeros(samples), None) if return_controls else np.zeros(samples)
    lead = e["lead_seconds"]
    shifted = [{**r, "time": r["time"] + lead, "off": r["off"] + lead} for r in notes]
    total = (seconds + lead) if seconds is not None else None
    k = controls(shifted, c, total, midi_controls, lead)
    if f0_map is not None:
        mapped = np.asarray(f0_map(k["t"] - lead, k["f0"].copy()), dtype=float)
        if mapped.shape != k["f0"].shape or not np.isfinite(mapped).all() or np.any(mapped <= 0):
            raise ValueError("f0_map must return positive finite Hz with the control track's shape")
        k["f0"] = mapped
    step = round(sr * k["dt"])
    samples = (k["steps"] - 1) * step
    ctrl_t = np.arange(k["steps"]) * step
    n = np.arange(samples)
    f0 = np.interp(n, ctrl_t, k["f0"])
    amplitude = np.interp(n, ctrl_t, k["amplitude"])
    effort = np.interp(n, ctrl_t, k["effort"])
    expression = sorted((r for r in midi_controls if r['control']==11),key=lambda r:r['time'])
    if expression:
        if any(not math.isfinite(r['time']) or r['time']<0 or not 0<=r['value']<=127 for r in expression):
            raise ValueError('invalid vocal expression control')
        amplitude *= np.interp(n/sr-lead,[r['time'] for r in expression],
                               [r['value']/127 for r in expression])
    if any('source_gain_db' in r for r in shifted):
        gains=np.array([r.get('source_gain_db',0) for r in shifted],dtype=float)
        if not np.isfinite(gains).all() or np.any(gains>0) or np.any(gains<-40):
            raise ValueError('source_gain_db must be finite within -40..0')
        index=np.maximum(0,np.searchsorted([r['time'] for r in shifted],n/sr,side='right')-1)
        lane=10**(gains[index]/20)
        amplitude *= sosfilt(butter(1,20,fs=sr,output='sos'),lane)
    seed = e["seed"]
    rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
    periods = 8192
    jitter = s["jitter"] * rng.standard_normal(periods)
    shimmer = s["shimmer"] * rng.standard_normal(periods)
    table, flow_table = lf_tables(tuple(s["soft_lf"]), tuple(s["loud_lf"]))
    os_ = e["oversample"]
    d, g = _glottis(f0, amplitude, effort, jitter, shimmer, table, flow_table, os_, sr)
    if os_ > 1:
        d, g = resample_poly(d, 1, os_), resample_poly(g, 1, os_)
    d, g = d[:samples], g[:samples]
    # Aspiration: high-passed noise pulsed by the glottal flow.
    noise = np.random.default_rng(np.random.SeedSequence([seed, 3])).standard_normal(samples)
    noise = sosfilt(butter(2, s["aspiration_hz"], "highpass", fs=sr, output="sos"), noise)
    whisper = np.interp(n, ctrl_t, k["whisper"])
    src = d + noise * (s["breathiness"] * (0.3 * amplitude + 0.7 * g) + c["aero"]["aspiration_gain"] * whisper)
    del d, g, noise, whisper, f0, effort
    # Frication: each consonant's noise colour weighted by its window.
    white = np.random.default_rng(np.random.SeedSequence([seed, 4])).standard_normal(samples)
    colour_sum = np.zeros(samples)
    weight_sum = np.zeros(samples)
    for event, window in k["noise_colours"]:
        cc = c["consonants"][event["name"]]
        w = np.interp(n, ctrl_t, window)
        live = np.nonzero(w > 0)[0]
        if not len(live):
            continue
        lo, hi = live[0], live[-1] + 1
        colour_sum[lo:hi] += w[lo:hi] * _coloured(white[lo:hi], cc["noise_hz"], cc["noise_q"], sr)
        weight_sum[lo:hi] += w[lo:hi]
    generic = sosfilt(butter(2, 1500, "highpass", fs=sr, output="sos"), white)
    generic /= np.std(generic) + 1e-12
    mix = np.minimum(weight_sum, 1)
    fric_noise = mix * colour_sum / np.maximum(weight_sum, 1e-9) + (1 - mix) * generic
    fric = np.interp(n, ctrl_t, k["turbulence"]) * fric_noise
    del white, colour_sum, weight_sum, generic, fric_noise, mix, n, amplitude
    out = _tract(src, fric, k["fric_at"], k["areas"], k["velum"], step, **tract_parameters(c))
    out = sosfilt(butter(4, e["output_lowpass_hz"], "lowpass", fs=sr, output="sos"), out)
    out = sosfilt(butter(1, 30, "highpass", fs=sr, output="sos"), out)
    out *= e["output_gain"]
    start = round(lead * sr)
    out = out[start:]
    if seconds is not None:
        out = np.pad(out, (0, max(0, round(seconds * sr) - len(out))))[:round(seconds * sr)]
    if return_controls:
        return out, k
    return out
