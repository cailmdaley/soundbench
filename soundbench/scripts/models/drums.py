"""Coupled membrane drum kit, contact wires, friction brushes and modal plates.

No samples or render normalization. Plate power transfers and contact rotations
are passive reduced coupling, not a geometrically nonlinear thin-plate PDE.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
from numba import njit
from scipy.signal import lfilter

from .mallorca_percussion import bordon_contact, contact, modal_signal, tambori_modes
from soundbench.scripts.paths import preset_path, read_preset

ROOT = Path(__file__).resolve().parents[2]
CONFIG = preset_path("drums")
KEYS = (35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 53, 57, 59, 60, 61, 62, 63)
HATS = (42, 44, 46, 63)
BRUSHES = (60, 61, 62)


def load_config(path=CONFIG):
    c = read_preset(path)
    if c.get("instrument") != "drums":
        raise ValueError("preset instrument must be drums")
    template = read_preset(CONFIG)
    if set(c) != set(template) or any(set(c[s]) != set(template[s]) for s in template if s != "instrument"):
        raise ValueError("unexpected drum preset schema")
    for s in c:
        if s == "instrument":
            continue
        for k, v in c[s].items():
            if not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                raise ValueError(f"invalid {s}.{k}")
            if k.endswith("hz") and not 20 <= v <= 12000:
                raise ValueError(f"invalid frequency {s}.{k}")
            if ("t60" in k or k == "t60") and not .01 <= v <= 5:
                raise ValueError(f"invalid decay {s}.{k}")
    e = c["engine"]
    if e["sample_rate"] != 48000 or type(e["seed"]) is not int or not 3 <= e["tail_seconds"] <= 10 or not 0 < e["output_gain"] <= 4:
        raise ValueError("invalid engine controls")
    if type(c["cymbals"]["modes"]) is not int or not 16 <= c["cymbals"]["modes"] <= 128 or c["cymbals"]["coupling"] > .2:
        raise ValueError("invalid modal plate controls")
    if not .01 <= c["kick"]["bottom_ratio"] <= 3 or not .01 <= c["snare"]["bottom_ratio"] <= 3 or c["snare"]["snare_threshold"] <= 0:
        raise ValueError("invalid head/wire controls")
    for k in ("hardness", "felt", "port"):
        if c["kick"][k] > 1:
            raise ValueError(f"kick.{k} must be 0..1")
    if c["mics"]["width"] > 1 or c["mics"]["overhead_ms"] > 20:
        raise ValueError("invalid microphone geometry")
    return c


def drum_modes(b, radius=.15):
    # Feed the shared two-head + shell cavity eigenproblem an exact schema.
    body = {k: b[k] for k in ("fundamental_hz", "bottom_ratio", "air_hz", "t60", "damping_exponent", "max_hz", "shell_body_hz", "shell_body_t60", "shell_volume")}
    return tambori_modes({"body": body}, radius)


def _drum(row, c, rng):
    sr = c["engine"]["sample_rate"]
    p, v = row["pitch"], row["velocity"]/127
    brush = p in BRUSHES
    b = copy.deepcopy(c["kick"] if p in (35, 36) else c["snare"])
    if p in (41, 43, 45, 47, 48, 50):
        b.update(fundamental_hz=c["toms"]["low_hz"] if p in (41, 43, 45) else c["toms"]["high_hz"], bottom_ratio=1.1, air_hz=60., t60=c["toms"]["t60"], gain=c["toms"]["gain"])
    kick = p in (35, 36)
    snare = p in (38, 39, 40, *BRUSHES)
    if kick:
        # Felt damps the heads; a vent weakens the cavity compression spring.
        b["t60"] /= 1+2*b["felt"]
        b["air_hz"] *= 1-.65*b["port"]
        b["stick_ms"] *= 1.5-b["hardness"]
    held = row["off"]-row["time"] if p in (60, 61) else 0
    n = round((held+c["engine"]["tail_seconds"])*sr)
    amp = v**1.4
    force = contact(n, sr, b["stick_ms"]*(1.5-.65*v))
    if p in (60, 61):
        h = max(1, round(held*sr))
        t = np.arange(h)/sr
        # Bristle friction noise follows the hand's periodic sweep pressure.
        motion = (.4+.6*np.sin(np.pi*t/ max(held, 1e-6))**2)
        noise = lfilter([1, -1], [1, -.65], rng.normal(size=h))
        force[:] = 0
        force[:h] = noise*motion*b["brush_gain"]
        fade = min(h//2, round(.02*sr))
        if fade:
            force[:fade] *= np.linspace(0, 1, fade)
            force[h-fade:h] *= np.linspace(1, 0, fade)
    elif p == 62:
        force = lfilter([1], [1, -.8], rng.normal(size=n)*contact(n, sr, 8))*.6
    f, t60, w, lower_w = drum_modes(b, .7 if p in (37, 39, 40, 61) else .15)
    y = modal_signal(force*amp, f, t60, w, sr)
    if snare:
        lower = modal_signal(force*amp, f, t60, lower_w, sr)
        buzz = bordon_contact(lower, b["snare_threshold"], rng)
        y += b["snare_gain"]*modal_signal(buzz, [1900, 3300, 5700, 8100], [b["wire_t60"], b["wire_t60"]*.7, b["wire_t60"]*.5, b["wire_t60"]*.3], [.5, .35, .2, .12], sr)
    rim = modal_signal(contact(n, sr, .18)*amp, [b["shell_hz"], b["shell_hz"]*2.31, b["shell_hz"]*3.78], [.065, .04, .02], [.6, .25, .1], sr)
    if p == 37:
        y = rim*.7
    elif p in (39, 40):
        y = .85*y+rim*.65
    # A smooth, velocity-dependent stick/head contact radiation bandwidth.
    corner = 450+6500*v*v if not kick else 500+4000*b["hardness"]*v
    a = math.exp(-2*math.pi*corner/sr)
    y = lfilter([1-a], [1, -a], y)*b["gain"]
    return y


def plate_modes(c, kind):
    b = c["cymbals"]
    count = b["modes"]
    low = b[f"{kind}_low_hz"]
    # Deterministic quasi-random spacing breaks harmonic combs; frequencies
    # stand in for measured plate eigenmodes, not a membrane Bessel series.
    u = np.linspace(0, 1, count)
    f = low+(14500-low)*u**1.25
    f *= 1+.018*np.sin(np.arange(count)*2.399963)
    t60 = b[f"{kind}_t60"]*(f/low)**(-.2)
    w = (f/low)**b["tilt"]
    if kind == "hat":
        # Two slightly different plates, coupled pairwise by pedal contact.
        f = np.r_[f, f*1.037]
        t60 = np.r_[t60, t60*.92]
        w = np.r_[w, w*.85]
    return f, t60, w/np.sqrt(len(w))


@njit(cache=True, inline="always")
def _cascade(z, coupling, is_hat):
    """Phase-preserving upward transfer conserves the sum of modal powers."""
    half = len(z)//2
    for j in range(len(z)-1):
        if is_hat and j == half-1:
            continue
        a, b = z[j], z[j+1]
        ea, eb = abs(a)**2, abs(b)**2
        fraction = coupling*.02*math.tanh(8*math.sqrt(ea))
        transfer = fraction*ea
        z[j] = a*math.sqrt(1-fraction)
        z[j+1] = b*math.sqrt(1+transfer/eb) if eb > 1e-30 else complex(math.sqrt(transfer), 0)


@njit(cache=True)
def _plate(force, poles, weights, coupling, closure, closed_rate, is_hat):
    """Contracting poles and phase-preserving upward power transfer are passive."""
    z = np.zeros(len(poles), np.complex128)
    y = np.zeros(len(force))
    half = len(poles)//2
    nonzero = np.flatnonzero(force)
    last_force = nonzero[-1] if len(nonzero) else -1
    for i in range(len(force)):
        close = closure[i]
        damp = math.exp(-closed_rate*close*close) if is_hat else 1.
        z = z*poles*damp + force[i]*weights
        # Transfer power upward within each plate, preserving recipient phase.
        # This is an energy-envelope approximation, not von Karman forces.
        _cascade(z, coupling, is_hat)
        if is_hat:
            angle = .08*close
            co, si = math.cos(angle), math.sin(angle)
            for j in range(half):
                a, b = z[j], z[j+half]
                z[j], z[j+half] = co*a-si*b, si*a+co*b
        y[i] = np.sum(z.real*weights)
        if i > max(2048, last_force) and np.max(np.abs(z)) < 1e-11:
            break
    return y


def _cymbal(row, c, rng, pedal):
    sr, b = c["engine"]["sample_rate"], c["cymbals"]
    p = row["pitch"]
    kind = "hat" if p in HATS else "ride" if p in (51, 53, 59) else "crash"
    n = round(c["engine"]["tail_seconds"]*sr)
    f, t, w = plate_modes(c, kind)
    v = row["velocity"]/127
    force = contact(n, sr, .3 if p == 53 else .8)
    if p != 53:
        force = force + lfilter([1, -1], [1, -.25], rng.normal(size=n))*contact(n, sr, 1.8)*.5
    if p == 53:
        w *= np.exp(-((f-2300)/1800)**2)
    closure = np.full(n, row.get("pedal", {42: 1., 44: 1., 46: 0., 63: .5}.get(p, 0.)))
    if kind == "hat":
        for time, value in pedal:
            if time >= row["time"]:
                closure[max(0, round((time-row["time"])*sr)):] = value
        # Contact release is smoothed over 2 ms to avoid waveform steps.
        a = math.exp(-1/(.002*sr))
        closure = lfilter([1-a], [1, -a], closure, zi=[closure[0]*a])[0]
    poles = np.exp(-math.log(1000)/(t*sr)+2j*np.pi*f/sr)
    y = _plate(force*v**1.4, poles, w, b["coupling"], closure, math.log(1000)/(b["closed_t60"]*sr), kind == "hat")
    # Intermittent two-plate rattle is tied to contact and plate vibration.
    if kind == "hat":
        rattle = abs(y)*closure*(1-closure*.8)*rng.normal(size=n)
        y += lfilter([1, -1], [1, -.4], rattle)*.25
    return y*b["gain"]*(b["hat_gain"] if kind == "hat" else 1)*(.7 if p == 44 else 1)


def hit(row, c=None, seed=None, pedal=()):
    c = load_config() if c is None else c
    if row["pitch"] not in KEYS or not 1 <= row["velocity"] <= 127 or not np.isfinite([row["time"], row["off"]]).all() or row["time"] < 0 or row["off"] <= row["time"]:
        raise ValueError("invalid drum gesture; pitches select kit pieces, explicit releases required")
    rng = np.random.default_rng(c["engine"]["seed"] if seed is None else seed)
    y = _cymbal(row, c, rng, pedal) if row["pitch"] in (*HATS, 49, 51, 53, 57, 59) else _drum(row, c, rng)
    # Only an inaudible terminal tail is tapered; note-off stops friction,
    # while struck drums/cymbals continue their natural decay.
    fade = min(len(y), round(.04*c["engine"]["sample_rate"]))
    y[-fade:] *= np.linspace(1, 0, fade)
    return y


def _pan(pitch):
    if pitch in HATS:
        return -.65
    if pitch in (41, 43, 45):
        return .45
    if pitch in (47, 48, 50):
        return -.25
    if pitch in (51, 53, 59):
        return .6
    if pitch in (49, 57):
        return -.45
    return -.08 if pitch in (37, 38, 39, 40, *BRUSHES) else 0.


def render(rows, seconds, config=None, pedal=()):
    """Fixed-gain stereo close/overhead/bleed mix; room belongs to the studio."""
    c = load_config() if config is None else config
    sr, m = c["engine"]["sample_rate"], c["mics"]
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("duration must be positive")
    if any(not math.isfinite(t) or t < 0 or not 0 <= v <= 1 for t, v in pedal):
        raise ValueError("invalid pedal timeline")
    explicit_pedal = list(pedal)
    pedal = list(pedal)
    # GM hat keys set the shared physical pedal until CC4 takes control.
    # A foot chick explicitly closes the plates, including ringing open hits.
    defaults = {42: 1., 44: 1., 46: 0., 63: .5}
    for row in sorted(rows, key=lambda r: r["time"]):
        if row["pitch"] in HATS and (row["pitch"] == 44 or not any(t <= row["time"] for t, _ in explicit_pedal)):
            pedal.append((row["time"], defaults[row["pitch"]]))
    pedal.sort(key=lambda event: event[0])
    out = np.zeros((round(seconds*sr), 2))
    def add(y, start, channel, gain):
        size = min(len(y), len(out)-start)
        if size > 0:
            out[start:start+size, channel] += y[:size]*gain
    for ordinal, row in enumerate(sorted(rows, key=lambda r: (r["time"], r["pitch"]))):
        if row["off"]+c["engine"]["tail_seconds"] > seconds+1/sr:
            raise ValueError("render duration must include final release and tail")
        r = dict(row)
        prior = [v for t, v in pedal if t <= r["time"]]
        if prior and r["pitch"] in HATS:
            r["pedal"] = prior[-1]
        y = hit(r, c, c["engine"]["seed"]+ordinal, pedal)
        pos = _pan(r["pitch"])*m["width"]
        start = round(r["time"]*sr)
        oh = lfilter([.25], [1, -.75], y)
        for ch, side in enumerate((-1, 1)):
            angle = (pos+1)*math.pi/4
            direct = math.cos(angle) if ch == 0 else math.sin(angle)
            add(y, start, ch, direct)
            delay = round((m["overhead_ms"]+side*pos*.8)*sr/1000)
            add(oh, start+delay, ch, m["overhead_gain"]*(1+side*pos*.3))
            add(oh, start+round((6+ch*1.3)*sr/1000), ch, m["bleed_gain"])
    return out*c["engine"]["output_gain"]
