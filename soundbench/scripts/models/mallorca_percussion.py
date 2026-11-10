"""Modal Mallorcan tabor/castanets and a mechanically driven friction drum.

Percussion note numbers select gestures, not equal-tempered pitches.
All rendering uses fixed gain: no note-wise or whole-render normalization.
"""
from __future__ import annotations

import copy
import math
from functools import lru_cache
from pathlib import Path
import tomllib

import numpy as np
from numba import njit
from scipy.linalg import eigh
from scipy.signal import lfilter
from scipy.special import j1, jn_zeros

from .banjo import membrane_modes
from .note_random import note_rng

ROOT = Path(__file__).resolve().parents[2]
MODELS = ("tambori", "castanyoles", "ximbomba")
KEYS = {"tambori": tuple(range(36, 43)), "castanyoles": tuple(range(60, 65)),
        "ximbomba": (48, 49)}
XIMBOMBA_DEFAULTS = {"pot_hz": 200., "pot_coupling": 0., "pot_q": 5.}


def load_config(model, path=None):
    if model not in MODELS:
        raise ValueError(f"unknown percussion model {model!r}")
    path = Path(path) if path else ROOT / f"instruments/models/{model}.toml"
    c = tomllib.loads(path.read_text())
    if c.get("instrument") != model:
        raise ValueError("preset instrument does not match model")
    schema = {
        "tambori": {"fundamental_hz", "bottom_ratio", "air_hz", "t60", "damping_exponent",
                    "max_hz", "stick_ms", "snare_gain", "snare_threshold", "shell_hz",
                    "shell_body_hz", "shell_body_t60", "shell_volume"},
        "castanyoles": {"frequencies", "t60", "weights", "split_cents", "mascle_ratio",
                       "contact_ms", "dancers", "timing_ms", "tuning_cents"},
        "ximbomba": {"fundamental_hz", "modes", "cane_mass_ratio", "t60", "damping_exponent", "friction_static",
                     "friction_dynamic", "stribeck_velocity", "hand_speed", "pressure",
                     "release_s", "roughness", "direction_asymmetry", "pot_hz", "pot_coupling", "pot_q"}}
    if set(c) != {"instrument", "engine", "body"}:
        raise ValueError("unexpected percussion preset schema")
    # A ximbomba hand either lifts between strokes or rubs continuously.
    hands = [{"stroke_peak"}, {"turn_s", "ease"}] if model == "ximbomba" else [set()]
    if model == "ximbomba":
        # Presets written before the pot or the hand profile existed keep their behaviour:
        # a closed pot and the symmetric lifting bell.
        for key, value in XIMBOMBA_DEFAULTS.items():
            c["body"].setdefault(key, value)
        if not set(c["body"]) & {"stroke_peak", "turn_s", "ease"}:
            c["body"]["stroke_peak"] = .5
    if not any(set(c["body"]) == schema[model] | h for h in hands):
        expected = min((schema[model] | h for h in hands), key=lambda keys: len(keys ^ set(c["body"])))
        raise ValueError(f"unexpected {model} preset body: missing {sorted(expected-set(c['body']))}, "
                         f"unknown {sorted(set(c['body'])-expected)}")
    if set(c["engine"]) != {"sample_rate", "seed", "tail_seconds", "output_gain"}:
        raise ValueError("unexpected engine schema")
    for section in (c["engine"], c["body"]):
        for value in section.values():
            a = np.asarray(value)
            if a.dtype.kind not in "ifu" or not np.all(np.isfinite(a)) or np.any(a < 0):
                raise ValueError("parameters must be finite nonnegative numbers")
    e, b = c["engine"], c["body"]
    if e["sample_rate"] != 48000 or type(e["seed"]) is not int or e["output_gain"] <= 0 or not .5 <= e["tail_seconds"] <= 10:
        raise ValueError("invalid engine parameters")
    for key, value in b.items():
        if key.endswith("hz") and not 10 <= value <= 8000:
            raise ValueError(f"invalid body.{key}")
    if model == "tambori":
        if not .5 <= b["bottom_ratio"] <= 2 or not .02 <= b["t60"] <= 2 or not .1 <= b["stick_ms"] <= 5:
            raise ValueError("invalid drum tension/decay/contact")
        if b["max_hz"] <= b["fundamental_hz"] or b["snare_threshold"] <= 0 or not .02 <= b["shell_body_t60"] <= 1 or not 0 <= b["shell_volume"] <= 2:
            raise ValueError("invalid membrane bandwidth/snare threshold/shell")
    elif model == "castanyoles":
        f, t, w = (np.asarray(b[k]) for k in ("frequencies", "t60", "weights"))
        if f.ndim != 1 or not 2 <= len(f) <= 16 or f.shape != t.shape or f.shape != w.shape or np.any(f < 100) or np.any(f > 10000) or np.any(t < .003) or np.any(t > 1) or not np.any(w > 0):
            raise ValueError("invalid shell modes")
        if type(b["dancers"]) is not int or not 1 <= b["dancers"] <= 16 or not .5 <= b["mascle_ratio"] <= 1 or not .05 <= b["contact_ms"] <= 3 or b["timing_ms"] > 30 or b["tuning_cents"] > 100 or b["split_cents"] > 100:
            raise ValueError("invalid castanet pair/ensemble parameters")
    else:
        if not 30 <= b["fundamental_hz"] <= 400 or not .02 <= b["t60"] <= 2 or not .01 <= b["release_s"] <= .3 or not 0 <= b["cane_mass_ratio"] <= 5 or type(b["modes"]) is not int or not 1 <= b["modes"] <= 64 or not 0 <= b["damping_exponent"] <= 4:
            raise ValueError("invalid friction drum mechanics")
        if not 0 < b["friction_dynamic"] < b["friction_static"] <= 2 or not .001 <= b["stribeck_velocity"] <= 1 or not .001 <= b["hand_speed"] <= 1 or not .01 <= b["pressure"] <= 10:
            raise ValueError("invalid friction curve")
        if not 0 <= b["roughness"] <= .5 or not 0 <= b["direction_asymmetry"] <= .5 or not .05 <= b.get("stroke_peak", .15) <= .5:
            raise ValueError("invalid wet-contact controls")
        if not 0 <= b["pot_coupling"] <= 10 or not .5 <= b["pot_q"] <= 100:
            raise ValueError("invalid pot")
        if not 0 <= b.get("turn_s", 0) <= .5 or not .05 <= b.get("ease", 1) <= 1:
            raise ValueError("invalid hand turn")
    return c


def modal_signal(force, frequencies, times, weights, sr=48000):
    """Exact sampled damped-mode impulse responses, with a finite contact pulse."""
    out = np.zeros(len(force))
    for f, t60, weight in zip(frequencies, times, weights):
        r = math.exp(-math.log(1000) / (t60 * sr))
        w = 2 * math.pi * f / sr
        # Each mode's first velocity-response sample has the same force scale.
        out += weight * lfilter([1, -r * math.cos(w)], [1, -2*r*math.cos(w), r*r], force)
    return out


def contact(n, sr, width_ms):
    force = np.zeros(n)
    width = max(2., width_ms * sr / 1000)
    k = min(n, math.ceil(width+.5))
    phase = np.minimum(1., (np.arange(k)+.5)/width)
    pulse = np.sin(np.pi*phase)
    force[:k] = pulse / pulse.sum()
    return force


def tambori_modes(c, radius=.15):
    """Two Bessel heads coupled by the cavity's compression spring.

    The air spring couples only modes with net volume displacement (m=0).
    The bottom head has its own tension. Symmetric stiffness guarantees
    positive eigenvalues; eigenvectors retain driving/radiating modal signs.
    """
    b = c["body"]
    return _tambori_modes(tuple(sorted(b.items())), radius)


@lru_cache(maxsize=32)
def _tambori_modes(parameters, radius):
    b = dict(parameters)
    head = {"fundamental_hz": b["fundamental_hz"], "bridge_radius": radius,
            "max_hz": b["max_hz"], "t60_fundamental": b["t60"],
            "damping_exponent": b["damping_exponent"], "weight_tilt": -1.5}
    ratios, weights, times = membrane_modes(head)
    ratios, weights, times = map(np.asarray, (ratios, weights, times))
    # Keep the acoustically strongest modes, retaining the lowest as a tuning anchor.
    selected = sorted(set([0, *np.argsort(weights)[-15:].tolist()]))
    ratios, weights, times = (a[selected] for a in (ratios, weights, times))
    f = b["fundamental_hz"] * ratios
    roots = jn_zeros(0, 30) / jn_zeros(0, 1)[0]
    volume = np.array([1/r if np.min(abs(roots-r)) < 1e-6 else 0 for r in ratios])
    # A flexible shell's volume coordinate participates in sealed-cavity
    # compression. This is shell breathing, not an invented Helmholtz vent.
    air = np.r_[volume, -volume, -b["shell_volume"]]
    k = np.diag((2*np.pi*np.r_[f, f*b["bottom_ratio"], b["shell_body_hz"]])**2)
    k += (2*np.pi*b["air_hz"])**2 * np.outer(air, air)
    values, vectors = eigh(k)
    drive = np.r_[weights, np.zeros(len(f)), 0.]
    pickup = np.r_[weights, -.45*weights, .6]
    bottom = np.r_[np.zeros(len(f)), weights, 0.]
    damping = np.r_[1/times, 1/(times*.85), 1/b["shell_body_t60"]]
    t60 = 1/(vectors**2).T.dot(damping)
    return np.sqrt(values)/(2*np.pi), t60, (vectors.T @ drive)*(vectors.T @ pickup), (vectors.T @ drive)*(vectors.T @ bottom)


def bordon_contact(lower, threshold, rng):
    """Intermittent wound-wire contact force driven by snare-side velocity."""
    gate = np.maximum(np.abs(lower)-threshold, 0)
    buzz = gate * np.tanh(lower/threshold) * rng.normal(0, 1, len(lower))
    return lfilter([1, -1], [1, -.3], buzz)


def _tambori_hit(pitch, velocity, c, rng):
    sr, b = c["engine"]["sample_rate"], c["body"]
    n = round(c["engine"]["tail_seconds"] * sr)
    amp = (velocity/127)**1.35
    f, t, w, bottom = tambori_modes(c, .72 if pitch in (37, 41) else .15)
    force = contact(n, sr, b["stick_ms"] * (1.25-.4*velocity/127))
    y = modal_signal(force, f, t, w, sr)
    lower = modal_signal(force, f, t, bottom, sr)
    # A wound string's intermittent head contacts excite a short buzzing resonator.
    # The rectified contact gate is an approximation, not a solved snare/head port.
    buzz = bordon_contact(lower, b["snare_threshold"], rng)
    y += b["snare_gain"] * modal_signal(buzz, [1700, 3100], [.014, .009], [.5, .2], sr)
    shell = modal_signal(contact(n, sr, .2), [b["shell_hz"], b["shell_hz"]*2.31], [.045, .023], [.7, .2], sr)
    if pitch == 41:
        y = .6*y + shell
    elif pitch == 42:
        y = shell
    if pitch == 38:
        amp *= 1.25
    return y * amp


def _castanet_hit(pitch, velocity, c, detune=0):
    sr, b = c["engine"]["sample_rate"], c["body"]
    n = round(c["engine"]["tail_seconds"] * sr)
    ratio = (b["mascle_ratio"] if pitch in (61, 63) else 1) * 2**(detune/1200)
    split = 2**(b["split_cents"]/2400)
    f = np.asarray(b["frequencies"])*ratio
    t, w = np.asarray(b["t60"]), np.asarray(b["weights"])
    force = contact(n, sr, b["contact_ms"]*(1.2-.3*velocity/127))
    y = modal_signal(force, f/split, t, w*.5, sr)
    y += modal_signal(force, f*split, t*.88, w*.5, sr)
    return y * (velocity/127)**1.35


def friction_drum_modes(b):
    """Center-driven (0,n) membrane modes, loaded by the rigid cane, over a ported pot.

    The cane rides with the center of the head and adds mass to every mode.
    Modes have unit center amplitude, so their modal masses scale as J1(z)^2
    and their volume displacements as 2 J1(z)/z. Head stiffness is scaled so
    the cane-loaded head's lowest mode sounds at fundamental_hz. Upper modes
    keep the stroke's transitions sharp: a truncated head is too smooth an
    admittance to snap out of stick.

    The pot's air is a spring between the head's net volume and the air plug
    in the hole through its base, the last degree of freedom: pot_coupling is
    that spring's stiffness against the head's first mode, pot_hz the plug's
    resonance on it with the head held still, pot_q its loss. Head and hole
    radiate together, so below pot_hz the air sloshing through the hole cancels
    the head and the groan's fundamental comes out weak. pot_coupling 0 is a
    closed pot whose air does not load the head.

    Returns mass, stiffness and damping matrices, the radiating volume weights,
    the cane's drive vector and the cane-loaded head's mode frequencies.
    """
    zeros = jn_zeros(0, b["modes"])
    j = j1(zeros)
    membrane = .005*j**2/j[0]**2
    m = np.diag(membrane)+b["cane_mass_ratio"]*membrane[0]
    head = np.diag(membrane*(zeros/zeros[0])**2)
    lowest = math.sqrt(eigh(head, m, eigvals_only=True)[0])
    k = head*(2*np.pi*b["fundamental_hz"]/lowest)**2
    hz = np.sqrt(eigh(k, m, eigvals_only=True))/(2*np.pi)
    # Damping acts on the loaded modes, cane mass included.
    damping = np.diag(np.diag(m)*2*math.log(1000)/(b["t60"]*(zeros/zeros[0])**-b["damping_exponent"]))
    volume = 2*j/zeros
    volume /= volume[0]
    nm = b["modes"]
    air = b["pot_coupling"]*k[0, 0]
    omega = 2*np.pi*b["pot_hz"]
    plug = air/omega**2 if air > 0 else membrane[0]
    weights = np.append(volume, 1.)
    mass = np.zeros((nm+1, nm+1)); mass[:nm, :nm] = m; mass[nm, nm] = plug
    stiffness = air*np.outer(weights, weights); stiffness[:nm, :nm] += k
    loss = np.zeros((nm+1, nm+1)); loss[:nm, :nm] = damping; loss[nm, nm] = plug*omega/b["pot_q"]
    drive = np.append(np.ones(nm), 0.)
    return mass, stiffness, loss, weights, drive, hz


@njit(cache=True)
def _mechanical_stepper(mass, stiffness, damping, sr, drive):
    """Trapezoidal q/v maps: no artificial damping of unforced linear modes.

    The third map is the velocity response to a unit force along `drive`.
    """
    dt = 1/sr
    a = mass + dt*damping/2 + dt*dt*stiffness/4
    inv = np.linalg.inv(a)
    vv = inv @ (mass - dt*damping/2 - dt*dt*stiffness/4)
    vq = inv @ (-dt*stiffness)
    return vv, vq, inv @ drive*dt


@njit(cache=True)
def _friction(x, mu_s, mu_d, vs):
    """Stribeck curve: static friction decays to dynamic with sliding speed."""
    return mu_d+(mu_s-mu_d)*math.exp(-x/vs)


@njit(cache=True)
def _friction_junction(held, hand, normal, sr, mass, stiffness, damping, volume, drive,
                       mu_s, mu_d, vs, release, roughness, randoms):
    """A wet hand sliding along the cane: stick-slip against the drum's admittance.

    Each sample solves the friction junction against the membrane's
    instantaneous center admittance, as in a bowed string: the hand holds the
    cane while the needed force stays under static friction, then the cane
    slips along the Stribeck curve until the relative velocity returns to zero.
    The slip rate locks to the loaded membrane, so the hand speaks as a
    pitched groan; where the hand is slow, the sticks lengthen into creaks.
    Wet-skin roughness perturbs the friction coefficient. After `held`
    samples the hand lets go and the head's velocity is damped with `release`.
    """
    vv, vq, response = _mechanical_stepper(mass, stiffness, damping, sr, drive)
    admittance = drive @ response
    n, nm = len(hand), len(volume)
    q, v = np.zeros(nm), np.zeros(nm)
    y, slipping = np.zeros(n), np.zeros(n)
    stuck = True
    for i in range(n):
        p = normal[i]
        free = vv @ v + vq @ q
        force = 0.
        if i < held and p > 1e-8:
            # Roughness never turns friction negative.
            grip = max(0., 1+roughness*randoms[i])
            relative = hand[i]-drive @ free
            needed = relative/admittance
            if stuck and abs(needed) <= mu_s*p*grip:
                force = needed
            else:
                # The junction line meets the friction curve where the cane
                # slides. Without a sliding root the bisection collapses to
                # zero relative velocity and the hand grips the cane again.
                s = 1. if relative > 0 else -1.
                hi = abs(relative)
                lo = 0.
                for _ in range(40):
                    mid = .5*(lo+hi)
                    if (abs(relative)-mid)/admittance > p*grip*_friction(mid, mu_s, mu_d, vs):
                        lo = mid
                    else:
                        hi = mid
                force = s*(abs(relative)-hi)/admittance
                stuck = hi < 1e-6
                slipping[i] = hi
        else:
            stuck = True
        next_v = free + response*force
        q += (v+next_v)/(2*sr)
        a = (next_v-v)*sr
        v = next_v
        # Head and hole radiate their net volume acceleration;
        # millimetre-scale head motion maps to unit-order sample values.
        y[i] = 1e-3*(volume*a).sum()
        if i >= held:
            v *= math.exp(-1/(release*sr))
    return y, slipping


def hand_profile(b, duration, sr):
    """Hand speed over one stroke, 0..1, and whether the grip lasts through it.

    A preset with `stroke_peak` lifts the hand between strokes: speed follows
    sin(pi*u^w)^1.5 peaking at stroke_peak of the stroke, and the grip follows
    the speed. A preset with `turn_s` rubs continuously: the hand reverses in
    turn_s (raised-cosine half turns at each end of the stroke), eases from
    full speed to `ease` of it along the stroke, and keeps its grip.
    """
    held = max(1, round(duration*sr))
    u = np.arange(held)/held
    if "stroke_peak" in b:
        return np.sin(np.pi*u**(math.log(.5)/math.log(b["stroke_peak"])))**1.5, False
    t = np.arange(held)/sr
    half = min(b["turn_s"]/2, duration/2)
    speed = 1-(1-b["ease"])*np.sin(np.pi/2*u)**2
    if half > 0:
        speed *= np.sin(np.pi/2*np.minimum(1, np.minimum(t, duration-t)/half))**2
    return speed, True


def ximbomba_chain(rows, c, rng=None, diagnostics=False):
    """One player's run of joined strokes, rendered through one hand and one head.

    Each row is a stroke (pitch 48 up, 49 down); a stroke that starts before
    the previous one ends takes the hand over at its onset. The head keeps
    ringing from stroke to stroke, so legato strokes join into one groan.
    Returns audio from the first stroke's onset.
    """
    b, e = c["body"], c["engine"]
    sr = e["sample_rate"]
    start = rows[0]["time"]
    ends = [min(r["off"], nxt["time"]) for r, nxt in zip(rows, rows[1:])]+[rows[-1]["off"]]
    held = max(1, round((rows[-1]["off"]-start)*sr))
    n = held+round(e["tail_seconds"]*sr)
    hand, normal, gain = np.zeros(n), np.zeros(n), np.ones(n)
    asymmetry = b["direction_asymmetry"]
    for k, (row, end) in enumerate(zip(rows, ends)):
        a = round((row["time"]-start)*sr)
        speed, grips = hand_profile(b, end-row["time"], sr)
        speed = speed[:max(0, n-a)]
        span = slice(a, a+len(speed))
        u = np.arange(len(speed))/len(speed)
        direction = 1 if row["pitch"] == 48 else -1
        # Opposite directions differ in speed, profile skew and grip.
        skew = 1+direction*asymmetry*(2*u-1)
        hand[span] = direction*b["hand_speed"]*(1+direction*asymmetry)*speed*skew
        # A row's pressure holds until the next stroke; CC2 events move it.
        until = n if k == len(rows)-1 else round((rows[k+1]["time"]-start)*sr)
        pressure = np.full(until-a, row.get("pressure", .7))
        for time, value in row.get("pressure_events", []):
            pressure[max(0, round((time-row["time"])*sr)):] = value
        lift = speed**.6 if not grips else 1.
        normal[a:until] = b["pressure"]*pressure*(1-direction*asymmetry*.2)
        normal[span] *= lift
        if not grips:
            normal[a+len(speed):until] = 0
        # MIDI velocity sets radiated intensity, not the hand speed: changing
        # speed at every dynamic mark would move the stick/slip regime.
        gain[a:until] = (row["velocity"]/80)**1.35
    mass, stiffness, damping, volume, drive, _ = friction_drum_modes(b)
    rng = np.random.default_rng(e["seed"]) if rng is None else rng
    # Wet-skin friction fluctuates over sub-millisecond patches.
    grip = lfilter([1-math.exp(-2*math.pi*2500/sr)], [1, -math.exp(-2*math.pi*2500/sr)], rng.normal(size=n))
    grip /= np.std(grip) or 1
    y, slipping = _friction_junction(held, hand, normal, sr, mass, stiffness, damping, volume, drive,
                                     b["friction_static"], b["friction_dynamic"], b["stribeck_velocity"],
                                     b["release_s"], b["roughness"], grip)
    # Intensity changes glide over 5 ms rather than stepping.
    pole = math.exp(-1/(.005*sr))
    y *= lfilter([1-pole], [1, -pole], gain, zi=[gain[0]*pole])[0]
    if diagnostics:
        return y, dict(slip_velocity=slipping, hand_velocity=hand, normal_force=normal)
    return y


def ximbomba_stroke(row, c, rng=None, diagnostics=False):
    """A single stroke: a chain of one."""
    return ximbomba_chain([row], c, rng, diagnostics)


def ximbomba_chains(rows, join_s=.05):
    """Group a track's strokes into runs the hand plays without lifting off."""
    chains = []
    for row in sorted(rows, key=lambda r: r["time"]):
        if chains and row["time"] <= max(r["off"] for r in chains[-1])+join_s:
            chains[-1].append(row)
        else:
            chains.append([row])
    return chains


def ximbomba_source_sha256():
    """Hash of the code a ximbomba render runs, so its fit evidence survives edits elsewhere."""
    import hashlib
    import inspect
    parts = [load_config, friction_drum_modes, _mechanical_stepper, _friction, _friction_junction,
             hand_profile, ximbomba_chain, ximbomba_chains, render]
    text = "".join(inspect.getsource(getattr(f, "py_func", f)) for f in parts)+repr(XIMBOMBA_DEFAULTS)
    return hashlib.sha256(text.encode()).hexdigest()


def render(rows, seconds, config, track_id=0):
    """Render independent tracks from time/off/pitch/velocity gesture dictionaries.

    Hits ring naturally beyond note-off; a roll stops creating hits at off.
    Ximbomba strokes less than 50 ms apart share one hand and one head;
    friction ends at each run's last off. CC64 has no percussion meaning.
    """
    c = copy.deepcopy(config)
    model, e, b = c["instrument"], c["engine"], c["body"]
    sr = e["sample_rate"]
    if not math.isfinite(seconds) or not 0 < seconds <= 3600:
        raise ValueError("render duration must be finite and in (0, 3600]")
    out = np.zeros(round(seconds*sr))
    rng = np.random.default_rng(np.random.SeedSequence([e['seed'], track_id, 4099]))
    dancers = b.get("dancers", 1)
    detunes = rng.normal(0, b.get("tuning_cents", 0), dancers)
    biases = rng.normal(0, b.get("timing_ms", 0)/1000, dancers)

    def place(at, y):
        start = round(at*sr)
        count = min(len(y), len(out)-start)
        if count > 0:
            out[start:start+count] += y[:count]

    for row in rows:
        if row["pitch"] not in KEYS[model] or not 1 <= row["velocity"] <= 127 or not all(math.isfinite(row[k]) for k in ("time", "off")) or not 0 <= row["time"] < row["off"] <= seconds:
            raise ValueError(f"invalid {model} gesture {row}")
        if not 0 <= row.get("pressure", .7) <= 1 or any(not math.isfinite(t) or not 0 <= p <= 1 for t, p in row.get("pressure_events", [])):
            raise ValueError("pressure must be 0..1 with finite event times")
        pitch, vel = row["pitch"], row["velocity"]
        rng = note_rng(e['seed'], track_id, row['time'], sr, pitch)
        if model == "ximbomba":
            # Friction strokes render per chain below.
            continue
        roll = pitch == 40 if model == "tambori" else pitch in (62, 63)
        times = [0.]
        levels = [1.]
        if model == "tambori" and pitch == 39:
            times, levels = [0., .027], [.48, 1.]
        elif roll:
            interval = .034 if model == "tambori" else .042
            times = list(np.arange(0, row["off"]-row["time"], interval))
            levels = [(.82**(i % 4) if model == "tambori" else [1., .8, .7, .88][i % 4]) for i in range(len(times))]
        hits = []
        for d in range(dancers):
            for offset, level in zip(times, levels):
                jitter = rng.normal(0, b.get("timing_ms", 0)/2000)
                at = max(0., row["time"]+offset+biases[d]+jitter)
                if roll and at >= row["off"]:
                    continue
                if model == "tambori":
                    y = _tambori_hit(pitch, vel, c, rng)
                else:
                    y = _castanet_hit(pitch, vel, c, detunes[d])
                    if pitch == 64:
                        y += _castanet_hit(61, vel, c, detunes[d])
                hits.append((at, y*level/math.sqrt(dancers)))
        for at, y in hits:
            place(at, y)
    if model == "ximbomba":
        for chain in ximbomba_chains(rows):
            place(chain[0]["time"], ximbomba_chain(chain, c, note_rng(e['seed'], track_id, chain[0]['time'], sr, chain[0]['pitch'])))
    # Static DC rejection, no peak normalization or limiter.
    out = lfilter([1, -1], [1, math.exp(-2*math.pi*10/sr)*-1], out)
    return out*e["output_gain"]
