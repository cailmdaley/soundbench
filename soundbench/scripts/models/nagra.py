"""A film crew's mono tape recorder in a studio: one room microphone, then tape.

The band is heard the way the Nagra rolls of a film shoot hear a session: one
microphone somewhere in the room, every player at their own distance from it,
then a quarter-inch tape machine. The chain, per roll (one machine and mic
position):

Microphone. Each stem is mono-summed, delayed by distance / speed of sound,
scaled by (reference / distance) ** direct_exponent, and darkened by a one-pole
lowpass whose corner falls with distance (hf_corner_hz at the reference
distance, divided by (distance / reference) ** hf_exponent). The room's
reverberant field is the same everywhere in the room, so every stem feeds the
reverb at its undistanced level: the room IR's tail (the direct peak and the
first ir_skip_ms removed, unit energy) scaled by wet_db relative to the direct
sound at the reference distance. A distant player is mostly room; a close one
mostly direct. Room tone (brown noise below a few hundred hertz) is added at
the mic, and the mic's highpass shapes it all.

Record. The program is set to the tape's operating level (the 95th-percentile
400 ms RMS at operating_level_dbfs), pre-emphasised, soft-saturated with a
small bias asymmetry (unity small-signal gain, so only peaks compress and pick
up even and odd harmonics), and de-emphasised: high frequencies saturate first,
as they do on tape.

Playback. The tape is read at a time-varying position: wow and flutter are
sinusoids whose rate and depth wander slowly (smoothed noise), given in cents
of peak pitch deviation; each roll has its own constant speed offset. A tape
change ends one roll (the machine stopping: speed falls over stop_s, then a
short fade), leaves gap_s of silence in which music is lost, and starts the
next roll mid-performance (speed rising over start_s). Playback adds the head
bump (a peaking filter near 60-100 Hz), the HF rolloff (a Butterworth lowpass)
and hiss (white noise, highpassed and lowpassed, at hiss_dbfs RMS), each roll
with its own. Hiss runs under the music, so a silence within a roll is tape,
not digital zero; only the gap at a tape change is empty. A final change to
`stop` winds down at that boundary and truncates the program, followed by
exactly 1.5 seconds of digital silence. At most one stop is allowed, last.

Everything random comes from one seed, so a night is reproducible.
"""
from __future__ import annotations

import math
import tomllib
from pathlib import Path

import numpy as np
from scipy.signal import butter, lfilter, oaconvolve, resample_poly, sosfilt
from soundbench.scripts.paths import ROOMS

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOMS / "nagra.toml"
SAMPLE_RATE = 48_000
CENTS = math.log(2) / 1200


def load_config(path: Path = CONFIG) -> dict:
    return tomllib.loads(Path(path).read_text())


def preset_config(cfg: dict, name: str | None) -> dict:
    """Select a named room/machine preset; no selection preserves the base IR."""
    if name is None:
        return cfg
    if name not in cfg.get("preset", {}):
        raise ValueError(f"unknown Nagra preset: {name}")
    return merged(cfg, cfg["preset"][name])


def shape_room_ir(ir, rt60_s, sr=SAMPLE_RATE):
    """Time-scale an IR to its effective Schroeder -5 to -25 dB decay target."""
    if rt60_s is None or not np.any(ir):
        return ir
    if rt60_s <= 0:
        raise ValueError("room rt60_s must be positive")
    energy = np.cumsum((ir ** 2)[::-1])[::-1]
    db = 10 * np.log10(np.maximum(energy / energy[0], 1e-30))
    mask = (db <= -5) & (db >= -25)
    if mask.sum() < 2:
        raise ValueError("IR has no measurable decay for room preset")
    slope = np.polyfit(np.flatnonzero(mask) / sr, db[mask], 1)[0]
    scale = rt60_s / (-60 / slope)
    shaped = np.interp(np.arange(max(1, round(len(ir) * scale))) / scale,
                       np.arange(len(ir)), ir, right=0)
    return shaped / (np.linalg.norm(shaped) + 1e-20)


def transfer_stereo(mono, cfg, sr=SAMPLE_RATE):
    """Complementary delayed lowpass side; preserve mid and normalize side energy.

    A causal fractional-order one-pole cascade approximates the fitted 1.5 poles.
    Digital-zero gaps stay zero, including the terminal transport silence.
    Coherence is ideal (unity); the measured 0.98 is not independent noise.
    """
    from scipy.special import gammaln
    delay = float(cfg["delay_ms"]) * sr / 1000
    corner, poles = float(cfg["lowpass_hz"]), float(cfg["poles"])
    if delay < 0 or not 0 < corner < sr / 2 or poles <= 0:
        raise ValueError("invalid transfer delay, lowpass or poles")
    a = math.exp(-2 * math.pi * corner / sr)
    k = np.arange(max(2, math.ceil(20 * sr / (2 * math.pi * corner))))
    kernel = np.exp(gammaln(k + poles) - gammaln(poles) - gammaln(k + 1)
                    + k * math.log(a) + poles * math.log1p(-a))
    filtered = lfilter(kernel, [1], mono)
    side = np.interp(np.arange(len(mono)) - delay, np.arange(len(mono)), filtered, left=0)
    side[mono == 0] = 0
    power = np.dot(side, side)
    if power > 0:
        side *= 10 ** (float(cfg["side_mid_db"]) / 20) * math.sqrt(np.dot(mono, mono) / power)
    return np.column_stack((mono + side, mono - side))


def merged(base: dict, override: dict | None) -> dict:
    """Machine defaults with a roll's overrides (one level of tables)."""
    out = {k: dict(v) if isinstance(v, dict) else v for k, v in base.items()}
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key].update(value)
        else:
            out[key] = value
    return out


# ------------------------------------------------------------------ filters

def one_pole_lowpass(x, hz, sr=SAMPLE_RATE):
    a = math.exp(-2 * math.pi * min(hz, 0.45 * sr) / sr)
    return lfilter([1 - a], [1, -a], x)


def peaking(x, hz, db, q, sr=SAMPLE_RATE):
    """RBJ peaking EQ."""
    a = 10 ** (db / 40)
    w = 2 * math.pi * hz / sr
    alpha = math.sin(w) / (2 * q)
    b = np.array([1 + alpha * a, -2 * math.cos(w), 1 - alpha * a])
    den = np.array([1 + alpha / a, -2 * math.cos(w), 1 - alpha / a])
    return lfilter(b / den[0], den / den[0], x)


def shelf(x, hz, db, sr=SAMPLE_RATE, inverse=False):
    """First-order high shelf: unity at DC, `db` above `hz` (bilinear, prewarped)."""
    g = 10 ** (db / 20)
    k = math.tan(math.pi * hz / sr)
    # H(s) = (g s + w) / (s + w) with w = 1 (normalised); bilinear s = (1 - z^-1)/(k(1 + z^-1))
    b = np.array([g + k, k - g])
    a = np.array([1 + k, k - 1])
    if inverse:
        b, a = a, b
    return lfilter(b / a[0], a / a[0], x)


def butter_sos(order, hz, kind, sr=SAMPLE_RATE):
    return butter(order, hz, btype=kind, fs=sr, output="sos")


def smooth_noise(rng, n, hz, sr=SAMPLE_RATE):
    """Unit-SD lowpassed noise: the slow wandering of a machine's speed."""
    # Generated at a 100 Hz control rate and interpolated: cheap and smooth.
    rate = 100
    steps = int(n / sr * rate) + 4
    w = rng.standard_normal(steps)
    a = math.exp(-2 * math.pi * hz / rate)
    w = lfilter([1 - a], [1, -a], lfilter([1 - a], [1, -a], w))
    w = (w - w.mean()) / (w.std() + 1e-12)
    return np.interp(np.arange(n) / sr * rate, np.arange(steps), w)


# ------------------------------------------------------------------ the room mic

def room_ir(path: Path, skip_ms: float, sr=SAMPLE_RATE) -> np.ndarray:
    """The reverberant tail of an IR file: W channel, at sr, direct peak removed, unit energy."""
    from scipy.io import wavfile
    rate, data = wavfile.read(path)
    data = np.asarray(data, dtype=np.float64)
    if data.ndim > 1:
        data = data[:, 0]
    if rate != sr:
        g = math.gcd(rate, sr)
        data = resample_poly(data, sr // g, rate // g)
    peak = int(np.argmax(np.abs(data)))
    start = peak + int(skip_ms / 1000 * sr)
    fade = int(0.002 * sr)
    tail = data.copy()
    tail[:start] = 0
    tail[start:start + fade] *= np.linspace(0, 1, fade)[: len(tail[start:start + fade])]
    return tail / (np.sqrt(np.sum(tail ** 2)) + 1e-20)


def mic(stems, distances, cfg, ir, rng, sr=SAMPLE_RATE):
    """One microphone hearing every stem (mono arrays) at its distance in the room."""
    m, room = cfg["mic"], cfg["room"]
    dry_n = max(len(s) for s in stems)
    def furthest(r):
        if not isinstance(r, dict):
            return max(float(r), .1)
        low = min(0., min(p[1] for p in r['gain_db']))
        return r['far_m'] / 10**(low/(20*m['direct_exponent']))
    max_delay = round(max(furthest(r) for r in distances) / m["speed_of_sound"] * sr)
    n = max(dry_n + len(ir) - 1, dry_n + max_delay)
    direct = np.zeros(n)
    dry = np.zeros(n)
    r0 = m["reference_m"]
    for signal, distance in zip(stems, distances):
        dry[: len(signal)] += signal
        def shaped(r):
            r = max(float(r), .1)
            corner = m['hf_corner_hz'] / (r/r0) ** m['hf_exponent']
            return one_pole_lowpass(signal, corner, sr) * (r0/r) ** m['direct_exponent']
        def heard(r):
            delay = round(max(float(r), .1) / m['speed_of_sound'] * sr)
            field = shaped(r)
            path = np.zeros(n)
            end = min(n, delay+len(field))
            path[delay:end] = field[:end-delay]
            return path
        if isinstance(distance, dict):
            points = np.asarray(distance['gain_db'])
            gain = np.interp(np.arange(n)/sr, points[:,0], points[:,1])
            # Signed approach is relative to the reference distance; retreat
            # attenuates and darkens the direct field without boosting the room.
            low, high = min(0., points[:,1].min()), max(5., points[:,1].max())
            far = distance['far_m'] / 10**(low/(20*m['direct_exponent']))
            near = distance['far_m'] / 10**(high/(20*m['direct_exponent']))
            weight = np.clip((10**(gain/20)-10**(low/20)) /
                             (10**(high/20)-10**(low/20)), 0, 1)
            position = distance['far_m'] / 10**(gain/(20*m['direct_exponent']))
            del gain
            movement = np.clip((far-position)/(far-near), 0, 1)
            del position
            far_delay = round(far/m['speed_of_sound']*sr)
            near_delay = round(near/m['speed_of_sound']*sr)
            source = np.arange(n) - (far_delay+(near_delay-far_delay)*movement)
            del movement
            axis = np.arange(len(signal))
            # One moving propagation path, not two delayed copies that comb-filter.
            moving = np.interp(source,axis,shaped(far),left=0,right=0)*(1-weight)
            moving += np.interp(source,axis,shaped(near),left=0,right=0)*weight
            direct += moving
            del source, axis, weight, moving
        else:
            direct += heard(distance)
    # Overlap-add: one full-length FFT of the program would hold several copies of it at once.
    wet = oaconvolve(dry, ir)[:n] * 10 ** (room["wet_db"] / 20)
    out = direct + wet
    if room.get("tone_dbfs") is not None:
        tone = lfilter([1], [1, -0.995], rng.standard_normal(n))
        tone = one_pole_lowpass(tone, room["tone_hz"], sr)
        tone *= 10 ** (room["tone_dbfs"] / 20) / (tone.std() + 1e-20)
        out = out + tone * _program_rms(out)  # room tone relative to the program's level
    return sosfilt(butter_sos(2, m["highpass_hz"], "highpass", sr), out)


def _program_rms(x, sr=SAMPLE_RATE):
    """95th-percentile RMS over 400 ms blocks: the level of the music when it plays."""
    block = int(0.4 * sr)
    usable = len(x) // block * block
    if usable == 0:
        rms = float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0
        return rms if rms > 1e-4 else 0.0
    rms = np.sqrt(np.mean(x[:usable].reshape(-1, block) ** 2, axis=1))
    if usable < len(x):
        rms = np.append(rms, np.sqrt(np.mean(x[usable:] ** 2)))
    # Below -80 dBFS is not playing; also exclude blocks 60 dB below the peak.
    active = rms[rms > max(1e-4, float(rms.max(initial=0)) * 1e-3)]
    return float(np.percentile(active, 95)) if len(active) else 0.0


# ------------------------------------------------------------------ tape

def record(x, cfg, sr=SAMPLE_RATE):
    """Program at operating level, saturated through the record emphasis."""
    t = cfg["tape"]
    level = _program_rms(x, sr)
    if level == 0:
        return np.zeros_like(x)
    x = x * 10 ** (t["operating_level_dbfs"] / 20) / level
    e = shelf(x, t["emphasis_hz"], t["emphasis_db"], sr)
    k, b = t["drive"], t["asymmetry"]
    slope = k * (1 - math.tanh(k * b) ** 2)
    e = (np.tanh(k * (e + b)) - math.tanh(k * b)) / slope
    return shelf(e, t["emphasis_hz"], t["emphasis_db"], sr, inverse=True)


def speed_curve(n, cfg, rng, sr=SAMPLE_RATE):
    """Relative tape speed (1 = nominal) per output sample: offset, wow and flutter."""
    t = np.arange(n) / sr
    ratio = np.full(n, 2 ** (cfg["speed_cents"] / 1200))
    for name in ("wow", "flutter"):
        w = cfg[name]
        rate = w["hz"] * (1 + w["wander"] * 0.5 * smooth_noise(rng, n, w["wander_hz"], sr))
        depth = w["cents"] * np.clip(1 + w["wander"] * smooth_noise(rng, n, w["wander_hz"], sr), 0, None)
        phase = 2 * np.pi * np.cumsum(rate) / sr + rng.uniform(0, 2 * np.pi)
        ratio *= np.exp(CENTS * depth * np.sin(phase))
    return ratio, t


def splice(rolls, changes, cfg_by_roll, rng, sr=SAMPLE_RATE):
    """Play the recorded rolls in turn, cutting at each tape change.

    rolls: {name: recorded program (same program time base)}; changes: sorted
    [{"at_s", "roll", "gap_s"}]; the first segment plays roll "A" (or the only roll).
    Returns the mono output and a list of segment records.
    """
    stops = [i for i, c in enumerate(changes) if c["roll"] == "stop"]
    if stops and stops != [len(changes) - 1]:
        raise ValueError("tape stop must occur at most once and be last")
    if any(c["roll"] not in rolls and c["roll"] != "stop" for c in changes):
        raise ValueError("unknown tape roll")
    if any(float(c.get("gap_s", 0)) < 0 for c in changes):
        raise ValueError("tape gaps must be nonnegative")
    if any(float(c["at_s"]) <= (float(changes[i - 1]["at_s"]) if i else 0)
           for i, c in enumerate(changes)):
        raise ValueError("tape changes must be strictly increasing")
    first = "A" if "A" in rolls else next(iter(rolls))
    plan = [{"roll": first, "start": 0.0, "gap": 0.0}]
    for c in changes:
        plan.append({"roll": c["roll"], "start": float(c["at_s"]), "gap": float(c.get("gap_s", 0.0))})
    length = max(len(r) for r in rolls.values())
    pieces, records = [], []
    for i, seg in enumerate(plan):
        if seg["roll"] == "stop":
            records.append({"roll": "stop", "output_start_s": round(sum(len(q) for q in pieces) / sr, 4),
                            "seconds": 1.5})
            pieces.append(np.zeros(round(1.5 * sr)))
            break
        cfg = cfg_by_roll[seg["roll"]]
        p = cfg["transport"]
        source = rolls[seg["roll"]]
        position = seg["start"] + seg["gap"]
        if i:
            gap = round(seg["gap"] * sr)
            pieces.append(np.zeros(gap))
        end = plan[i + 1]["start"] if i + 1 < len(plan) else length / sr + 0.5
        out_n = (round(end * sr) - round(seg["start"] * sr) - (gap if i else 0)
                 if i + 1 < len(plan) else int(round((length / sr - position) * sr)))
        if out_n <= 0:
            raise ValueError("tape gap consumes the next segment")
        if i + 1 == len(plan):
            # Generate enough playback to reach the source end, then trim at its
            # integrated speed. Keep the start ramp inside that integration.
            budget = max(out_n, round(out_n / 2 ** (cfg["speed_cents"] / 1200) * 1.25 + sr))
            while True:
                ratio, _ = speed_curve(budget, cfg, rng, sr)
                if i:
                    k = min(budget, round(p["start_s"] * sr))
                    ratio[:k] *= np.linspace(p["start_speed"], 1.0, k)
                integrated = np.cumsum(ratio)
                if integrated[-1] >= len(source) - position * sr:
                    break
                budget *= 2
            out_n = int(np.searchsorted(integrated, len(source) - position * sr)) + 1
            ratio = ratio[:out_n]
        else:
            ratio, _ = speed_curve(out_n, cfg, rng, sr)
        env = np.ones(out_n)
        if i:  # the machine starts: speed rises into place
            k = min(out_n, int(p["start_s"] * sr))
            ramp = np.linspace(p["start_speed"], 1.0, k)
            if i + 1 < len(plan):
                ratio[:k] *= ramp
            env[: int(0.004 * sr)] = np.linspace(0, 1, int(0.004 * sr))[: min(out_n, int(0.004 * sr))]
        if i + 1 < len(plan):  # the machine stops: speed falls, then the sound is gone
            k = min(out_n, int(p["stop_s"] * sr))
            ratio[out_n - k:] *= np.linspace(1.0, p["stop_speed"], k) ** 2
            f = min(out_n, int(p["stop_fade_s"] * sr))
            env[out_n - f:] *= np.linspace(1, 0, f) ** 2
        read = position * sr + np.concatenate(([0.0], np.cumsum(ratio[:-1])))
        y = np.interp(read, np.arange(len(source)), source, left=0.0,
                      right=float(source[-1]) if i + 1 == len(plan) else 0.0)
        y = playback_colour(y, cfg, rng, sr) * env
        records.append({"roll": seg["roll"], "program_start_s": round(position, 4),
                        "output_start_s": round(sum(len(q) for q in pieces) / sr, 4),
                        "seconds": round(out_n / sr, 4)})
        pieces.append(y)
    return np.concatenate(pieces), records


def playback_colour(y, cfg, rng, sr=SAMPLE_RATE):
    t = cfg["tape"]
    y = peaking(y, t["head_bump_hz"], t["head_bump_db"], t["head_bump_q"], sr)
    y = sosfilt(butter_sos(2, t["lowpass_hz"], "lowpass", sr), y)
    hiss = rng.standard_normal(len(y))
    hiss = sosfilt(butter_sos(1, t["hiss_highpass_hz"], "highpass", sr), hiss)
    hiss = sosfilt(butter_sos(2, t["hiss_lowpass_hz"], "lowpass", sr), hiss)
    hiss *= 10 ** (t["hiss_dbfs"] / 20) / (hiss.std() + 1e-20)
    return y + hiss


def render(stems, rolls, changes, cfg, ir, seed, sr=SAMPLE_RATE):
    """stems: [(name, mono array)]; rolls: {roll: {"distances": {name: m}, overrides…}}.

    Returns (mono output, report). Each roll's mic hears the whole program; the
    splice decides which roll is heard when.
    """
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x7A9E]))
    recorded, cfg_by_roll, report = {}, {}, {"rolls": {}}
    for name, roll in rolls.items():
        rcfg = merged(cfg, {k: v for k, v in roll.items() if k != "distances"})
        default = rcfg["mic"]["default_distance_m"]
        dist = [roll.get("distances", {}).get(n, default) for n, _ in stems]
        roll_ir = shape_room_ir(ir, rcfg["room"].get("rt60_s"), sr)
        recorded[name] = record(mic([s for _, s in stems], dist, rcfg, roll_ir, rng, sr), rcfg, sr)
        cfg_by_roll[name] = rcfg
        report["rolls"][name] = {n: d for (n, _), d in zip(stems, dist)}
    y, segments = splice(recorded, changes, cfg_by_roll, rng, sr)
    report["segments"] = segments
    return y, report
