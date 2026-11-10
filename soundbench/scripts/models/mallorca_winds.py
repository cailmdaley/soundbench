"""Continuous wind waveguides and explicit score controls.

Flute: convected jet delay, cubic edge nonlinearity, lossy bore return.
Reed: quasi-static closing reed with Bernoulli flow driving a positive,
full-period bore return (equivalent harmonic cone, not measured geometry).
The drone uses its own reed and negative half-period cylindrical return.
No sampled audio, generated notes, or per-note phase-reset oscillators.
"""
from __future__ import annotations
import math
from pathlib import Path
import tomllib
import numpy as np
from numba import njit
from scipy.signal import butter, sosfilt, sosfiltfilt, lfilter
from .note_random import wind_noise

ROOT = Path(__file__).resolve().parents[2]
# Raw-reed and breath controls that piece-scoped presets inherit unless they
# set them, with their accepted ranges (shared by both winds).
XEREMIA_DEFAULTS = {"reed_turbulence": 0.4, "tremor_cents": 3.0, "repeat_cut_ms": 30.0, "radiation": 3.0,
                    "formant_hz": 2200.0, "formant_db": 9.0}
FLABIOL_DEFAULTS = {"tremor_cents": 3.0}
WIND_RANGES = {"reed_turbulence": (0, .6), "tremor_cents": (0, 10), "repeat_cut_ms": (0, 80), "radiation": (0, 4),
               "formant_hz": (1000, 4000), "formant_db": (0, 12)}


def load_config(instrument="flabiol", path=None):
    if instrument not in ("flabiol", "xeremia"):
        raise ValueError("instrument must be flabiol or xeremia")
    c = tomllib.loads(Path(path or ROOT / f"instruments/models/{instrument}.toml").read_text())
    required = {"sample_rate", "seed", "output_gain", "lowpass_hz", "noise", "loss_pole",
                "jet_ratio", "pressure", "attack_s", "release_s", "tail_seconds",
                "vibrato_cents", "vibrato_hz", "wobble_cents", "drone_gain", "drone_pitch",
                "squawk_cents", "sag_cents", "tuning_cents", "delay_scale", "bore_correction",
                "pressure_pitch_cents", "breath_noise", "chiff_noise", "squawk_noise"}
    required.add('pitch_range')
    c.setdefault('pitch_range', [74,98] if instrument=='flabiol' else [59,72])
    for key, default in (XEREMIA_DEFAULTS if instrument == 'xeremia' else FLABIOL_DEFAULTS).items():
        required.add(key)
        c.setdefault(key, default)
    if set(c) != required:
        raise ValueError(f"wind config keys must be {sorted(required)}")
    for key, value in c.items():
        if key not in ("tuning_cents", "bore_correction", "pitch_range") and (not isinstance(value, (int, float)) or not math.isfinite(value)):
            raise ValueError(f"invalid {key}")
    if c["sample_rate"] != 48000 or not isinstance(c["seed"], int) or c["seed"] < 0:
        raise ValueError("winds require 48 kHz and nonnegative integer seed")
    if len(c["bore_correction"]) != 2 or not all(math.isfinite(x) for x in c["bore_correction"]):
        raise ValueError("bore_correction needs intercept cents and cents/Hz")
    if len(c["tuning_cents"]) != 12 or not all(math.isfinite(x) and abs(x) <= 100 for x in c["tuning_cents"]):
        raise ValueError("tuning_cents must contain twelve finite offsets within +/-100 cents")
    if not (0 <= c["pressure_pitch_cents"] <= 500 and 0 <= c["wobble_cents"] <= 20
            and all(0 <= c[k] <= .1 for k in ("breath_noise", "chiff_noise", "squawk_noise"))):
        raise ValueError("invalid wind pressure sensitivity or radiation noise")
    if not (0 < c["attack_s"] < 1 and 0 < c["release_s"] <= c["tail_seconds"] <= 5):
        raise ValueError("tail must contain positive attack/release")
    if not (0 <= c["loss_pole"] < .95 and 0 < c["jet_ratio"] < 1 and .5 < c["delay_scale"] < 2):
        raise ValueError("invalid waveguide parameters")
    if not (0 < c["lowpass_hz"] < 20000 and 0 < c["pressure"] < 2 and 0 <= c["noise"] <= .3
            and 0 < c["output_gain"] <= 2 and 0 <= c["drone_gain"] <= 1 and 36 <= c["drone_pitch"] <= 60):
        raise ValueError("invalid pressure, noise, gain, filter or drone")
    for key, (low, high) in WIND_RANGES.items():
        if key in c and not low <= c[key] <= high:
            raise ValueError(f"{key} must be within {low}..{high}")
    if (len(c['pitch_range']) != 2 or any(not isinstance(p,int) for p in c['pitch_range'])
            or not 48 <= c['pitch_range'][0] < c['pitch_range'][1] <= 100):
        raise ValueError('pitch_range must be two ordered MIDI pitches within 48..100')
    return c


def frequency(pitch, config):
    return 440 * 2 ** ((pitch - 69 + config["tuning_cents"][int(pitch) % 12] / 100) / 12)


def _cc_lane(controls, cc, n, sr, default):
    lane = np.full(n, float(default))
    for event in sorted(controls, key=lambda r: r["time"]):
        if event["control"] == cc:
            lane[min(n, round(event["time"] * sr)):] = event["value"]
    return lane


def bag_wander(n, sr, seed, track_id=0):
    """Smooth squeeze/refill variation with seeded 2–4-second knot intervals."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, track_id, 4099]))
    times, values = [0.0], [0.0]
    while times[-1] < n / sr:
        times.append(times[-1] + rng.uniform(2, 4))
        values.append(rng.uniform(-1, 1))
    t = np.arange(n) / sr
    index = np.minimum(np.searchsorted(times, t, side="right") - 1, len(times)-2)
    times, values = np.array(times), np.array(values)
    u = (t-times[index]) / (times[index+1]-times[index])
    u = u*u*u*(10+u*(-15+6*u))
    return values[index] + (values[index+1]-values[index])*u


def reed_formant(hz, db, sr, q=1.2):
    """Peaking biquad: the double reed's own cane resonance colours what the chanter radiates."""
    a = 10**(db/40)
    w = 2*np.pi*hz/sr
    alpha = np.sin(w)/(2*q)
    b = np.array([1+alpha*a, -2*np.cos(w), 1-alpha*a])
    den = np.array([1+alpha/a, -2*np.cos(w), 1-alpha/a])
    return b/den[0], den/den[0]


def bag_tremor(n, sr, seed, track_id=0):
    """Unit-variance 1.5–6 Hz pressure flutter: a player's arm or breath is never still."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, track_id, 5153]))
    rate = 100
    knots = rng.standard_normal(int(n/sr*rate)+rate+2)
    knots = sosfiltfilt(butter(2, [1.5, 6], btype="bandpass", fs=rate, output="sos"), knots)
    knots /= np.std(knots) or 1
    return np.interp(np.arange(n)/sr, np.arange(len(knots))/rate, knots)


def plan(rows, seconds, config, instrument, controls=(), details=False, track_id=0, *, compass=None, cut_s=None):
    """Build sample-rate controls. Xeremia holds last fingering through rests.

    CC2 pressure, CC1 vibrato, CC68 flute slur, CC20 drone gate,
    CC21 drone MIDI pitch, CC22 squawk depth, CC23 end-sag depth.
    CC24 bag gate: default on from first note to last note-off.
    CC25 chanter gate: independent of the drone, default on.
    """
    sr = config["sample_rate"]
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("seconds must be finite and positive")
    n = round(seconds * sr)
    for e in controls:
        if not (math.isfinite(e["time"]) and e["time"] >= 0 and isinstance(e["control"], int)
                and 0 <= e["control"] <= 127 and 0 <= e["value"] <= 127):
            raise ValueError("invalid control event")
        if e["control"] == 21 and not 36 <= e["value"] <= 60:
            raise ValueError("drone pitch CC21 must be MIDI 36..60")
    rows = sorted(rows, key=lambda r: r["time"])
    low, high = compass or config.get('pitch_range', (74,98) if instrument=='flabiol' else (59,72))
    for r in rows:
        if not (math.isfinite(r["time"]) and math.isfinite(r["off"]) and 0 <= r["time"] < r["off"] <= seconds
                and isinstance(r["pitch"], int) and low <= r["pitch"] <= high
                and 1 <= r["velocity"] <= 127):
            raise ValueError(f"invalid {instrument} note: {r}")
    # Deliberate grace overlap is last-onset-wins; simultaneous chords fail.
    if any(a["time"] == b["time"] for a, b in zip(rows, rows[1:])):
        raise ValueError("winds are monophonic; simultaneous notes are not allowed")
    # The chanter's fingers sit on the first note before the bag fills.
    first_pitch = rows[0]["pitch"] if rows and instrument == "xeremia" else low
    freq = np.full(n, frequency(first_pitch, config))
    drive = np.zeros(n)
    loud = np.zeros(n)
    transient = np.zeros(n)
    slur = _cc_lane(controls, 68, n, sr, 0)
    if not rows:
        result = (freq, drive, loud, np.full(n, frequency(config["drone_pitch"], config)), np.zeros(n))
        return (*result, transient) if details else result
    first, last = round(rows[0]["time"] * sr), round(max(r["off"] for r in rows) * sr)
    release = config["release_s"]
    for k, r in enumerate(rows):
        a = round(r["time"] * sr)
        b = round(r["off"] * sr)
        nxt = round(rows[k+1]["time"] * sr) if k+1 < len(rows) else n
        freq[a:nxt] = frequency(r["pitch"], config)
        amp = .35 + .65 * r["velocity"] / 127
        if instrument == "xeremia" and k and rows[k-1]["pitch"] == r["pitch"] and config.get("repeat_cut_ms"):
            # A chanter cannot tongue: a piper parts repeated notes with a
            # quick cut from the next finger up, as written graces would.
            above = next((p for p in (r["pitch"]+2, r["pitch"]+1) if p <= high), None)
            cut = above if above is not None else r["pitch"]-2
            # A very short repeat keeps at least half its length at its own pitch.
            freq[a:a+min(round(config["repeat_cut_ms"]/1000*sr), (nxt-a)//2)] = frequency(cut, config)
        if instrument == "xeremia":
            # Velocity alters effort gently, never tongues or silences the reed.
            loud[a:nxt] = .85 + .15 * amp
            continue
        end = min(n, nxt, b + round(release * sr))
        age = np.arange(end-a) / sr
        env = np.minimum(1, age / config["attack_s"])
        if k and slur[a] >= 64 and rows[k-1]["off"] >= r["time"] - .025:
            env[:] = 1
        else:
            transient[a:end] = np.minimum(1, age/.003)*np.exp(-age/.015)
        tail = np.maximum(0, (np.arange(a, end) - b) / sr)
        env *= np.maximum(0, 1-tail/release) ** 2
        # Every new tongued note explicitly owns the breath envelope.
        drive[a:end] = env
        loud[a:end] = amp
    pressure = _cc_lane(controls, 2, n, sr, 96) / 96
    vib = _cc_lane(controls, 1, n, sr, 0) / 127
    t = np.arange(n) / sr
    cents = config["vibrato_cents"] * vib * np.sin(2*np.pi*config["vibrato_hz"]*t)
    drone_f = np.full(n, frequency(config["drone_pitch"], config))
    drone_gate = np.zeros(n)
    if instrument == "xeremia":
        bag = np.zeros(n)
        bag[first:last] = 1
        if any(e["control"] == 24 for e in controls):
            bag *= _cc_lane(controls, 24, n, sr, 127) / 127
        drive = bag
        # Finite pressure ramps at every bag on/off edge; pitch effects follow them.
        changes = np.diff(np.r_[0, bag > 0, 0].astype(int))
        starts, ends = np.where(changes == 1)[0], np.where(changes == -1)[0]
        squawk = _cc_lane(controls, 22, n, sr, 127) / 127
        sag = _cc_lane(controls, 23, n, sr, 127) / 127
        for k, (a, b) in enumerate(zip(starts, ends)):
            count = min(b-a, round(config["attack_s"] * sr))
            u = np.linspace(0, 1, count)
            smooth = u*u*(3-2*u)
            age = np.arange(count)/sr
            drive[a:a+count] *= (.86+.14*smooth)*np.minimum(1, age/.025)
            settling = -(1-u)**1.3 + .16*np.sin(2*np.pi*5*u)*(1-u)
            cents[a:a+count] += config["squawk_cents"] * squawk[a:a+count] * settling
            transient[a:a+count] = squawk[a:a+count]*(1-u)*np.minimum(1, age/.02)
            next_start = starts[k+1] if k+1 < len(starts) else n
            explicit_cut = cut_s is not None and any(
                e["control"] == 24 and e["value"] == 0 and round(e["time"] * sr) == b for e in controls)
            fall = cut_s if cut_s is not None and (b < last or explicit_cut) else release
            end = min(n, next_start, b + round(fall * sr))
            u = np.arange(end-b) / (fall * sr)
            drive[b:end] = (1-u) ** 2
            if cut_s is None or (b >= last and not explicit_cut):
                cents[b:end] -= config["sag_cents"] * sag[b:end] * u
        sensitivity = config["pressure_pitch_cents"]
        wander = bag_wander(n, sr, config["seed"], track_id)
        if sensitivity:
            pressure *= 1 + config["wobble_cents"] / sensitivity * wander
            # Callers that build their own control dict (the skinpipe) leave these off.
            if config.get("tremor_cents"):
                pressure *= 1 + config["tremor_cents"] / sensitivity * bag_tremor(n, sr, config["seed"], track_id)
        # Both pipes see the same bag pressure; no independent detuning oscillator.
        # Startup and shutdown have their own pressure/pitch curves above.
        cents += sensitivity * (pressure-1)
        pitch_lane = _cc_lane(controls, 21, n, sr, config["drone_pitch"])
        for p in np.unique(pitch_lane):
            drone_f[pitch_lane == p] = frequency(p, config)
        drone_f *= 2 ** (cents / 1200)
        drone_gate = _cc_lane(controls, 20, n, sr, 127) / 127
    elif config.get("tremor_cents"):
        # A player's breath is never perfectly steady: the jet's pressure and
        # the pitch it pulls flutter together.
        flutter = bag_tremor(n, sr, config["seed"], track_id)
        pressure *= 1 + config["tremor_cents"] / max(config["pressure_pitch_cents"], 1) * flutter
        cents += config["tremor_cents"] * flutter
    if instrument == 'xeremia':
        # Ten-millisecond effort slew avoids gain steps on the running reed.
        alpha = -math.expm1(-1/(.01*sr))
        loud = lfilter([alpha], [1, -(1-alpha)], loud)
    freq *= 2 ** (cents / 1200)
    result = (freq, drive * pressure, loud, drone_f, drone_gate)
    return (*result, transient) if details else result


@njit(cache=True)
def _read(buf, pos, delay):
    x = (pos-delay) % len(buf)
    i = int(x)
    f = x-i
    return buf[i]*(1-f) + buf[(i+1) % len(buf)]*f


@njit(cache=True)
def _flute(freq, pressure, loud, noise, sr, pole, jet_ratio, delay_scale):
    bore, jet = np.zeros(4096), np.zeros(4096)
    out = np.zeros(len(freq))
    lp = prev = dc = breath = gain = 0.0
    for i in range(len(out)):
        # Slew controls, not bore history: slurs retain the stored acoustic wave.
        breath += .003 * (pressure[i]-breath)
        gain += .003 * (loud[i]-gain)
        f = freq[i]
        w = 2*np.pi*f/sr
        phase_delay = np.arctan2(pole*np.sin(w), 1-pole*np.cos(w)) / w
        delay = max(3.0, 1.5*sr/f*delay_scale-phase_delay-1)
        pos = i % 4096
        returned = _read(bore, pos, delay)
        lp = (1-pole)*returned + pole*lp
        reflected = -lp
        jet[pos] = (1.1 + .18*gain)*breath*(1+noise[i]) - .5*reflected
        x = _read(jet, pos, delay*jet_ratio)
        edge = max(-1.0, min(1.0, x*(x*x-1)))
        dc = edge-prev+.995*dc
        prev = edge
        bore[pos] = dc+.5*reflected
        out[i] = returned*.3*gain
    return out


@njit(cache=True)
def reed_wave(gamma, incoming):
    """Bernoulli closing-reed scattering, shared by coupled reed instruments."""
    target = gamma-2*incoming
    dp = 0.0
    if target != 0:
        lo, hi = -4.0, 4.0
        for _ in range(18):
            dp = .5*(lo+hi)
            flow = .5*max(0.0, 1-dp)*np.sqrt(abs(dp))*(1 if dp >= 0 else -1)
            if dp+flow > target:
                hi = dp
            else:
                lo = dp
        dp = .5*(lo+hi)
    return gamma-dp-incoming


@njit(cache=True)
def _reed(freq, pressure, noise, sr, pole, cone, delay_scale, turbulence, radiation=0.0):
    """Implicit Bernoulli closing-reed flow in pressure-wave scattering form.

    A cone's effective round trip returns in phase after one period.
    A stopped cylindrical drone returns inverted after half a period.
    Turbulence in the reed channel scales with the flow through it, so the
    reed's noise pulses with each opening rather than hissing steadily.
    The bore's low-pass reflection keeps the low harmonics; its complement,
    the high-passed wave that leaves through the open holes and bell, is what
    radiates the reedy upper partials. `radiation` adds that transmitted
    wave to the internal pressure: a high shelf above the loss corner.
    """
    bore = np.zeros(4096)
    out = np.zeros(len(freq))
    lp = hp = dc = prev = breath = flow = 0.0
    f = freq[0] if len(freq) else 0.0
    # Fingers take a few milliseconds to cover or lift a hole.
    finger = 1-np.exp(-1/(.004*sr))
    smooth = 1-np.exp(-2*np.pi*4000/sr)
    for i in range(len(out)):
        breath += .002 * (pressure[i]-breath)
        f += finger*(freq[i]-f)
        w = 2*np.pi*f/sr
        phase_delay = np.arctan2(pole*np.sin(w), 1-pole*np.cos(w)) / w
        delay = max(3.0, sr/f*(1 if cone else .5)*delay_scale-phase_delay-(1 if cone else -.33))
        pos = i % 4096
        returned = _read(bore, pos, delay)
        lp = (1-pole)*returned+pole*lp
        incoming = .97*lp*(1 if cone else -1)
        # Peak quasi-static flow is about 0.19, at dp = 1/3.
        # Clamping the flow keeps a noisy transient from driving gamma negative.
        gamma = .65*breath*(1+noise[i]+turbulence[i]*min(abs(flow), .25)/.19)
        # Solve dp + flow(dp) = gamma - 2*incoming for outgoing pressure.
        # The closure flow law's negative slope above dp=1/3 supplies gain.
        wave = reed_wave(gamma, incoming)
        dp = gamma-wave-incoming
        flow = .5*max(0.0, 1-dp)*np.sqrt(abs(dp))
        # Remove static flow; the loop retains its full harmonic/odd-mode impedance.
        dc = wave-prev+(.995 if cone else .9999)*dc
        prev = wave
        bore[pos] = dc
        # The divisor keeps a chanter stem's loudness roughly independent of radiation.
        # Holes radiate the reed's 2–4 kHz edge; the transmitted wave is
        # smoothed above 4 kHz, where a gaita's energy is hiss.
        hp += smooth*((returned-lp)-hp)
        out[i] = .3*(returned+radiation*hp)/(1+.1*radiation)
    return out


def render(rows, seconds, config=None, instrument="flabiol", controls=(), track_id=0):
    if instrument not in ("flabiol", "xeremia"):
        raise ValueError("instrument must be flabiol or xeremia")
    c = config or load_config(instrument)
    f, p, loud, df, gate, transient = plan(rows, seconds, c, instrument, controls, details=True, track_id=track_id)
    sounding_f = f.copy()
    sr = c["sample_rate"]
    a, b = c["bore_correction"]
    f = f / 2 ** ((a+b*f)/1200)
    noise = wind_noise(rows, len(f), sr, c['seed'], track_id, 0) * c["noise"]
    if instrument == "flabiol":
        audio = _flute(f, p*c["pressure"], loud, noise, sr, c["loss_pole"], c["jet_ratio"], c["delay_scale"])
    else:
        reed_noise = noise + wind_noise(rows, len(f), sr, c['seed'], track_id, 1)*6*c["squawk_noise"]*transient
        chanter_gate = _cc_lane(controls, 25, len(f), sr, 127) / 127
        chanter_turbulence = wind_noise(rows, len(f), sr, c['seed'], track_id, 3)*c["reed_turbulence"]
        audio = _reed(f, p*chanter_gate*c["pressure"], reed_noise*chanter_gate, sr, c["loss_pole"], True,
                      c["delay_scale"], chanter_turbulence*chanter_gate, c["radiation"]) * loud
        if c["formant_db"]:
            # About 0.27 dB of stem loudness per dB of peak; undo it so the stem keeps its level.
            audio = lfilter(*reed_formant(c["formant_hz"], c["formant_db"], sr), audio)*10**(-.27*c["formant_db"]/20)
        drone_noise = wind_noise(rows, len(f), sr, c['seed'], track_id, 2)*(c["noise"]+6*c["squawk_noise"]*transient)
        drone_turbulence = wind_noise(rows, len(f), sr, c['seed'], track_id, 4)*c["reed_turbulence"]
        drone = _reed(df, p*gate*c["pressure"], drone_noise, sr, .65, False, 1.0, drone_turbulence, c["radiation"])
        audio += c["drone_gain"]*drone
    audio = sosfilt(butter(3, c["lowpass_hz"], fs=sr, output="sos"), audio)
    # Turbulent flow at the exposed fipple/reed radiates outside the bore loop.
    # Keep its seed independent of the acoustic excitation's draws.
    air = sosfilt(butter(2, [1800, 14000], btype="bandpass", fs=sr, output="sos"),
                  wind_noise(rows, len(f), sr, c['seed'], track_id, 8171, normal=True))
    # This fixed filter's noise-power gain at 48 kHz is about 0.52.
    air /= np.sqrt(.52)
    if instrument == "flabiol":
        register = 1.45 + .35*np.clip(np.log2(sounding_f/frequency(74, c))/2, 0, 1)
        audio += air*p*loud*(c["breath_noise"]*register+c["chiff_noise"]*transient)
    else:
        audio += air*p*c["squawk_noise"]*transient
    # Written expression shapes radiated level; CC2 independently controls the bag/jet.
    expression = _cc_lane(controls, 11, len(f), sr, 127)/127
    if any(e['control']==11 for e in controls):
        expression = sosfilt(butter(1, 12, fs=sr, output='sos'), expression)
    audio *= c["output_gain"] * expression
    if not np.isfinite(audio).all() or np.max(np.abs(audio), initial=0) >= 1:
        raise ValueError("unstable or clipping wind render")
    return audio
