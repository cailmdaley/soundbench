"""Five traveling-wave loops sharing a damped membrane and bridge port.

The hot loop is compiled by numba. The loop phase (loss, dispersion and Thiran
allpass) is compensated at the fundamental, not approximated by a half-sample.
The bridge's symmetric contraction transfers energy to all five strings;
modal feedback is bounded and delayed one sample to avoid an algebraic loop.
"""
from __future__ import annotations

import math
from pathlib import Path
import tomllib

import numpy as np
from numba import njit
from scipy.signal import lfilter
from scipy.special import jn_zeros, jv

CONFIG = Path(__file__).resolve().parents[2] / "instruments/models/banjo.toml"


MEMBRANE_KEYS = {"fundamental_hz", "bridge_radius", "max_hz", "t60_fundamental",
                 "damping_exponent", "weight_tilt"}


def membrane_modes(head, radius_m=0.14):
    """Bridge-driven, radiating modes of an ideal circular membrane.

    f_mn = f1 * j_mn/j_01, with j_mn a zero of J_m. The mass-normalized
    radial shape is J_m(j_mn*r)/J_(m+1)(j_mn); its magnitude drives the
    bridge at radius r (a fraction of the head radius). Nonaxisymmetric
    modes have zero net volume velocity: radiation efficiency is
    eta_m = (ka)**(2*m)/(1 + (ka)**(2*m)), k = 2*pi*f/343 m/s,
    a = radius_m (0.14 m by default). Amplitude uses sqrt(eta_m); m=0 is a monopole (eta=1).
    The radiated weight is |shape| * sqrt(eta_m) * (f/f1)**weight_tilt,
    expressed relative to the fundamental. T60 = T1*(f1/f)**damping_exponent.
    max_hz is the roll-off knee: a half-cosine amplitude taper reaches zero
    at min(2*max_hz, (max_hz+Nyquist)/2). Bessel-order and radial-root bounds follow
    that frequency ceiling (j_mn > m and j_mn >= (n-1/4)*pi), not fixed caps.

    A force-driven mode obeys q'' + 2*gamma*q' + omega**2*q = F/M.
    Its velocity response peaks at 1/(2*M*gamma) = Q/(M*omega),
    gamma = ln(1000)/T60. modal_coefficients scales the resonator's
    input so its peak is proportional to T60, rather than damping-independent.
    """
    j01 = jn_zeros(0, 1)[0]
    ceiling = min(2 * head["max_hz"], (head["max_hz"] + 24000) / 2)
    limit = ceiling / head["fundamental_hz"]
    zero_limit = limit * j01
    radial_count = math.ceil(zero_limit / math.pi + 1)
    modes = []
    for m in range(math.ceil(zero_limit)):
        zeros = jn_zeros(m, radial_count) / j01
        if zeros[0] > limit:
            break
        for ratio, zero in zip(zeros, zeros * j01):
            if ratio > limit:
                break
            frequency = head["fundamental_hz"] * ratio
            ka = 2 * math.pi * frequency * radius_m / 343
            # Logistic form avoids overflow for large multipole orders.
            radiation = 1.0 if m == 0 else math.exp(-0.5 * np.logaddexp(0, -2 * m * math.log(ka)))
            drive = abs(jv(m, zero * head["bridge_radius"]) / jv(m + 1, zero))
            taper = 1.0 if frequency <= head["max_hz"] else 0.5 * (
                1 + math.cos(math.pi * (frequency - head["max_hz"]) / (ceiling - head["max_hz"])))
            weight = drive * radiation * ratio ** head["weight_tilt"] * taper
            modes.append((float(ratio), float(weight),
                          float(head["t60_fundamental"] * ratio ** -head["damping_exponent"])))
    modes.sort()
    fundamental_weight = modes[0][1]
    return ([r for r, _, _ in modes], [w / fundamental_weight for _, w, _ in modes],
            [t for _, _, t in modes])


def load_config(path=CONFIG):
    config = tomllib.loads(Path(path).read_text())
    head = config.get("head", {})
    if isinstance(head, dict) and set(head) == MEMBRANE_KEYS:
        if not (0 < head["bridge_radius"] < 1 and 0 < head["fundamental_hz"] < head["max_hz"] < 24000
                and head["t60_fundamental"] > 0 and head["damping_exponent"] >= 0):
            raise ValueError("invalid membrane head parameters")
        ratios, weights, t60 = membrane_modes(head)
        config["head"] = {"fundamental_hz": head["fundamental_hz"], "ratios": ratios,
                          "weights": weights, "t60_seconds": t60}
    schema = {
        "engine": "sample_rate seed tail_seconds output_gain silence_threshold",
        "strings": "open_midi max_fret t60 loss_mix dispersion release_seconds choke_seconds pitch_smoothing_ms",
        "excitation": "pluck_position position_jitter pick_ms lowpass_hz velocity_corner_exponent velocity_amplitude_exponent thumb_gain thumb_pick_ms hammer_gain pull_gain",
        "head": "fundamental_hz ratios weights t60_seconds",
        "pot": "frequency_hz t60_seconds weight",
        "bridge": "coupling head_feedback",
        "pickups": "head string",
    }
    if set(config) != set(schema):
        raise ValueError(f"model tables must be {sorted(schema)}")
    for name, keys in schema.items():
        optional = {'decay_hz', 'decay_t60', 'polarization_mix', 'polarization_rate', 'polarization_cents', 'dispersion_stages'} if name == 'strings' else ({'exponential_pick', 'pick_ms_strings', 'lowpass_hz_strings', 'pluck_position_strings', 'fractional_pluck'} if name == 'excitation' else ({'peak_knee', 'peak_ceiling'} if name == 'engine' else set()))
        if not isinstance(config[name], dict) or not set(keys.split()) <= set(config[name]) or set(config[name]) - set(keys.split()) - optional:
            raise ValueError(f"{name} parameters must be {keys}")
    for group_name, group in config.items():
        for key, value in group.items():
            if group_name == 'excitation' and key == 'fractional_pluck' and type(value) is bool:
                continue
            values = np.asarray(value).ravel().tolist() if isinstance(value, list) else [value]
            if not all(type(v) in (int, float) and math.isfinite(v) for v in values):
                raise ValueError("model parameters must be finite numbers")
    s, e, h, b, p = (config[k] for k in ("strings", "excitation", "head", "bridge", "pot"))
    if ('decay_hz' in s) != ('decay_t60' in s):
        raise ValueError('decay_hz and decay_t60 must be supplied together')
    if 'decay_hz' in s:
        hz, times = np.asarray(s['decay_hz']), np.asarray(s['decay_t60'])
        if hz.ndim != 1 or len(hz) < 2 or hz[0] <= 0 or hz[-1] > 24000 or np.any(np.diff(hz) <= 0) or times.shape != (5, len(hz)) or np.any(times <= 0):
            raise ValueError('invalid per-string decay curve')
    polar = {'polarization_mix', 'polarization_rate', 'polarization_cents'}
    if polar & set(s):
        if not polar <= set(s) or 'decay_hz' not in s:
            raise ValueError('polarization requires all three arrays and a decay curve')
        if any(np.shape(s[k]) != (5,) for k in polar) or any(not 0 <= x <= .5 for x in s['polarization_mix']) or any(x <= 0 for x in s['polarization_rate']) or any(abs(x) > 20 for x in s['polarization_cents']):
            raise ValueError('invalid polarization parameters')
    if 'dispersion_stages' in s:
        stages = np.asarray(s['dispersion_stages'])
        if 'decay_hz' not in s or stages.shape != (5, 3) or np.any(stages < -.98) or np.any(stages > 0):
            raise ValueError('dispersion_stages needs five rows of three coefficients in -.98..0')
    for key, low, high in [('pick_ms_strings', 0, 5), ('lowpass_hz_strings', 0, 24000), ('pluck_position_strings', 0, .5)]:
        if key in e and (np.shape(e[key]) != (5,) or any(not low < x < high for x in e[key])):
            raise ValueError(f'invalid excitation.{key}')
    if e.get('exponential_pick', 0) not in (0, 1):
        raise ValueError('exponential_pick must be 0 or 1')
    if 'fractional_pluck' in e and type(e['fractional_pluck']) is not bool:
        raise ValueError('excitation.fractional_pluck must be boolean')
    if config["engine"]["sample_rate"] != 48000:
        raise ValueError("banjo uses 48000 Hz")
    for key in ("open_midi", "t60", "loss_mix", "dispersion"):
        if len(s[key]) != 5:
            raise ValueError(f"strings.{key} needs five values")
    if any(not 0 <= x <= 0.5 for x in s["loss_mix"]):
        raise ValueError("loss_mix must be 0..0.5")
    if any(not -0.5 <= x <= 0 for x in s["dispersion"]):
        raise ValueError("dispersion must be -0.5..0 (stiff-string allpass coefficient)")
    if not 0 <= b["coupling"] <= 0.1 or not 0 <= b["head_feedback"] <= 0.01:
        raise ValueError("bridge coupling must be 0..0.1, feedback 0..0.01")
    if not 0 < e["pluck_position"] < 0.5 or not 0 <= e["position_jitter"] < 1:
        raise ValueError("invalid pluck position/jitter")
    if len(h["ratios"]) != len(h["weights"]) or len(h["ratios"]) != len(h["t60_seconds"]):
        raise ValueError("head modal arrays must have equal lengths")
    if any(h["fundamental_hz"] * r >= 24000 for r in h["ratios"]):
        raise ValueError("head modes must be below Nyquist")
    for value in [*s["t60"], *h["t60_seconds"], config["pot"]["t60_seconds"],
                  s["release_seconds"], s["choke_seconds"], s["pitch_smoothing_ms"],
                  e["pick_ms"], e["thumb_pick_ms"], e["lowpass_hz"]]:
        if value <= 0:
            raise ValueError("time constants and cutoff must be positive")
    if sum(h["weights"]) <= 0 or any(x < 0 for x in h["weights"]):
        raise ValueError("modal weights must be nonnegative and nonzero")
    if not 0 < config["pot"]["frequency_hz"] < 24000 or s["max_fret"] < 0:
        raise ValueError("invalid pot frequency/max fret")
    engine = config["engine"]
    if ('peak_knee' in engine) != ('peak_ceiling' in engine):
        raise ValueError('peak_knee and peak_ceiling must be supplied together')
    if 'peak_ceiling' in engine and not 0 < engine['peak_knee'] < engine['peak_ceiling'] < 1:
        raise ValueError('require 0 < peak_knee < peak_ceiling < 1')
    if not 0 <= engine["tail_seconds"] <= 30:
        raise ValueError("tail_seconds must be 0..30")
    if not isinstance(engine["seed"], int) or engine["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not 1e-200 <= engine["silence_threshold"] <= 1e-8 or engine["output_gain"] <= 0:
        raise ValueError("invalid silence threshold/output gain")
    if any(not isinstance(x, int) or not 21 <= x <= 96 for x in s["open_midi"]):
        raise ValueError("open_midi needs integer pitches 21..96")
    if not isinstance(s["max_fret"], int) or max(s["open_midi"]) + s["max_fret"] > 108:
        raise ValueError("max_fret must be an integer; highest pitch must be <=108")
    if not 0 < e["lowpass_hz"] < 24000 or e["velocity_corner_exponent"] < 0 or e["velocity_amplitude_exponent"] <= 0:
        raise ValueError("invalid excitation cutoff/velocity exponents")
    if max(e["pick_ms"], e["thumb_pick_ms"]) > 5:
        raise ValueError("pick pulses must be <=5 ms")
    if any(e[k] < 0 for k in ("thumb_gain", "hammer_gain", "pull_gain")) or p["weight"] < 0:
        raise ValueError("excitation and pot gains must be nonnegative")
    if h["fundamental_hz"] <= 0 or any(r <= 0 for r in h["ratios"]) or any(v < 0 for v in config["pickups"].values()):
        raise ValueError("mode frequencies must be positive; pickup gains nonnegative")
    return config


def assign_string(pitch, thumb, config, explicit=None, *, sounding=None,
                  started=None, last_frets=None):
    """Prefer a free playable string, then low fret with a mild hand-jump cost.

    Held/CC64-sustained strings count as sounding. If all candidates sound,
    steal the earliest-started note. Explicit pins bypass these preferences;
    ordinary voices exclude the short fifth string.
    """
    opens = config["strings"]["open_midi"]
    choices = [explicit - 1] if explicit is not None else ([4] if thumb else list(range(4)))
    if any(i not in range(5) for i in choices):
        raise ValueError("string numbers are 1..5")
    playable = [i for i in choices if 0 <= pitch - opens[i] <= config["strings"]["max_fret"]]
    if not playable:
        raise ValueError(f"MIDI {pitch} cannot be played on the selected banjo strings")
    free = [i for i in playable if sounding is None or not sounding[i]]
    if free:
        return min(free, key=lambda i: (pitch - opens[i] +
                   (0.15 * abs(pitch - opens[i] - last_frets[i]) if last_frets is not None else 0), i))
    return min(playable, key=lambda i: (started[i], i))


@njit(cache=True)
def _coefficients(pitch, sr, mix, dispersion, t60):
    f = 440.0 * 2.0 ** ((pitch - 69.0) / 12.0)
    w = 2.0 * math.pi * f / sr
    loss_phase = math.atan2(mix * math.sin(w), 1.0 - mix + mix * math.cos(w))
    disp_phase = 2.0 * math.atan((1.0 - dispersion) / (1.0 + dispersion) * math.tan(w / 2.0))
    delay = sr / f - (loss_phase + disp_phase) / w
    # Center the Thiran delay on one sample: D in [0.5, 1.5), away from
    # its near-zero-delay pole. Integer delay crossings keep the same tap.
    integer = int(math.floor(delay - 0.5))
    fractional_phase = w * (delay - integer)
    t, u = math.tan(w / 2.0), math.tan(fractional_phase / 2.0)
    a = (t - u) / (t + u)
    gain = math.exp(-math.log(1000.0) / (t60 * f))
    return integer, a, gain


@njit(cache=True)
def _rich_coefficients(pitch, sr, dispersion, bank, lane, stages=np.zeros(3)):
    index = min(len(bank[0])-2, max(0, int(math.floor(pitch-21))))
    fraction = min(1., max(0., pitch-21-index))
    h = bank[lane, index]*(1-fraction) + bank[lane, index+1]*fraction
    f = 440.*2.**((pitch-69.)/12.)
    w = 2*math.pi*f/sr
    real, imag = 0., 0.
    for k in range(len(h)):
        real += h[k]*math.cos(w*k)
        imag -= h[k]*math.sin(w*k)
    phase = -math.atan2(imag, real)
    phase += 2*math.atan((1-dispersion)/(1+dispersion)*math.tan(w/2))
    for coefficient in stages:
        phase += 2*math.atan((1-coefficient)/(1+coefficient)*math.tan(w/2))
    delay = sr/f-phase/w
    if delay < 1.5:
        raise ValueError('loss/dispersion phase exceeds the positive delay budget at this pitch')
    integer = int(math.floor(delay-.5))
    t, u = math.tan(w/2), math.tan(w*(delay-integer)/2)
    return integer, (t-u)/(t+u), h


@njit(cache=True)
def _render(events, pulses, samples, owners_count, sr, opens, t60, mix, dispersion,
            release, choke, smoothing, coupling, feedback, modal_c, modal_r2,
            modal_in, modal_weights, head_pickup, string_pickup, output_gain, threshold, bank, stages,
            continuous=False, glide=1.0, contact=0, contact_keep=1.0, fractional_pluck=False):
    """Run the shared traveling-wave strings, bridge and modal body.

    By default a pluck clears its string and a damping change applies its
    whole per-period loss at the read point at once, so the output steps.
    With continuous=True no state is ever cleared: a pluck adds to the
    string's motion (a new pitch glides over `contact` samples) and the pick,
    for those samples, dips the wave passing the bridge to contact_keep in a
    raised cosine. Damping is a
    per-period loss g(t) approached by a one-pole `glide` per sample, and
    each sample read is attenuated by exp of the mean of log g over its own
    round trip, so a choke is a continuous exponential, never a staircase.
    Damping only accumulates until the string is plucked or fretted again;
    a pluck lifts it at once. Kind-5 rows take an optional T60 override in
    column 9, and every damping gain is floored at 1e-12.
    """
    size = int(sr / (440.0 * 2.0 ** ((np.min(opens) - 69.0) / 12.0))) + 128
    lanes = len(opens)
    rich = bank.shape[1] > 0
    filters = np.zeros((lanes, 65))
    history = np.zeros((lanes, 65))
    stage_x, stage_y = np.zeros((lanes, 3)), np.zeros((lanes, 3))
    delay = np.zeros((lanes, size))
    injection = np.zeros((lanes, size))
    ap_y = np.zeros(lanes)
    disp_x, disp_y, loss_prev = np.zeros(lanes), np.zeros(lanes), np.zeros(lanes)
    pitch = opens.copy()
    start_pitch, target = opens.copy(), opens.copy()
    ramp_start, ramp_length, shape = np.zeros(lanes), np.ones(lanes), np.zeros(lanes)
    owner, token = np.zeros(lanes, np.int64), np.full(lanes, -1, np.int64)
    extra_loss = np.ones(lanes)
    loss_target = np.ones(lanes)
    # Running sum of per-sample log loss; its difference over one period is a round trip's loss.
    log_loss = np.zeros((lanes, size))
    contact_start = np.full(lanes, -(1 << 40), np.int64)
    ints, aps, gains = np.zeros(lanes, np.int64), np.zeros(lanes), np.zeros(lanes)
    for s in range(lanes):
        ints[s], aps[s], gains[s] = _coefficients(pitch[s], sr, mix[s], dispersion[s], t60[s])
        if rich:
            ints[s], aps[s], filters[s] = _rich_coefficients(pitch[s], sr, dispersion[s], bank, s, stages[s])
    modes = len(modal_c)
    y1, y2 = np.zeros((owners_count, modes)), np.zeros((owners_count, modes))
    out = np.zeros((owners_count, samples))
    forces = np.zeros(owners_count)
    waves = np.zeros(lanes)
    head_previous = 0.0
    event_index = 0
    for n in range(samples):
        pos = n % size
        while event_index < len(events) and events[event_index, 0] <= n:
            ev = events[event_index]
            kind, s = int(ev[1]), int(ev[2])
            if kind == 0:
                # A physical string has one loop, including across named MIDI voices.
                if continuous and token[s] >= 0 and abs(pitch[s]-ev[4]) > 1e-9:
                    start_pitch[s], target[s] = pitch[s], ev[4]
                    ramp_start[s], ramp_length[s], shape[s] = n, max(1, contact), 1
                else:
                    if not continuous:
                        delay[s, :] = 0.0
                        history[s, :] = 0.0
                        stage_x[s, :] = stage_y[s, :] = 0.0
                        injection[s, :] = 0.0
                        ap_y[s] = disp_x[s] = disp_y[s] = loss_prev[s] = 0.0
                    pitch[s] = start_pitch[s] = target[s] = ev[4]
                    ramp_length[s] = 1
                    ramp_start[s] = n
                    ints[s], aps[s], gains[s] = _coefficients(pitch[s], sr, mix[s], dispersion[s], t60[s])
                    if rich:
                        ints[s], aps[s], filters[s] = _rich_coefficients(pitch[s], sr, dispersion[s], bank, s, stages[s])
                if continuous and token[s] >= 0:
                    contact_start[s] = n
                owner[s], token[s] = int(ev[3]), int(ev[10])
                # A pick frees the string at once; the round-trip mean keeps the old wave's change smooth.
                extra_loss[s] = loss_target[s] = 1.0
            elif kind == 1:
                if token[s] == int(ev[10]):
                    # Floored so an extreme damp cannot underflow to a zero gain (log 0 in continuous mode).
                    gain = max(1e-12, math.exp(-math.log(1000.0) / ((ev[9] if ev[9] > 0 else release) * (440.0 * 2.0 ** ((pitch[s] - 69.0) / 12.0)))))
                    if continuous:
                        loss_target[s] = min(loss_target[s], gain)
                    else:
                        extra_loss[s] = loss_target[s] = gain
            elif kind in (2, 3, 4):
                if kind == 2 and token[s] != int(ev[10]):
                    event_index += 1
                    continue
                start_pitch[s], target[s] = pitch[s], ev[4]
                ramp_start[s], ramp_length[s], shape[s] = n, max(1, ev[6]), ev[7]
                if not continuous:
                    extra_loss[s] = loss_target[s] = 1.0
                elif kind in (3, 4):
                    # A new fret lifts the damping; a slide carries the string's damping with it.
                    loss_target[s] = 1.0
                if kind in (3, 4):
                    owner[s], token[s] = int(ev[3]), int(ev[10])
            elif kind == 5:
                if token[s] == int(ev[10]) and owner[s] == int(ev[3]):
                    gain = max(1e-12, math.exp(-math.log(1000.0) / ((ev[9] if ev[9] > 0 else choke) * (440.0 * 2.0 ** ((pitch[s] - 69.0) / 12.0)))))
                    if continuous:
                        loss_target[s] = min(loss_target[s], gain)
                    else:
                        extra_loss[s] = loss_target[s] = gain
            if kind in (0, 3, 4):
                travel = max(1.0, ev[8] * sr / (440.0 * 2.0 ** ((ev[4] - 69.0) / 12.0)))
                comb = int(travel)
                fraction = travel - comb if fractional_pluck else 0.0
                for j in range(pulses.shape[1]):
                    injection[s, (pos + j) % size] += pulses[event_index, j]
                    injection[s, (pos + j + comb) % size] -= pulses[event_index, j]
                    if fractional_pluck and fraction > 0.0:
                        injection[s, (pos + j + comb + 1) % size] -= pulses[event_index, j] * fraction
                        injection[s, (pos + j + comb) % size] += pulses[event_index, j] * fraction
            event_index += 1
        common = 0.0
        for s in range(lanes):
            frac = min(1.0, (n - ramp_start[s]) / ramp_length[s])
            if shape[s] == 1:
                frac = frac * frac * (3.0 - 2.0 * frac)
            desired = start_pitch[s] + frac * (target[s] - start_pitch[s])
            new_pitch = pitch[s] + smoothing * (desired - pitch[s])
            if abs(new_pitch - pitch[s]) > 1e-10:
                pitch[s] = new_pitch
                ints[s], aps[s], gains[s] = _coefficients(pitch[s], sr, mix[s], dispersion[s], t60[s])
                if rich:
                    ints[s], aps[s], filters[s] = _rich_coefficients(pitch[s], sr, dispersion[s], bank, s, stages[s])
            read = (pos - ints[s]) % size
            scale = extra_loss[s]
            if continuous:
                extra_loss[s] += glide * (loss_target[s] - extra_loss[s])
                if abs(loss_target[s] - extra_loss[s]) < 1e-12:
                    extra_loss[s] = loss_target[s]
                dent = 1.0
                # The contact never spans a whole round trip, so a high note keeps its new pulse.
                width = min(contact, max(1, ints[s] - 2))
                if n - contact_start[s] < width:
                    dent -= (1.0 - contact_keep) * 0.5 * (1.0 - math.cos(2.0 * math.pi * (n - contact_start[s]) / width))
                total = log_loss[s, (pos - 1) % size] + math.log(extra_loss[s]) / max(1, ints[s])
                log_loss[s, pos] = total
                scale = dent * math.exp(total - log_loss[s, read])
            x = delay[s, read]
            y = aps[s] * x + delay[s, (read - 1) % size] - aps[s] * ap_y[s]
            ap_y[s] = y
            d = dispersion[s] * y + disp_x[s] - dispersion[s] * disp_y[s]
            disp_x[s], disp_y[s] = y, d
            v = ((1.0 - mix[s]) * d + mix[s] * loss_prev[s]) * gains[s] * scale
            loss_prev[s] = d
            if rich:
                for stage in range(3):
                    filtered = stages[s, stage]*d + stage_x[s, stage] - stages[s, stage]*stage_y[s, stage]
                    stage_x[s, stage] = d if abs(d) >= threshold else 0.
                    stage_y[s, stage] = filtered if abs(filtered) >= threshold else 0.
                    d = stage_y[s, stage]
                history[s, n % 65] = d if abs(d) >= threshold else 0.
                v = 0.
                for k in range(65):
                    v += filters[s, k]*history[s, (n-k) % 65]
                v *= scale
            if abs(ap_y[s]) < threshold:
                ap_y[s] = 0.0
            if abs(disp_x[s]) < threshold:
                disp_x[s] = 0.0
            if abs(disp_y[s]) < threshold:
                disp_y[s] = 0.0
            if abs(loss_prev[s]) < threshold:
                loss_prev[s] = 0.0
            waves[s] = v
            common += v / lanes
        forces[:] = 0.0
        for s in range(lanes):
            drive = injection[s, pos]
            injection[s, pos] = 0.0
            v = (1.0 - coupling - feedback) * waves[s] + coupling * common + feedback * head_previous
            delay[s, pos] = v + drive
            # Flush far above IEEE subnormal range, below any audible level.
            if abs(delay[s, pos]) < threshold:
                delay[s, pos] = 0.0
            forces[owner[s]] += waves[s] + drive
        head_previous = 0.0
        for o in range(owners_count):
            head = 0.0
            for m in range(modes):
                y = modal_c[m] * y1[o, m] - modal_r2[m] * y2[o, m] + modal_in[m] * forces[o]
                if abs(y) < threshold:
                    y = 0.0
                y2[o, m], y1[o, m] = y1[o, m], y
                head += modal_weights[m] * y
            head_previous += head
            out[o, n] = output_gain * (head_pickup * head + string_pickup * forces[o])
            if abs(out[o, n]) < threshold:
                out[o, n] = 0.0
    return out


def modal_coefficients(config):
    """Head weights use fundamental-relative units, with a fixed gain reference.

    The fixed force gain, 0.039315, sets the head's amplitude unit, independent
    of the generated mode count. It calibrates the membrane bank to the
    low-mode reference's radiated impulse energy (6.76e-5 at 48 kHz).
    Pot weight is relative to the head fundamental, not the whole modal sum.
    For poles r*exp(±i*theta), H(z) = b/(1 - 2*r*cos(theta)/z + r**2/z**2).
    b = (T60/T1)*(1-r)*|1-r*exp(-2i*theta)|/2 gives
    |H(exp(i*theta))| = T60/(2*T1), the force-to-velocity peak scaling.
    T1 = 0.12 s is a fixed amplitude-unit reference, not the mode's damping;
    changing any T60 changes that mode's response in proportion.
    """
    sr = config["engine"]["sample_rate"]
    h, p = config["head"], config["pot"]
    frequencies = np.array([h["fundamental_hz"] * r for r in h["ratios"]] + [p["frequency_hz"]])
    times = np.array(h["t60_seconds"] + [p["t60_seconds"]])
    radii = np.exp(-np.log(1000) / (sr * times))
    angles = 2 * np.pi * frequencies / sr
    fundamental_weight = h["weights"][np.argmin(h["ratios"])]
    if fundamental_weight <= 0:
        raise ValueError("head fundamental must have nonzero weight")
    weights = 0.03931491719806966 * np.array([w / fundamental_weight for w in h["weights"]] + [p["weight"]])
    reference_time = 0.12
    modal_in = 0.5 * (times / reference_time) * (1 - radii) * np.abs(1 - radii * np.exp(-2j * angles))
    return 2 * radii * np.cos(angles), radii ** 2, modal_in, weights


@njit(cache=True)
def _output_guard(audio, knee, ceiling):
    """Odd, continuously differentiable peak protection, linear below the knee."""
    width = ceiling-knee
    for owner in range(len(audio)):
        for n in range(audio.shape[1]):
            x = audio[owner, n]
            if x > knee:
                audio[owner, n] = knee + width*math.tanh((x-knee)/width)
            elif x < -knee:
                audio[owner, n] = -knee - width*math.tanh((-x-knee)/width)
    return audio


def render(events, seconds, owners=1, config=None):
    """Rows: sample, kind, string(0..4), owner, pitch, velocity, duration,
    shape(0 linear/1 ease), pluck position, release T60 override (0 default),
    unique note token.
    """
    c = config or load_config()
    sr = c["engine"]["sample_rate"]
    e, s, h, p, b = (c[k] for k in ("excitation", "strings", "head", "pot", "bridge"))
    rows = np.asarray(events, dtype=np.float64).reshape(-1, 11)
    if len(rows):
        rows = rows[np.argsort(rows[:, 0], kind="stable")].copy()
    pulse_size = math.ceil(sr * max(e["pick_ms"], e["thumb_pick_ms"]) / 1000) + 64
    if e.get('exponential_pick', 0):
        pulse_size = 512
    pulses = np.zeros((len(rows), pulse_size))
    for i, row in enumerate(rows):
        if row[1] not in (0, 3, 4):
            continue
        rng = np.random.default_rng(np.random.SeedSequence([c["engine"]["seed"], int(row[10]), int(row[2]), int(row[3])]))
        string = int(row[2])
        position = e.get('pluck_position_strings', [e['pluck_position']]*5)[string]
        pick_ms = e.get('pick_ms_strings', [e['thumb_pick_ms'] if j == 4 else e['pick_ms'] for j in range(5)])[string]
        corner = e.get('lowpass_hz_strings', [e['lowpass_hz']]*5)[string]
        row[8] = position * rng.uniform(1 - e["position_jitter"], 1 + e["position_jitter"])
        thumb = row[2] == 4
        length = max(2, round(sr * pick_ms / 1000))
        pulse = np.sin(np.pi * (np.arange(length) + 0.5) / length)
        # Fixed impulse area; hardness changes spectrum rather than note loudness.
        pulse /= pulse.sum()
        amplitude = (row[5] / 127) ** e["velocity_amplitude_exponent"] * (e["thumb_gain"] if thumb else 1)
        if row[1] == 3:
            amplitude *= e["hammer_gain"]
        elif row[1] == 4:
            amplitude *= e["pull_gain"]
        cutoff = corner * max(row[5] / 127, 0.01) ** e["velocity_corner_exponent"]
        alpha = 1 - math.exp(-2 * math.pi * cutoff / sr)
        force = np.zeros(pulse_size)
        if e.get('exponential_pick', 0):
            pole = math.exp(-2/(sr*pick_ms/1000))
            force[:] = (1-pole)*pole**np.arange(pulse_size)
        else:
            force[:length] = pulse
        pulses[i] = amplitude * lfilter([alpha], [1, -(1-alpha)], force)
    modal_c, modal_r2, modal_in, weights = modal_coefficients(c)
    bank = np.empty((5, 0, 65))
    opens = np.array(s['open_midi'], dtype=float)
    times, mixes, dispersions = np.array(s['t60']), np.array(s['loss_mix']), np.array(s['dispersion'])
    if 'decay_hz' in s:
        from soundbench.scripts.models.string_loss import loss_bank
        bank = loss_bank(s)
        if 'polarization_mix' in s:
            second = rows.copy()
            second[:, 2] += 5
            cents = np.array(s['polarization_cents'])
            second[:, 4] += cents[rows[:, 2].astype(int)]/100
            weights2 = np.array(s['polarization_mix'])[rows[:, 2].astype(int)]
            pulses = np.concatenate((pulses*(1-weights2[:, None]), pulses*weights2[:, None]))
            rows = np.concatenate((rows, second))
            order = np.argsort(rows[:, 0], kind='stable')
            rows, pulses = rows[order], pulses[order]
            opens = np.r_[opens, opens+cents/100]
            times, mixes, dispersions = np.tile(times, 2), np.tile(mixes, 2), np.tile(dispersions, 2)
    stages = np.array(s.get('dispersion_stages', [[0.]*3]*5))
    if len(opens) == 10:
        stages = np.tile(stages, (2, 1))
    audio = _render(rows, pulses, round(seconds * sr), owners, sr,
                   opens, times, mixes, dispersions,
                   s["release_seconds"], s["choke_seconds"],
                   1 - math.exp(-1 / (s["pitch_smoothing_ms"] * sr / 1000)),
                   b["coupling"], b["head_feedback"], modal_c,
                   modal_r2, modal_in, weights,
                   c["pickups"]["head"], c["pickups"]["string"],
                   c["engine"]["output_gain"], c["engine"]["silence_threshold"], bank, stages,
                   fractional_pluck=e['fractional_pluck'] if 'fractional_pluck' in e else False)
    if 'peak_ceiling' in c['engine']:
        return _output_guard(audio, c['engine']['peak_knee'], c['engine']['peak_ceiling'])
    return audio
