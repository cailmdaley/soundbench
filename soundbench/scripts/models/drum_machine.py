"""Circuit-modelled drum machines: TR-808, TR-909 and a Linn-style ROM machine.

Every machine renders one fixed-gain stereo stem at 48 kHz from a list of hits
(time, off, pitch, velocity). The voices share a few primitives:

- a bridged-T resonator rung by a trigger pulse: a TPT state-variable bandpass
  whose state persists across hits (a retrigger adds to the ringing), whose
  frequency rises while the pulse loads the network (the 808 pitch blip), and
  whose output passes an asymmetric soft clip scaled by the trigger level;
- six free-running band-limited square oscillators, never phase-reset, so the
  808's metal catches them at a new relative phase on every hit;
- seeded white noise through analog-style filters and exponential VCAs;
- a frozen low-resolution ROM (the 909's cymbals, the Linn's whole kit) read
  through a zero-order hold, with the VCA after the DAC.

Velocity is a trigger level, not an output gain: on the 808 and 909 it deepens
the bass drum's blip, drives its clipper and makes the snare's noise bark.
A unit name seeds per-box component tolerances; the night's seed changes only
noise and oscillator phases.
"""
from __future__ import annotations
import copy
import math
from pathlib import Path
import tomllib
import zlib

from numba import njit
import numpy as np
from scipy.signal import lfilter, resample_poly

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = {m: ROOT / f"instruments/models/{m}.toml" for m in ("tr808", "tr909", "linn")}
LN1000 = math.log(1000.0)

# Shared GM-following key map; a key a machine lacks is a render error.
KEYMAP = {
    "tr808": {35: "bd", 36: "bd", 37: "rs", 38: "sd", 39: "cp", 40: "sd", 41: "lt", 42: "ch", 43: "mt",
              44: "ch", 45: "ht", 46: "oh", 49: "cy", 56: "cb", 70: "ma", 75: "cl"},
    "tr909": {35: "bd", 36: "bd", 37: "rs", 38: "sd", 39: "cp", 40: "sd", 41: "lt", 42: "ch", 43: "mt",
              44: "ch", 45: "ht", 46: "oh", 49: "cy", 51: "rd"},
    "linn": {35: "bd_soft", 36: "bd", 37: "rs", 38: "sd", 39: "cp", 40: "sd_rim", 41: "lt", 42: "ch", 43: "mt",
             44: "ph", 45: "ht", 46: "oh", 49: "cy", 51: "rd", 56: "cb", 70: "ma"},
}
HAT_CHOKERS = (42, 44)


# ---------------------------------------------------------------- configuration

def _merge(base, over, where):
    for key, value in over.items():
        if key not in base:
            raise ValueError(f"unknown drum-machine setting {where}{key}")
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"{where}{key} must be a table")
            _merge(base[key], value, f"{where}{key}.")
        else:
            kind = type(base[key])
            number = isinstance(value, (int, float)) and not isinstance(value, bool)
            if not (isinstance(value, kind) if kind in (bool, str) else number and (kind is float or isinstance(value, int))):
                raise ValueError(f"{where}{key} must be {kind.__name__}")
            base[key] = float(value) if isinstance(base[key], float) else value


def load_config(machine, path=None, overrides=None):
    """A machine preset, with sidecar overrides merged into its exact schema."""
    if machine not in CONFIGS:
        raise ValueError(f"unknown drum machine {machine!r}")
    template = tomllib.loads(CONFIGS[machine].read_text())
    c = copy.deepcopy(template)
    if path is not None:
        preset = tomllib.loads(Path(path).read_text())
        if preset.get("machine") != machine:
            raise ValueError(f"preset is not a {machine} preset")
        _merge(c, preset, "")
    if overrides:
        _merge(c, overrides, "")
    e = c["engine"]
    if e["sample_rate"] != 48000 or not 0 < e["output_gain"] <= 4 or not 1 <= e["tail_seconds"] <= 10:
        raise ValueError("invalid drum-machine engine controls")
    if e["velocity"] not in ("continuous", "accent") or not 0 <= e["accent"] <= 1:
        raise ValueError("velocity must be 'continuous' or 'accent'; accent 0..1")
    for name, v in c["voices"].items():
        for k, x in v.items():
            if not isinstance(x, (int, float)) or not math.isfinite(x) or (x < 0 and k not in ("pan", "tilt")):
                raise ValueError(f"invalid {machine} voices.{name}.{k}")
        if not -1 <= v.get("pan", 0) <= 1:
            raise ValueError(f"{machine} voices.{name}.pan must be -1..1")
        for k in ("level", "tone", "decay", "tune", "snappy"):
            if k in v and v[k] > 1:
                raise ValueError(f"{machine} voices.{name}.{k} is a 0..1 knob")
    return c


def _seed(*parts):
    return [zlib.crc32(str(p).encode()) for p in parts]


def unit_tolerances(c, machine):
    """Per-box component draws: fixed for a named unit, independent of the night."""
    rng = np.random.default_rng(_seed(machine, c["engine"]["unit"]))
    u = c["unit"]

    def draw(sigma):
        return float(np.clip(rng.normal(0, sigma), -2*sigma, 2*sigma))
    tol = {name: dict(freq=1+draw(u["freq"]), decay=1+draw(u["decay"]), level=10**(draw(u["level_db"])/20))
           for name in sorted(c["voices"])}
    if "metal" in c:
        tol["squares"] = [f*(1+draw(u["squares"])) for f in c["metal"]["hz"]]
    return tol


# ---------------------------------------------------------------- primitives

def _biquad(kind, f, q, sr):
    w = 2*math.pi*min(f, .45*sr)/sr
    alpha, cw = math.sin(w)/(2*q), math.cos(w)
    if kind == "bp":
        b = [alpha, 0, -alpha]
    elif kind == "lp":
        b = [(1-cw)/2, 1-cw, (1-cw)/2]
    else:
        b = [(1+cw)/2, -(1+cw), (1+cw)/2]
    a = [1+alpha, -2*cw, 1-alpha]
    return np.array(b)/a[0], np.array(a)/a[0]


def filt(x, kind, f, q=.7071, sr=48000):
    b, a = _biquad(kind, f, q, sr)
    return lfilter(b, a, x)


def decay(n, t60, sr=48000):
    return np.exp(-LN1000*np.arange(n)/(t60*sr))


def attack(n, ms, sr=48000):
    k = max(1, round(ms*sr/1000))
    e = np.ones(n)
    e[:min(k, n)] = np.linspace(0, 1, k, endpoint=False)[:n]
    return e


def tone_lp(x, tone, sr=48000):
    """A dark-able top: tone 1 is the circuit's own spectrum; lower tones close a 4-pole
    Butterworth low-pass (2 kHz at 0, 11 kHz at 0.75), so the band below the corner keeps its level."""
    if tone >= .999:
        return x
    return filt(filt(x, "lp", 2000*10**tone, .5412, sr), "lp", 2000*10**tone, 1.3066, sr)


def length_for(t60, sr=48000, floor_db=80, cap=6.0):
    return int(min(cap, t60*floor_db/60+.01)*sr)


@njit(cache=True)
def _ring(total, starts, levels, freqs, glides, legato, gates, t60s, blips, snaps, drives, clicks,
          blip_tau, snap_tau, pulse_n, click_hz, click_hp, bias, release_t60, sr):
    """Bridged-T resonator rung by trigger pulses, stateful across every hit of one voice.

    The pulse raises the network's frequency by a fast `snaps[i]` (decaying with `snap_tau`, the
    pulse loading the network) and a slower `blips[i]` (decaying with `blip_tau`);
    `legato` notes glide without re-exciting; other notes glide in during the
    `glides[i]` samples before their onset. A gate closes by raising the damping.
    """
    out = np.zeros(total)
    m = len(starts)
    if m == 0:
        return out
    a_pulse = 1-math.exp(-1/(0.00015*sr))
    a_click = 1-math.exp(-2*math.pi*click_hz/sr)
    a_hp = 1-math.exp(-2*math.pi*click_hp/sr)
    blip_dec = math.exp(-1/(blip_tau*sr))
    snap_dec = math.exp(-1/(snap_tau*sr))
    snap = 0.0
    ic1 = 0.0
    ic2 = 0.0
    blip = 0.0
    px = 0.0
    lc = 0.0
    hc = 0.0
    pulse_end = -1
    pulse_level = 0.0
    tb = math.tanh(bias)
    i = 0
    cur = -1
    n = starts[0]
    while n < total:
        while i+1 < m and starts[i+1] <= n:
            i += 1
        if i != cur:
            cur = i
            if not legato[i]:
                blip = blips[i]
                snap = snaps[i]
                pulse_end = starts[i]+pulse_n
                pulse_level = levels[i]
        f = freqs[i]
        if legato[i] and i > 0 and n < starts[i]+glides[i]:
            u = (n-starts[i])/glides[i]
            f = freqs[i-1]*(freqs[i]/freqs[i-1])**u
        elif i+1 < m and not legato[i+1] and glides[i+1] > 0 and n >= starts[i+1]-glides[i+1]:
            u = (n-(starts[i+1]-glides[i+1]))/glides[i+1]
            f = freqs[i]*(freqs[i+1]/freqs[i])**u
        f *= 1+blip+snap
        x = pulse_level if n < pulse_end else 0.0
        px += (x-px)*a_pulse
        t60 = release_t60 if (gates[i] >= 0 and n >= gates[i]) else t60s[i]
        k = 1/(math.pi*f*t60/6.907755)
        g = math.tan(math.pi*min(f, .45*sr)/sr)
        a1 = 1/(1+g*(g+k))
        a2 = g*a1
        a3 = g*a2
        v3 = px-ic2
        v1 = a1*ic1+a2*v3
        v2 = ic2+a2*ic1+a3*v3
        ic1 = 2*v1-ic1
        ic2 = 2*v2-ic2
        y = v1
        s = drives[i]
        if s > 1e-6:
            # Unity at full scale (|y| = 1) for every drive, so a harder trigger is never quieter.
            y = (math.tanh(s*y+bias)-tb)/(math.tanh(s+bias)-tb)
        lc += (px-lc)*a_click
        hc += (lc-hc)*a_hp
        out[n] = y+clicks[i]*(lc-hc)
        blip *= blip_dec
        snap *= snap_dec
        if n >= pulse_end and abs(ic1)+abs(ic2)+abs(lc)+abs(hc) < 1e-9:
            if i+1 < m:
                nxt = starts[i+1]
                if nxt > n+1:
                    ic1 = 0.0
                    ic2 = 0.0
                    lc = 0.0
                    hc = 0.0
                    px = 0.0
                    blip = 0.0
                    snap = 0.0
                    n = nxt
                    continue
            else:
                break
        n += 1
    return out


def ring(total, notes, v, sr=48000, release_t60=.03, click_hz=4000., click_hp=10., bias=0.):
    """Run one resonator voice; `notes` are dicts with n, L, hz, t60 and optional gate/glide/legato."""
    notes = sorted(notes, key=lambda h: h["n"])
    arr = lambda key, default, dtype=np.float64: np.array([h.get(key, default) for h in notes], dtype=dtype)
    return _ring(total, arr("n", 0, np.int64), arr("L", 1.), arr("hz", 50.), arr("glide", 0, np.int64),
                 arr("legato", False, np.bool_), arr("gate", -1, np.int64), arr("t60", .1), arr("blip", 0.), arr("snap", 0.),
                 arr("drive", 0.), arr("click", 0.), max(v.get("blip_ms", 5.), .1)/1000, max(v.get("snap_ms", 3.), .1)/1000,
                 max(1, round(v.get("pulse_ms", 1.)*sr/1000)), click_hz, click_hp, bias, release_t60, float(sr))


def squares(start, n, freqs, phases, duty, sr=48000):
    """Sum of free-running PolyBLEP squares, sampled at absolute samples start..start+n."""
    t = (start+np.arange(n, dtype=np.float64))/sr
    y = np.zeros(n)

    def blep(p, dt):
        r = np.zeros_like(p)
        lo = p < dt
        x = p[lo]/dt
        r[lo] = x+x-x*x-1
        hi = p > 1-dt
        x = (p[hi]-1)/dt
        r[hi] = x*x+x+x+1
        return r
    for f, ph in zip(freqs, phases):
        p = (ph+f*t) % 1.0
        dt = f/sr
        y += np.where(p < duty, 1.0, -1.0)+blep(p, dt)-blep((p-duty) % 1.0, dt)
    return y


# ---------------------------------------------------------------- trigger levels

def trigger_levels(hits, c, machine):
    """Velocity -> trigger level, with the shared accent bus's coupling and sag."""
    e = c["engine"]
    if machine == "linn":
        for h in hits:
            h["L"] = (h["velocity"]/100)**1.5
        return
    for h in hits:
        v = h["velocity"]
        if e["velocity"] == "accent":
            h["L"] = 1+.8*e["accent"] if v >= 100 else 1.0
        else:
            h["L"] = v/96 if v <= 96 else 1+.8*e["accent"]*(v-96)/31
    by_n = {}
    for h in hits:
        by_n.setdefault(h["n"], []).append(h)
    for group in by_n.values():
        top = max(h["L"] for h in group)
        accented = sum(h["L"] > 1.001 for h in group)
        for h in group:
            h["L"] += e["accent_coupling"]*(top-h["L"])
            if h["L"] > 1:
                h["L"] = 1+(h["L"]-1)*max(.5, 1-e["accent_sag"]*(accented-1))


# ---------------------------------------------------------------- 808 voices

def _resonant(total, hs, v, tol, sr, blip_scale=1., drive=0., t60=None, hz=None, gate=False):
    notes = []
    for h in hs:
        notes.append(dict(n=h["n"], L=h["L"]*v["gain"], hz=(hz or v["hz"])*tol["freq"],
                          t60=(t60 or v["t60"])*tol["decay"], blip=v.get("blip", 0.)*h["L"]**blip_scale,
                          drive=drive*h["L"], click=v.get("click", 0.)*h["L"], gate=h["off_n"] if gate else -1))
    return ring(total, notes, v, sr, click_hz=v.get("click_hz", 4000.), click_hp=v.get("click_hp", 10.))


def bd808(total, hs, v, tol, sr, c):
    """Bass drum: one resonator stream for keys 35 and 36; 35 is the long, gated boom."""
    hz = v["hz_lo"]*(v["hz_hi"]/v["hz_lo"])**v["tune"]
    notes = []
    for h in hs:
        k = {**v, **h.get("knobs", {})}
        long = h["pitch"] == 35
        t60 = k["decay_s"] if k["decay_s"] > 0 else k["t60_lo"]*(k["t60_hi"]/k["t60_lo"])**k["decay"]
        notes.append(dict(n=h["n"], L=h["L"]*k["gain"], hz=hz*tol["freq"],
                          t60=(k["long_t60"] if long else t60)*tol["decay"],
                          blip=(2**(k["sweep_semitones"]/12)-1)*h["L"]**k["blip_accent"],
                          snap=(2**(k["snap_semitones"]/12)-1)*h["L"]**k["blip_accent"], drive=k["drive"]*h["L"],
                          click=k["click"]*h["L"], gate=h["off_n"] if long else -1))
    click_hz = 200*25**v["tone"]
    return [(0, ring(total, notes, dict(blip_ms=v["sweep_ms"]/3, snap_ms=v["snap_ms"], pulse_ms=v["pulse_ms"]), sr,
                     click_hz=click_hz, bias=v["asymmetry"]))]


def bass808(total, hs, v, tol, sr, c):
    """Bass mode: the bass drum's resonator played as a monophonic pitched voice.

    The pulse's ring grows with frequency (a short pulse into a band-pass gives about
    area * 2 pi f), so the pulse height is scaled by 55 Hz / f to keep the level flat across the range.
    """
    hs = sorted(hs, key=lambda h: h["n"])
    notes = []
    for j, h in enumerate(hs):
        k = {**v, **h.get("knobs", {})}
        legato = j > 0 and h["n"] < hs[j-1]["off_n"] and not k["retrigger"]
        hz = 440*2**((h["pitch"]-69)/12)
        glide = min(round(k["glide_ms"]*sr/1000), h["n"]-hs[j-1]["n"]) if j else 0
        # Bass mode is tuned by its player: the pitch is exact, the tolerances shape only decay and level.
        notes.append(dict(n=h["n"], L=h["L"]*k["gain"]*55/hz, hz=hz,
                          t60=k["decay_s"]*tol["decay"], blip=0. if legato else (2**(k["sweep_semitones"]/12)-1)*h["L"]**k["blip_accent"],
                          snap=0. if legato else (2**(k["snap_semitones"]/12)-1)*h["L"]**k["blip_accent"],
                          drive=k["drive"]*h["L"], click=k["click"]*h["L"], gate=h["off_n"],
                          glide=glide, legato=legato))
    # A legato note's gate belongs to the note it continues.
    for j in range(len(notes)-1, 0, -1):
        if notes[j]["legato"]:
            notes[j-1]["gate"] = -1
    return [(0, ring(total, notes, dict(blip_ms=v["sweep_ms"]/3, snap_ms=v["snap_ms"], pulse_ms=v["pulse_ms"]), sr,
                     click_hz=200*25**v["tone"], bias=v["asymmetry"]))]


def tom808(total, hs, v, tol, sr, c):
    hz = v["hz"]*2**((v["tune"]-.5)*v["tune_range"]/12) if v["hz_abs"] <= 0 else v["hz_abs"]
    y = _resonant(total, hs, v, tol, sr, blip_scale=1., hz=hz, t60=v["t60"]*(.5+v["decay"]))
    segs = [(0, y)]
    if v["noise"] > 0:
        for h in hs:
            rng = np.random.default_rng(h["rng"])
            n = length_for(v["noise_t60"], sr)
            x = lfilter([.06], [1, -.94], rng.standard_normal(n))
            segs.append((h["n"], x*decay(n, v["noise_t60"], sr)*attack(n, .5, sr)*v["noise"]*h["L"]))
    return segs


def sd808(total, hs, v, tol, sr, c):
    """Snare: two bridged-T bodies (TONE balances them) plus high-passed snappy noise."""
    lo = dict(v, hz=v["lo_hz"], t60=v["lo_t60"])
    hi = dict(v, hz=v["hi_hz"], t60=v["hi_t60"])
    w_lo, w_hi = 1-.5*v["tone"], .4+.6*v["tone"]
    body = w_lo*_resonant(total, hs, lo, tol, sr)+w_hi*_resonant(total, hs, hi, tol, sr)
    segs = [(0, body)]
    for h in hs:
        rng = np.random.default_rng(h["rng"])
        snappy = min(1., v["snappy"]+(.3 if h["pitch"] == 40 else 0))
        t60 = (v["decay_s"] if v["decay_s"] > 0 else v["noise_t60"])*tol["decay"]
        n = length_for(t60, sr)
        x = filt(filt(rng.standard_normal(n+256), "hp", v["noise_hp"], .7, sr), "lp", v["noise_lp"]*(.6+.6*v["tone"]), .7, sr)[256:]
        segs.append((h["n"], x*decay(n, t60, sr)*attack(n, .3, sr)*snappy*v["noise_gain"]*h["L"]**v["noise_accent"]))
    return segs


def cp808(total, hs, v, tol, sr, c):
    """Clap: band-passed noise through a sawtooth burst envelope plus a reverb tail."""
    segs = []
    for h in hs:
        rng = np.random.default_rng(h["rng"])
        t60 = v["tail_s"]*tol["decay"]
        n = length_for(t60, sr)+round(.05*sr)
        x = filt(rng.standard_normal(n+512), "bp", v["bp_hz"]*tol["freq"], v["bp_q"], sr)[512:]
        hp = filt(filt(rng.standard_normal(n+512), "hp", v["hp_hz"], .7, sr), "lp", v["air_lp"], .7, sr)[512:]
        env = np.zeros(n)
        t = np.arange(n)/sr
        at = 0.
        for k in range(int(v["bursts"])):
            if k:
                at += (v["spread_ms"]+rng.uniform(-1, 1)*v["jitter_ms"])/1000
            u = t-at
            env += np.where(u >= 0, (1-np.exp(-np.maximum(u, 0)/3e-4))*np.exp(-np.maximum(u, 0)/(v["burst_ms"]/1000)), 0)
        u = t-at
        env += v["tail"]*np.where(u >= 0, np.exp(-LN1000*np.maximum(u, 0)/t60)*(1-np.exp(-np.maximum(u, 0)/.002)), 0)
        segs.append((h["n"], (x+v["air"]*hp)*env*v["gain"]*h["L"]))
    return segs


def rs808(total, hs, v, tol, sr, c):
    lo = _resonant(total, hs, dict(v, hz=v["lo_hz"], t60=v["lo_t60"]), tol, sr)
    hi = _resonant(total, hs, dict(v, hz=v["hi_hz"], t60=v["hi_t60"]), tol, sr)
    return [(0, filt(lo+v["hi_mix"]*hi, "hp", v["hp_hz"], .7, sr))]


def cl808(total, hs, v, tol, sr, c):
    return [(0, _resonant(total, hs, v, tol, sr))]


def ma808(total, hs, v, tol, sr, c):
    segs = []
    for h in hs:
        rng = np.random.default_rng(h["rng"])
        n = length_for(v["t60"], sr)
        x = filt(filt(rng.standard_normal(n+256), "hp", v["hp_hz"], .7, sr), "hp", v["hp_hz"], .7, sr)[256:]
        segs.append((h["n"], tone_lp(x*decay(n, v["t60"]*tol["decay"], sr)*attack(n, v["attack_ms"], sr), v["tone"], sr)*v["gain"]*h["L"]))
    return segs


class Metal:
    """The 808's six free-running squares: frequencies from the unit, phases from the night."""

    def __init__(self, c, tol, seed):
        m = c["metal"]
        self.freqs = tol["squares"]
        self.duty = m["duty"]
        self.phases = np.random.default_rng(_seed("metal-phase", seed)).random(len(self.freqs))

    def __call__(self, start, n, which=None, warm=256, oversample=4):
        """The squares from absolute sample start-warm, rendered at 4x and decimated (PolyBLEP alone
        leaves aliases about 20 dB down in the 6-16 kHz hat band)."""
        idx = range(len(self.freqs)) if which is None else which
        x = squares((start-warm)*oversample, (n+warm)*oversample, [self.freqs[i] for i in idx],
                    [self.phases[i] for i in idx], self.duty, sr=48000*oversample)
        return resample_poly(x, 1, oversample)/len(idx)


def vca(x, env, drive, bias=.3):
    """The metal's transistor VCA: a soft, asymmetric clip of the enveloped signal. Loud, it
    intermodulates the squares into the dense cloud of sum and difference tones; quiet, it is linear."""
    if drive <= 0:
        return x*env
    rms = np.sqrt(np.mean(x*x))+1e-12
    tb = math.tanh(bias)
    return (np.tanh(drive*x/rms*env+bias)-tb)/(drive*(1-tb*tb))*rms


def _choke(env, h, chokers, sr, tau_ms):
    later = [n for n in chokers if h["n"] < n < h["n"]+len(env)]
    if later:
        k = later[0]-h["n"]
        env[k:] *= np.exp(-np.arange(len(env)-k)/(tau_ms*sr/1000))
    return env


def hats808(total, hs, v, tol, sr, c, metal=None, chokers=(), opened=False):
    segs, warm = [], 256
    for h in hs:
        t60 = (v["decay_s"] if v["decay_s"] > 0 else v["t60_lo"]*(v["t60_hi"]/v["t60_lo"])**v["decay"])*tol["decay"]
        if h["pitch"] == 44:
            t60 *= .6
        n = length_for(t60, sr)
        x = metal(h["n"], n, warm=warm)
        x = filt(x, "bp", v["bp_hz"]*tol["freq"], v["bp_q"], sr)[warm:]
        env = decay(n, t60, sr)*attack(n, v["attack_ms"], sr)
        if opened:
            env = _choke(env, h, chokers, sr, c["engine"]["choke_ms"])
        y = filt(vca(x, env, v["drive"]), "hp", v["hp_hz"], .7, sr)
        segs.append((h["n"], tone_lp(y, v["tone"], sr)*v["gain"]*h["L"]))
    return segs


def cy808(total, hs, v, tol, sr, c, metal=None, **_):
    """Cymbal: the metal split into three bands, each with its own VCA; TONE trades low for high."""
    segs, warm = [], 256
    for h in hs:
        long = v["t60_lo"]*(v["t60_hi"]/v["t60_lo"])**v["decay"]*tol["decay"]
        n = length_for(long, sr)
        x = metal(h["n"], n, warm=warm)
        low = filt(x, "bp", v["low_hz"], v["band_q"], sr)[warm:]*decay(n, v["low_t60"], sr)
        mid = filt(x, "bp", v["mid_hz"], v["band_q"], sr)[warm:]*decay(n, v["mid_ratio"]*long, sr)
        high = filt(filt(x, "hp", v["high_hz"], .7, sr), "hp", v["high_hz"], .7, sr)[warm:]*decay(n, long, sr)
        y = (1-v["tone"])*low+v["mid"]*mid+(.3+.7*v["tone"])*high
        y = filt(vca(y, attack(n, v["attack_ms"], sr), v["drive"]), "hp", v["post_hp"], .7, sr)
        segs.append((h["n"], tone_lp(y, v["top"], sr)*v["gain"]*h["L"]))
    return segs


def cb808(total, hs, v, tol, sr, c, metal=None, **_):
    """Cowbell: two of the squares through a band-pass and a two-stage envelope."""
    segs, warm = [], 512
    for h in hs:
        n = length_for(v["tail_t60"], sr)
        x = metal(h["n"], n, which=(4, 5), warm=warm)
        x = filt(filt(x, "bp", v["bp_hz"]*tol["freq"], v["bp_q"], sr), "lp", v["lp_hz"], .7, sr)[warm:]
        env = (1-v["tail"])*decay(n, v["fast_t60"], sr)+v["tail"]*decay(n, v["tail_t60"]*tol["decay"], sr)
        segs.append((h["n"], vca(x, env*attack(n, .3, sr), v["drive"])*v["gain"]*h["L"]))
    return segs


# ---------------------------------------------------------------- 909 voices

def _vco(f, shaper):
    """A triangle core waveshaped toward a sine; the VCO starts at phase zero on every trigger."""
    phase = 2*math.pi*np.cumsum(f)/48000
    tri = 2/math.pi*np.arcsin(np.sin(phase))
    return (1-shaper)*tri+shaper*np.sin(phase)


def _mono(segs, fade_n=96):
    """One voice circuit: a new trigger cuts the previous hit's tail."""
    segs = sorted(segs, key=lambda s: s[0])
    for j in range(len(segs)-1):
        start, y = segs[j]
        k = segs[j+1][0]-start
        if k < len(y):
            y = y[:k+fade_n].copy()
            y[k:] *= np.linspace(1, 0, len(y)-k)
            segs[j] = (start, y)
    return segs


def bd909(total, hs, v, tol, sr, c):
    segs = []
    f_end = v["hz_lo"]*(v["hz_hi"]/v["hz_lo"])**v["tune"]*tol["freq"]
    t60 = (v["decay_s"] if v["decay_s"] > 0 else v["t60_lo"]*(v["t60_hi"]/v["t60_lo"])**v["decay"])*tol["decay"]
    for h in hs:
        rng = np.random.default_rng(h["rng"])
        long = h["pitch"] == 35
        n = length_for(t60, sr)
        t = np.arange(n)/sr
        depth = (v["sweep_ratio"]-1)*(1+v["accent_sweep"]*(h["L"]-1))
        f = f_end*(1+depth*np.exp(-t/(v["sweep_ms"]/1000))+v["knock"]*np.exp(-t/.0015))
        y = _vco(f, v["shaper"])*decay(n, t60, sr)
        y *= 1-v["punch"]+v["punch"]*np.exp(-t/(v["punch_ms"]/1000))
        s = v["drive"]*h["L"]
        if s > 1e-6:
            y = np.tanh(s*y)/s
        if not long:
            click = np.zeros(n)
            k = round(.001*sr)
            click[:k] = 1
            burst = filt(rng.standard_normal(n), "lp", v["click_lp"], .7, sr)*np.exp(-t/.003)
            y += v["attack"]*(filt(click, "lp", v["click_lp"], .7, sr)+.5*burst)
        segs.append((h["n"], y*v["gain"]*h["L"]))
    return _mono(segs)


def sd909(total, hs, v, tol, sr, c):
    segs = []
    for h in hs:
        rng = np.random.default_rng(h["rng"])
        n = length_for(max(v["noise_t60"], v["lo_t60"]), sr)
        t = np.arange(n)/sr
        bend = 1+v["bend"]*np.exp(-t/(v["bend_ms"]/1000))
        tone = min(1., v["tone"]+(.3 if h["pitch"] == 40 else 0))
        body = (_vco(v["lo_hz"]*tol["freq"]*bend, .6)*decay(n, v["lo_t60"], sr)
                + v["hi_mix"]*_vco(v["hi_hz"]*tol["freq"]*bend, .6)*decay(n, v["hi_t60"], sr))
        x = filt(filt(rng.standard_normal(n), "lp", 3000*3**tone, .7, sr), "hp", v["noise_hp"], .7, sr)
        noise = x*decay(n, v["noise_t60"]*tol["decay"], sr)*attack(n, .3, sr)*v["snappy"]*h["L"]**v["noise_accent"]
        segs.append((h["n"], (body*h["L"]+v["noise_gain"]*noise)*v["gain"]))
    return _mono(segs)


def tom909(total, hs, v, tol, sr, c):
    segs = []
    hz = v["hz"]*2**((v["tune"]-.5)*v["tune_range"]/12)*tol["freq"] if v["hz_abs"] <= 0 else v["hz_abs"]
    t60 = v["t60"]*(.4+1.2*v["decay"])*tol["decay"]
    for h in hs:
        rng = np.random.default_rng(h["rng"])
        n = length_for(t60, sr)
        t = np.arange(n)/sr
        y = _vco(hz*(1+v["sweep"]*np.exp(-t/(v["sweep_ms"]/1000))), .8)*decay(n, t60, sr)
        y += v["noise"]*filt(rng.standard_normal(n), "lp", 2500, .7, sr)*decay(n, .08, sr)
        segs.append((h["n"], y*v["gain"]*h["L"]))
    return _mono(segs)


# ---------------------------------------------------------------- ROMs

_ROM_CACHE: dict = {}


def modal_plate(rng, seconds, modes, lo, hi, t60_lo, t60_hi, tilt, bloom_ms, noise, noise_t60, sr=48000):
    """A struck plate: inharmonic modes denser at high frequency, high modes blooming late."""
    n = round(seconds*sr)
    t = np.arange(n)/sr
    f = np.sort(lo*(hi/lo)**(rng.random(modes)**.7))
    t60 = t60_lo*(f/lo)**(math.log(t60_hi/t60_lo)/math.log(hi/lo))*rng.uniform(.7, 1.3, modes)
    amp = rng.rayleigh(1, modes)*(f/lo)**tilt
    ph = rng.random(modes)*2*math.pi
    y = np.zeros(n)
    for j in range(0, modes, 16):
        sl = slice(j, j+16)
        tau = bloom_ms/1000*(f[sl]/hi)
        y += np.sum(amp[sl, None]*np.sin(2*math.pi*f[sl, None]*t+ph[sl, None])
                    * np.exp(-LN1000*t/t60[sl, None])*(1-np.exp(-t/np.maximum(tau[:, None], 1e-4))), axis=0)
    y /= np.max(abs(y))+1e-12
    hiss = filt(rng.standard_normal(n), "hp", lo*2, .7, sr)*decay(n, noise_t60, sr)
    y += noise*hiss/(np.max(abs(hiss))+1e-12)
    return y/(np.max(abs(y))+1e-12)


def freeze(source, rom_sr, bits, mu_law, sr=48000):
    """Resample to the ROM rate and quantize with no dither: 6-bit linear or 8-bit companded."""
    g = math.gcd(rom_sr, sr)
    x = resample_poly(source, rom_sr//g, sr//g)
    x /= np.max(abs(x))+1e-12
    steps = 2**(bits-1)-1
    if mu_law:
        mu = 255.
        y = np.sign(x)*np.log1p(mu*abs(x))/math.log1p(mu)
        y = np.round(y*steps)/steps
        return np.sign(y)*np.expm1(abs(y)*math.log1p(mu))/mu
    return np.round(x*steps)/steps


def play_rom(rom, rom_sr, clock, rc_hz, sr=48000, oversample=4):
    """Zero-order hold at the ROM clock, a gentle 2-pole RC, then decimation to the output rate."""
    n = int(len(rom)/(rom_sr*clock)*sr)
    t = np.arange(n*oversample)/(sr*oversample)
    z = rom[np.minimum((t*rom_sr*clock).astype(np.int64), len(rom)-1)]
    a = 1-math.exp(-2*math.pi*rc_hz/(sr*oversample))
    z = lfilter([a], [1, a-1], lfilter([a], [1, a-1], z))
    return resample_poly(z, 1, oversample)


def _plate_source(c, voice, rng):
    v = c["voices"][voice]
    return modal_plate(rng, v["rom_s"], int(v["modes"]), v["lo_hz"], v["hi_hz"], v["mode_t60_lo"], v["mode_t60_hi"],
                       v["tilt"], v["bloom_ms"], v["hiss"], v["hiss_t60"])


def _linn_sources(c, voice, rng):
    """The Linn's ROMs, recorded from the studio's own physical models."""
    from soundbench.scripts.models import drums
    v = c["voices"][voice]
    if "modes" in v:
        return _plate_source(c, voice, rng)
    if voice in ("cb", "ma", "cp"):
        n = round(v["rom_s"]*48000)
        t = np.arange(n)/48000
        if voice == "cb":
            ratios, amps = (1, 2.065, 3.13, 6.06), (1, .5, .3, .15)
            return sum(a*np.sin(2*math.pi*v["hz"]*r*t)*decay(n, v["t60"]/(1+k*.4)) for k, (r, a) in enumerate(zip(ratios, amps)))*attack(n, .4)
        if voice == "ma":
            x = filt(rng.standard_normal(n), "bp", v["hz"], 1.2)
            return x*attack(n, v["attack_ms"])*decay(n, v["t60"])
        hits = cp808(0, [dict(n=0, L=1., rng=_seed("linn-clap", c["engine"]["unit"]))], v, dict(freq=1., decay=1.), 48000, c)
        y = hits[0][1]
        return y[:n] if len(y) >= n else np.pad(y, (0, n-len(y)))
    kc = drums.load_config()
    kc["engine"]["seed"] = int(rng.integers(1 << 30))
    row = dict(pitch=int(v["source_key"]), velocity=int(v["source_velocity"]), time=0., off=.1)
    if row["pitch"] in drums.HATS:
        row["pedal"] = 1. if row["pitch"] in (42, 44) else 0.
    y = drums.hit(row, kc)[:round(v["rom_s"]*48000)]
    if v.get("wires", 0) > 0:
        # The recorded snare's broadband wire rattle, which the model's few wire resonances lack.
        n = len(y)
        rattle = filt(rng.standard_normal(n), "hp", 200, .7)*decay(n, v["wires_t60"])*attack(n, .5)
        y = y/np.max(abs(y))+v["wires"]*rattle/np.max(abs(rattle))
    if v.get("click", 0) > 0:
        # The beater's contact click, which the modal head truncates.
        burst = filt(rng.standard_normal(len(y)), "bp", v["click_hz"], .8)*np.exp(-np.arange(len(y))/(.0025*48000))
        y = y/np.max(abs(y))+v["click"]*burst/np.max(abs(burst))
    return y


def _rom_key(c, machine, voice):
    e = c["engine"]
    return (machine, e["unit"], voice, e["rom_rate"], e["rom_bits"], e["rom_mu_law"], e["rom_rc_hz"], e["rom_flatten"],
            tuple(sorted(c["voices"][voice].items())))


def rom(c, machine, voice):
    """The frozen ROM for one voice of this unit, built once and identical on every hit: (codes, envelope).

    With `rom_flatten` the source's own decay is divided out before quantizing and returned as an
    envelope that the analog VCA re-applies after the DAC, so a 6-bit tail keeps its 6 bits instead
    of truncating to silence and grit.
    """
    key = _rom_key(c, machine, voice)
    if key not in _ROM_CACHE:
        rng = np.random.default_rng(_seed(machine, c["engine"]["unit"], voice, "rom"))
        src = (_plate_source if machine == "tr909" else _linn_sources)(c, voice, rng)
        fade = min(len(src), round(.01*48000))
        src = src.copy()
        src[-fade:] *= np.linspace(1, 0, fade)
        e = c["engine"]
        env = np.ones(len(src))
        if e["rom_flatten"]:
            k = round(.01*48000)
            env = np.sqrt(np.convolve(src*src, np.ones(k)/k, "same"))+1e-4*np.max(abs(src))
            env /= env.max()
        _ROM_CACHE[key] = (freeze(src/env, int(e["rom_rate"]), int(e["rom_bits"]), bool(e["rom_mu_law"])), env)
    return _ROM_CACHE[key]


def rom_voice(total, hs, v, tol, sr, c, machine=None, voice=None, chokers=(), opened=False, mono=True):
    """A ROM read through the DAC and an analog VCA whose decay is the voice's knob."""
    clock = 2**((v["tune"]-.5)*v["tune_range"]/12)*tol["freq"]
    key = ("played", clock, *_rom_key(c, machine, voice))
    if key not in _ROM_CACHE:
        codes, env = rom(c, machine, voice)
        played = play_rom(codes, c["engine"]["rom_rate"], clock, c["engine"]["rom_rc_hz"], sr)
        # The flattened source's envelope, stretched by the playback clock like the ROM itself.
        t = np.arange(len(played))*clock
        _ROM_CACHE[key] = played*np.interp(t, np.arange(len(env)), env, right=0.)
    played = _ROM_CACHE[key]
    t60 = v["vca_t60_lo"]*(v["vca_t60_hi"]/v["vca_t60_lo"])**v["decay"]*tol["decay"]
    segs = []
    for h in hs:
        env = decay(len(played), t60*(.6 if h["pitch"] == 44 and machine != "linn" else 1), sr)
        if opened:
            env = _choke(env, h, chokers, sr, c["engine"]["choke_ms"])
        segs.append((h["n"], tone_lp(played*env, v["tone"], sr)*v["gain"]*h["L"]))
    return _mono(segs) if mono else segs


# ---------------------------------------------------------------- the machines

VOICES = {
    "tr808": dict(bd=bd808, sd=sd808, rs=rs808, cp=cp808, lt=tom808, mt=tom808, ht=tom808, lc=tom808, mc=tom808,
                  hc=tom808, ch=hats808, oh=hats808, cy=cy808, cb=cb808, ma=ma808, cl=cl808, bass=bass808),
    "tr909": dict(bd=bd909, sd=sd909, rs=rs808, cp=cp808, lt=tom909, mt=tom909, ht=tom909,
                  ch=rom_voice, oh=rom_voice, cy=rom_voice, rd=rom_voice),
    "linn": {k: rom_voice for k in ("bd", "bd_soft", "sd", "sd_rim", "rs", "cp", "lt", "mt", "ht",
                                    "ch", "ph", "oh", "cy", "rd", "cb", "ma")},
}


def voice_of(machine, pitch, c, bass=False):
    if bass:
        if not 24 <= pitch <= 60:
            raise ValueError("808 bass mode plays sounding MIDI pitches 24..60 (C1..C4)")
        return "bass"
    name = KEYMAP[machine].get(pitch)
    if name is None:
        raise ValueError(f"{machine} has no voice on MIDI key {pitch}")
    if machine == "tr808" and name in ("lt", "mt", "ht") and c["engine"]["toms"] == "congas":
        name = {"lt": "lc", "mt": "mc", "ht": "hc"}[name]
    return name


def _knob_groups(hits):
    """Hits per voice; voices without per-hit knob handling split further by their knob rides."""
    groups = {}
    for h in hits:
        streamed = h["voice"] in ("bd", "bass")
        key = (h["voice"], () if streamed else tuple(sorted(h.get("knobs", {}).items())))
        groups.setdefault(key, []).append(h)
    return [(key[0], hs) for key, hs in sorted(groups.items())]


def render(rows, seconds, machine, config=None, bass=False):
    """Fixed-gain stereo stem for one track's hits; `bass` plays the 808 bass drum as a pitched voice.

    A row may carry `knobs`, overrides of its voice's knobs for that hit (the sidecar's rides).
    """
    c = load_config(machine) if config is None else config
    sr, e = c["engine"]["sample_rate"], c["engine"]
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("duration must be positive")
    total = round(seconds*sr)
    hits = []
    for j, r in enumerate(sorted(rows, key=lambda r: (r["time"], r["pitch"]))):
        if (not 1 <= r["velocity"] <= 127 or not np.isfinite([r["time"], r["off"]]).all()
                or r["time"] < 0 or r["off"] <= r["time"]):
            raise ValueError("invalid drum-machine hit; every note needs a velocity and an explicit release")
        if r["off"]+e["tail_seconds"] > seconds+1/sr:
            raise ValueError("render duration must include the final release and tail")
        voice = voice_of(machine, r["pitch"], c, bass)
        for k, x in r.get("knobs", {}).items():
            if k not in c["voices"][voice] or isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0:
                raise ValueError(f"invalid knob ride {voice}.{k} = {x!r}")
        hits.append(dict(r, n=round(r["time"]*sr), off_n=round(r["off"]*sr), voice=voice,
                         rng=_seed(machine, e["seed"], j, r["pitch"])))
    trigger_levels(hits, c, machine)
    tol = unit_tolerances(c, machine)
    metal = Metal(c, tol, e["seed"]) if machine == "tr808" else None
    chokers = sorted(h["n"] for h in hits if h["pitch"] in HAT_CHOKERS)
    out = np.zeros((total, 2))
    for name, hs in _knob_groups(hits):
        v, fn = c["voices"][name], VOICES[machine][name]
        if fn not in (bd808, bass808):
            v = {**v, **hs[0].get("knobs", {})}
        extra = {}
        if fn in (hats808, cy808, cb808):
            extra = dict(metal=metal, chokers=chokers, opened=name == "oh") if fn is hats808 else dict(metal=metal)
        elif fn is rom_voice:
            extra = dict(machine=machine, voice=name, chokers=chokers, opened=name == "oh",
                         mono=name not in ("cy", "rd"))
        segs = fn(total, hs, v, tol[name], sr, c, **extra)
        gain = v["level"]*tol[name]["level"]
        angle = (v["pan"]+1)*math.pi/4
        for start, y in segs:
            size = min(len(y), total-start)
            if size > 0:
                out[start:start+size, 0] += y[:size]*gain*math.cos(angle)
                out[start:start+size, 1] += y[:size]*gain*math.sin(angle)
    return out*e["output_gain"]
