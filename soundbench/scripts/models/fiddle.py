"""Four bowed velocity waveguides, passive bridge mixing and modal radiation.

The bow junction solves static adhesion or a Stribeck sliding branch against
string impedance. No oscillator schedules slip events and no samples synthesize
sound. String numbers are high to low; all event times are seconds.
"""
from __future__ import annotations
from functools import lru_cache
import copy
import json
import math
from pathlib import Path
import tomllib

import numpy as np
from numba import njit
from scipy.signal import butter, firwin, firwin2, freqz, lfilter, minimum_phase, oaconvolve, sosfilt
from soundbench.scripts.models.string_loss import design_loss
import soundbench.scripts.models.fiddle_player as fiddle_player
from soundbench.scripts.models.fiddle_player import normal_force, bow_shape, grit_controls, helmholtz, helmholtz_score

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'instruments/models/fiddle.toml'
TUNINGS = {'standard': [76, 69, 62, 55], 'AEAE': [76, 69, 64, 57], 'ADAE': [76, 69, 62, 57],
           'GDGD': [74, 67, 62, 55]}


def load_config(path=CONFIG, tuning='standard'):
    c = tomllib.loads(Path(path).read_text())
    if set(c) != {'engine', 'strings', 'bow', 'body'}:
        raise ValueError('expected engine, strings, bow, body')
    for section in c.values():
        for value in section.values():
            if not np.all(np.isfinite(np.asarray(value, dtype=float))):
                raise ValueError('parameters must be finite')
    e, s, b, body = (c[k] for k in ('engine', 'strings', 'bow', 'body'))
    if e['sample_rate'] != 48000 or e['output_gain'] <= 0 or not 0.5 <= e['tail_seconds'] <= 10 or type(e['seed']) is not int or e['seed'] < 0:
        raise ValueError('invalid engine')
    if tuning not in TUNINGS:
        raise ValueError('tuning must be standard, AEAE, ADAE or GDGD')
    s['open_midi'] = TUNINGS[tuning].copy()
    if np.shape(s['impedance']) != (4,) or min(s['impedance']) <= 0 or not 0 <= s['coupling'] <= .05 or not -.3 <= s['dispersion'] <= 0 or not .03 <= s['release_seconds'] <= 1 or not 0 <= s['max_semitones'] <= 24:
        raise ValueError('invalid strings')
    if len(s['decay_hz']) != len(s['decay_t60']) or len(s['decay_hz']) < 2 or min(s['decay_hz']+s['decay_t60']) <= 0 or np.any(np.diff(s['decay_hz']) <= 0):
        raise ValueError('invalid loss curve')
    if not 0 < b['mu_dynamic'] < b['mu_static'] <= 2 or not .002 <= b['stribeck_velocity'] <= .1 or not .002 <= b['force'] <= 1 or not .01 <= b['speed'] <= .5 or not .04 <= b['position'] <= .3 or not .001 <= b['attack_seconds'] <= .1 or not .001 <= b['change_seconds'] <= .05:
        raise ValueError('invalid bow')
    if not 0 <= b['roughness'] <= .2 or not 200 <= b['roughness_hz'] <= 12000:
        raise ValueError('invalid bow roughness')
    if not 0 <= b['pressure_wobble'] <= .3 or not 0 <= b['speed_wobble'] <= .2 or not 1 <= b['wobble_hz'][0] < b['wobble_hz'][1] <= 400:
        raise ValueError('invalid bow wobble')
    f, db = np.asarray(body['envelope_hz']), np.asarray(body['envelope_db'])
    if f.ndim != 1 or len(f) < 4 or f.shape != db.shape or np.any(np.diff(f) <= 0) or f[0] < 50 or f[-1] > 20000 or np.any(abs(db) > 60):
        raise ValueError('invalid body envelope')
    if type(body['ripple_modes']) is not int or not 0 <= body['ripple_modes'] <= 200 or not 0 <= body['ripple_gain'] <= 3 or type(body['ripple_seed']) is not int or not -6 <= body['tilt_db_per_octave'] <= 6:
        raise ValueError('invalid body ripple or tilt')
    return c


def assign_string(pitch, config, explicit=None, occupied=()):
    if explicit is not None and (type(explicit) is not int or explicit not in range(1, 5)):
        raise ValueError('string must be 1..4, high to low')
    opens = config['strings']['open_midi']
    choices = [explicit-1] if explicit is not None else range(4)
    choices = [s for s in choices if 0 <= pitch-opens[s] <= config['strings']['max_semitones']]
    free = [s for s in choices if s not in occupied]
    if not free:
        raise ValueError(f'no free playable string for MIDI {pitch}; pin double stops')
    return min(free, key=lambda s: (pitch-opens[s], s))


# The bow junction runs at twice the output rate. Slip times fall on the
# kernel's sample grid; at 48 kHz that quantization repeats every few periods
# and leaves sidebands about 20 dB under the harmonics in the 2-5 kHz band.
OVERSAMPLE = 2
RATE = 48000*OVERSAMPLE


@lru_cache(maxsize=8)
def _loss(serialized):
    s = json.loads(serialized)
    bank, delay = [], []
    for p in range(55, 101):
        f = 440*2**((p-69)/12)
        h = design_loss(s['decay_hz'], np.log(1000)/np.array(s['decay_t60']), f, fs=RATE)
        _, response = freqz(h, worN=[2*np.pi*f/RATE])
        a = s['dispersion']
        z = np.exp(-2j*np.pi*f/RATE)
        ap = (a+z)/(1+a*z)
        bank.append(h)
        delay.append(-np.angle(response[0]*ap)/(2*np.pi*f/RATE))
    return np.array(bank), np.array(delay)


@njit(cache=True)
def _read(buf, index, delay):
    pos = index-delay
    i = math.floor(pos)
    frac = pos-i
    return (1-frac)*buf[i % len(buf)] + frac*buf[(i+1) % len(buf)]


@njit(cache=True)
def _kernel(pitches, force, speed, beta, damping, rough, bank, phase, impedance, dispersion, coupling, mu_s, mu_d, vs, rate, state=None):
    n = pitches.shape[1]
    if state is None:
        left, right = np.zeros((4, 2048)), np.zeros((4, 2048))
        hist = np.zeros((4, 65))
        apx, apy, previous = np.zeros(4), np.zeros(4), np.zeros(4)
        adhered = np.ones(4, dtype=np.uint8)
        start = 0
    else:
        left, right, hist = state[0].copy(), state[1].copy(), state[2].copy()
        apx, apy, previous = state[3].copy(), state[4].copy(), state[5].copy()
        adhered, start = state[6].copy(), state[7]
    out = np.zeros((4, n))
    stick = np.zeros((4, n), dtype=np.uint8)
    for local in range(n):
        i = start+local
        # Convex bridge scattering is passive, including the unbowed strings.
        common = np.sum(previous)/4
        next_bridge = np.zeros(4)
        for s in range(4):
            pitch = pitches[s, local]
            f = 440*2**((pitch-69)/12)
            k = min(44, max(0, int(math.floor(pitch))-55))
            frac = min(1., max(0., pitch-(k+55)))
            comp = phase[k]*(1-frac)+phase[k+1]*frac
            period = rate/f
            dl = max(2., beta[s, local]*period)
            dr = max(2., (1-beta[s, local])*period-comp-.15)
            a = -_read(left[s], i, dl)
            bridge = _read(right[s], i, dr)
            hist[s, i % 65] = (1-coupling)*bridge+coupling*common
            filtered = 0.
            for j in range(65):
                filtered += ((1-frac)*bank[k, j]+frac*bank[k+1, j])*hist[s, (i-j) % 65]
            dispersed = dispersion*filtered+apx[s]-dispersion*apy[s]
            apx[s], apy[s] = filtered, dispersed
            b = -dispersed
            free = a+b
            relative = speed[s, local]-free
            normal = force[s, local]
            required = 2*impedance[s]*relative
            friction = 0.
            # Rosin and hair irregularity scale both friction coefficients, so
            # bow noise is born at the contact and jitters each slip.
            grain = 1.+rough[s, local]
            ms, md = mu_s*grain, mu_d*grain
            if normal > 1e-10:
                if abs(required) <= (ms if adhered[s] else md)*normal:
                    friction = required
                    stick[s, local] = 1
                    adhered[s] = 1
                else:
                    # On this branch relative speed stays away from zero.
                    # Bisection solves F=N*mu(v_b-v_free-F/(2Z)).
                    adhered[s] = 0
                    sign = 1. if relative >= 0 else -1.
                    lo, hi = md*normal, min(ms*normal, abs(required))
                    for iteration in range(12):
                        mid = (lo+hi)/2
                        slip = max(0., abs(relative)-mid/(2*impedance[s]))
                        mu = md+(ms-md)*math.exp(-(slip/vs)**2)
                        if mid > normal*mu:
                            hi = mid
                        else:
                            lo = mid
                    friction = sign*(lo+hi)/2
            added = friction/(2*impedance[s])
            gain = damping[s, local]
            left[s, i % 2048] = (b+added)*gain
            right[s, i % 2048] = (a+added)*gain
            next_bridge[s] = bridge
            out[s, local] = bridge
        previous = next_bridge
    return out, stick, (left, right, hist, apx, apy, previous, adhered, start+n)


@lru_cache(maxsize=8)
def _body(serialized):
    """Minimum-phase radiation envelope and seeded fine-structure modes."""
    b = json.loads(serialized)
    knots, db = np.log(b['envelope_hz']), np.asarray(b['envelope_db'], float)
    grid = np.linspace(0, 24000, 8193)
    x = np.log(np.maximum(grid, 20))
    # Outside the fitted band the envelope falls at 12 dB per octave.
    level = np.interp(x, knots, db)-12*(np.maximum(0, knots[0]-x)+np.maximum(0, x-knots[-1]))/math.log(2)
    level += b['tilt_db_per_octave']*np.maximum(0, x-math.log(1000))/math.log(2)
    level -= np.interp(math.log(1000), knots, db)
    fir = minimum_phase(firwin2(2047, grid, 10**(level/10), fs=48000), n_fft=65536)
    rng = np.random.default_rng(b['ripple_seed'])
    count = b['ripple_modes']
    frequency = np.sort(np.exp(rng.uniform(math.log(250), math.log(7000), count)))
    q = rng.uniform(12, 30, count)
    gain = b['ripple_gain']*rng.lognormal(0, .5, count)
    modes = []
    for f, quality, g in zip(frequency, q, gain):
        r = math.exp(-math.pi*f/(quality*48000))
        angle = 2*math.pi*f/48000
        # Peak-normalized resonator: unit gain at its centre frequency.
        numerator = (1-r)*abs(1-r*np.exp(-2j*angle))
        modes.append((g*numerator, (1., -2*r*math.cos(angle), r*r)))
    return fir, modes


class _Body:
    """Bridge-to-air radiation as a stream: fine-structure modes, then the
    fitted envelope (overlap-add), then a 10 Hz high-pass."""

    def __init__(self, config, shape):
        self.fir, self.modes = _body(json.dumps(config['body'], sort_keys=True))
        lead = tuple(shape[:-1])
        self.mode_state = [np.zeros(lead+(2,)) for _ in self.modes]
        self.tail = np.zeros(lead+(len(self.fir)-1,))
        self.pole = math.exp(-2*np.pi*10/48000)
        self.highpass_state = np.zeros(lead+(1,))

    def __call__(self, signal):
        y = np.array(signal, dtype=float, copy=True)
        for k, (numerator, denominator) in enumerate(self.modes):
            mode, self.mode_state[k] = lfilter([numerator], denominator, signal, axis=-1, zi=self.mode_state[k])
            y += mode
        full = oaconvolve(y, self.fir.reshape((1,)*(y.ndim-1)+(-1,)), axes=-1)
        full[..., :self.tail.shape[-1]] += self.tail
        n = y.shape[-1]
        y, self.tail = full[..., :n], full[..., n:].copy()
        y, self.highpass_state = lfilter([1, -1], [1, -self.pole], y, axis=-1, zi=self.highpass_state)
        return y


def body_filter(signal, config):
    """Bridge-to-air radiation of a whole signal."""
    return _Body(config, np.shape(signal))(signal)


@lru_cache(maxsize=8)
def _noise_filter(band, btype):
    sos = butter(2, band, btype=btype, fs=48000, output='sos')
    impulse = np.zeros(1 << 20)
    impulse[0] = 1
    return sos, math.sqrt(np.sum(sosfilt(sos, impulse)**2))


class _Noise:
    """Seeded filtered noise of unit variance, continuous across blocks.

    It is scaled by the filter's noise gain rather than by the sample
    deviation, so the level doesn't depend on block or render length.
    """

    def __init__(self, seed, band, btype):
        self.rng = np.random.default_rng(np.random.SeedSequence(seed))
        self.sos, self.scale = _noise_filter(band, btype)
        self.state = np.zeros((self.sos.shape[0], 2))

    def __call__(self, n):
        y, self.state = sosfilt(self.sos, self.rng.normal(size=n), zi=self.state)
        return y/self.scale


def _curves(note, i, j):
    """Pitch, bow force, bow speed and contact point of one note over samples [i, j)."""
    t = (np.arange(i, j)-note['on'])/48000
    pitch = note['base']+note['slide']*np.maximum(0, 1-t/note['slide_seconds'])
    vibrato_env = np.clip((t-note['vibrato_delay'])/.12, 0, 1)
    pitch += note['vibrato']/100*np.sin(2*np.pi*note['vibrato_rate']*np.maximum(0, t-note['vibrato_delay'])+note['phase'])*vibrato_env
    force = np.full(j-i, note['force'])
    speed = np.full(j-i, note['direction']*note['speed'])
    beta = np.full(j-i, note['position'])
    for a, control, value in note['controls']:
        k = max(a, i)-i
        if control == 'force':
            force[k:] = value
        elif control == 'speed':
            speed[k:] = note['direction']*value
        else:
            beta[k:] = value
    if note['played']:
        speed *= bow_shape(t, note['duration'], note['phrase_end'])
        if not note['manual_force'] and not note['chop']:
            force = normal_force(note['pitch'], abs(speed), beta, note['impedance'])
            if note['style'] == 'surface':
                force *= .12
            elif note['style'] == 'crunch':
                force = np.minimum(1., force*3)
    attack_env = np.ones(j-i) if note['slur'] else np.minimum(1., t/note['attack'])
    release_env = np.ones(j-i) if note['continued'] else np.minimum(1., (note['duration']-t)/note['change'])
    env = attack_env*release_env
    # The bow leaves the string while still moving: weight comes off ahead
    # of speed, so a slowing bow doesn't grip and stop the string.
    lift = attack_env*release_env**3
    if note['chop']:
        env *= np.exp(-t/.018)
        lift = env
    return pitch, force*lift, speed*env, beta


def plan(rows, seconds, config, owners, player=True):
    """Check every note and settle its string, stroke and envelope, before any sample exists."""
    c = config
    n = round(seconds*48000)
    opens = c['strings']['open_midi']
    end = np.zeros(4)
    directions = np.ones(4)
    # One fiddle, one bow: strokes that start together share its direction.
    bow, stroke_start = 1., -1.
    assignment, notes = [], []
    rng = np.random.default_rng(c['engine']['seed'])
    ordered = sorted(rows, key=lambda r: r['time'])
    planned_end = np.zeros(4)
    planned_strings = []
    for row in ordered:
        s = assign_string(row['pitch'], c, row.get('string'), np.where(planned_end > row['time']+1e-6)[0])
        planned_strings.append(s)
        planned_end[s] = row['off']
    slur_starts = [(row['time'], s) for row, s in zip(ordered, planned_strings) if row.get('slur', False)]
    following = [None]*len(ordered)
    next_on_string = {}
    for j in range(len(ordered)-1, -1, -1):
        s = planned_strings[j]
        following[j] = next_on_string.get(s)
        next_on_string[s] = ordered[j]
    for row_index, row in enumerate(ordered):
        allowed = {'time', 'off', 'pitch', 'velocity', 'owner', 'string', 'slur', 'force', 'speed', 'position', 'vibrato', 'vibrato_delay', 'vibrato_rate', 'slide', 'slide_seconds', 'chop', 'controls', 'player', 'bow_style', 'phrase_end', 'grit'}
        if set(row)-allowed:
            raise ValueError(f'unknown fiddle note keys: {set(row)-allowed}')
        if any(not np.isfinite(row[k]) for k in ('time', 'off', 'pitch', 'velocity')) or not 0 <= row['time'] < row['off'] <= seconds or not 1 <= row['velocity'] <= 127:
            raise ValueError('invalid note duration/pitch/velocity')
        owner = row.get('owner', 0)
        if type(owner) is not int or not 0 <= owner < owners:
            raise ValueError('invalid owner')
        s = planned_strings[row_index]
        on, off = round(row['time']*48000), round(row['off']*48000)
        if off-on < 1:
            raise ValueError('note shorter than one sample')
        slur = bool(row.get('slur', False)) and end[s] > 0 and abs(end[s]-row['time']) <= .015
        # A slur onto another string: the same stroke rolls across, no bow change.
        crossing = bool(row.get('slur', False)) and not slur and any(
            k != s and end[k] > 0 and abs(end[k]-row['time']) <= .015 for k in range(4))
        # A connected bow change: the previous stroke on this string ends as this begins.
        joined = not slur and not crossing and end[s] > 0 and abs(end[s]-row['time']) <= .015
        if crossing:
            directions[s] = bow
        elif not slur:
            if abs(stroke_start-row['time']) > .015:
                bow, stroke_start = -bow, row['time']
            directions[s] = bow
        end[s] = row['off']
        assignment.append({'time': row['time'], 'pitch': row['pitch'], 'string': s+1, 'owner': owner, 'slur': slur, 'crossing': crossing})
        style = row.get('bow_style', 'normal')
        if style not in ('normal', 'surface', 'crunch', 'raw'):
            raise ValueError('bow_style must be normal, surface, crunch or raw')
        played = bool(row.get('player', player)) and style != 'raw'
        vibrato = float(row.get('vibrato', 12 if played and off-on > 33600 else 0))
        vibrato_delay = float(row.get('vibrato_delay', .25))
        vibrato_rate = float(row.get('vibrato_rate', 5.5))
        if not 0 <= vibrato_delay <= 2 or not 4 <= vibrato_rate <= 8:
            raise ValueError('vibrato delay 0..2 seconds and rate 4..8 Hz')
        slide = float(row.get('slide', 0))
        slide_s = float(row.get('slide_seconds', .12))
        if not 0 <= vibrato <= 35 or not -3 <= slide <= 3 or not .01 <= slide_s <= 1:
            raise ValueError('vibrato 0..35 cents; slide -3..3 semitones, 0.01..1 seconds')
        if row['pitch']+slide < opens[s]:
            raise ValueError('slide starts below open string')
        if abs(row['pitch']-opens[s]) < .01:
            vibrato = 0.
        # The player hears the bow's flattening and places the finger (or
        # tunes the peg) to correct it.
        base = row['pitch']+(fiddle_player.intonation(row['pitch']) if played else 0.)
        f = row.get('force', c['bow']['force']*max(.12, c['strings']['impedance'][s])/.12)
        v = row.get('speed', c['bow']['speed'])
        grit = float(row.get('grit', .25))
        if not 0 <= grit <= 1:
            raise ValueError('grit 0..1')
        noise_scale, position_scale = grit_controls(grit)
        pos = row.get('position', c['bow']['position']*(position_scale if played else 1.))
        if not np.isfinite(f+v+pos) or not .002 <= f <= 1 or not .01 <= v <= .5 or not .04 <= pos <= .3:
            raise ValueError('bow force .002..1 N, speed .01..0.5 m/s, position .04..0.3')
        events = []
        for dt, control, value in row.get('controls', []):
            if not np.isfinite(dt+value) or not 0 <= dt <= row['off']-row['time']:
                raise ValueError('invalid control time')
            if not (control == 'force' and .002 <= value <= 1 or control == 'speed' and .01 <= value <= .5
                    or control == 'position' and .04 <= value <= .3):
                raise ValueError('invalid continuous bow control')
            events.append((min(off, on+round(dt*48000)), control, value))
        manual_force = 'force' in row or any(control == 'force' for _, control, _ in events)
        chop = bool(row.get('chop', False))
        assignment[-1].update(off=row['off'], adaptive=played and style == 'normal' and not manual_force and not chop)
        next_row = following[row_index]
        phrase_end = bool(row.get('phrase_end', next_row is None or next_row['time'] > row['off']+.25))
        continued = bool(next_row and next_row.get('slur', False) and abs(next_row['time']-row['off']) <= .015)
        handed = bool(next_row and not continued and abs(next_row['time']-row['off']) <= .015)
        rolled = not continued and any(k != s and abs(t0-row['off']) <= .015 for t0, k in slur_starts)
        # The player changes bow quickly between connected strokes, rather
        # than lifting and setting down again as at a phrase start.
        connected_in, connected_out = played and joined, played and handed
        # Rolling across strings within a stroke takes about 8 ms each way.
        attack = .008 if crossing else .015 if connected_in else max(c['bow']['attack_seconds'], .035) if played else c['bow']['attack_seconds']
        change = .008 if rolled else .010 if connected_out else max(c['bow']['change_seconds'], .025) if played else c['bow']['change_seconds']
        note = dict(string=s, on=on, off=off, duration=row['off']-row['time'], pitch=row['pitch'], base=base,
                    slide=slide, slide_seconds=slide_s, vibrato=vibrato, vibrato_delay=vibrato_delay,
                    vibrato_rate=vibrato_rate, phase=rng.uniform(-.1, .1), force=f, speed=v, position=pos,
                    direction=directions[s], controls=events, played=played, style=style,
                    manual_force=manual_force, chop=chop, phrase_end=phrase_end, slur=slur,
                    continued=continued, attack=attack, change=change, impedance=c['strings']['impedance'][s])
        pitch_end, force_end, speed_end, _ = _curves(note, off-1, off)
        # Finger release restores the open pitch after a stopped-note tail.
        # Damping is roundtrip-scaled, not a per-sample envelope on the output.
        release = max(.24, c['strings']['release_seconds']) if played else c['strings']['release_seconds']
        ring = .055 if played else 0.
        # The scored pitch, not the intonation-corrected one, says whether a finger is down.
        stopped = abs(row['pitch']-opens[s]) > .01
        note.update(pitch_end=pitch_end[0], force_end=force_end[0], speed_end=speed_end[0],
                    gap_end=min(n, round(next_row['time']*48000)) if continued else off,
                    ring_end=off+round(ring*48000), tail=min(n, off+round((ring+release)*48000)),
                    damping=np.exp(-math.log(1000)/(release*(440*2**((pitch_end[0]-69)/12)))) if stopped else 1.,
                    owner=owner, intensity=(row['velocity']/90)**1.2, grain=noise_scale,
                    sway=noise_scale*(fiddle_player.steadiness(row['pitch']) if played else 1.))
        notes.append(note)
    return notes, assignment


def _fill(notes, a, b, c):
    """Unsmoothed control signals of the four strings over samples [a, b)."""
    m = b-a
    p = np.repeat(np.asarray(c['strings']['open_midi'], float)[:, None], m, axis=1)
    force, speed = np.zeros((4, m)), np.zeros((4, m))
    beta = np.full((4, m), c['bow']['position'])
    damping = np.ones((4, m))
    attribution = np.zeros((4, m), dtype=np.int32)
    intensity, grain, sway = np.ones((4, m)), np.ones((4, m)), np.ones((4, m))

    def put(x, s, i, j, value):
        i, j = max(i, a), min(j, b)
        if i < j:
            x[s, i-a:j-a] = value
    # Later notes on a string overwrite what earlier ones left behind.
    for note in notes:
        if note['on'] >= b:
            break
        s, on, off = note['string'], note['on'], note['off']
        i, j = max(on, a), min(off, b)
        put(beta, s, on, b, note['position'])
        if i < j:
            p[s, i-a:j-a], force[s, i-a:j-a], speed[s, i-a:j-a], beta[s, i-a:j-a] = _curves(note, i, j)
        put(p, s, off, b, note['pitch_end'])
        # A slur onto the next note holds the bow through the gap.
        put(force, s, off, note['gap_end'], note['force_end'])
        put(speed, s, off, note['gap_end'], note['speed_end'])
        put(damping, s, on, note['ring_end'], 1.)
        put(damping, s, note['ring_end'], note['tail'], note['damping'])
        put(p, s, note['tail'], b, c['strings']['open_midi'][s])
        put(attribution, s, on, b, note['owner'])
        put(intensity, s, on, b, note['intensity'])
        put(grain, s, on, b, note['grain'])
        put(sway, s, on, b, note['sway'])
    return p, force, speed, beta, damping, attribution, intensity, grain, sway


class _Controls:
    """The strings' control signals block by block, carrying every filter and
    noise state across block edges.

    The rosin grain (seeded per string, present only while the bow moves)
    jitters the friction. The arm sway is band-limited noise (`wobble_hz`)
    that moves bow speed by a relative deviation `speed_wobble`, with the
    pressure following it as an arm does, so the Helmholtz motion's amplitude
    breathes; `pressure_wobble` adds independent pressure variation. Both
    scale per note by the grit's grain and by how much room the note's
    corridor leaves.
    """

    def __init__(self, notes, config, noise=True):
        self.notes, self.c, self.noise = notes, config, noise
        self.state = {k: np.zeros((4, 1)) for k in ('intensity', 'force', 'speed')}
        if noise:
            seed, b = config['engine']['seed'], config['bow']
            self.grain = [_Noise([seed, 911, s], float(b['roughness_hz']), 'lowpass') for s in range(4)]
            self.sway = [[_Noise([seed, 1213, k, s], tuple(b['wobble_hz']), 'bandpass') for s in range(4)] for k in (0, 1)]

    def __call__(self, a, b):
        p, force, speed, beta, damping, attribution, intensity, grain, sway = _fill(self.notes, a, b, self.c)
        smooth = math.exp(-1/240)
        intensity, self.state['intensity'] = lfilter([1-smooth], [1, -smooth], intensity, axis=-1, zi=self.state['intensity'])
        # Slew continuous controls to keep CC boundaries from injecting impulses.
        pole = math.exp(-1/(.002*48000))
        force, self.state['force'] = lfilter([1-pole], [1, -pole], force, axis=-1, zi=self.state['force'])
        speed, self.state['speed'] = lfilter([1-pole], [1, -pole], speed, axis=-1, zi=self.state['speed'])
        rough = np.zeros_like(force)
        if self.noise:
            bow = self.c['bow']
            arm, hand = np.ones_like(force), np.ones_like(force)
            for s in range(4):
                moving = (force[s] > 1e-5) & (abs(speed[s]) > 1e-4)
                rough[s] = np.clip(bow['roughness']*grain[s]*self.grain[s](b-a), -.5, .5)*moving
                arm[s] = 1+np.clip(bow['speed_wobble']*np.sqrt(sway[s])*self.sway[0][s](b-a), -.5, .5)
                hand[s] = 1+np.clip(bow['pressure_wobble']*np.sqrt(sway[s])*self.sway[1][s](b-a), -.5, .5)
            force, speed = force*arm*hand, speed*arm
        return dict(pitch=p, force=force, speed=speed, beta=beta, damping=damping, roughness=rough,
                    attribution=attribution, intensity=intensity)


def controls(rows, seconds, config, owners, player=True):
    """The four strings' smoothed control signals over a whole render, without bow noise."""
    notes, assignment = plan(rows, seconds, config, owners, player)
    k = _Controls(notes, config, noise=False)(0, round(seconds*48000))
    return (*(k[name] for name in ('pitch', 'force', 'speed', 'beta', 'damping', 'attribution', 'intensity')), assignment)


class _Queue:
    """Samples waiting along the last axis."""

    def __init__(self, shape, dtype=float):
        self.data = np.zeros(tuple(shape)+(0,), dtype)

    def put(self, x):
        self.data = np.concatenate([self.data, x], axis=-1)

    def take(self, k):
        x, self.data = self.data[..., :k], self.data[..., k:]
        return x

    def __len__(self):
        return self.data.shape[-1]


class _Decimator:
    """`resample_poly(x, 1, OVERSAMPLE)` as a stream: the same centred Kaiser FIR."""

    def __init__(self, rows):
        self.delay = 10*OVERSAMPLE
        self.h = firwin(2*self.delay+1, 1/OVERSAMPLE, window=('kaiser', 5.))
        self.state = np.zeros((rows, len(self.h)-1))
        self.count = 0

    def __call__(self, x):
        y, self.state = lfilter(self.h, 1, x, axis=-1, zi=self.state)
        # Output k is filter sample OVERSAMPLE*k+delay.
        first = max(0, self.delay-self.count)
        first += (self.delay-self.count-first) % OVERSAMPLE
        self.count += x.shape[-1]
        return y[:, first::OVERSAMPLE]

    def flush(self):
        return self(np.zeros((self.state.shape[0], self.delay)))


HELMHOLTZ_FACTORS = (.6, .75, .88, 1., 1.15, 1.33, 1.6)
BLOCK_SECONDS = 2.


PREVIEW_SECONDS = 2.


def _spans(assignment, total, adaptive, piece):
    """Kernel pieces at the junction rate.

    Note boundaries cut the timeline into spans. Where the span holds
    adaptive notes, its first piece is the player's preview window, the
    first PREVIEW_SECONDS (or the whole span when shorter), so a held drone
    costs no more memory than a short note; every other piece is at most
    `piece` long. Each piece carries the span's adaptive notes, whether it
    is the preview, and the span's end.
    """
    boundaries = sorted({0, total, *(min(total, round(r[k]*RATE)) for r in assignment for k in ('time', 'off'))})
    rows = [(round(r['time']*RATE), round(r['off']*RATE), r) for r in assignment if r['adaptive']] if adaptive else []
    for on, off in zip(boundaries[:-1], boundaries[1:]):
        active = [r for a, b, r in rows if a <= on < b] if off-on >= .1*RATE else []
        start = on
        if active:
            start = min(off, on+round(PREVIEW_SECONDS*RATE))
            yield on, start, active, True, off
        for begin in range(start, off, piece):
            yield begin, min(off, begin+piece), active, False, off


def _choose_force(x, force, active, parameters, state, decisions, on, off):
    """Choose force with the actual string history, using deterministic lookahead.

    The preview forks mechanical state, not audio: only the chosen bow force
    enters the committed waveguide evolution. Explicit force controls bypass it.
    It runs over the span's preview window and returns each string's factor
    for the whole span.
    """
    n = force.shape[1]
    factors = []
    for row in active:
        s = row['string']-1
        best, candidates = None, []
        for factor in HELMHOLTZ_FACTORS:
            trial = force.copy()
            trial[s] = np.clip(trial[s]*factor, 0, 1)
            _, adhered, _ = _kernel(x[0], trial, x[2], x[3], x[4], x[5], *parameters, state)
            # Check the bow contact mid-note and at its end, after the entry.
            checks = [helmholtz(adhered[s, round(a*n):round(b*n)], x[0][s, round(a*n):round(b*n)], RATE)
                      for a, b in ((.3, .6), (.65, .94))]
            # Prefer the calibrated centre of the Helmholtz corridor.
            score = helmholtz_score(checks)-.5*abs(math.log(factor))
            candidates.append(dict(factor=factor, score=score, checks=checks))
            if best is None or score > best[0]:
                best = score, trial, factor
        force = best[1]
        factors.append((s, best[2]))
        decisions.append(dict(time=on/RATE, off=off/RATE, string=s+1, pitch=row['pitch'], factor=best[2], candidates=candidates))
    return factors


def render(rows, seconds, owners=1, config=None, diagnostics=False, player=True, noise=True, feedback=True,
           trace=None, channels=2, block_seconds=BLOCK_SECONDS):
    """Render note rows to per-owner audio, (owners, samples, channels).

    The strings run in blocks of `block_seconds`, carrying every control,
    noise, waveguide and filter state across, so memory stays bounded at any
    length and the audio doesn't depend on the block size. With
    `diagnostics` it also returns the string assignment and the player's
    decisions; `trace` (on with diagnostics unless set) adds the per-sample
    bridge, stick, force, speed, pitch and roughness. `channels=1` returns
    (owners, samples), one channel of the two.
    """
    c = copy.deepcopy(config or load_config())
    if not math.isfinite(seconds) or seconds <= 0 or type(owners) is not int or owners < 1 or channels not in (1, 2):
        raise ValueError('invalid duration, owners or channels')
    trace = diagnostics if trace is None else trace
    notes, assignment = plan(rows, seconds, c, owners, player)
    n = round(seconds*48000)
    block = max(1, round(block_seconds*48000))
    source = _Controls(notes, c, noise)
    bank, phase = _loss(json.dumps(c['strings'], sort_keys=True))
    # Friction coefficients are kernel arguments, not synthesis-layer gains.
    parameters = (bank, phase, np.array(c['strings']['impedance']), c['strings']['dispersion'], c['strings']['coupling'],
                  c['bow']['mu_static'], c['bow']['mu_dynamic'], c['bow']['stribeck_velocity'], float(RATE))
    inputs = _Queue((6, 4))
    weights, places, bridge = _Queue((4,)), _Queue((4,), np.int32), _Queue((4,))
    decimate, radiate = _Decimator(4), _Body(c, (4, 1))
    mono = np.zeros((owners, n))
    traced = {k: [] for k in ('bridge', 'stick', 'force', 'speed', 'pitch', 'roughness')}
    produced = done = 0
    peak = 0.
    state = None
    decisions = []

    def emit(final=False):
        nonlocal done, peak
        if len(bridge) < block and not (final and len(bridge)):
            return
        raw = bridge.take(len(bridge))
        k = raw.shape[1]
        if trace:
            traced['bridge'].append(raw)
        tones = radiate(raw*weights.take(k))
        attribution = places.take(k)
        for owner in range(owners):
            mono[owner, done:done+k] = np.sum(tones*(attribution == owner), axis=0)*c['engine']['output_gain']
        peak = np.maximum(peak, np.max(np.abs(mono[:, done:done+k]), initial=0))
        done += k

    factors = []
    for on, off, active, first, end in _spans(assignment, n*OVERSAMPLE, player and feedback, block*OVERSAMPLE):
        while on+len(inputs) < off:
            k = source(produced, min(n, produced+block))
            # Controls are smooth at 48 kHz; hold them for the oversampled junction.
            # Per-sample finger damping is re-rooted to keep its decay per second.
            held = np.repeat(np.stack([k[name] for name in ('pitch', 'force', 'speed', 'beta', 'damping', 'roughness')]), OVERSAMPLE, axis=2)
            held[4] **= 1/OVERSAMPLE
            inputs.put(held)
            weights.put(k['intensity'])
            places.put(k['attribution'])
            if trace:
                for name in ('speed', 'pitch', 'roughness'):
                    traced[name].append(k[name])
            produced = min(n, produced+block)
        x = inputs.take(off-on)
        if active and first:
            factors = _choose_force(x, x[1].copy(), active, parameters, state, decisions, on, end)
        force = x[1]
        if active:
            force = force.copy()
            for s, factor in factors:
                force[s] = np.clip(force[s]*factor, 0, 1)
        # This is the only committed mechanical state, including all passive
        # strings and the adhesion history at slurred finger changes.
        result, adhered, state = _kernel(x[0], force, x[2], x[3], x[4], x[5], *parameters, state)
        if trace:
            traced['stick'].append(adhered)
            traced['force'].append(force)
        bridge.put(decimate(result))
        emit()
    if n:
        bridge.put(decimate.flush()[:, :n-done-len(bridge)])
        emit(final=True)
    mono /= math.sqrt(2)
    # The limit is on the written channel. np.maximum carries a NaN through,
    # and a NaN peak fails the comparison.
    if not peak/math.sqrt(2) < .98:
        raise ValueError('nonfinite or clipping output: reduce preset output_gain')
    audio = mono if channels == 1 else np.repeat(mono[:, :, None], 2, axis=2)
    if not diagnostics:
        return audio
    details = {'assignment': assignment, 'player_decisions': decisions}
    if trace:
        joined = {k: np.concatenate(v, axis=1) if v else np.zeros((4, 0)) for k, v in traced.items()}
        joined['stick'] = joined['stick'][:, ::OVERSAMPLE].astype(np.uint8)
        joined['force'] = joined['force'][:, ::OVERSAMPLE]
        details.update(joined)
    return audio, details
