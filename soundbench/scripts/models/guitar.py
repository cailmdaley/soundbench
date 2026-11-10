"""Coupled, dispersive waveguides with paired lanes and a modal body.

The traveling-wave kernel is shared with the banjo; the guitar owns its
excitation, six-string loss bank, body response and fingering conventions.
String 1 is high E; string 6 is low E. No samples enter synthesis.
"""
from __future__ import annotations

from functools import lru_cache
import json
import math
from pathlib import Path
import tomllib

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import lfilter
from soundbench.scripts.models.banjo import _render, _output_guard, _rich_coefficients
from soundbench.scripts.models.string_loss import design_loss, PITCHES

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'instruments/models/guitar-steel.toml'
TUNINGS = {'standard': [64, 59, 55, 50, 45, 40],
           'dadgad': [62, 57, 55, 50, 45, 38],
           'open-g': [62, 59, 55, 50, 43, 38]}


def load_config(path=CONFIG):
    c = tomllib.loads(Path(path).read_text())
    s = c['strings']
    s.pop('release_seconds', None)
    s.setdefault('release_hz', [80, 200, 500, 1200])
    s.setdefault('release_t60', [.65, .55, .45, .35])
    s.setdefault('release_residual_multiplier', 3.0)
    s.setdefault('inharmonicity', [0.]*len(s['open_midi']))
    s.setdefault('damping_glide_ms', 1.0)
    s.setdefault('refret_ms', 25.0)
    s.setdefault('refret_t60', .04)
    c['excitation'].setdefault('slide_noise', .002)
    c['excitation'].setdefault('contact_ms', 1.5)
    c['excitation'].setdefault('restrike_keep', .5)
    if set(c) != {'engine', 'strings', 'excitation', 'body', 'bridge'}:
        raise ValueError('expected engine, strings, excitation, body, bridge tables')
    for table in c.values():
        for value in table.values():
            if not np.all(np.isfinite(np.asarray(value, dtype=float))):
                raise ValueError('parameters must be finite')
    e, s, x, b, bridge = (c[k] for k in ('engine', 'strings', 'excitation', 'body', 'bridge'))
    if e['sample_rate'] != 48000 or e['output_gain'] <= 0 or not 0 <= e['tail_seconds'] <= 10:
        raise ValueError('invalid engine configuration')
    if not isinstance(e['seed'], int) or e['seed'] < 0 or not 0 < e['silence_threshold'] <= 1e-8:
        raise ValueError('invalid seed or silence threshold')
    count = len(s['open_midi'])
    if count not in (4, 6):
        raise ValueError('expected four courses or six strings')
    for key in ('open_midi', 't60', 'loss_mix', 'dispersion', 'polarization_mix', 'polarization_rate', 'polarization_cents', 'inharmonicity'):
        if np.shape(s[key]) != (count,):
            raise ValueError(f'strings.{key} needs {count} values')
    if any(not 28 <= p <= 88 for p in s['open_midi']) or not 0 <= s['max_fret'] <= 24:
        raise ValueError('invalid string range')
    if any(t <= 0 for t in s['t60']) or any(not 0 <= m <= .5 for m in s['loss_mix']) or any(not -.5 <= d <= 0 for d in s['dispersion']) or any(not 0 <= v <= 1e-4 for v in s['inharmonicity']):
        raise ValueError('invalid waveguide parameters')
    hz, times = np.array(s['decay_hz']), np.array(s['decay_t60'])
    if hz.ndim != 1 or len(hz) < 2 or hz[0] <= 0 or hz[-1] > 24000 or np.any(np.diff(hz) <= 0) or times.shape != (count, len(hz)) or np.any(times <= 0):
        raise ValueError('invalid decay curves')
    if any(not 0 <= m <= .5 for m in s['polarization_mix']) or any(r <= 0 for r in s['polarization_rate']) or any(abs(v) > 20 for v in s['polarization_cents']):
        raise ValueError('invalid polarizations')
    if len(s['release_hz']) != len(s['release_t60']) or len(s['release_hz']) < 2 or np.any(np.diff(s['release_hz']) <= 0) or min(s['release_hz']+s['release_t60']) <= 0 or s['release_residual_multiplier'] < 1:
        raise ValueError('invalid register-dependent release')
    if not 0 <= x['slide_noise'] <= .02:
        raise ValueError('invalid slide noise gain')
    if min(s['pitch_smoothing_ms'], s['damping_glide_ms']) <= 0 or min(s['choke_seconds'], s['refret_t60'], *s['release_t60']) < .005 or s['refret_ms'] < 0:
        raise ValueError('invalid smoothing, or a damping T60 under 5 ms')
    if not 0 < x['contact_ms'] <= 5 or not 0 <= x['restrike_keep'] <= 1:
        raise ValueError('invalid pick contact')
    if not 0 < x['position'] < .5 or not 0 <= x['jitter'] < .5 or not 0 < x['pick_ms'] <= 5 or not 0 < x['lowpass_hz'] < 24000:
        raise ValueError('invalid excitation')
    if x['velocity_exponent'] <= 0 or x['velocity_corner_exponent'] < 0:
        raise ValueError('invalid velocity response')
    if not 0 <= x.get('course_delay_ms',0) <= 4 or not 0 <= x.get('course_position_spread',0) <= .08:
        raise ValueError('invalid course crossing or position spread')
    if x.get('course_delay_ms',0) and count != 4:
        raise ValueError('course crossing requires a four-course preset')
    if not (len(b['frequency_hz']) == len(b['t60_seconds']) == len(b['weights'])) or not len(b['weights']):
        raise ValueError('body arrays must have equal nonzero length')
    if any(not 20 < f < 20000 for f in b['frequency_hz']) or any(not 0 < t <= .5 for t in b['t60_seconds']) or any(w < 0 for w in b['weights']):
        raise ValueError('invalid body modes')
    if not 0 <= bridge['coupling'] <= .02 or bridge['direct'] < 0:
        raise ValueError('invalid bridge')
    return c


def configure(c, tuning='standard', capo=0):
    """Apply a named tuning or six absolute MIDI open pitches, high to low."""
    c = json.loads(json.dumps(c))
    opens = TUNINGS.get(tuning) if isinstance(tuning, str) else tuning
    if opens is None or np.shape(opens) != (6,) or not isinstance(capo, int) or not 0 <= capo <= 12:
        raise ValueError('unknown tuning or invalid capo')
    c['strings']['open_midi'] = [float(p) + capo for p in opens]
    if any(not 28 <= p <= 88 for p in c['strings']['open_midi']):
        raise ValueError('open pitches must be in MIDI 28..88')
    return c


def assign_string(pitch, config, explicit=None, sounding=None, started=None):
    opens = config['strings']['open_midi']
    if explicit is not None and (not isinstance(explicit, int) or explicit not in range(1, len(opens)+1)):
        raise ValueError(f'string numbers are 1..{len(opens)}')
    choices = [explicit-1] if explicit is not None else range(len(opens))
    choices = [s for s in choices if 0 <= pitch-opens[s] <= config['strings']['max_fret']]
    if not choices:
        raise ValueError(f'MIDI {pitch} is unplayable on the selected guitar string')
    free = [s for s in choices if sounding is None or not sounding[s]]
    return min(free, key=lambda s: (pitch-opens[s], s)) if free else min(choices, key=lambda s: (started[s], s))


@lru_cache(maxsize=8)
def _bank(serialized):
    s = json.loads(serialized)
    count = len(s['open_midi'])
    bank = np.empty((2*count, len(PITCHES), 65))
    for lane in range(2*count):
        string = lane % count
        rates = np.log(1000) / np.array(s['decay_t60'][string])
        if lane >= count:
            rates *= s['polarization_rate'][string]
        for i, pitch in enumerate(PITCHES):
            bank[lane, i] = design_loss(s['decay_hz'], rates, 440*2**((pitch-69)/12))
    return bank


def _allpass_phase(a, w):
    return np.angle((a+np.exp(-1j*w))/(1+a*np.exp(-1j*w)))


@lru_cache(maxsize=8)
def _stages(serialized):
    """Three equal allpass stages per lane that stretch the partials like a stiff string.

    The target is f_n = n f1 sqrt(1+B n^2)/sqrt(1+B) at the open pitch
    (B = strings.inharmonicity). The loop phase, including the Thiran tuning
    and minimum-phase loss, is solved for its partials below 4 kHz and fitted
    in cents with weight 1/n^2. First-order stages follow the stiff-string
    curve to about the 30th partial and stretch higher partials less.
    """
    s = json.loads(serialized)
    count, sr = len(s['open_midi']), 48000
    bank = _bank(serialized)
    stages = np.zeros((2*count, 3))
    w = np.linspace(1e-4, 2*np.pi*4500/sr, 20000)
    for string, (pitch, B) in enumerate(zip(s['open_midi'], s['inharmonicity'])):
        if B <= 0:
            continue
        f0 = 440*2**((pitch-69)/12)
        n = np.arange(1, int(4000/f0)+1)
        target = 1200*np.log2(np.sqrt(1+B*n*n)/np.sqrt(1+B))
        def error(a):
            integer, thiran, h = _rich_coefficients(pitch, sr, s['dispersion'][string], bank, string, np.full(3, a))
            phase = -w*integer+_allpass_phase(thiran, w)+_allpass_phase(s['dispersion'][string], w)+3*_allpass_phase(a, w)
            phase += np.angle(np.polyval(h[::-1], np.exp(-1j*w)))
            cents = 1200*np.log2(np.interp(2*np.pi*n, np.unwrap(-phase), w)*sr/(2*np.pi)/(n*f0))
            return np.sum((cents-target)**2/n**2)
        a = minimize_scalar(error, bounds=(-.9, 0), method='bounded', options=dict(xatol=1e-3)).x
        stages[string] = stages[string+count] = a
    return stages


def render(events, seconds, owners=1, config=None, *, member_stems=False, excite_member=None):
    """Render banjo-compatible 11-column events, with strings 0..5.

    Kinds: 0 pluck, 1 release, 2 pitch ramp, 3 hammer, 4 pull, 5 damp.
    On attacks, column 8 is pluck position; column 9 selects hardness
    (0 preset, 1 flesh, 2 nail). On releases, column 8 is policy (0 normal,
    1 honour sustain-extended off, 2 force finger release).
    Open strings ignore normal and sustain-extended releases.
    Every note has a unique token. Explicit kind-5 damping always applies;
    its column 9 is a T60 of 5 ms to 2 s (0 is the preset choke).
    No string state is ever cleared: a pluck adds to the string's motion,
    and a pluck at a new pitch is preceded by a refret damp of the old note.
    With course crossing enabled, attack column 7 is 0 down / 1 up.
    member_stems returns primary then secondary tonal/body owner stems without
    common slide noise. excite_member=0/1 excites only that physical member;
    both sets of waveguides remain active and coupled for transfer diagnostics.
    """
    c = config or load_config()
    if excite_member not in (None,0,1):
        raise ValueError('excite_member is None, 0 or 1')
    sr = c['engine']['sample_rate']
    count = len(c['strings']['open_midi'])
    if not math.isfinite(seconds) or seconds < 0 or not isinstance(owners, int) or owners < 1:
        raise ValueError('invalid duration or owner count')
    rows = np.asarray(events, dtype=float).reshape(-1, 11)
    if not np.all(np.isfinite(rows)):
        raise ValueError('events must be finite')
    for row in rows:
        if row[0] < 0 or row[1] not in range(6) or row[2] not in range(count) or row[3] not in range(owners) or not 0 <= row[5] <= 127 or row[10] < 0 or row[10] != round(row[10]) or row[6] < 0 or row[7] not in (0, 1) or (row[9] not in (0, 1, 2) if row[1] != 5 else not (row[9] == 0 or .005 <= row[9] <= 2)):
            raise ValueError('invalid event')
        if row[1] == 1 and row[8] not in (0, 1, 2):
            raise ValueError('release policy must be normal, let ring, or force')
        if row[1] in (0, 2, 3, 4):
            assign_string(row[4], c, int(row[2])+1)
    rows = rows[np.argsort(rows[:, 0], kind='stable')].copy()
    s, x, b, bridge, e = (c[k] for k in ('strings', 'excitation', 'body', 'bridge', 'engine'))
    current, choked, kept, moves, held = {}, set(), [], {}, {}
    for row in rows:
        token = int(row[10])
        if row[1] == 2:
            moves[int(row[0]), token] = abs(row[4]-current.get(token,row[4]))
        string = int(row[2])
        if row[1] == 0 and string in held:
            # The fretting hand leaves the old note before the pick arrives.
            start, old, owner = held[string]
            if abs(current[old]-row[4]) > .01:
                kept.append(np.array([max(start+1, row[0]-round(s['refret_ms']*sr/1000)), 5, string, owner,
                                      0, 0, 0, 0, 0, s['refret_t60'], old]))
        if row[1] in (0, 3, 4):
            held[string] = (row[0], token, row[3])
        if row[1] in (0, 2, 3, 4):
            current[token] = row[4]
        elif row[1] == 5:
            choked.add(token)
        elif row[1] == 1:
            p = current.get(token, s['open_midi'][int(row[2])]+1)
            is_open = abs(p-s['open_midi'][int(row[2])]) < .01
            if token in choked or (is_open and row[8] != 2):
                continue
            f = 440*2**((p-69)/12)
            row[9] = np.interp(np.log(f), np.log(s['release_hz']), s['release_t60'])
        kept.append(row)
    rows = np.asarray(kept, dtype=float).reshape(-1, 11)
    nominal_positions = rows[:,8].copy()
    crossing = x.get('course_delay_ms',0)
    spread = x.get('course_position_spread',0) if crossing else 0.
    pulses = np.zeros((len(rows), 512))
    slides = np.zeros((owners, round(seconds*sr)))
    for i, row in enumerate(rows):
        if row[1] == 2 and moves.get((int(row[0]),int(row[10])),0) >= 2:
            rng = np.random.default_rng(np.random.SeedSequence([e['seed'], int(row[10]), int(row[0])]))
            if rng.random() < .35:
                length = min(2400,max(144,int(row[6])))
                burst = lfilter([1,-1],[1,-.8],rng.normal(size=length))*np.sin(np.pi*np.arange(length)/length)**2
                burst /= max(np.sqrt(np.sum(burst*burst)),1e-12)
                onset, owner = int(row[0]), int(row[3])
                end = min(slides.shape[1],onset+length)
                if end > onset:
                    slides[owner,onset:end] += burst[:end-onset]*x['slide_noise']
        if row[1] not in (0, 3, 4):
            continue
        rng = np.random.default_rng(np.random.SeedSequence([e['seed'], int(row[10]), int(row[2]), int(row[3])]))
        row[8] = ((row[8] or x['position'])-spread/2)*rng.uniform(1-x['jitter'], 1+x['jitter'])
        if not 0 < row[8] < .5:
            raise ValueError('pluck position must be between 0 and .5')
        hardness = int(row[9])
        width = x['pick_ms'] * (2.2 if hardness == 1 else .7 if hardness == 2 else 1)
        pole = math.exp(-2/(sr*width/1000))
        pulse = (1-pole)*pole**np.arange(512)
        cutoff = x['lowpass_hz']*(.55 if hardness == 1 else 1.3 if hardness == 2 else 1)
        cutoff = min(20000, cutoff*max(row[5]/127, .01)**x['velocity_corner_exponent'])
        alpha = 1-math.exp(-2*math.pi*cutoff/sr)
        amplitude = (row[5]/127)**x['velocity_exponent']
        amplitude *= .3 if row[1] == 3 else .45 if row[1] == 4 else 1
        pulses[i] = amplitude*lfilter([alpha], [1, -(1-alpha)], pulse)
    secondary = rows.copy()
    secondary[:, 2] += count
    secondary[secondary[:, 1] == 1, 9] *= s['release_residual_multiplier']
    cents = np.array(s['polarization_cents'])
    weights = np.array(s['polarization_mix'])[rows[:, 2].astype(int)]
    secondary[:, 4] += cents[rows[:, 2].astype(int)]/100
    if crossing:
        offsets = {}
        for i,row in enumerate(rows):
            if row[1] == 0:
                rng = np.random.default_rng(np.random.SeedSequence([e['seed'],int(row[10]),int(row[2])+count,int(row[3])]))
                secondary[i,8] = ((nominal_positions[i] or x['position'])+spread/2)*rng.uniform(1-x['jitter'],1+x['jitter'])
                if not 0 < secondary[i,8] < .5:
                    raise ValueError('secondary pluck position must be between 0 and .5')
                delay = round(sr*crossing/1000)
                offsets[int(row[10])] = (delay,0) if row[7] else (0,delay)
        # Follow each member's own pick onset through its slides and release.
        for i,row in enumerate(rows):
            first_delay,second_delay = offsets.get(int(row[10]),(0,0))
            rows[i,0] += first_delay
            secondary[i,0] += second_delay
    pulses = np.concatenate((pulses*(1-weights[:, None]), pulses*weights[:, None]))
    rows = np.concatenate((rows, secondary))
    if excite_member is not None:
        pulses[(rows[:,2]>=count) != bool(excite_member)] = 0.
    if member_stems:
        rows[rows[:,2]>=count,3] += owners
        owners *= 2
    if member_stems or excite_member is not None:
        slides = np.zeros((owners,round(seconds*sr)))
    order = np.argsort(rows[:, 0], kind='stable')
    rows, pulses = rows[order], pulses[order]
    opens = np.array(s['open_midi'], float)
    frequency, times = np.array(b['frequency_hz']), np.array(b['t60_seconds'])
    radii = np.exp(-np.log(1000)/(sr*times))
    angles = 2*np.pi*frequency/sr
    # Peak-normalized parallel causal resonators: short body tails cannot hold notes.
    modal_in = (1-radii)*np.abs(1-radii*np.exp(-2j*angles))
    audio = _render(rows, pulses, round(seconds*sr), owners, sr,
        np.r_[opens, opens+cents/100], np.tile(s['t60'], 2),
        np.tile(s['loss_mix'], 2), np.tile(s['dispersion'], 2),
        s['release_t60'][0], s['choke_seconds'],
        1-math.exp(-1/(s['pitch_smoothing_ms']*sr/1000)),
        bridge['coupling'], 0., 2*radii*np.cos(angles), radii**2,
        modal_in, np.array(b['weights']), 1., bridge['direct'],
        e['output_gain'], e['silence_threshold'],
        _bank(json.dumps(s, sort_keys=True)), _stages(json.dumps(s, sort_keys=True)),
        continuous=True, glide=1-math.exp(-1/(s['damping_glide_ms']*sr/1000)),
        contact=round(x['contact_ms']*sr/1000), contact_keep=x['restrike_keep'])
    return _output_guard(audio + slides*e['output_gain'], .85, .98)


def note(on, off, pitch, velocity, string, token, sr=48000, owner=0, kind=0, hardness=0, let_ring=False):
    """A pluck or legato destination with an explicit release, in seconds."""
    return [[round(on*sr), kind, string-1, owner, pitch, velocity, .008*sr, 1, 0, hardness, token],
            [round(off*sr), 1, string-1, owner, 0, 0, 0, 0, int(let_ring), 0, token]]


@lru_cache(maxsize=1)
def stroke_profile():
    return tomllib.loads((ROOT/'instruments/models/guitar-stroke.toml').read_text())


def strum(on, duration, pitches, config, velocity=90, direction='down', speed_ms=None,
          strings=None, accent=1., muted=False, fingers=1, token=0, owner=0, hardness=0,
          let_ring=True, tempo_bpm=100., reach=None, humanize=True):
    """One hand stroke across explicit chord strings, or a rasgueado of sweeps.

    Pitches and strings pair in any order. A downstroke crosses from the
    lowest string upward. speed_ms is the authored first-to-last string time;
    without it the hand's speed comes from guitar-stroke.toml, scaled by
    tempo and velocity, and the stroke takes as long as its span of strings.
    Gaps lengthen after heavier strings and vary with a seeded hand, and the
    force falls off across the stroke; accent scales the first string only.
    reach is how many strings a stroke strikes from its starting side;
    upstrokes default to the top three or four.
    muted=True chokes each string just after it sounds (a chuck); a number
    damps the strings with that T60 in seconds (a palm-muted strum).
    Rasgueado sweeps (fingers > 1) alternate direction, begin at least 35 ms
    apart with a 3 ms turnaround, and lose 8% force per finger.
    Each call uses tokens token .. token+23.
    """
    if direction not in ('down', 'up') or (speed_ms is not None and not 10 <= speed_ms <= 80) or not isinstance(fingers, int) or not 1 <= fingers <= 4:
        raise ValueError('direction down/up, speed 10..80 ms, fingers 1..4')
    if not 0 < accent <= 1.5 or not 1 <= velocity <= 127 or duration <= 0 or not 30 <= tempo_bpm <= 300:
        raise ValueError('invalid strum dynamics, duration or tempo')
    if strings is None or len(strings) != len(pitches) or len(set(strings)) != len(strings) or not len(strings):
        raise ValueError('strum needs distinct explicit strings paired with pitches')
    if reach is not None and (not isinstance(reach, int) or not 1 <= reach <= len(strings)):
        raise ValueError('reach is a string count within the chord')
    if muted is not True and muted is not False and not .005 <= muted <= 2:
        raise ValueError('muted is a boolean or a damping T60 of 5 ms to 2 s')
    pairs = sorted(zip(strings, pitches))
    for string, pitch in pairs:
        assign_string(pitch, config, string)
    hand = stroke_profile()
    rng = np.random.default_rng(np.random.SeedSequence([config['engine']['seed'], token, owner]))
    jitter = lambda sigma, size=None: rng.lognormal(0, sigma, size) if humanize else np.ones(size) if size else 1.
    exponent = config['excitation']['velocity_exponent']
    t60 = config['strings']['choke_seconds'] if muted is True else muted
    rows, start = [], on
    for finger in range(fingers):
        down = (direction == 'down') != bool(finger % 2)
        struck = sorted(pairs, reverse=down)
        count = reach or (len(struck) if down else min(len(struck), int(rng.integers(hand['up_reach'][0], hand['up_reach'][1]+1)) if humanize else hand['up_reach'][1]))
        struck = struck[:count]
        span = speed_ms
        if span is None:
            six = hand['six_string_ms']*(90/velocity)**hand['velocity_exponent']*(100/tempo_bpm)**hand['tempo_exponent']
            span = six*(1 if down else hand['up_factor'])*jitter(hand['spread_sigma'])*abs(struck[-1][0]-struck[0][0])/5
        # The pick crosses string spacings, held longer by the heavier string it leaves.
        weights = np.array([abs(b[0]-a[0])*(1+hand['resistance']*(a[0]-1)/5) for a, b in zip(struck, struck[1:])])
        weights = weights*jitter(hand['gap_sigma'], len(weights)) if len(weights) else weights
        offsets = np.r_[0, np.cumsum(weights)/max(weights.sum(), 1e-12)*span/1000]
        forces = hand['last_first_force']**(np.arange(count)/max(1, count-1))*.92**finger*jitter(hand['force_sigma'], count)
        for i, (string, pitch) in enumerate(struck):
            t = start+offsets[i]
            vel = min(127, max(1, round(velocity*(forces[i]*(accent if i == 0 else 1))**(1/exponent))))
            ident = token+finger*6+i
            rows += note(t, t+(.065 if muted is True else duration), pitch, vel, string, ident,
                         owner=owner, hardness=hardness, let_ring=let_ring and not muted)
            if muted:
                rows.append([round((t+hand['mute_ms']/1000)*48000), 5, string-1, owner, 0, 0, 0, 0, 0, t60, ident])
        start += max(.035, offsets[-1]+.003)
    return rows
