"""Seeded singing gestures and Fant-Rd LF excitation through voice's tract.

Geometry and wave propagation are imported from voice. No hum output filter:
its losses and frequency-dependent radiation live inside the nasal branch.
"""
from __future__ import annotations
from functools import lru_cache
import math
from pathlib import Path
import tomllib

import numpy as np
from scipy.signal import butter, sosfilt, resample_poly
from . import voice as v1

CONFIG = v1.CONFIG.with_name('voice2.toml')
PRESETS = {'tenor': CONFIG, 'alto': CONFIG.with_name('voice2-alto.toml')}
HUMS = {'hm': 'm', 'mm': 'm', 'nn': 'n', 'ng': 'g'}
VOWELS = {'ah': 'aa', 'eh': 'eh', 'ee': 'iy', 'oh': 'ow', 'oo': 'uw',
          'uh': 'ax', 'aw': 'ao', 'ae': 'ae', 'ai': 'aa'}
CRY_SYLLABLE = {vowel: key for key, vowel in reversed(list(VOWELS.items()))}
ART_KEYS = {'slide': {'to', 'duration_s'}, 'cry': {'vowel'},
            'onset': {'style', 'depth'}, 'fall': {'depth', 'duration_s'},
            'creak': set(), 'breath': set(), 'straight': set(),
            'quiver': {'depth'}, 'flick': {'to'}}


def syllable_key(text):
    """Canonical hum/vowel key, sharing v1's punctuation convention."""
    if not isinstance(text, str):
        raise ValueError('voice2 syllable must be text')
    key = ''.join(c for c in text.lower() if 'a' <= c <= 'z')
    if key not in HUMS and key not in VOWELS and key not in v1.LEXICON:
        raise ValueError(f'unsupported voice2 syllable {text!r}')
    return key


def load_config(preset='tenor'):
    if preset not in PRESETS:
        raise ValueError('voice2 preset must be tenor or alto')
    q = tomllib.loads(CONFIG.read_text())['voice2']
    if preset == 'alto':
        q.update(tomllib.loads(PRESETS[preset].read_text())['voice2'])
    c = v1.load_config()
    c['voice2'] = q
    c['tract']['length_cm'] = q['length_cm']
    c['nasal']['length_cm'] *= q['nasal_scale']
    c['nasal']['velum_from_glottis_cm'] *= q['nasal_scale']
    c['nasal']['loss'] = q['nasal_loss']
    c['nasal']['nostril_radiation_hz'] = q['nostril_radiation_hz']
    c['engine']['output_gain'] = q['output_gain']
    c['engine']['seed'] = q['seed']
    c['engine']['tail_seconds'] = q['tail_seconds']
    return c


def rd_timing(rd):
    """Fant (1995) regressions, fractions of one period (DESIGN coefficients)."""
    if not math.isfinite(rd) or not 0.3 <= rd <= 2.7:
        raise ValueError('Rd must be 0.3..2.7')
    ra = (-1 + 4.8 * rd) / 100
    rk = (22.4 + 11.8 * rd) / 100
    rg = 1 / (4 * (0.11 * rd / (0.5 + 1.2 * rk) - ra) / rk)
    tp = 1 / (2 * rg)
    return tp, tp * (1 + rk), ra


@lru_cache(maxsize=1)
def rd_tables():
    pulses = [v1._lf_pulse(*rd_timing(float(rd)), 4096) for rd in np.linspace(.3, 2.7, 97)]
    return np.array([p[0] for p in pulses]), np.array([p[1] for p in pulses])


def validate_articulation(art):
    kind = art.get('kind')
    if kind not in ART_KEYS or set(art) - ({'kind'} | ART_KEYS.get(kind, set())):
        raise ValueError(f'unknown voice2 articulation kind/keys: {art}')
    if kind == 'slide' and not {'to', 'duration_s'} <= set(art):
        raise ValueError('slide needs to and duration')
    if kind == 'cry' and art.get('vowel', 'ah') not in VOWELS:
        raise ValueError('cry vowel must be an open vowel')
    if kind == 'onset' and art.get('style') not in ('breathy', 'glottal', 'scoop', 'hum'):
        raise ValueError('onset style must be breathy, glottal, scoop or hum')
    for key in ('to', 'depth', 'duration_s'):
        if key in art:
            x = art[key]
            lo, hi = ((.005, 10) if key == 'duration_s' else
                      ((-12, 12) if key == 'to' else (0, 8 if kind == 'onset' else 6)))
            if type(x) not in (int, float) or not math.isfinite(x) or not lo <= x <= hi:
                raise ValueError(f'invalid voice2 articulation {key}')


_BALANCE = {}


def _a_weighted_db(x, sr=48000):
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f2 = np.fft.rfftfreq(len(x), 1 / sr) ** 2
    weight = (12194 ** 4 * f2 ** 2) / ((f2 + 20.6 ** 2) * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2)) * (f2 + 12194 ** 2)) ** 2
    return 10 * np.log10(np.sum(spectrum * weight) + 1e-30)


def cry_balance_db(config, vowel, f0):
    """How much louder (A-weighted dB) this voice sings `vowel` than its hum,
    at the same velocity and pitch, measured by singing both.

    A cry subtracts this so it sits cry_lift_db above the hum it opens from,
    rather than the 10-25 dB an open mouth adds over closed lips. The table is
    sung once per preset and vowel on a semitone grid, then interpolated.
    """
    q = config['voice2']
    key = (q['length_cm'], q['nasal_scale'], q['velum_cm2'], q['rd_soft'], q['rd_loud'], vowel)
    if key not in _BALANCE:
        quiet = {**config, 'voice2': {**q, 'vibrato': 0.0, 'flutter_cents': 0.0, '_calibrating': True}}
        grid = np.arange(45, 91)
        balance = []
        for pitch in grid:
            notes = [{'time': 0.0, 'off': 0.6, 'pitch': int(pitch), 'velocity': 80, 'syllable': 'hm'},
                     {'time': 0.6, 'off': 1.2, 'pitch': int(pitch), 'velocity': 80, 'syllable': vowel}]
            audio = render(notes, 1.3, quiet)
            balance.append(_a_weighted_db(audio[int(.85 * 48000):int(1.1 * 48000)])
                           - _a_weighted_db(audio[int(.2 * 48000):int(.45 * 48000)]))
        _BALANCE[key] = (grid, np.array(balance))
    grid, balance = _BALANCE[key]
    return np.interp(69 + 12 * np.log2(np.maximum(f0, 1.0) / 440), grid, balance)


def _walk(rng, t, spacing, scale):
    knots = np.arange(0, t[-1] + spacing, spacing)
    return np.interp(t, knots, rng.normal(0, scale, len(knots)))


def controls(notes, config, seconds=None, midi_controls=(), articulations=()):
    """Gesture lanes on a 1-ms grid. Articulations select zero-based note indices."""
    q = config['voice2']
    rows = sorted([dict(r) for r in notes], key=lambda r: r['time'])
    for i, r in enumerate(rows):
        if any(type(r.get(k)) not in (int, float) or not math.isfinite(r[k]) for k in ('time', 'off', 'pitch', 'velocity')):
            raise ValueError('notes require finite time, off, pitch, velocity')
        if r['time'] < 0 or r['off'] <= r['time'] or not 0 <= r['velocity'] <= 127 or not 0 <= r['pitch'] <= 127:
            raise ValueError('invalid voice2 note')
        r['arts'] = []
        r['_syllable'] = syllable_key(r['syllable']) if r.get('syllable') is not None else None
    # Phrase context survives windows starting on a vowel cry. Syllable
    # spans survive melismas, so a cry opens/closes once, not at every pitch.
    phrases, units = [], []
    for i, r in enumerate(rows):
        if not phrases or r['time'] - rows[i - 1]['off'] > .08:
            phrases.append([])
        phrases[-1].append(r)
        if r.get('syllable') is not None or not units:
            units.append([])
        units[-1].append(r)
    for phrase in phrases:
        hums = [r for r in phrase if r['_syllable'] in HUMS]
        context = float(np.mean([24 * (r['velocity'] / 127 - 1) for r in hums])) if hums else None
        for r in phrase:
            r['_hummed_context'] = bool(hums)
            r['_hum_db'] = context
    for unit in units:
        for r in unit:
            r['_unit_start'] = unit[0]['time']
            r['_unit_end'] = max(x['off'] for x in unit)
    for event in articulations:
        if set(event) != {'note', 'art'} or type(event['note']) is not int or not 0 <= event['note'] < len(rows):
            raise ValueError('articulation needs a valid note index and art')
        validate_articulation(event['art'])
        rows[event['note']]['arts'].append(event['art'])
    dt = .001
    seconds = seconds if seconds is not None else max((r['off'] for r in rows), default=0) + q['tail_seconds']
    t = np.arange(math.ceil(seconds / dt) + 2) * dt
    count = len(t)
    shapes = v1.vowel_shapes(config)
    areas = np.tile(shapes['ax'], (count, 1))
    pitch = np.full(count, float(rows[0]['pitch']) if rows else 60.)
    envelope = np.zeros(count)
    level = np.full(count, -60.)
    effort = np.zeros(count)
    vibrato = np.zeros(count)
    velum = np.zeros(count)
    onset_rd = np.zeros(count)
    inhale = np.zeros(count)
    creak = np.zeros(count)
    hum = np.zeros(count)
    cry = np.zeros(count)
    cry_open = np.zeros(count)
    cry_vowel = np.full(count, '', dtype=object)
    rng = np.random.default_rng(np.random.SeedSequence([q['seed'], 1]))
    carried_syllable = None
    delays, dips, transitions, breaths = [], [], [], []
    for i, r in enumerate(rows):
        if r['_syllable'] is not None:
            carried_syllable = r['_syllable']
        syllable = carried_syllable
        if syllable is None:
            raise ValueError('first note needs a syllable')
        start, off = r['time'], r['off']
        prev = rows[i - 1] if i else None
        nxt = rows[i + 1] if i + 1 < len(rows) else None
        entry = prev is None or start - prev['off'] > .08
        final = nxt is None or nxt['time'] - off > .08
        # A legato phrase keeps breath/phonation through short MIDI gaps;
        # only a real phrase end owns a release envelope.
        boundary = off if final else nxt['time']
        duration = boundary - start
        if duration <= 0:
            raise ValueError('simultaneous singing notes are unsupported')
        arts = {a['kind']: a for a in r['arts']}
        r['_destination'] = r['pitch'] + arts.get('slide', {}).get('to', 0)
        is_hum = syllable in HUMS
        is_cry = 'cry' in arts or (syllable in ('ah', 'ai', 'oh') and r['_hummed_context'])
        if 'cry' in arts:
            syllable = arts['cry'].get('vowel', 'ah')
            is_hum = False
        r['_hum'] = is_hum
        cry_entry = is_cry and (r.get('syllable') is not None or entry or 'cry' in arts)
        mask = (t >= start) & (t < (boundary + (q['release_s'] if final else 0)))
        age = np.maximum(t - start, 0)
        local = t[mask] - start
        pitch[mask] = r['pitch']
        # Analytic unit-step response of a damped second-order system.
        if not entry:
            previous_pitch = prev['_destination']
            interval = r['pitch'] - previous_pitch
            settling = np.clip(.06 + abs(interval) * .022, .06, .25)
            omega = 8 / settling
            zeta = .68
            wd = omega * math.sqrt(1 - zeta ** 2)
            response = 1 - np.exp(-zeta * omega * local) * (np.cos(wd * local) + zeta * omega / wd * np.sin(wd * local))
            pitch[mask] = previous_pitch + interval * response
            prep = (t >= start - .035) & (t < start)
            pitch[prep] -= np.sign(interval) * min(.15, abs(interval) * .025) * np.sin(np.pi * (t[prep] - start + .035) / .035)
            if abs(interval) >= 3:
                transitions.append({'time': start, 'interval': interval, 'settling_s': settling})
        style = arts.get('onset', {}).get('style', 'glottal' if is_cry or r['velocity'] >= 95 else ('hum' if is_hum else 'breathy'))
        attack = .01 if style == 'glottal' else .055
        env = v1.smoothstep(age / attack) if entry or cry_entry or 'onset' in arts else np.ones(count)
        if final:
            env *= 1 - v1.smoothstep((t - off) / q['release_s'])
        envelope[mask] = env[mask]
        # Source loudness and quality both follow CC11 and the swell.
        db = 24 * (r['velocity'] / 127 - 1)
        swell = rng.uniform(2, 6) * np.sin(np.pi * np.clip(age / duration, 0, 1)) if duration > .8 else np.zeros(count)
        level[mask] = db + swell[mask]
        if not entry and is_hum:
            dip, width = rng.uniform(2, 8), rng.uniform(.04, .09)
            if r['pitch'] == prev['pitch']:
                dip = max(dip, 6)
            shape = np.exp(-.5 * ((t - start) / (width / 3)) ** 2)
            level -= dip * shape
            dips.append({'time': start, 'depth_db': dip, 'width_s': width})
        # Per-note rate and depth wandering, separate from straight-tone flutter.
        delay = rng.uniform(.2, min(.8, max(.21, duration * .2)))
        delays.append(delay if duration >= .5 else None)
        if duration >= .5 and 'straight' not in arts:
            rate = np.clip(rng.uniform(5, 6) + _walk(rng, age, .16, 1.05), 3.2, 8.5)
            phase = 2 * np.pi * np.cumsum(rate) * dt + rng.uniform(0, 2 * np.pi)
            depth = np.clip(rng.uniform(30, 58) + _walk(rng, age, .22, 22), 5, 95)
            lane = np.sin(phase) * depth * v1.smoothstep((age - delay) / .4) * q['vibrato']
            vibrato[mask] = lane[mask]
        if 'quiver' in arts:
            vibrato[mask] += 100 * arts['quiver'].get('depth', .6) * np.sin(2 * np.pi * rng.uniform(5, 7) * age[mask]) * v1.smoothstep(age[mask] / .1)
        if style == 'scoop' or (entry and style != 'glottal'):
            depth = arts.get('onset', {}).get('depth', rng.uniform(.5, 1.5) if not is_hum else .3)
            pitch[mask] -= depth * (1 - v1.smoothstep(age[mask] / .16))
        if 'slide' in arts:
            art = arts['slide']
            pitch[mask] += art['to'] * v1.smoothstep(age[mask] / art['duration_s'])
        if 'flick' in arts:
            pitch[mask] += arts['flick'].get('to', 2) * np.exp(-.5 * ((age[mask] - .10) / .025) ** 2)
        if final or 'fall' in arts:
            depth = arts.get('fall', {}).get('depth', rng.uniform(.2, .8))
            length = arts.get('fall', {}).get('duration_s', .2)
            lane = v1.smoothstep((t - boundary + length) / length)
            pitch[mask] -= depth * lane[mask]
            level[mask] -= 4 * lane[mask]
            onset_rd[mask] += .4 * lane[mask]
        if style == 'breathy':
            onset_rd[mask] += 1.0 * (1 - v1.smoothstep(age[mask] / .12))
            ahead = (t >= start - .055) & (t < start)
            inhale[ahead] = .003 * np.sin(np.pi * (t[ahead] - start + .055) / .055)
        elif style == 'glottal' and (entry or cry_entry or 'onset' in arts):
            onset_rd[mask] -= .6 * (1 - v1.smoothstep(age[mask] / .04))
        if 'creak' in arts or (final and r['pitch'] <= 60):
            creak[mask] = v1.smoothstep((t[mask] - boundary + .12) / .16)
        vowel = VOWELS.get(syllable, 'ax')
        if syllable in v1.LEXICON and syllable not in VOWELS and not is_hum:
            vowel = v1.parse_syllable(syllable)[1][0][0]
        target = shapes[vowel].copy()
        if is_hum:
            hum[mask] = 1
            cc = config['consonants'][HUMS[syllable]]
            target = v1.constrict(shapes['ax'], len(target), [[cc['place_cm'], cc['width_cm'], 0]], v1.section_length_cm(config))
            velum[mask] = q['velum_cm2']
        if syllable == 'ai':
            vowel_start = start if 'cry' in arts else r['_unit_start']
            vowel_end = boundary if 'cry' in arts else r['_unit_end']
            blend = v1.smoothstep(((t[mask] - vowel_start) / max(vowel_end - vowel_start, 1e-6) - .55) / .45)[:, None]
            areas[mask] = target * (1 - blend) + shapes['iy'] * blend
        else:
            areas[mask] = target
        if is_cry:
            cry[mask] = 1
            closed = v1.constrict(shapes['ax'], len(target), [[.36, .73, 0]], v1.section_length_cm(config))
            cry_start = start if 'cry' in arts else r['_unit_start']
            cry_end = boundary if 'cry' in arts else r['_unit_end']
            opening = v1.smoothstep((t[mask] - cry_start) / .06) * (1 - v1.smoothstep((t[mask] - cry_end + .09) / .09))
            areas[mask] = closed * (1 - opening[:, None]) + areas[mask] * opening[:, None]
            velum[mask] = q['velum_cm2'] * (1 - opening)
            cry_open[mask] = opening
            cry_vowel[mask] = vowel
            if r['_hum_db'] is not None and not q.get('_calibrating'):
                # In a hummed phrase the cry's level is the phrase's hum level
                # plus cry_lift_db (radiation-balanced below), not its own velocity.
                level[mask] += r['_hum_db'] - db + q['cry_lift_db']
        if final and nxt and nxt['time'] - off >= .5:
            gap = nxt['time'] - off
            # Opening the tract while release is still voiced makes a
            # vowel burst, not an inhalation. Reserve the release first.
            length = min(rng.uniform(.25, .6), max(0., gap - q['release_s'] - .065))
            end = nxt['time'] - .045
            bm = (t >= end - length) & (t < end)
            inhale[bm] = 10 ** ((db + q['inhalation_db'] + (0 if is_hum else 10)) / 20) * np.sin(np.pi * (t[bm] - end + length) / length)
            areas[bm] = shapes['ax']
            velum[bm] = 0
            breaths.append({'start': end - length, 'end': end, 'gap_s': gap})
        if 'breath' in arts:
            bm = (t >= start - .35) & (t < start - .03)
            inhale[bm] = 10 ** ((db + q['inhalation_db'] + (0 if is_hum else 10)) / 20) * np.sin(np.pi * (t[bm] - start + .35) / .32)
            areas[bm] = shapes['ax']
            velum[bm] = 0
    cc = sorted((c for c in midi_controls if c['control'] == 11), key=lambda c: c['time'])
    if cc:
        if any(not math.isfinite(c['time']) or c['time'] < 0 or not 0 <= c['value'] <= 127 for c in cc):
            raise ValueError('invalid voice2 CC11')
        expression = np.interp(t, [c['time'] for c in cc], [c['value'] / 127 for c in cc])
    else:
        expression = np.ones(count)
    level += 20 * np.log10(np.maximum(expression, 1e-9))
    effort = np.maximum(np.clip((level + 17) / 17, 0, 1), .85 * cry)
    for vowel in {v for v in cry_vowel if v} if not q.get('_calibrating') else ():
        m = (cry_vowel == vowel) & (cry_open > 0)
        if not m.any():
            continue
        balance = cry_balance_db(config, CRY_SYLLABLE[vowel], 440 * 2 ** ((pitch[m] - 69) / 12))
        # Radiation jumps as soon as the lips part, so the balance leads the opening.
        level[m] -= np.maximum(balance, 0) * v1.smoothstep(cry_open[m] / .3)
    flutter = sum(np.sin(2 * np.pi * hz * t + rng.uniform(0, 2 * np.pi)) for hz in (.71, 1.13, 2.37)) * q['flutter_cents'] * math.sqrt(2 / 3)
    head = v1.smoothstep((440 * 2 ** ((pitch - 69) / 12) - q['register_hz'][0]) / np.diff(q['register_hz'])[0])
    cents = 100 * pitch + flutter + vibrato * (1 - .2 * head)
    f0 = 440 * 2 ** ((cents / 100 - 69) / 12)
    # Final fry blends into irregular subharmonic periods, not added oscillators.
    fry = np.clip(48 + _walk(rng, t, .025, 15), 30, 70)
    f0 = f0 * (1 - creak) + fry * creak
    rd = np.clip(q['rd_soft'] + effort * (q['rd_loud'] - q['rd_soft']) + q['head_rd'] * head + onset_rd - q['tension'], .3, 2.7)
    rd = rd * (1 - hum) + np.clip(rd, 1.15, 1.4) * hum
    return {'t': t, 'dt': dt, 'steps': count, 'areas': areas, 'velum': velum, 'hum': hum,
            'f0': f0, 'pitch': pitch, 'amplitude': 10 ** (level / 20) * envelope * (expression > 0),
            'effort': effort, 'rd': rd, 'head': head, 'inhale': inhale, 'creak': creak,
            'flutter': flutter, 'vibrato': vibrato, 'delays': delays, 'dips': dips,
            'transitions': transitions, 'breaths': breaths}


def sounding_spans(k, samples, step, pad_s=.4, join_s=1.5):
    """Sample spans around the voice's sounding stretches, on control-step boundaries.

    Outside them the source is exactly zero (no amplitude, no inhalation), so each span
    renders from rest; the pad covers the release, the tract's ring and the output
    filters' tails. Long pieces are synthesized span by span to bound memory.
    """
    active = (k['amplitude'] > 0) | (k['inhale'] > 0)
    if not active.any():
        return []
    pad, join = round(pad_s / k['dt']), round(join_s / k['dt'])
    idx = np.flatnonzero(active)
    breaks = np.flatnonzero(np.diff(idx) > join + 2 * pad)
    firsts, lasts = np.r_[idx[0], idx[breaks + 1]], np.r_[idx[breaks], idx[-1]]
    limit = math.ceil(samples / step)
    return [(max(f - pad, 0) * step, min(min(l + pad, limit) * step, samples)) for f, l in zip(firsts, lasts)]


def _synthesize(k, q, a, b, sr, step, rng, jitter, shimmer, table, flow, params, filters):
    """One sounding span of the voice, samples [a, b), from a tract at rest."""
    samples = b - a
    t = (a + np.arange(samples)) / sr
    lane = lambda key: np.interp(t, k['t'], k[key])
    f0, amp, rd = lane('f0'), lane('amplitude'), lane('rd')
    # Alternating pulse amplitudes give creak an irregular double-period component.
    amp *= 1 - lane('creak') * .35 * (1 + np.sin(2 * np.pi * 24 * t))
    d, g = v1._glottis(f0, amp, (rd - .3) / 2.4, jitter, shimmer, table, flow, 4, sr)
    d, g = resample_poly(d, 1, 4)[:samples], resample_poly(g, 1, 4)[:samples]
    # Effort-dependent closure sharpening complements LF timing; Ee is
    # still the excitation scale. This is a chosen source tilt, not a filter
    # on the radiated voice or a substitute for nasal geometry.
    sharp = np.r_[0., np.diff(d)] * sr / (2 * np.pi * q['closure_hz'])
    d += sharp * lane('effort') ** 2 * (1 - lane('hum'))
    # Head voice rounds the glottal source, before any tract scattering.
    rounded = sosfilt(butter(1, 1600, fs=sr, output='sos'), d)
    head = lane('head')
    d = d * (1 - head) + rounded * head
    white = rng.standard_normal(samples)
    oral_noise = sosfilt(butter(2, 800, 'highpass', fs=sr, output='sos'), white)
    hum_noise = sosfilt(butter(1, q['hum_air_hz'], fs=sr, output='sos'), white)
    hum_noise = sosfilt(butter(1, 100, 'highpass', fs=sr, output='sos'), hum_noise)
    hum = lane('hum')
    noise = oral_noise * (1 - hum) + hum_noise * hum
    air = q['breathiness'] * (rd / 2.7) ** 2
    air *= 1 + q['hum_soft_air'] * hum * (1 - lane('effort')) ** 2
    src = d + noise * (air * (q['aspiration'] * g + q['leak'] * amp) + lane('inhale'))
    first = a // step
    areas, velum = k['areas'][first:], k['velum'][first:]
    y = v1._tract(src, np.zeros(samples), np.zeros(len(areas), np.int64), np.ascontiguousarray(areas), np.ascontiguousarray(velum), step, **params)
    # Mirror-image rejection shared with v1, not a hum-darkening filter.
    for sos in filters:
        y = sosfilt(sos, y)
    return y


def render(notes, seconds=None, config=None, return_controls=False, midi_controls=(), articulations=()):
    c = config or load_config()
    q = c['voice2']
    if seconds is None:
        seconds = max(r['off'] for r in notes) + q['tail_seconds'] if notes else 0.
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError('seconds must be finite and nonnegative')
    samples = round(seconds * 48000)
    if not notes or samples == 0:
        out = np.zeros(samples)
        return (out, None) if return_controls else out
    k = controls(notes, c, seconds, midi_controls, articulations)
    sr, step = 48000, 48
    rng = np.random.default_rng(np.random.SeedSequence([q['seed'], 2]))
    jitter = rng.normal(0, q['jitter'], 32768)
    shimmer = rng.normal(0, q['shimmer'], 32768)
    table, flow = rd_tables()
    params = v1.tract_parameters(c)
    params.update(sinus_hz=q['sinus_hz'], sinus_area=q['sinus_area'], sinus_q=q['sinus_q'], nasal_wall_hz=q['nasal_wall_hz'])
    filters = [butter(4, 10000, fs=sr, output='sos'), butter(1, 30, 'highpass', fs=sr, output='sos')]
    out = np.zeros(samples)
    for a, b in sounding_spans(k, samples, step):
        out[a:b] = _synthesize(k, q, a, b, sr, step, rng, jitter, shimmer, table, flow, params, filters)
    out *= c['engine']['output_gain']
    return (out, k) if return_controls else out
