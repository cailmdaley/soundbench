#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["mido>=1.3,<2", "numpy>=1.26", "numba>=0.61", "scipy>=1.11"]
# ///
"""Render voice2 with score-addressed articulations and hashed preset provenance."""
from __future__ import annotations
import argparse
from collections import defaultdict, deque
import hashlib
import json
import math
from pathlib import Path
import time
import mido
import numpy as np
from soundbench.scripts.instrumentation import matching_tracks
from soundbench.scripts.models import voice2
from soundbench.scripts.perform import TempoMap, BarMap
from soundbench.scripts.render_model import notes as track_notes, write_wav
from soundbench.scripts.render_voice import lyrics_for

LIMITS = {'vibrato': (0, 3), 'breathiness': (0, 4), 'tension': (-1, 1),
          'release_s': (.01, 5), 'tail_seconds': (.1, 10), 'cry_lift_db': (-6, 18)}


def selected_config(settings):
    options = settings.get('voice2_model', {})
    if not isinstance(options, dict) or set(options) - (set(LIMITS) | {'preset', 'seed', 'unknown_syllable'}):
        raise ValueError('unknown voice2_model setting')
    c = voice2.load_config(options.get('preset', 'tenor'))
    for key, value in options.items():
        if key == 'preset':
            continue
        if key == 'unknown_syllable':
            voice2.syllable_key(value)
            continue
        if key == 'seed':
            if type(value) is not int or value < 0:
                raise ValueError('voice2 seed must be a nonnegative integer')
        elif type(value) not in (int, float) or not math.isfinite(value) or not LIMITS[key][0] <= value <= LIMITS[key][1]:
            raise ValueError(f'invalid voice2_model.{key}')
        c['voice2'][key] = value
    if c['voice2']['tail_seconds'] < c['voice2']['release_s']:
        raise ValueError('voice2 tail must contain release')
    files = [voice2.v1.CONFIG, voice2.CONFIG]
    if options.get('preset') == 'alto':
        files.append(voice2.PRESETS['alto'])
    digest = hashlib.sha256(b''.join(p.read_bytes() for p in files) + json.dumps(options, sort_keys=True).encode()).hexdigest()
    return c, options, digest


def sung_notes(performed, score, index, unknown_syllable=None):
    """Scored notes at performed times; an unpronounceable lyric sings `unknown_syllable` if given."""
    tempo, lyrics = TempoMap(performed), lyrics_for(score, index)
    scored = defaultdict(deque)
    for r in track_notes(score, index):
        scored[r['pitch']].append(r)
    rows = []
    for r in track_notes(performed, index):
        if not scored[r['pitch']]:
            raise ValueError('score/performed note correspondence differs')
        s = scored[r['pitch']].popleft()
        text = word = lyrics.get(s['tick'])
        if text is not None and unknown_syllable is not None:
            try:
                voice2.syllable_key(text)
            except ValueError:
                text = unknown_syllable
        rows.append({'time': tempo.seconds_at(r['tick']), 'off': tempo.seconds_at(r['off']),
                     'pitch': r['pitch'], 'velocity': r['velocity'], 'syllable': text,
                     'score_tick': s['tick'], **({'word': word} if word != text else {})})
    if any(scored.values()):
        raise ValueError('unmatched score notes')
    return rows


def schedule(performed, score, settings, indices, rows):
    """Interpolate score/performed anchors, in the meter's beat units."""
    bars, source = BarMap(score), TempoMap(score)
    names = {i: performed.tracks[i].name for i in indices}
    out = {i: [] for i in indices}
    occupied = set()
    for art in settings.get('articulation', []):
        if not isinstance(art, dict) or not {'track', 'bar', 'kind'} <= set(art):
            raise ValueError('articulation needs track, bar, kind')
        matches = matching_tracks(art['track'], indices, names)
        if not matches:
            # Other instruments own their articulations.
            all_indices = [i for i, tr in enumerate(performed.tracks) if any(m.type == 'note_on' and m.velocity for m in tr)]
            if not matching_tracks(art['track'], all_indices, {i: tr.name for i, tr in enumerate(performed.tracks)}):
                raise ValueError('articulation track matched no notes')
            continue
        if len(matches) != 1 or type(art['bar']) is not int:
            raise ValueError('articulation must select one track and integer bar')
        index = matches[0]
        kind = art['kind']
        if kind not in voice2.ART_KEYS:
            raise ValueError(f'unknown voice2 articulation {kind!r}')
        allowed = (voice2.ART_KEYS[kind] - {'duration_s'}) | {'track', 'bar', 'beat', 'kind'}
        if 'duration_s' in voice2.ART_KEYS[kind]:
            allowed.add('duration_beats')
        if set(art) - allowed:
            raise ValueError(f'unknown voice2 articulation keys: {art}')
        beat = art.get('beat', 1)
        if type(beat) not in (int, float) or not math.isfinite(beat):
            raise ValueError('invalid articulation beat')
        tick = bars.position(art['bar'], beat)
        candidates = [j for j, r in enumerate(rows[index]) if r['score_tick'] == tick]
        if len(candidates) != 1:
            raise ValueError('voice2 articulation must address one scored note onset')
        note = candidates[0]
        a = {k: v for k, v in art.items() if k not in ('track', 'bar', 'beat')}
        if 'duration_beats' in a:
            dur = a.pop('duration_beats')
            if type(dur) not in (int, float) or not math.isfinite(dur) or dur <= 0:
                raise ValueError('duration_beats must be positive')
            x = np.array([source.seconds_at(r['score_tick']) for r in rows[index]])
            y = np.array([r['time'] for r in rows[index]])
            end = source.seconds_at(tick + dur * bars.beat_length_at(tick))
            mapped = float(np.interp(end, x, y)) if end <= x[-1] else end + y[-1] - x[-1]
            a['duration_s'] = float(mapped - rows[index][note]['time'])
        voice2.validate_articulation(a)
        key = (index, note, a['kind'])
        if key in occupied:
            raise ValueError('duplicate voice2 articulation kind at note')
        occupied.add(key)
        out[index].append({'note': note, 'art': a})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    args = parser.parse_args()
    data = json.loads(args.manifest.read_text())
    c, options, digest = selected_config(data.get('settings', {}))
    performed, score = mido.MidiFile(data['performed_midi']), mido.MidiFile(data['score_midi'])
    indices = [j['index'] for j in data['jobs']]
    rows = {i: sung_notes(performed, score, i, options.get('unknown_syllable')) for i in indices}
    arts = schedule(performed, score, data.get('settings', {}), indices, rows)
    seconds = TempoMap(performed).seconds_at(max(sum(m.time for m in tr) for tr in performed.tracks)) + c['voice2']['tail_seconds']
    start = time.perf_counter()
    voice2.render([dict(time=0, off=.05, pitch=62, velocity=70, syllable='ah')], .1, c)
    compile_seconds = time.perf_counter() - start
    stems, elapsed, peak = {}, 0., 0.
    for job in data['jobs']:
        tick, cc = 0, []
        for m in performed.tracks[job['index']]:
            tick += m.time
            if m.type == 'control_change' and m.control == 11:
                cc.append(dict(time=TempoMap(performed).seconds_at(tick), control=11, value=m.value))
        start = time.perf_counter()
        audio = voice2.render(rows[job['index']], seconds, c, midi_controls=cc, articulations=arts[job['index']])
        elapsed += time.perf_counter() - start
        write_wav(Path(job['output']), audio, 48000)
        peak = max(peak, float(np.max(np.abs(audio))))
        stems[str(job['index'])] = dict(engine='model', model='voice2', parameters=str(voice2.PRESETS[options.get('preset', 'tenor')]), parameters_sha256=digest, overrides=options)
    metrics = dict(stems=stems, audio_seconds=seconds, render_seconds=elapsed, compile_or_cache_seconds=compile_seconds,
                   realtime_factor=seconds * len(indices) / max(elapsed, 1e-9), peak=peak, articulations=arts)
    args.manifest.with_suffix('.metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print(f'Voice2: {seconds:.2f} s in {elapsed:.3f} s, peak {peak:.3f}')


if __name__ == '__main__':
    main()
