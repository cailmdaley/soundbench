#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["mido>=1.3,<2", "numpy>=1.26", "numba>=0.61", "scipy>=1.11"]
# ///
"""Render one shared physical guitar into source-attributed studio stems."""
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
from soundbench.scripts.paths import preset_bytes, preset_path
from soundbench.scripts.instrumentation import matching_tracks
from soundbench.scripts.models.guitar import load_config, configure, assign_string, render, note, strum
from soundbench.scripts.perform import TempoMap, BarMap
from soundbench.scripts.plucked import track_options
from soundbench.scripts.models.guitar_release import finish
from soundbench.scripts.render_model import notes, write_wav, assignment_statistics


def schedule(performed, score, settings, indices, config):
    sr = 48000
    tempo, source_tempo, bars = TempoMap(performed), TempoMap(score), BarMap(score)
    names = {i: performed.tracks[i].name for i in indices}
    note_tracks = [i for i,t in enumerate(performed.tracks) if any(m.type == 'note_on' and m.velocity for m in t)]
    options = track_options(settings, note_tracks, {i:t.name for i,t in enumerate(performed.tracks)})
    pins = {}
    for selector, string in settings.get('guitar_strings', {}).items():
        matches = matching_tracks(selector, indices, names)
        if not matches or not isinstance(string, int) or string not in range(1, 7):
            raise ValueError('guitar_strings needs matching tracks and strings 1..6')
        for index in matches:
            if index in pins:
                raise ValueError('duplicate guitar string mapping')
            pins[index] = string
    rows, anchors = [], {}
    for owner, index in enumerate(indices):
        scored = defaultdict(deque)
        for r in notes(score, index):
            scored[r['pitch']].append(r)
        pairs = []
        for r in notes(performed, index):
            if not scored[r['pitch']]:
                raise ValueError('score/performance note mismatch')
            src = scored[r['pitch']].popleft()
            r.update(owner=owner, index=index, token=index*1000000+r['token'],
                     score_tick=src['tick'], time=tempo.seconds_at(r['tick']),
                     end=tempo.seconds_at(r['off']), explicit=pins.get(index), art={})
            rows.append(r)
            pairs.append((source_tempo.seconds_at(src['tick']), r['time']))
        if any(scored.values()):
            raise ValueError('unmatched scored guitar notes')
        groups = defaultdict(list)
        for x, y in pairs:
            groups[x].append(y)
        anchors[index] = (np.array(sorted(groups)), np.array([np.mean(groups[t]) for t in sorted(groups)]))
    def at(index, tick):
        x, y = anchors[index]
        t = source_tempo.seconds_at(tick)
        if not len(x):
            raise ValueError('gesture track has no notes')
        if t < x[0]:
            return max(0., t+y[0]-x[0])
        if t > x[-1]:
            return t+y[-1]-x[-1]
        return float(np.interp(t, x, y))
    damps = []
    occupied = set()
    allowed = {'track', 'bar', 'beat', 'kind', 'string', 'hardness', 'position',
               'to', 'duration_beats', 'shape', 'direction', 'speed_ms', 'accent', 'fingers', 'strings', 'reach', 'mute'}
    for art in settings.get('guitar_articulation', []):
        if set(art)-allowed or not {'track', 'bar', 'kind'} <= set(art):
            raise ValueError(f'invalid guitar articulation: {art}')
        matches = matching_tracks(art['track'], indices, names)
        if len(matches) != 1 or not isinstance(art['bar'], int):
            raise ValueError('guitar articulation needs one track and integer bar')
        index = matches[0]
        tick = bars.position(art['bar'], float(art.get('beat', 1)))
        if (index, tick) in occupied:
            raise ValueError('one guitar gesture per track/onset')
        occupied.add((index, tick))
        kind = art['kind']
        if kind not in ('pluck', 'hammer', 'pull', 'slide', 'bend', 'strum', 'chuck', 'rasgueado', 'damp', 'palm-mute'):
            raise ValueError(f'unknown guitar gesture {kind}')
        if 'string' in art and (not isinstance(art['string'], int) or art['string'] not in range(1, 7)):
            raise ValueError('gesture string must be 1..6')
        if kind == 'damp':
            damps.append((index, tick, art.get('string')))
            continue
        selected = [r for r in rows if r['index'] == index and r['score_tick'] == tick]
        if not selected or (kind not in ('strum', 'chuck', 'rasgueado') and len(selected) != 1):
            raise ValueError('gesture needs an exact scored onset; pitch gestures need one note')
        strumming = kind in ('strum', 'chuck', 'rasgueado')
        if 'fingers' in art and (not isinstance(art['fingers'], int) or art['fingers'] not in range(1, 5)):
            raise ValueError('fingers must be an integer 1..4')
        if 'strings' in art:
            strings = art['strings']
            if not strumming or not isinstance(strings, list) or len(strings) != len(selected) or len(set(strings)) != len(strings) or any(not isinstance(s, int) or s not in range(1, 7) for s in strings):
                raise ValueError('strum strings must pair distinct strings 1..6 with pitches, highest first')
        if strumming and 'string' in art:
            raise ValueError('strums use strings, not string')
        if ('reach' in art and not strumming) or ('mute' in art and kind not in ('strum', 'rasgueado')):
            raise ValueError('reach belongs to strokes; mute to strum and rasgueado')
        if 'reach' in art and (not isinstance(art['reach'], int) or isinstance(art['reach'], bool) or not 1 <= art['reach'] <= len(selected)):
            raise ValueError('reach is a string count within the chord')
        if 'mute' in art and (isinstance(art['mute'], bool) or not isinstance(art['mute'], (int, float)) or not .005 <= art['mute'] <= 2):
            raise ValueError('mute is a damping T60 of 0.005..2 s; use kind = "chuck" for a choked brush')
        for ordinal, r in enumerate(sorted(selected, key=lambda r: -r['pitch'])):
            r['art'] = art
            r['explicit'] = art['strings'][ordinal] if 'strings' in art else art.get('string', r['explicit'])
        if kind == 'bend' and not abs(float(art.get('to', 99))) <= config['strings']['max_bend_semitones']:
            raise ValueError(f"this preset's strings bend at most {config['strings']['max_bend_semitones']} semitones")
        if kind in ('slide', 'bend'):
            if not {'to', 'duration_beats'} <= set(art) or float(art['duration_beats']) <= 0 or art.get('shape', 'linear') not in ('linear', 'ease'):
                raise ValueError('pitch ramp needs to, positive duration_beats and linear/ease shape')
            if not math.isfinite(float(art['to'])) or not math.isfinite(float(art['duration_beats'])):
                raise ValueError('pitch ramp must be finite')
    current = [None]*6
    events, assignment = [], []
    for r in sorted(rows, key=lambda r: (r['time'], -r['pitch'], r['owner'], r['token'])):
        if r.get('handled', False):
            continue
        art, t = r['art'], r['time']
        kind = art.get('kind', 'pluck')
        hardness = {'preset': 0, 'pick': 0, 'flesh': 1, 'nail': 2}.get(art.get('hardness', 'preset'))
        if hardness is None:
            raise ValueError('hardness is preset/pick/flesh/nail')
        chord = [q for q in rows if q['index'] == r['index'] and q['score_tick'] == r['score_tick']] if kind in ('strum', 'chuck', 'rasgueado') else [r]
        used = set()
        for q in sorted(chord, key=lambda q: -q['pitch']):
            sounding = [s in used or (old is not None and old['end'] > t) for s, old in enumerate(current)]
            started = [old['time'] if old else -math.inf for old in current]
            s = assign_string(q['pitch'], config, q['explicit'], sounding, started)
            if s in used:
                raise ValueError('chord cannot fit distinct strings; split/pin voices')
            if kind in ('hammer', 'pull'):
                if q['explicit'] is None:
                    raise ValueError('hammer/pull needs an explicit string or pinned track')
                old = current[s]
                if old is None or old['index'] != q['index'] or old['end'] < t-.001 or ((q['pitch'] > old['pitch']) != (kind == 'hammer')) or q['pitch'] == old['pitch']:
                    raise ValueError('hammer/pull needs a held source on the same track/string and correct pitch direction')
            q['string'] = s+1
            used.add(s)
            q['previous'] = current[s]
            current[s] = q
            q['handled'] = True
            assignment.append(dict(track=names[q['index']], time_s=t, off_time_s=q['end'], pitch=q['pitch'], string=s+1, kind=kind, token=q['token']))
        if kind in ('strum', 'chuck', 'rasgueado'):
            # Reserve a separate token range for sweep repeats; stale MIDI offs cannot damp them.
            generated = strum(t, max(.02, max(q['end'] for q in chord)-t),
                [q['pitch'] for q in chord], config, r['velocity'],
                art.get('direction', 'down'), float(art['speed_ms']) if 'speed_ms' in art else None,
                [q['string'] for q in chord], float(art.get('accent', 1)),
                True if kind == 'chuck' else float(art['mute']) if 'mute' in art else False,
                int(art.get('fingers', 3 if kind == 'rasgueado' else 1)),
                100000000 + r['token']*32, r['owner'], hardness,
                let_ring=options[r['index']].get('let_ring', True),
                tempo_bpm=min(300, max(30, 60/(tempo.seconds_at(r["tick"]+performed.ticks_per_beat)-tempo.seconds_at(r["tick"])))),
                reach=art.get('reach'))
            if kind != 'chuck':
                # Stroke offsets change attacks, not the authored sustain boundary.
                ends = {q['string']-1: round(q['end']*sr) for q in chord}
                starts = {int(ev[10]): int(ev[0]) for ev in generated if ev[1] == 0}
                for ev in generated:
                    if ev[1] == 1:
                        ev[0] = max(ends[int(ev[2])], starts[int(ev[10])]+1)
            events += generated
            # Track final sweep tokens for left-hand damping.
            for q in chord:
                struck = [ev for ev in generated if ev[1] == 0 and ev[2] == q['string']-1]
                if not struck:
                    q['silent'] = True
                    current[q['string']-1] = q['previous']
                    continue
                q['token'] = int(max(struck,key=lambda ev:ev[0])[10])
        else:
            k = 3 if kind == 'hammer' else 4 if kind == 'pull' else 0
            generated = note(t, max(t+.001, r['end']), r['pitch'], r['velocity'], r['string'], r['token'], owner=r['owner'], kind=k, hardness=hardness, let_ring=options[r['index']].get('let_ring', False))
            generated[0][8] = float(art.get('position', 0))
            events += generated
            if kind == 'palm-mute':
                events.append([round((t+.03)*sr), 5, r['string']-1, r['owner'], 0, 0, 0, 0, 0, 0, r['token']])
            if kind in ('slide', 'bend'):
                target = r['pitch']+float(art['to'])
                assign_string(target, config, r['string'])
                end_tick = r['score_tick']+float(art['duration_beats'])*bars.beat_length_at(r['score_tick'])
                length = round((at(r['index'], end_tick)-t)*sr)
                events.append([round(t*sr), 2, r['string']-1, r['owner'], target, 0, max(1, length), art.get('shape') == 'ease', 0, 0, r['token']])
    for index, tick, string in damps:
        t = at(index, tick)
        candidates = [r for r in rows if r['index'] == index and r['time'] < t and not r.get('silent',False) and (string is None or r['string'] == string)]
        if not candidates:
            raise ValueError('damp needs a preceding note')
        for s in set(r['string'] for r in candidates):
            r = max((q for q in candidates if q['string'] == s), key=lambda q: q['time'])
            events.append([round(t*sr), 5, s-1, r['owner'], 0, 0, 0, 0, 0, 0, r['token']])
    events.sort(key=lambda ev: (ev[0], 0 if ev[1] == 1 else 1))
    end_tick = max(sum(m.time for m in track) for track in performed.tracks)
    end = max(tempo.seconds_at(end_tick), max((ev[0]+ev[6] for ev in events), default=0)/sr)
    events = finish(events, end + config['engine']['tail_seconds']/2)
    events.sort(key=lambda ev: (ev[0], 0 if ev[1] == 1 else 1))
    return events, end+config['engine']['tail_seconds'], assignment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    args = parser.parse_args()
    data = json.loads(args.manifest.read_text())
    options = data['settings'].get('guitar_model', {})
    if set(options)-{'preset', 'output_gain', 'tuning', 'capo'}:
        raise ValueError('unknown guitar_model option')
    path = preset_path('guitar', options.get('preset'))
    c = configure(load_config(path), options.get('tuning', 'standard'), options.get('capo', 0))
    if 'output_gain' in options:
        gain = float(options['output_gain'])
        if not math.isfinite(gain) or gain <= 0:
            raise ValueError('output_gain must be positive and finite')
        c['engine']['output_gain'] = gain
    performed, score = mido.MidiFile(data['performed_midi']), mido.MidiFile(data['score_midi'])
    indices = [job['index'] for job in data['jobs']]
    events, seconds, assignment = schedule(performed, score, data['settings'], indices, c)
    render([], .001, len(indices), c)
    start = time.perf_counter()
    audio = render(events, seconds, len(indices), c)
    elapsed = time.perf_counter()-start
    stems = {}
    for owner, job in enumerate(data['jobs']):
        write_wav(Path(job['output']), audio[owner])
        stems[str(job['index'])] = dict(engine='model', model='guitar',
            parameters=str(path), parameters_sha256=hashlib.sha256(preset_bytes(path)+json.dumps(options, sort_keys=True).encode()).hexdigest(), overrides=options)
    metrics = dict(stems=stems, audio_seconds=seconds, render_seconds=elapsed,
        realtime_factor=seconds/elapsed, assignment=assignment,
        peak=float(np.max(np.abs(audio), initial=0)), assignment_statistics=assignment_statistics(events))
    args.manifest.with_suffix('.metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    print(f'Guitar ({path.stem}): {seconds:.2f}s in {elapsed:.3f}s ({seconds/elapsed:.1f}x realtime)')


if __name__ == '__main__':
    main()
