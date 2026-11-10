#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["mido>=1.3,<2", "numpy>=1.26", "numba>=0.61", "scipy>=1.11"]
# ///
"""Render four shared fretless bass strings into aligned source-attributed stems."""
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
from soundbench.scripts.models.bass import load_config, assign_string, render, note
from soundbench.scripts.perform import TempoMap, BarMap
from soundbench.scripts.render_model import notes, write_wav


def schedule(performed, score, settings, indices, config):
    tempo, source_tempo, bars = TempoMap(performed), TempoMap(score), BarMap(score)
    names = {i:performed.tracks[i].name for i in indices}
    pins = {}
    for selector,string in settings.get('bass_strings',{}).items():
        matches = matching_tracks(selector,indices,names)
        if not matches or type(string) is not int or string not in range(1,5):
            raise ValueError('bass_strings needs matching tracks and strings 1..4')
        for i in matches:
            if i in pins:
                raise ValueError('duplicate bass string assignment')
            pins[i] = string
    rows, maps = [], {}
    for owner,index in enumerate(indices):
        scored = defaultdict(deque)
        for r in notes(score,index):
            scored[r['pitch']].append(r)
        anchors = defaultdict(list)
        for r in notes(performed,index):
            if not scored[r['pitch']]:
                raise ValueError('score/performance bass note mismatch')
            src = scored[r['pitch']].popleft()
            r.update(owner=owner,index=index,token=index*1000000+r['token'],score_tick=src['tick'],
                     time=tempo.seconds_at(r['tick']),end=tempo.seconds_at(r['off']),explicit=pins.get(index),art={})
            rows.append(r)
            anchors[source_tempo.seconds_at(src['tick'])].append(r['time'])
        if any(scored.values()):
            raise ValueError('unmatched scored bass notes')
        maps[index] = (np.array(sorted(anchors)),np.array([np.mean(anchors[t]) for t in sorted(anchors)]))
    def at(index,tick):
        x,y = maps[index]
        if not len(x):
            raise ValueError('gesture track has no notes')
        t = source_tempo.seconds_at(tick)
        if t < x[0]:
            return max(0.,t+y[0]-x[0])
        if t > x[-1]:
            return t+y[-1]-x[-1]
        return float(np.interp(t,x,y))
    occupied, damps = set(), []
    for art in settings.get('bass_articulation',[]):
        kind = art.get('kind')
        allowed = {'track','bar','beat','kind','string'}
        if kind in ('pizz','slap','slap-back','dead','ghost','slide'):
            allowed |= {'position','cents'}
        if kind == 'slide':
            allowed |= {'to','duration_beats','shape'}
        if kind not in ('pizz','slap','slap-back','dead','ghost','slide','damp') or set(art)-allowed or not {'track','bar','kind'} <= set(art):
            raise ValueError(f'invalid bass articulation: {art}')
        matches = matching_tracks(art['track'],indices,names)
        if len(matches) != 1 or type(art['bar']) is not int:
            raise ValueError('bass gesture needs one track and integer bar')
        index = matches[0]
        tick = bars.position(art['bar'],float(art.get('beat',1)))
        if (index,tick) in occupied:
            raise ValueError('one bass gesture per onset')
        occupied.add((index,tick))
        if 'string' in art and (type(art['string']) is not int or art['string'] not in range(1,5)):
            raise ValueError('bass string must be 1..4')
        if kind == 'damp':
            damps.append((index,tick,art.get('string')))
            continue
        selected = [r for r in rows if r['index'] == index and r['score_tick'] == tick]
        if len(selected) != 1:
            raise ValueError('bass articulation needs exactly one scored note at its position')
        r = selected[0]
        r['art'],r['explicit'] = art,art.get('string',r['explicit'])
        if kind == 'slide':
            if not {'to','duration_beats'} <= set(art) or not all(math.isfinite(float(art[k])) for k in ('to','duration_beats')) or float(art['duration_beats']) <= 0 or art.get('shape','ease') not in ('linear','ease'):
                raise ValueError('slide needs finite to, positive duration_beats, linear/ease shape')
    current = [None]*4
    events,assignment = [],[]
    for r in sorted(rows,key=lambda r:(r['time'],r['owner'],r['token'])):
        art,t = r['art'],r['time']
        kind = art.get('kind','pizz')
        sounding = [old is not None and old['end'] > t for old in current]
        started = [old['time'] if old else -math.inf for old in current]
        s = assign_string(r['pitch'],config,r['explicit'],sounding,started)
        r['string'] = s+1
        current[s] = r
        events += note(t,max(t+.001,r['end']),r['pitch'],r['velocity'],s+1,r['token'],r['owner'],
                       'pizz' if kind == 'slide' else kind,float(art.get('position',0)),float(art.get('cents',0)))
        if kind == 'slide':
            target = r['pitch']+float(art['to'])
            assign_string(target,config,s+1)
            endtick = r['score_tick']+float(art['duration_beats'])*bars.beat_length_at(r['score_tick'])
            end = at(r['index'],endtick)
            if end > r['end']+.001:
                raise ValueError('slide must finish before the scored note release; tie/lengthen the source')
            events.append([round(t*48000),2,s,r['owner'],target,0,max(1,round((end-t)*48000)),
                           art.get('shape','ease') == 'ease',0,0,r['token']])
        assignment.append(dict(track=names[r['index']],time_s=t,off_time_s=r['end'],pitch=r['pitch'],
                               string=s+1,kind=kind,token=r['token']))
    for index,tick,string in damps:
        t = at(index,tick)
        candidates = [r for r in rows if r['index'] == index and r['time'] < t and (string is None or string == r['string'])]
        if not candidates:
            raise ValueError('bass damp needs a preceding note')
        for s in set(r['string'] for r in candidates):
            r = max((r for r in candidates if r['string'] == s),key=lambda r:r['time'])
            events.append([round(t*48000),6,s-1,r['owner'],0,0,0,0,0,0,r['token']])
    # Releases precede replacement attacks, pitch ramps follow their source attack.
    events.sort(key=lambda r:(r[0],0 if r[1] == 1 else 2 if r[1] == 2 else 1))
    endtick = max((sum(m.time for m in track) for track in performed.tracks),default=0)
    end = max(tempo.seconds_at(endtick),max((r['end'] for r in rows),default=0))
    return events,end+config['engine']['tail_seconds'],assignment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest',type=Path)
    args = parser.parse_args()
    data = json.loads(args.manifest.read_text())
    options = data['settings'].get('bass_model',{})
    if set(options)-{'preset','output_gain','seed','intonation_cents'}:
        raise ValueError('unknown bass_model option')
    path = preset_path('bass', options.get('preset'))
    c = load_config(path)
    for key in ('output_gain','seed'):
        if key in options:
            c['engine'][key] = options[key]
    if 'intonation_cents' in options:
        c['excitation']['intonation_cents'] = options['intonation_cents']
    if not math.isfinite(c['engine']['output_gain']) or c['engine']['output_gain'] <= 0 or type(c['engine']['seed']) is not int or c['engine']['seed'] < 0 or not 0 <= c['excitation']['intonation_cents'] <= 15:
        raise ValueError('invalid bass_model override')
    performed,score = mido.MidiFile(data['performed_midi']),mido.MidiFile(data['score_midi'])
    indices = [job['index'] for job in data['jobs']]
    events,seconds,assignment = schedule(performed,score,data['settings'],indices,c)
    render([],0.001,len(indices),c)
    start = time.perf_counter()
    audio = render(events,seconds,len(indices),c)
    elapsed = time.perf_counter()-start
    stems = {}
    for owner,job in enumerate(data['jobs']):
        write_wav(Path(job['output']),audio[owner])
        stems[str(job['index'])] = dict(engine='model',model='bass',parameters=str(path),
            parameters_sha256=hashlib.sha256(preset_bytes(path)+json.dumps(options,sort_keys=True).encode()).hexdigest(),overrides=options)
    metrics = dict(stems=stems,audio_seconds=seconds,render_seconds=elapsed,realtime_factor=seconds/elapsed,
                   assignment=assignment,peak=float(np.max(abs(audio),initial=0)))
    args.manifest.with_suffix('.metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(f'Bass ({path.stem}): {seconds:.2f}s in {elapsed:.3f}s ({seconds/elapsed:.1f}x realtime)')


if __name__ == '__main__':
    main()
