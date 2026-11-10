#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["mido>=1.3,<2", "numpy>=1.26", "numba>=0.61", "scipy>=1.11"]
# ///
"""Render mandolin or dobro MIDI with explicit course and gesture controls."""
import argparse
from collections import defaultdict, deque
import hashlib
import json
import math
from pathlib import Path
import time
import mido
import numpy as np
from soundbench.scripts.paths import parameters_path
from soundbench.scripts.instrumentation import matching_tracks
from soundbench.scripts.perform import TempoMap
from soundbench.scripts.models import guitar
from soundbench.scripts.models.folk_strings import load_config, note, tremolo, render, SR, ROOT
from soundbench.scripts.models.guitar_release import finish
from soundbench.scripts.render_model import write_wav, assignment_statistics


def schedule(midi, settings, indices, config, instrument):
    tempo = TempoMap(midi)
    count = len(config['strings']['open_midi'])
    names = {i:midi.tracks[i].name for i in indices}
    pins = {}
    for selector, string in settings.get(f'{instrument}_strings', {}).items():
        matches = matching_tracks(selector, indices, names)
        if not matches or not isinstance(string,int) or not 1 <= string <= count:
            raise ValueError('invalid string/course mapping')
        for index in matches:
            if index in pins:
                raise ValueError('duplicate string mapping')
            pins[index] = string
    options = settings.get(f'{instrument}_model', {})
    bend_range = float(options.get('bend_range',12))
    if not math.isfinite(bend_range) or not 0 < bend_range <= 24:
        raise ValueError('bend_range must be in (0,24] semitones')
    records, controls, duration = [], {}, 0.
    for owner, index in enumerate(indices):
        state = defaultdict(lambda:{20:51,21:114,22:30,23:0,24:0,25:0})
        pending = defaultdict(deque)
        bends = defaultdict(list)
        tick = 0
        for order,msg in enumerate(midi.tracks[index]):
            tick += msg.time
            t = tempo.seconds_at(tick)
            duration = max(duration,t)
            if msg.type == 'control_change' and msg.control in (20,21,22,23,24,25):
                state[msg.channel][msg.control] = msg.value
            elif msg.type == 'pitchwheel':
                bends[msg.channel].append((t,msg.pitch/8192*bend_range))
            elif msg.type == 'note_on' and msg.velocity:
                r = dict(on=t,off=None,pitch=msg.note,velocity=msg.velocity,owner=owner,index=index,
                         channel=msg.channel,token=index*1000000+order,state=dict(state[msg.channel]),explicit=pins.get(index))
                records.append(r)
                pending[msg.channel,msg.note].append(r)
            elif msg.type in ('note_on','note_off') and pending[msg.channel,msg.note]:
                pending[msg.channel,msg.note].popleft()['off'] = t
        if any(pending.values()):
            raise ValueError('every mandolin/dobro MIDI note needs an explicit note-off')
        controls[index] = bends
    current = [None]*count
    events, assignment = [], []
    for r in sorted(records,key=lambda r:(r['on'],-r['pitch'],r['owner'],r['token'])):
        t, off, state = r['on'],r['off'],r['state']
        if off <= t:
            raise ValueError('notes need positive duration')
        sounding = [old is not None and old['off'] > t for old in current]
        started = [old['on'] if old else -math.inf for old in current]
        s = guitar.assign_string(r['pitch'],config,r['explicit'],sounding,started)
        kind = state[24]
        if kind not in (0,1,2):
            raise ValueError('CC24 is 0 pluck, 1 hammer, 2 pull')
        old = current[s]
        if kind and (r['explicit'] is None or old is None or old['index'] != r['index'] or old['off'] < t or r['pitch'] == old['pitch'] or ((r['pitch']>old['pitch']) != (kind==1))):
            raise ValueError('hammer/pull needs overlapping source and pinned string with correct direction')
        mode = state[23]
        if instrument == 'dobro' and mode:
            raise ValueError('CC23 tremolo/chop belongs to mandolin')
        if mode not in (0,64,127):
            raise ValueError('CC23 is 0 ordinary, 64 tremolo, 127 chop')
        if mode and kind:
            raise ValueError('tremolo/chop cannot combine with hammer/pull')
        if mode == 64:
            generated = tremolo(t,off,r['pitch'],r['velocity'],s+1,100000000+r['token']*4096,
                rate=4+20*state[20]/127,evenness=state[21]/127,accent=1+.5*state[22]/127,
                seed=config['engine']['seed'],owner=r['owner'])
        else:
            generated = note(t,off,r['pitch'],r['velocity'],s+1,r['token'],owner=r['owner'],kind=3 if kind==1 else 4 if kind==2 else 0,direction='up' if state[25]>=64 else 'down')
            if mode == 127:
                generated.append([round((t+.02)*SR),5,s,r['owner'],0,0,0,0,0,0,r['token']])
        bends = controls[r['index']][r['channel']]
        held = next((p for at,p in reversed(bends) if at <= t),0.)
        # Split one continuous curve at repeat picks without restarting its pitch.
        curve = [(t,held)]+[(at,p) for at,p in bends if t < at <= off]
        curve_times,curve_pitches = np.array(curve).T
        for a in [ev for ev in generated if ev[1] in (0,3,4)]:
            start = a[0]/SR
            stop = min(off,min((ev[0]/SR for ev in generated if ev[1]==0 and ev[0]>a[0]),default=off))
            offset = float(np.interp(start,curve_times,curve_pitches))
            a[4] += offset
            guitar.assign_string(a[4],config,s+1)
            points = [(start,offset)]+[(at,p) for at,p in curve if start < at < stop]
            points.append((stop,float(np.interp(stop,curve_times,curve_pitches))))
            for (at,p),(end,q) in zip(points,points[1:]):
                guitar.assign_string(r['pitch']+q,config,s+1)
                if end > at and q != p:
                    generated.append([round(at*SR),2,s,r['owner'],r['pitch']+q,0,max(1,round((end-at)*SR)),0,0,0,a[10]])
        events += generated
        current[s] = r
        assignment.append(dict(track=names[r['index']],time_s=t,off_time_s=off,pitch=r['pitch'],string=s+1,kind=mode or kind,token=r['token']))
    seconds = duration+config['engine']['tail_seconds']
    events = finish(events,duration+config['engine']['tail_seconds']/2)
    events.sort(key=lambda ev:(ev[0],0 if ev[1]==1 else 1))
    return events,seconds,assignment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest',type=Path)
    args = parser.parse_args()
    data = json.loads(args.manifest.read_text())
    instrument = next((i for i in ('mandolin','dobro') if args.manifest.stem.endswith('.'+i)),None)
    instrument = data.get('instrument',instrument)
    if instrument not in ('mandolin','dobro'):
        raise ValueError('manifest suffix must be .mandolin.json or .dobro.json')
    options = data['settings'].get(f'{instrument}_model',{})
    if set(options)-{'parameters','output_gain','bend_range','vibrato'}:
        raise ValueError('unknown model options')
    path = parameters_path(options.get('parameters',f'instruments/models/{instrument}.toml'))
    c = load_config(instrument,path)
    if 'output_gain' in options:
        gain = float(options['output_gain'])
        if not math.isfinite(gain) or gain <= 0:
            raise ValueError('output_gain must be positive and finite')
        c['engine']['output_gain'] = gain
    midi = mido.MidiFile(data['performed_midi'])
    indices = [job['index'] for job in data['jobs']]
    events,seconds,assignment = schedule(midi,data['settings'],indices,c,instrument)
    render([],.001,len(indices),c,instrument,vibrato=options.get('vibrato'))
    start = time.perf_counter()
    audio = render(events,seconds,len(indices),c,instrument,vibrato=options.get('vibrato'))
    elapsed = time.perf_counter()-start
    stems = {}
    for owner,job in enumerate(data['jobs']):
        write_wav(Path(job['output']),audio[owner])
        stems[str(job['index'])] = dict(engine='model',model=instrument,preset=instrument,parameters=str(path),
            parameters_sha256=hashlib.sha256(path.read_bytes()+json.dumps(options,sort_keys=True).encode()).hexdigest(),overrides=options)
    metrics = dict(stems=stems,audio_seconds=seconds,render_seconds=elapsed,assignment=assignment,
        peak=float(np.max(abs(audio),initial=0)),assignment_statistics=assignment_statistics(events))
    args.manifest.with_suffix('.metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(f'{instrument}: {seconds:.2f}s in {elapsed:.3f}s')


if __name__ == '__main__':
    main()
