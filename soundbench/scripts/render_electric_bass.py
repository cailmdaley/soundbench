#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["mido>=1.3,<2", "numpy>=1.26", "numba>=0.61", "scipy>=1.11"]
# ///
"""Render a shared four-string electric bass from MIDI notes and controllers."""
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
from soundbench.scripts.models.electric_bass import ROOT, load_config, assign_string, render, note, SR
from soundbench.scripts.perform import TempoMap
from soundbench.scripts.render_model import write_wav


def schedule(midi, indices, config):
    tempo = TempoMap(midi)
    actions=[]
    end=0.
    for owner,index in enumerate(indices):
        tick=0
        for order,msg in enumerate(midi.tracks[index]):
            tick+=msg.time
            t=tempo.seconds_at(tick); end=max(end,t)
            actions.append((t,owner,order,msg))
    cc=defaultdict(lambda: defaultdict(int))
    bend=defaultdict(float)
    active=defaultdict(deque)
    current=[None]*4
    events=[]; assignment=[]; token=0
    for t,owner,order,msg in sorted(actions,key=lambda a:(a[0],a[1],a[2])):
        if msg.is_meta:continue
        key=(owner,msg.channel)
        if msg.type=='control_change':
            if msg.control in (20,21,22,23,68):
                if msg.control==23 and msg.value not in range(5):
                    raise ValueError('CC23 is auto=0 or physical string 1..4')
                cc[key][msg.control]=msg.value
            elif msg.control in (120,123):
                for s,old in enumerate(current):
                    if old and old['key']==key:
                        events.append([round(t*SR),5,s,owner,0,0,0,0,0,0,old['token']])
            elif msg.control==64 and msg.value>=64:
                raise ValueError('electric bass uses explicit note lengths, not CC64')
        elif msg.type=='pitchwheel':
            bend[key]=12*msg.pitch/8192
            for s,old in enumerate(current):
                if old and old['key']==key and old['held']:
                    target=old['pitch']+bend[key]
                    assign_string(target,config,s+1)
                    events.append([round(t*SR),2,s,owner,target,0,.010*SR,0,0,0,old['token']])
        elif msg.type=='note_on' and msg.velocity:
            pin=cc[key][23] or None
            pitch=msg.note+bend[key]
            s=assign_string(pitch,config,pin)
            # Legato must be deliberately pinned; automatic string allocation
            # cannot establish which physical string a slur means.
            old=current[s]
            kind=6 if cc[key][20]>=64 else 7 if cc[key][21]>=64 else 0
            if cc[key][68]>=64:
                if pin is None or old is None or old['key']!=key or not old['held'] or pitch==old['pitch']:
                    raise ValueError('CC68 legato needs a held source, pinned string and changed pitch')
                kind=3 if pitch>old['pitch'] else 4
            r=dict(key=key,token=token,pitch=msg.note,held=True,string=s)
            active[key,msg.note].append(r); current[s]=r
            attack=note(t,t+.001,pitch,msg.velocity,s+1,token,kind=kind,hardness=int(cc[key][22]>=64),owner=owner)[0]
            events.append(attack)
            assignment.append(dict(track=midi.tracks[indices[owner]].name,time_s=t,pitch=msg.note,string=s+1,kind=kind,token=token))
            token+=1
        elif msg.type in ('note_off','note_on'):
            queue=active[key,msg.note]
            if queue:
                old=queue.popleft();old['held']=False
                events.append([round(t*SR),1,old['string'],owner,0,0,0,0,0,0,old['token']])
    # Unterminated MIDI notes fail instead of being hidden by a global fade.
    if any(active.values()):
        raise ValueError('electric-bass MIDI has unreleased notes')
    return sorted(events,key=lambda e:e[0]),end+config['engine']['tail_seconds'],assignment


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest',type=Path)
    args=parser.parse_args(); data=json.loads(args.manifest.read_text())
    options=data['settings'].get('electric_bass_model',{})
    if set(options)-{'preset','parameters','output_gain'}:
        raise ValueError('unknown electric_bass_model option')
    preset=options.get('preset','flat')
    if preset not in ('flat','round'):raise ValueError('preset is flat or round')
    path=parameters_path(options.get('parameters',f'instruments/models/electric-bass-{preset}.toml'))
    config=load_config(path)
    if 'output_gain' in options:
        gain=float(options['output_gain'])
        if not math.isfinite(gain) or gain<=0:raise ValueError('positive finite output gain required')
        config['engine']['output_gain']=gain
    midi=mido.MidiFile(data['performed_midi'])
    indices=[j['index'] for j in data['jobs']]
    events,seconds,assignment=schedule(midi,indices,config)
    start=time.perf_counter();audio=render(events,seconds,len(indices),config);elapsed=time.perf_counter()-start
    stems={}
    for owner,job in enumerate(data['jobs']):
        write_wav(Path(job['output']),audio[owner])
        stems[str(job['index'])]=dict(engine='model',model='electric-bass',preset=preset,parameters=str(path),parameters_sha256=hashlib.sha256(path.read_bytes()+json.dumps(options,sort_keys=True).encode()).hexdigest(),overrides=options)
    metrics=dict(stems=stems,audio_seconds=seconds,render_seconds=elapsed,peak=float(np.max(abs(audio),initial=0)),assignment=assignment)
    args.manifest.with_suffix('.metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(f'Electric bass: {seconds:.2f}s in {elapsed:.2f}s')


if __name__=='__main__':main()
