#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["mido>=1.3,<2", "numpy>=1.26", "numba>=0.61", "scipy>=1.11"]
# ///
"""Render machine-voice tracks with scored lyrics and score-addressed sob gestures."""
from __future__ import annotations
import argparse
from collections import defaultdict, deque
import hashlib
import json
from pathlib import Path
import time

import mido
import numpy as np
from soundbench.scripts.instrumentation import matching_tracks
from soundbench.scripts.models import machine_voice, piano
from soundbench.scripts.models.machine_voice import CONFIG, load_config, render, validate_gesture
from soundbench.scripts.models.voice import CONFIG as VOICE_CONFIG, syllable_key
from soundbench.scripts.perform import TempoMap, BarMap
from soundbench.scripts.render_model import notes as track_notes, write_wav
from soundbench.scripts.render_voice import selected_config as selected_voice_config, sung_notes


def selected_config(settings, score=None):
    options=settings.get('machine_voice',{})
    if not isinstance(options,dict):
        raise ValueError('machine_voice must be a table')
    overrides={k:v for k,v in options.items() if k!='preset'}
    # A MIDI key signature provides the piece scale; explicit scale wins.
    if 'scale' not in overrides and score is not None:
        key=next((m.key for track in score.tracks for m in track if m.type=='key_signature'),None)
        if key:
            tonic={'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}[key[0]]
            if len(key)>1 and key[1] in ('#','b'):
                tonic+=1 if key[1]=='#' else -1
            steps=[0,2,3,5,7,8,10] if key.endswith('m') else [0,2,4,5,7,9,11]
            overrides['scale']=[(tonic+s)%12 for s in steps]
    config=load_config(preset=options.get('preset','corrected'),overrides=overrides)
    return config,options


def addressed_gestures(performed,score,settings,indices):
    """An exact scored onset maps to its pitch-matched performed note onset."""
    bars=BarMap(score); tempo=TempoMap(performed)
    all_note_tracks=[i for i,t in enumerate(performed.tracks) if any(m.type=='note_on' and m.velocity for m in t)]
    names={i:t.name for i,t in enumerate(performed.tracks)}
    anchors={}
    for index in indices:
        by_pitch=defaultdict(deque)
        for row in track_notes(score,index):
            by_pitch[row['pitch']].append(row)
        for row in track_notes(performed,index):
            if not by_pitch[row['pitch']]:
                raise ValueError('score/performed voice correspondence differs')
            source=by_pitch[row['pitch']].popleft()
            key=(index,source['tick'])
            if key in anchors:
                raise ValueError('machine voice is monophonic; chord onset found')
            anchors[key]=tempo.seconds_at(row['tick'])
        if any(by_pitch.values()):
            raise ValueError('unmatched scored voice notes')
    result={i:[] for i in indices}; occupied=set()
    for art in settings.get('voice_gesture',[]):
        if not isinstance(art,dict) or not {'track','bar','kind'}<=set(art):
            raise ValueError('voice_gesture needs track, bar, kind')
        matches=matching_tracks(art['track'],all_note_tracks,names)
        if len(matches)!=1 or matches[0] not in indices:
            raise ValueError('voice_gesture needs one machine-voice track')
        if type(art['bar']) is not int or art['bar']<1:
            raise ValueError('gesture bar must be a positive integer')
        beat=art.get('beat',1)
        if type(beat) not in (int,float) or not np.isfinite(beat) or beat<1:
            raise ValueError('gesture beat must be finite and one-based')
        key=(matches[0],bars.position(art['bar'],float(beat)))
        if key in occupied:
            raise ValueError('one explicit voice gesture per track/onset')
        if key not in anchors:
            raise ValueError('voice gesture needs an exact scored note onset')
        occupied.add(key)
        request={k:v for k,v in art.items() if k not in ('track','bar','beat')}
        request['time']=anchors[key]
        validate_gesture(request)
        result[key[0]].append(request)
    return result


def provenance(config,voice_digest,settings):
    """Hash effective parameters, gestures, source configuration and carrier data."""
    inputs={'machine_parameters':hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
            'voice_parameters':voice_digest}
    if config['carrier']=='piano':
        pc=piano.load_config()
        paths=[piano.CONFIG]+[piano.CONFIG.parent/v for k,v in pc['engine'].items() if k.endswith('_data')]
        inputs.update({str(p.relative_to(CONFIG.parents[2])):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    payload=dict(config=config,voice_gesture=settings.get('voice_gesture',[]),inputs=inputs)
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest(),inputs


def passages(rows, rest):
    """Runs of notes separated by rests of at least `rest` seconds."""
    runs=[]
    for row in sorted(rows,key=lambda r:r['time']):
        if not runs or row['time']-max(r['off'] for r in runs[-1])>=rest:
            runs.append([])
        runs[-1].append(row)
    return runs


def render_passages(rows, seconds, config, vc, controls, gestures, lead_in=1.0):
    """Render each passage between long rests on its own, placed on the track's timeline.

    A rest longer than the voice's tail plus the lead-in is silence in a whole-track
    render too, so passages are independent; rendering them apart bounds memory by the
    longest passage rather than the track. Each passage starts from the controller
    values in force at its start, and the vocoder's level and drive normalization
    is per passage. The sob's CC11 headroom is the whole track's, the first phrase
    of every later passage still breathes, and each passage draws its own random streams.
    """
    out=np.zeros(round(seconds*48000)); placed=[]
    tail=vc['engine']['tail_seconds']
    runs=passages(rows,tail+lead_in)
    c=machine_voice.validate(config or load_config())
    headroom=machine_voice.sob_headroom(rows,seconds,c['sob'],gestures,c['seed'],vc['expression']['legato_gap_s'])
    for stream,run in enumerate(runs):
        t0=max(0.,run[0]['time']-lead_in)
        t1=min(seconds,max(r['off'] for r in run)+tail)
        shift=lambda r:{**r,'time':r['time']-t0}
        notes=[{**shift(r),'off':r['off']-t0} for r in run]
        held={}
        for c in sorted(controls,key=lambda c:c['time']):
            if c['time']<=t0:
                held[c['control']]={**c,'time':0.}
        window=list(held.values())+[shift(c) for c in controls if t0<c['time']<=t1]
        asked=[shift(g) for g in gestures if run[0]['time']-1e-7<=g['time']<=run[-1]['time']+1e-7]
        audio,k=render(notes,t1-t0,config,vc,True,window,asked,
                         first=stream==0,headroom=headroom,stream=stream)
        at=round(t0*48000)
        size=min(len(audio),len(out)-at)
        out[at:at+size]+=audio[:size]
        placed+=[{**g,'time':g['time']+t0} for g in (k['gestures'] if k else [])]
    return out,placed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest',type=Path)
    args=parser.parse_args(); data=json.loads(args.manifest.read_text()); settings=data.get('settings',{})
    performed,score=mido.MidiFile(data['performed_midi']),mido.MidiFile(data['score_midi'])
    vc,voice_options,voice_digest=selected_voice_config(settings)
    config,options=selected_config(settings,score)
    if config['carrier']=='audio':
        raise ValueError('audio carrier is available through the Python array API, not sidecars')
    if config['carrier']=='organ':
        raise ValueError('organ carrier is not yet available')
    indices=[job['index'] for job in data['jobs']]
    explicit=addressed_gestures(performed,score,settings,indices)
    tempo=TempoMap(performed)
    seconds=tempo.seconds_at(max(sum(m.time for m in t) for t in performed.tracks))+vc['engine']['tail_seconds']
    digest,inputs=provenance(config,voice_digest,settings)
    stems={}; sung={}; gestures={}; peak=0.; elapsed=0.
    for job in data['jobs']:
        index=job['index']; rows=sung_notes(performed,score,index,voice_options.get('unknown_syllable'))
        # Registration cuts share precisely the plain voice renderer semantics.
        if 'ah_gain_db' in voice_options:
            syllable=None; profile=np.array(voice_options['ah_gain_db'])
            for row in rows:
                if row['syllable'] is not None: syllable=syllable_key(row['syllable'])
                row['source_gain_db']=float(np.interp(row['pitch'],profile[:,0],profile[:,1])) if syllable=='ah' else 0
        tick=0; controls=[]
        for message in performed.tracks[index]:
            tick+=message.time
            if message.type=='control_change' and message.control in (2,11):
                controls.append(dict(time=tempo.seconds_at(tick),control=message.control,value=message.value))
        start=time.perf_counter()
        audio,placed=render_passages(rows,seconds,config,vc,controls,explicit[index])
        elapsed+=time.perf_counter()-start
        write_wav(Path(job['output']),audio,48000)
        peak=max(peak,float(np.max(np.abs(audio),initial=0)))
        stems[str(index)]=dict(engine='model',model='machine-voice',preset=options.get('preset','corrected'),
            parameters=str(CONFIG.relative_to(CONFIG.parents[2])),parameters_sha256=digest,
            overrides=options,effective_parameters=config,voice_overrides=voice_options,inputs_sha256=inputs)
        sung[str(index)]=rows; gestures[str(index)]=placed
    metrics=dict(stems=stems,audio_seconds=seconds,render_seconds=elapsed,
        realtime_factor=seconds*len(indices)/max(elapsed,1e-9),sung=sung,gestures=gestures,peak=peak)
    args.manifest.with_suffix('.metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(f'Machine voice: {len(indices)} track(s), {seconds:.2f} s audio in {elapsed:.3f} s; peak {peak:.3f}')


if __name__=='__main__':
    main()
