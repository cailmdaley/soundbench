#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["mido>=1.3,<2", "numpy>=1.26", "numba>=0.61", "scipy>=1.11"]
# ///
"""Render chanter and thumb Voices into one shared physical-body stem."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import mido
import numpy as np
from soundbench.scripts.paths import preset_bytes, preset_label, preset_path
from soundbench.scripts.models.skinpipe import load_config, validate, render
from soundbench.scripts.render_winds import midi_controls, write_wav
from soundbench.scripts.instrumentation import aliases
from soundbench.scripts.perform import TempoMap


def selected_config(settings):
    options=settings.get('skinpipe_model',{})
    if not isinstance(options,dict) or set(options)-{'preset','output_gain','courses'}:
        raise ValueError('skinpipe_model accepts preset, output_gain and courses only')
    path=preset_path('skinpipe',options.get('preset'))
    config=load_config(path)
    for key in ('output_gain','courses'):
        if key in options: config[key]=options[key]
    validate(config)
    digest=hashlib.sha256(preset_bytes(path)+json.dumps(options,sort_keys=True).encode()).hexdigest()
    return config,path,digest,options


def groups(midi,jobs):
    thumbs=[j for j in jobs if 'skinthumb' in aliases(midi.tracks[j['index']].name)]
    chanters=[j for j in jobs if j not in thumbs]
    if len({midi.tracks[j['index']].name for j in chanters}) != len(chanters):
        raise ValueError('skinpipe chanter track identities must be unique')
    if thumbs and len(chanters)!=1:
        raise ValueError('skinthumb needs exactly one skinpipe chanter binding')
    if len(thumbs)>1: raise ValueError('one skinthumb Voice per skinpipe')
    return [(j,thumbs if len(chanters)==1 else []) for j in chanters]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest',type=Path)
    args=parser.parse_args()
    data=json.loads(args.manifest.read_text())
    c,path,digest,options=selected_config(data.get('settings',{}))
    midi=mido.MidiFile(data['performed_midi'])
    tempo=TempoMap(midi)
    seconds=tempo.seconds_at(max(sum(m.time for m in t) for t in midi.tracks))+c['tail_seconds']
    stems,schedule={},{}
    start=time.perf_counter();peak=0.
    for job,thumbs in groups(midi,data['jobs']):
        index=job['index']
        rows,controls=midi_controls(midi,index)
        plucks=[]
        for thumb in thumbs:
            r,_=midi_controls(midi,thumb['index'])
            plucks.extend(r)
            stems[str(thumb['index'])]=dict(merged_into=index,engine='model',model='skinpipe')
        audio=render(rows,seconds,c,controls,plucks,track_id=midi.tracks[index].name)
        write_wav(Path(job['output']),audio,c['sample_rate'])
        peak=max(peak,float(np.max(abs(audio),initial=0)))
        stems[str(index)]=dict(engine='model',model='skinpipe',parameters=preset_label(path),
            parameters_sha256=digest,overrides=options,track_indices=[index]+[j['index'] for j in thumbs],
            track_names=[midi.tracks[i].name for i in [index]+[j['index'] for j in thumbs]])
        schedule[str(index)]=dict(notes=rows,controls=controls,plucks=plucks)
    elapsed=time.perf_counter()-start
    metrics=dict(stems=stems,audio_seconds=seconds,render_seconds=elapsed,peak=peak,schedule=schedule,
                 includes_compile_or_cache=True)
    args.manifest.with_suffix('.metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(f'skinpipe: {len(schedule)} shared-body stem(s), {seconds:.2f}s, peak {peak:.3f}; render/cache {elapsed:.3f}s')

if __name__=='__main__': main()
