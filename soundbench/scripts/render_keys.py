#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["mido>=1.3,<2", "numpy", "numba>=0.61", "scipy"]
# ///
"""Render organ, pipe-organ or electric-piano MIDI to stereo stems and attributed metrics."""
from collections import defaultdict, deque
from pathlib import Path
import argparse
import hashlib
import json
import time
import mido
import numpy as np
from scipy.io.wavfile import write
from soundbench.scripts.paths import preset_bytes, preset_label, preset_path
from soundbench.scripts.models import organ, electric_piano, pipe_organ
from soundbench.scripts.models.keys_common import SR
from soundbench.scripts.perform import TempoMap

ENGINES={'organ':organ,'electric-piano':electric_piano,'pipe-organ':pipe_organ}
CONTROLS={'organ':{1,11,68,80,81,82,83,*range(12,21)},
          'electric-piano':{11,64,92,93,94},
          'pipe-organ':{11}}


def track_events(midi,index,model):
    tempo=TempoMap(midi); active=defaultdict(deque); notes=[]; controls=[]; tick=0
    for msg in midi.tracks[index]:
        tick+=msg.time; t=tempo.seconds_at(tick)
        if msg.type=='note_on' and msg.velocity:
            row={'time':t,'off':None,'pitch':msg.note,'velocity':msg.velocity,'channel':msg.channel}
            notes.append(row); active[msg.channel,msg.note].append(row)
        elif msg.type in ('note_off','note_on'):
            if active[msg.channel,msg.note]: active[msg.channel,msg.note].popleft()['off']=t
        elif msg.type=='control_change' and msg.control in CONTROLS[model]:
            controls.append((t,msg.control,msg.value,msg.channel))
    if any(active.values()): raise ValueError('keyboard MIDI has unterminated notes')
    return notes,controls


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('manifest',type=Path)
    a=p.parse_args(); data=json.loads(a.manifest.read_text())
    model=data.get('model',a.manifest.stem.rsplit('.',1)[-1])
    if model not in ENGINES: raise ValueError('manifest needs model organ, pipe-organ or electric-piano')
    engine=ENGINES[model]; options=data.get('settings',{}).get(model.replace('-','_')+'_model',{})
    allowed={'preset','output_gain'}|({'registration','tracks'} if model=='pipe-organ' else set())
    if set(options)-allowed: raise ValueError('unknown keyboard model setting')
    path=preset_path(model,options.get('preset'))
    cfg=engine.load_config(path)
    gain=float(options.get('output_gain',cfg['engine']['output_gain']))
    if not np.isfinite(gain) or not 0<gain<=4: raise ValueError('invalid keyboard output gain')
    cfg['engine']['output_gain']=gain
    registrations={k.strip().casefold():v for k,v in options.get('tracks',{}).items()}
    for name in [options.get('registration'),*registrations.values()]:
        if name is not None and name not in cfg.get('registration',{}): raise ValueError(f'unknown registration {name!r}')
    midi=mido.MidiFile(data['performed_midi']); tempo=TempoMap(midi)
    seconds=tempo.seconds_at(max(sum(m.time for m in tr) for tr in midi.tracks))+cfg['engine']['tail_seconds']
    digest=hashlib.sha256(preset_bytes(path)+json.dumps(options,sort_keys=True).encode()).hexdigest()
    stems={}; peak=0.; count=0; start=time.perf_counter()
    for job in data['jobs']:
        notes,controls=track_events(midi,job['index'],model)
        job_cfg=cfg
        if model=='pipe-organ':
            track=midi.tracks[job['index']].name.strip().casefold()
            job_cfg={**cfg,'active':registrations.get(track,options.get('registration',cfg['engine']['registration']))}
        audio=np.zeros((round(seconds*SR),2))
        for ch in sorted({x['channel'] for x in notes}):
            audio+=engine.render([x for x in notes if x['channel']==ch],seconds,job_cfg,
                                 [e[:3] for e in controls if e[3]==ch])
        this_peak=float(np.max(abs(audio),initial=0)); peak=max(peak,this_peak)
        if not np.all(np.isfinite(audio)) or this_peak>=1:
            raise ValueError('keyboard stem clips; reduce output_gain')
        output=Path(job['output']); output.parent.mkdir(parents=True,exist_ok=True)
        write(output,SR,np.rint(audio*2147483647).astype('<i4'))
        count+=len(notes)
        stems[str(job['index'])]={'engine':'model','model':model,**({'registration':job_cfg['active']} if model=='pipe-organ' else {}),'parameters':preset_label(path),
                                  'parameters_sha256':digest}
    metrics={'stems':stems,'notes':count,'peak':peak,'audio_seconds':seconds,'render_seconds':time.perf_counter()-start}
    a.manifest.with_suffix('.metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(f'{model}: {count} notes, {seconds:.2f}s; peak {peak:.3f}')

if __name__=='__main__': main()
