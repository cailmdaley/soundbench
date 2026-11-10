#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["mido>=1.3,<2", "numpy", "numba>=0.61", "scipy"]
# ///
"""Audio and routing invariants for both physical keyboards."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import mido
ROOT=Path(__file__).resolve().parents[1] / "soundbench"
sys.path.insert(0,str(ROOT/'scripts'))
from soundbench.scripts.models import organ, electric_piano as ep
from soundbench.scripts.models.keys_common import SR
from soundbench.scripts.render_keys import track_events
from soundbench.scripts.instrumentation import resolve_instruments


def note(p=60,v=80,t=0,off=.3):
    return {'pitch':p,'velocity':v,'time':t,'off':off}


def pitch_peak(y,hz):
    y=y[4800:38400].mean(axis=1); size=524288
    spec=abs(np.fft.rfft(y*np.hanning(len(y)),size)); freq=np.fft.rfftfreq(size,1/SR)
    ids=np.flatnonzero(abs(freq-hz)<hz*.03); i=ids[np.argmax(spec[ids])]
    a,b,c=np.log(spec[i-1:i+2]+1e-30); delta=.5*(a-c)/(a-2*b+c)
    return (i+delta)*SR/size


class KeysTests(unittest.TestCase):
    def test_wheel_gears_and_foldback(self):
        self.assertEqual(organ.wheel_frequency(69),440)
        self.assertAlmostEqual(organ.wheel_frequency(64),40*69/67*8)
        self.assertEqual(organ.wheel_frequency(126),organ.wheel_frequency(114))
        errors=[1200*np.log2(organ.wheel_frequency(p)/(440*2**((p-69)/12))) for p in range(36,97)]
        self.assertLess(max(abs(np.array(errors))),3)
        self.assertGreater(max(abs(np.array(errors))),.5)

    def test_audio_tuning(self):
        for engine in (organ,ep):
            cfg=engine.load_config()
            if engine is organ:
                cfg['leslie']['enabled']=False; cfg['drawbars']['registration']='008000000'
                cfg['contacts']['click_gain']=0
            errors=[]
            for p in (36,48,60,69,84,96):
                target=organ.wheel_frequency(p) if engine is organ else 440*2**((p-69)/12)
                y=engine.render([note(p,off=.9)],1,cfg)
                cents=1200*np.log2(pitch_peak(y,target)/target)
                self.assertLess(abs(cents),3,(engine.__name__,p,cents)); errors.append(cents)
            print(engine.__name__,'audio tuning cents:',errors)

    def test_release_seed_and_polyphony(self):
        for engine in (organ,ep):
            cfg=engine.load_config(); rows=[note(p) for p in (36,48,55,60,64,67,72,76)]
            a=engine.render(rows,2,cfg); b=engine.render(rows,2,cfg)
            self.assertTrue(np.array_equal(a,b)); self.assertTrue(np.isfinite(a).all())
            self.assertLess(abs(a).max(),1)
            self.assertGreater(np.std(a[:14400]),1e-4)
            self.assertLess(np.std(a[72000:]),np.std(a[:14400])*1e-4)
            cfg['engine']['seed']+=1
            self.assertFalse(np.array_equal(a,engine.render(rows,2,cfg)))

    def test_single_trigger(self):
        rows=[note(t=0,off=.5),note(64,t=0,off=.4),note(67,t=.2,off=.7),note(t=.8,off=1)]
        self.assertEqual([t for t,_ in organ.percussion_triggers(rows)],[0,.8])

    def test_rotor_acceleration(self):
        target=np.full(SR*4,127.)
        rates,phase=organ.rotor(target,.8,6.7,.8,SR)
        self.assertLess(rates[0],.801)
        self.assertTrue(np.all(np.diff(rates)>0))
        self.assertAlmostEqual(rates[int(.8*SR)],.8+(6.7-.8)*(1-np.exp(-1)),places=3)
        self.assertGreater(rates[-1],6.6)

    def test_organ_velocity_independent(self):
        cfg=organ.load_config()
        self.assertTrue(np.array_equal(organ.render([note(v=20)],1,cfg),organ.render([note(v=127)],1,cfg)))

    def test_velocity_continuity(self):
        cfg=ep.load_config(); rms=[]
        for v in range(1,128):
            y=ep.render([note(v=v,off=.1)],.2,cfg); rms.append(np.std(y[:4800]))
        jumps=np.diff(rms)
        self.assertLess(np.max(abs(jumps)),.035)
        self.assertGreater(rms[-1],rms[31]*2)
        self.assertTrue(np.isfinite(rms).all())
        print('Electric piano velocity max adjacent RMS change:',max(abs(jumps)))

    def test_pedal_channel_and_modes(self):
        cfg=ep.load_config(); a=ep.render([note()],1,cfg)
        b=ep.render([note()],1,cfg,[(0,64,127),(.7,64,0)])
        self.assertGreater(np.std(b[24000:28800]),np.std(a[24000:28800])*3)
        stereo=ep.render([note(off=.8)],1,cfg,[(0,92,127)])
        self.assertGreater(np.std(stereo[:,0]-stereo[:,1]),1e-3)
        self.assertEqual(ep.pedal_release(note(),[(0,64,127),(.8,64,0)],1),.8)

    def test_percussion_releases_each_key(self):
        cfg=organ.load_config(); cfg['drawbars']['registration']='000000000'
        cfg['leslie']['enabled']=False; cfg['contacts']['click_gain']=0
        cfg['percussion'].update(enabled=True,harmonic=2,fast=False,soft=False)
        y=organ.render([note(60,off=.1),note(64,off=.8)],1,cfg).mean(axis=1)[19200:33600]
        t=np.arange(len(y))/SR; win=np.hanning(len(y))
        amps=[abs(np.sum(y*win*np.exp(-2j*np.pi*organ.wheel_frequency(p)*t))) for p in (72,76)]
        self.assertLess(amps[0],amps[1]*.001)

    def test_scanner_modes_and_empty_silence(self):
        cfg=organ.load_config(); cfg['leslie']['enabled']=False
        renders=[organ.render([note(off=.8)],1,cfg,[(0,1,round(i*127/6))]) for i in range(7)]
        for i in range(1,7): self.assertGreater(np.std(renders[i]-renders[0]),1e-4)
        for engine in (organ,ep): self.assertTrue(np.all(engine.render([],1)==0))

    def test_pickup_brightness_depends_on_velocity(self):
        cfg=ep.load_config()
        ratios=[]
        for v in (40,120):
            y=ep.render([note(60,v,off=.5)],.6,cfg).mean(axis=1)[4800:19200]
            t=np.arange(len(y))/SR; f=440*2**((60-69)/12); win=np.hanning(len(y))
            amps=[abs(np.sum(y*win*np.exp(-2j*np.pi*f*k*t))) for k in (1,2,3)]
            ratios.append((amps[1]+amps[2])/amps[0])
        self.assertGreater(ratios[1],ratios[0]*2)

    def test_evidence_and_probe_gates(self):
        metrics=json.loads((ROOT/'references/electric-piano/fit.json').read_text())
        self.assertLess(metrics['feature_loss_after'],metrics['feature_loss_before'])
        self.assertEqual([n['midi_measured_octave'] for n in metrics['notes']],[43,55])
        for n in metrics['notes']:
            for label in ('before','after'):
                levels=n[label]['loudness']
                self.assertLess(abs(levels['reference_lufs']-levels['model_lufs_after']),.5)
        for instrument in ('organ','electric-piano'):
            path=ROOT/f'build/{instrument}-probe/metrics.json'
            if not path.exists(): continue
            probes=json.loads(path.read_text())['probes']
            for row in probes:
                self.assertLess(row['peak'],1)
                self.assertLess(row['final_second_rms'],1e-4,row['name'])
                self.assertTrue((path.parent/(row['name']+'.mp3')).exists())
                self.assertTrue((path.parent/(row['name']+'.spec.png')).exists())
            levels={r['name']:r['preview_lufs'] for r in probes}
            pair=('pickup-rounded','pickup-bark') if instrument=='electric-piano' else ('gospel-clean','gospel-driven')
            self.assertLess(abs(levels[pair[0]]-levels[pair[1]]),.5)

    def test_invalid(self):
        for engine in (organ,ep):
            for row in (note(off=None),note(p=5),note(v=0),note(t=.4,off=.2)):
                with self.assertRaises((ValueError,TypeError)): engine.render([row],1)

    def test_midi_and_routing(self):
        midi=mido.MidiFile(); tr=mido.MidiTrack(); midi.tracks.append(tr)
        tr.extend([mido.MetaMessage('track_name',name='keys'),mido.Message('note_on',note=60,velocity=80),
                   mido.Message('note_on',note=60,velocity=90,time=20),mido.Message('note_off',note=60,time=40),
                   mido.Message('note_off',note=60,time=40)])
        for model in ('organ','electric-piano'):
            binding=resolve_instruments(midi,{'instruments':{'keys':{'model':model}}})[0]
            self.assertEqual(binding.model,model)
            rows,_=track_events(midi,0,model); self.assertLess(rows[0]['off'],rows[1]['off'])
        tr.append(mido.Message('note_on',note=61,velocity=50))
        with self.assertRaises(ValueError): track_events(midi,0,'organ')

    def test_preset_stability(self):
        rows=[note(p,127,off=.5) for p in (36,48,55,60,64,67,72,76)]
        for engine,name in ((organ,'organ'),(ep,'electric-piano')):
            for path in (ROOT/'presets'/name).glob('*.toml'):
                cfg=engine.load_config(path); y=engine.render(rows,1.5,cfg)
                self.assertTrue(np.isfinite(y).all(),path.name)
                self.assertLess(abs(y).max(),1,path.name)
                self.assertLess(np.std(y[-4800:]),1e-4,path.name)

    def test_channel_pedal_isolation(self):
        midi=mido.MidiFile(); tr=mido.MidiTrack(); midi.tracks.append(tr)
        tr.extend([mido.Message('control_change',channel=0,control=64,value=127),
                   mido.Message('note_on',channel=0,note=60,velocity=80),
                   mido.Message('note_on',channel=1,note=64,velocity=80),
                   mido.Message('note_off',channel=0,note=60,time=96),
                   mido.Message('note_off',channel=1,note=64),
                   mido.Message('control_change',channel=0,control=64,value=0,time=384)])
        notes,controls=track_events(midi,0,'electric-piano')
        releases=[]
        for row in notes:
            channel=row['channel']
            releases.append(ep.pedal_release(row,[e[:3] for e in controls if e[3]==channel],1))
        self.assertEqual(releases,[.5,.1])

    def test_manifest_render(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'build') as td:
            td=Path(td); midi=mido.MidiFile(); tr=mido.MidiTrack(); midi.tracks.append(tr)
            tr.extend([mido.Message('note_on',note=60,velocity=80),mido.Message('note_off',note=60,time=480)])
            midi.save(td/'notes.mid')
            for model in ('organ','electric-piano'):
                manifest=td/f'test.{model}.json'
                manifest.write_text(json.dumps({'performed_midi':str(td/'notes.mid'),'jobs':[{'index':0,'output':str(td/f'{model}.wav')}]}))
                proc=subprocess.run([sys.executable,str(ROOT/'scripts/render_keys.py'),str(manifest)],capture_output=True,text=True)
                self.assertEqual(proc.returncode,0,proc.stderr)
                metrics=json.loads(manifest.with_suffix('.metrics.json').read_text())
                self.assertEqual(metrics['stems']['0']['model'],model)
                self.assertLess(metrics['peak'],1)

if __name__=='__main__':
    (ROOT/'build').mkdir(exist_ok=True)
    unittest.main()
