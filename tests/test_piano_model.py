#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["mido>=1.3,<2", "numpy", "numba>=0.61", "scipy", "soundfile"]
# ///
"""Audio-domain modal-piano tuning, stability, MIDI and controller invariants."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import subprocess
import wave
import numpy as np
import mido

ROOT=Path(__file__).resolve().parents[1] / "soundbench"
sys.path.insert(0,str(ROOT/'scripts'))
from soundbench.scripts.models.piano import coefficients, hammer_force, load_config, render
from soundbench.scripts.render_piano import track_events
from soundbench.scripts.audio_pitch import pitch_cents
from soundbench.scripts.instrumentation import resolve_instruments


def note(p=60,v=61,t=0,off=.25):
    return {'pitch':p,'velocity':v,'time':t,'off':off}


class PianoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=load_config()

    def test_audio_tuning(self):
        errors=[]
        tonal=copy.deepcopy(self.cfg)
        tonal['body_settings']['gain']=0
        for p in (36,42,48,54,60,66,72,78,84,90,96):
            y=render([note(p,off=.9)],1,tonal).mean(axis=1)
            cents=pitch_cents(y[4800:38400],p)
            errors.append(abs(cents))
            self.assertLess(abs(cents),3,(p,cents))
        print('Piano tuning max cents',max(errors))

    def test_full_audio_spectral_tuning(self):
        errors=[]
        for p in (36,48,60,72,84,96):
            y=render([note(p,off=1.3)],1.5,self.cfg).mean(axis=1)[14400:52800]
            size=524288; spectrum=abs(np.fft.rfft(y*np.hanning(len(y)),size))
            hz=440*2**((p-69)/12); f=np.fft.rfftfreq(size,1/48000)
            candidates=np.flatnonzero(abs(f-hz)<hz*.02)
            i=candidates[np.argmax(spectrum[candidates])]
            a,b,c=np.log(spectrum[i-1:i+2]+1e-20)
            delta=.5*(a-c)/(a-2*b+c)
            cents=1200*np.log2((i+delta)*48000/size/hz)
            errors.append(abs(cents)); self.assertLess(abs(cents),3,(p,cents))
        print('Full radiation spectral tuning max cents:',max(errors))

    def test_bridge_passivity_and_reciprocity(self):
        from soundbench.scripts.models.piano_bridge import bridge_matrix, coupled_poles, initial_state
        from scipy.linalg import expm
        matrix=bridge_matrix(np.array([-.7,0,.7]),.08,3+1j*.2)
        self.assertTrue(np.allclose(matrix,matrix.T))
        rng=np.random.default_rng(42)
        state=rng.normal(size=3)+1j*rng.normal(size=3)
        self.assertLess(np.linalg.norm(expm(matrix*.1)@state),np.linalg.norm(state))
        poles,info=coupled_poles(261.6,3,.6,.2,2)
        self.assertTrue(np.all(poles.real<0))
        residue=np.array([.2,.4,.1],complex)
        state=initial_state(info,residue)
        direct=info['radiation']@np.linalg.solve(info['vectors'],expm(bridge_matrix(info['offsets'],info['intrinsic'],info['admittance']+1j*info['reactance'])*.4)@state)
        self.assertAlmostEqual(abs(direct-np.sum(residue*np.exp(poles*.4))),0,places=10)
        self.assertGreater(-poles[info['fast']].real,-poles[info['slow']].real*3)

    def test_residual_removes_stretched_treble_and_retains_thump(self):
        from soundbench.scripts.models.piano_refine import sinusoidal_residual
        sr=24000; t=np.arange(sr*2)/sr
        y=(.1*np.sin(2*np.pi*2121.8*t)+.01*np.sin(2*np.pi*55*t))[:,None]
        residual,_=sinusoidal_residual(np.repeat(y,2,axis=1),sr,2121.8/np.sqrt(1.0001),.0001)
        def projection(x,hz):
            return abs(np.mean(x[12000:36000,0]*np.exp(-2j*np.pi*hz*t[12000:36000])))
        self.assertLess(projection(residual,2121.8),projection(y,2121.8)*.05)
        self.assertGreater(projection(residual,55),projection(y,55)*.9)
        fitted=json.loads((ROOT/'references/piano/fit.json').read_text())['tuning']['96']
        self.assertLess(abs(fitted['f0']*np.sqrt(1+fitted['B'])-fitted['frequencies'][0]),.3)

    def test_body_carrier_exclusions_do_not_leave_horizontal_tails(self):
        from soundbench.scripts.models.piano import body_audio
        cfg=copy.deepcopy(self.cfg)
        for profile in cfg['body']['profiles']:
            values=np.asarray(profile['rms'])
            values[:,:,np.array(cfg['body']['times'])>.2]=0
            profile['rms']=values.tolist()
            profile['knock']=[[70,.03,0,0] for _ in range(16)]
        frequencies=coefficients(60,61,cfg)[0]
        y=body_audio(60,61,3,cfg,0,frequencies)
        self.assertLess(np.std(y[48000:]),np.std(y[:9600])*.001)

    def test_body_attack_does_not_wrap_into_the_tail(self):
        from soundbench.scripts.models.piano import body_audio
        cfg=copy.deepcopy(self.cfg)
        for profile in cfg['body']['profiles']:
            values=np.full_like(np.asarray(profile['rms']),1e-5)
            values[:,:,np.array(cfg['body']['times'])<=.2]=.01
            profile['rms']=values.tolist()
            profile['knock']=[[70,.03,0,0] for _ in range(16)]
        y=body_audio(60,61,3,cfg,0,coefficients(60,61,cfg)[0])
        self.assertLess(np.std(y[120000:139200]),np.std(y[72000:96000])*4)

    def test_numerical_and_headroom(self):
        notes=[note(p,80,off=.5) for p in (36,48,55,60,64,67,72)]
        y=render(notes,2,self.cfg)
        self.assertTrue(np.isfinite(y).all()); self.assertLess(abs(y).max(),1)
        self.assertGreater(np.std(y),1e-4)

    def test_determinism(self):
        cfg=copy.deepcopy(self.cfg); cfg['engine']['seed']=42
        y=render([note()],1,cfg)
        self.assertTrue(np.array_equal(y,render([note()],1,cfg)))
        cfg['engine']['seed']=43
        self.assertFalse(np.array_equal(y,render([note()],1,cfg)))

    def test_release_and_half_pedal(self):
        levels=[]
        for value in (0,32,64,96,127):
            y=render([note()],2,self.cfg,[(0,64,value),(1.6,64,0)])
            levels.append(np.std(y[24000:48000]))
            self.assertLess(np.std(y[-4800:]),np.std(y[24000:48000])*.1+1e-8)
        self.assertTrue(all(a<b for a,b in zip(levels,levels[1:])),levels)
        self.assertGreater(levels[-1],levels[0]*10)

    def test_soft_pedal(self):
        plain=render([note()],1,self.cfg)
        soft=render([note()],1,self.cfg,[(0,67,127)])
        self.assertLess(np.std(soft),np.std(plain)*.9)

    def test_all_keys_and_coefficients(self):
        for p in range(21,109):
            f,a,b,r,s,stereo,B=coefficients(p,61,self.cfg)
            self.assertTrue(np.all(np.diff(f)>0))
            self.assertTrue(np.all(r>0)); self.assertTrue(np.all(s>0))
            self.assertTrue(np.all(np.isfinite(a))); self.assertGreaterEqual(B,0)

    def test_velocity_continuity(self):
        values=[]
        for v in range(1,128):
            f,a,b,*_=coefficients(60,v,self.cfg)
            values.append(np.sqrt(np.sum((a+b)**2)))
        jumps=20*np.log10(np.array(values[1:])/values[:-1])
        self.assertLess(abs(jumps[29:]).max(),2.0)
        print('Max adjacent velocity coefficient RMS step (v>=30):',abs(jumps[29:]).max())

    def test_rendered_velocity_ramp(self):
        rms=[]
        for v in range(1,128):
            y=render([note(v=v,off=.2)],.4,self.cfg)
            rms.append(np.std(y[960:7200]))
        step=20*np.log10(np.array(rms[1:])/rms[:-1])
        self.assertLess(abs(step[29:]).max(),2)
        print('Max rendered velocity RMS step (v>=30):',abs(step[29:]).max())

    def test_hammer_brightness_and_impulse(self):
        widths=[]
        for v in (30,61,100):
            force=hammer_force(v,self.cfg)
            self.assertTrue(np.isfinite(force).all()); self.assertTrue(np.all(force>=0))
            self.assertAlmostEqual(force.sum(),1)
            widths.append(len(force))
        self.assertGreater(widths[0],widths[-1])

    def test_midi_requires_release(self):
        midi=mido.MidiFile(); midi.tracks.append(mido.MidiTrack([mido.Message('note_on',note=60,velocity=61)]))
        with self.assertRaisesRegex(ValueError,'unterminated'): track_events(midi,0)

    def test_midi_off_is_not_pedal_extended(self):
        midi=mido.MidiFile(ticks_per_beat=1000)
        midi.tracks.append(mido.MidiTrack([mido.Message('control_change',control=64,value=64),mido.Message('note_on',note=60,velocity=61),mido.Message('note_off',note=60,time=500),mido.Message('control_change',control=64,value=0,time=1000)]))
        rows,cc=track_events(midi,0)
        self.assertAlmostEqual(rows[0]['off'],.25); self.assertEqual(cc[0][2],64)

    def test_invalid_inputs(self):
        for n in (note(20),note(109),note(v=0),note(off=-1),note(t=float('nan'))):
            with self.assertRaises(ValueError): render([n],1,self.cfg)

    def test_routing_and_resonance(self):
        midi=mido.MidiFile(); midi.tracks.append(mido.MidiTrack([mido.MetaMessage('track_name',name='piano'),mido.Message('note_on',note=60,velocity=61),mido.Message('note_off',note=60,time=480)]))
        binding=resolve_instruments(midi,{'instruments':{'piano':{'model':'piano'}}})[0]
        self.assertEqual(binding.instrument,'grand'); self.assertEqual(binding.model,'piano')
        plain=copy.deepcopy(self.cfg); plain['resonance']['gain']=0
        cc=[(0,64,127),(1.5,64,0)]
        y=render([note()],2,self.cfg,cc); baseline=render([note()],2,plain,cc)
        self.assertGreater(np.std(y-baseline),1e-8)

    def test_dispatch_manifest(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'build') as folder:
            folder=Path(folder); midi=mido.MidiFile()
            midi.tracks.append(mido.MidiTrack([mido.MetaMessage('track_name',name='piano'),mido.Message('note_on',note=60,velocity=61),mido.Message('note_off',note=60,time=480)]))
            path=folder/'input.mid'; midi.save(path)
            manifest=folder/'piano.json'
            manifest.write_text(json.dumps({'performed_midi':str(path),'score_midi':str(path),'jobs':[{'index':0,'output':str(folder/'out.wav')}]}))
            subprocess.run([sys.executable,str(ROOT/'scripts/render_piano.py'),str(manifest)],check=True)
            metrics=json.loads(manifest.with_suffix('.metrics.json').read_text())
            self.assertEqual(metrics['notes'],1); self.assertEqual(metrics['stems']['0']['model'],'piano')
            self.assertTrue((folder/'out.wav').is_file())
            with wave.open(str(folder/'out.wav')) as audio:
                self.assertEqual(audio.getnchannels(),2)
                self.assertEqual(audio.getnframes(),round(metrics['audio_seconds']*48000))
            from soundbench.scripts.model_dispatch import render_model_stems
            settings={'instruments':{'piano':{'model':'piano'}}}
            stems=render_model_stems(path,path,settings,resolve_instruments(midi,settings),folder,'dispatch')
            self.assertEqual(stems['0']['model'],'piano')
            with wave.open(str(folder/'dispatch.stem.01-piano.raw.wav')) as audio:
                self.assertEqual(audio.getnframes(),round(metrics['audio_seconds']*48000))

if __name__=='__main__': unittest.main()
