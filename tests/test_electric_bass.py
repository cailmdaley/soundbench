#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["mido>=1.3,<2", "numpy>=1.26", "numba>=0.61", "scipy>=1.11", "soundfile>=0.12", "tomli-w>=1"]
# ///
"""Audio-domain tuning, damping, gestures, controller and routing invariants."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
from scipy.signal import find_peaks
import mido
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from soundbench.scripts.models.electric_bass import load_config, render, note, assign_string, SR
from soundbench.scripts.render_electric_bass import schedule
from soundbench.scripts.instrumentation import resolve_instruments


def pitch_hz(y):
    z=y[round(.15*SR):round(.65*SR)]
    size=2**19
    spectrum=abs(np.fft.rfft(z*np.hanning(len(z)),n=size))
    freq=np.fft.rfftfreq(size,1/SR)
    peaks,_=find_peaks(spectrum)
    peaks=[p for p in peaks if 35<freq[p]<450]
    peak=max(peaks,key=lambda p:spectrum[p])
    return freq[peak]


class BassTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c=load_config()
        render([],0.001,config=cls.c)

    def test_tuning(self):
        for p in (28,33,38,43,48,55,64):
            with self.subTest(pitch=p):
                s=assign_string(p,self.c)+1
                y=render(note(0,.7,p,90,s,0),1,config=self.c)[0]
                # Compare with nearest partial, since pickup cancellation can
                # make the fundamental weaker than the second harmonic.
                f=pitch_hz(y);target=440*2**((p-69)/12)
                harmonic=max(1,round(f/target))
                self.assertLess(abs(1200*np.log2(f/harmonic/target)),5)

    def test_finite_bounded_deterministic(self):
        rows=[]
        for i,p in enumerate((28,33,38,43)):
            rows+=note(0,.8,p,127,4-i,i)
        a=render(rows,1.5,config=self.c)
        np.testing.assert_array_equal(a,render(rows,1.5,config=self.c))
        self.assertTrue(np.isfinite(a).all());self.assertLess(np.max(abs(a)),.981)
        c=copy.deepcopy(self.c);c['engine']['seed']+=1
        self.assertFalse(np.array_equal(a,render(rows,1.5,config=c)))

    def test_open_string_release(self):
        y=render(note(0,.3,28,100,4,0),1,config=self.c)[0]
        early=np.sqrt(np.mean(y[int(.15*SR):int(.28*SR)]**2))
        late=np.sqrt(np.mean(y[int(.65*SR):]**2))
        self.assertLess(late/early,.001)

    def test_velocity_continuity(self):
        levels=[]
        for v in range(40,121,4):
            y=render(note(0,.25,40,v,3,0),.4,config=self.c)[0]
            levels.append(np.sqrt(np.mean(y*y)))
        self.assertTrue(np.all(np.diff(levels)>0))
        self.assertLess(np.max(np.diff(20*np.log10(levels))),1.3)

    def test_ghost_and_palm_decay(self):
        levels=[]
        for kind in (0,7,6):
            y=render(note(0,.6,40,100,3,0,kind=kind),.8,config=self.c)[0]
            levels.append(np.sqrt(np.mean(y[int(.2*SR):int(.45*SR)]**2)))
        self.assertLess(levels[1],levels[0]*.05)
        self.assertLess(levels[2],levels[0]*.005)

    def test_stale_release(self):
        base=note(0,.8,40,100,3,0)+note(.2,.7,42,100,3,1)
        stale=[int(.3*SR),1,2,0,0,0,0,0,0,0,0]
        np.testing.assert_array_equal(render(base,1,config=self.c),render(base+[stale],1,config=self.c))

    def test_slide_hammer(self):
        rows=note(0,.8,40,100,3,0)
        rows.append([int(.2*SR),2,2,0,43,0,.2*SR,1,0,0,0])
        y=render(rows,1,config=self.c)
        self.assertTrue(np.isfinite(y).all())
        rows=note(0,.8,40,100,3,0)+note(.3,.8,43,80,3,1,kind=3)
        self.assertGreater(np.max(abs(render(rows,1,config=self.c))),.01)

    def test_shared_amp_source_attribution(self):
        c=copy.deepcopy(self.c)
        c['excitation'].update(jitter=0.,noise=0.)
        c['engine']['output_gain']=.25
        rows=note(0,.4,40,100,3,0,owner=0)+note(0,.4,43,100,1,1,owner=1)
        split=render(rows,.6,owners=2,config=c)
        combined=[r.copy() for r in rows]
        for r in combined:r[3]=0
        joint=render(combined,.6,owners=1,config=c)
        np.testing.assert_allclose(split.sum(axis=0),joint[0],atol=1e-12)

    def test_output_gain_is_post_amp(self):
        c=copy.deepcopy(self.c);c['engine']['output_gain']=.25
        rows=note(0,.3,40,80,3,0)
        a=render(rows,.5,config=c)
        c['engine']['output_gain']=.5
        np.testing.assert_allclose(render(rows,.5,config=c),2*a,atol=1e-14)

    def test_reference_alignment(self):
        from soundbench.scripts.electric_bass_lab import align_reference
        z=np.zeros(round(.3*SR))
        onset=round(.08*SR)
        z[onset:]=np.sin(2*np.pi*100*np.arange(len(z)-onset)/SR)
        aligned,shift=align_reference(z)
        self.assertGreater(shift,.070);self.assertLess(shift,.080)
        self.assertEqual(len(aligned),len(z)-round(shift*SR))

    def test_fit_penalizes_model_only_ringing(self):
        from soundbench.scripts.electric_bass_lab import spectral_distance
        t=np.arange(round(.35*SR))/SR
        a=np.sin(2*np.pi*100*t)
        b=a+.25*np.sin(2*np.pi*1700*t)
        self.assertAlmostEqual(spectral_distance(a,a),0.)
        self.assertGreater(spectral_distance(a,b),3.)

    def test_invalid(self):
        with self.assertRaises(ValueError):assign_string(27,self.c)
        with self.assertRaises(ValueError):render([[0,0,4,0,40,90,0,0,0,0,0]],1,config=self.c)
        with self.assertRaises(ValueError):render([],float('nan'),config=self.c)

    def test_midi_routing_and_release_gate(self):
        m=mido.MidiFile();t=mido.MidiTrack();m.tracks.append(t)
        t.append(mido.MetaMessage('track_name',name='bass'))
        t.append(mido.Message('note_on',note=40,velocity=90))
        with self.assertRaises(ValueError):schedule(m,[0],self.c)
        t.append(mido.Message('note_off',note=40,time=240))
        rows,seconds,a=schedule(m,[0],self.c)
        self.assertEqual(len(a),1);self.assertTrue(any(r[1]==1 for r in rows))
        self.assertEqual(resolve_instruments(m,{'instruments':{'bass':{'model':'electric-bass'}}})[0].model,'electric-bass')

    def test_studio_dispatch(self):
        from soundbench.scripts.model_dispatch import render_model_stems
        import json
        import wave
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root/'build') as directory:
            out=Path(directory);m=mido.MidiFile();t=mido.MidiTrack();m.tracks.append(t)
            t.extend([mido.MetaMessage('track_name',name='bass'),mido.Message('note_on',note=40,velocity=90),mido.Message('note_off',note=40,time=240)])
            path=out/'probe.mid';m.save(path)
            settings={'instruments':{'bass':{'model':'electric-bass'}}}
            bindings=resolve_instruments(m,settings)
            stems=render_model_stems(path,path,settings,bindings,out,'probe')
            self.assertEqual(stems['0']['model'],'electric-bass')
            metrics=json.loads((out/'probe.electric-bass.metrics.json').read_text())
            self.assertEqual(len(metrics['assignment']),1)
            with wave.open(str(out/'probe.stem.01-bass.raw.wav')) as wav:
                self.assertEqual(wav.getframerate(),48000);self.assertEqual(wav.getnchannels(),2)

    def test_legato_requires_pin(self):
        m=mido.MidiFile();t=mido.MidiTrack();m.tracks.append(t)
        t.extend([mido.Message('control_change',control=68,value=127),mido.Message('note_on',note=40,velocity=90)])
        with self.assertRaises(ValueError):schedule(m,[0],self.c)


if __name__=='__main__':unittest.main()
