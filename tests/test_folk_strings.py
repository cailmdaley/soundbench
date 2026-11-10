# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=1.26", "scipy>=1.11", "numba>=0.61", "mido>=1.3,<2"]
# ///
import copy
import contextlib
import io
import json
import tempfile
from unittest.mock import patch
import sys
from pathlib import Path
import unittest
import numpy as np
from scipy.signal import find_peaks
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from soundbench.scripts.models.folk_strings import load_config, note, tremolo, chop, bar, render, SR, express_bar, vibrato
from soundbench.scripts.models import guitar
from soundbench.scripts.render_folk_strings import schedule
from soundbench.scripts.instrumentation import resolve_instruments
import mido


class FolkStrings(unittest.TestCase):
    def test_tuning(self):
        errors = []
        for inst in ('mandolin','dobro'):
            c = load_config(inst)
            for s,op in enumerate(c['strings']['open_midi'],1):
                for fret in (0,5,12):
                    p = op+fret
                    y = render(note(0,.6,p,65,s,10),.6,config=c,instrument=inst)[0]
                    z = y[2400:24000]*np.hanning(21600)
                    n = 262144
                    f = np.fft.rfftfreq(n,1/SR)
                    power = abs(np.fft.rfft(z,n))
                    f0 = 440*2**((p-69)/12)
                    # Course pair midpoint, not an accidentally selected upper partial.
                    sel = np.flatnonzero(abs(f-f0)<f0*.015)
                    k = sel[np.argmax(power[sel])]
                    a,b,d = np.log(power[k-1:k+2]+1e-30)
                    hz = (k+.5*(a-d)/(a-2*b+d))*SR/n
                    cents = 1200*np.log2(hz/f0)
                    errors.append(cents)
                    self.assertLess(abs(cents),6,(inst,s,p,cents))
        print('maximum primary/course tuning error:',max(abs(np.array(errors))),'cents')

    def test_finite_deterministic_seed_and_release(self):
        for inst in ('mandolin','dobro'):
            c = load_config(inst)
            rows = note(0,.3,67,75,1 if inst=='dobro' else 3,8)
            a = render(rows,3,config=c,instrument=inst)
            b = render(rows,3,config=c,instrument=inst)
            self.assertTrue(np.array_equal(a,b))
            self.assertTrue(np.all(np.isfinite(a)))
            self.assertLess(abs(a).max(),.98)
            tail = np.sqrt(np.mean(a[0,-4800:]**2))
            held = render(note(0,3,67,75,1 if inst=='dobro' else 3,8),3,config=c,instrument=inst)
            self.assertLess(tail,np.sqrt(np.mean(held[0,-4800:]**2))*.1)
            # Passive sympathetic lanes can retain a quiet residual after finger damping.
            self.assertLess(tail,np.sqrt(np.mean(a[0,4800:9600]**2))*.01)
            # The seed moves the pluck point; widen its jitter past one-sample comb quantization.
            d = copy.deepcopy(c);d['excitation']['jitter'] = .2
            jittered = render(rows,3,config=d,instrument=inst)
            d['engine']['seed'] += 1
            self.assertFalse(np.array_equal(jittered,render(rows,3,config=d,instrument=inst)))

    def test_velocity_ramp(self):
        for inst in ('mandolin','dobro'):
            c = load_config(inst)
            levels = []
            for v in range(1,128):
                y = render(note(0,.2,76, v,1,0),.2,config=c,instrument=inst)[0]
                levels.append(np.sqrt(np.mean(y*y)))
            steps = np.diff(20*np.log10(levels))
            self.assertTrue(np.all(steps>0))
            self.assertLess(max(steps[39:]),.5)
            print(inst,'maximum adjacent velocity step 40..127:',max(steps[39:]),'dB')

    def test_pair_and_controls(self):
        c = load_config('mandolin')
        self.assertTrue(all(.4 <= v < .5 for v in c['strings']['polarization_mix']))
        self.assertTrue(all(v != 0 for v in c['strings']['polarization_cents']))
        self.assertTrue(all(v != 1 for v in c['strings']['polarization_rate']))
        rows = tremolo(0,1,76,75,1,100,rate=12,evenness=1)
        self.assertEqual(sum(r[1]==0 for r in rows),12)
        self.assertEqual(sum(r[1]==1 for r in rows),1)
        self.assertAlmostEqual(next(r[0] for r in rows if r[1]==1)/SR,1)
        with self.assertRaises(ValueError): tremolo(0,1,76,75,1,0,rate=30)
        rows = chop(0,[79,74,67,59],[1,2,3,4],c)
        self.assertEqual(sum(r[1]==5 for r in rows),4)
        y = render(rows,.8,config=c)[0]
        self.assertLess(np.sqrt(np.mean(y[-4800:]**2)),1e-6)

    def test_bar_continuity_noise(self):
        c = load_config('dobro')
        rows = note(0,1,62,75,1,4)+bar(0,[(0,62),(.4,67.5),(.8,67)],1,4)
        y = render(rows,2,config=c,instrument='dobro')[0]
        self.assertTrue(np.isfinite(y).all())
        self.assertLess(np.max(abs(np.diff(y))),.3)
        d = copy.deepcopy(c);d['excitation']['slide_noise']=0
        self.assertFalse(np.array_equal(y,render(rows,2,config=d,instrument='dobro')[0]))
        with self.assertRaises(ValueError): bar(0,[(0,62),(0,67)],1,0)

    def test_growing_default_vibrato(self):
        c=load_config('dobro')
        rows=note(0,2,67,75,1,4)
        articulated=express_bar(rows,c,dict(depth_cents=20,rate=6,onset_delay=.3,rise_seconds=.4))
        ramps=[r for r in articulated if r[1]==2]
        t=np.array([(r[0]+r[6])/SR for r in ramps])
        cents=np.array([(r[4]-67)*100 for r in ramps])
        self.assertEqual(np.max(abs(cents[t<=.3])),0)
        self.assertLess(np.max(abs(cents[(t>.3)&(t<.4)])),4)
        self.assertGreater(np.max(cents[(t>.7)&(t<1.5)]),19)
        self.assertLess(np.min(cents[(t>.7)&(t<1.5)]),-19)
        self.assertLessEqual(np.max(abs(cents)),20.001)
        self.assertEqual(express_bar(rows,c,False),rows)
        self.assertFalse(any(r[1]==2 for r in express_bar(note(0,.65,67,75,1,4),c)))
        self.assertFalse(any(r[1]==2 for r in express_bar(note(0,2,62,75,1,4),c)))
        self.assertTrue(vibrato(0,1,67,1,4,depth_cents=25,rate=5))
        with self.assertRaises(ValueError):express_bar(rows,c,{'depth_cents':100})
        y=render(rows,2.5,config=c,instrument='dobro')
        self.assertFalse(np.array_equal(y,render(rows,2.5,config=c,instrument='dobro',vibrato=False)))
        self.assertTrue(np.array_equal(y,render(rows,2.5,config=c,instrument='dobro')))

    def test_tremolo_drift_and_downstroke_accents(self):
        rows=tremolo(0,4,79,80,1,20,rate=12,evenness=.9,accent=1.12)
        attacks=[r for r in rows if r[1]==0]
        intervals=np.diff([r[0]/SR for r in attacks])
        self.assertGreater(np.std(intervals)/np.mean(intervals),.007)
        self.assertGreater(np.mean([r[5] for r in attacks[::2]]),np.mean([r[5] for r in attacks[1::2]])*1.08)
        self.assertEqual(rows,tremolo(0,4,79,80,1,20,rate=12,evenness=.9,accent=1.12))
        self.assertFalse(rows==tremolo(0,4,79,80,1,20,rate=12,evenness=.9,accent=1.12,seed=818))
        self.assertEqual(sum(r[1]==1 for r in rows),1)
        print('tremolo interval coefficient of variation:',np.std(intervals)/np.mean(intervals))

    def test_tremolo_keeps_continuous_bend(self):
        midi=mido.MidiFile();t=mido.MidiTrack();midi.tracks.append(t)
        t.append(mido.MetaMessage('track_name',name='mandolin'))
        t.append(mido.Message('control_change',control=23,value=64))
        t.append(mido.Message('note_on',note=76,velocity=75))
        t.append(mido.Message('pitchwheel',pitch=2048,time=480))
        t.append(mido.Message('note_off',note=76,time=480))
        rows,_,_=schedule(midi,{},[0],load_config('mandolin'),'mandolin')
        attacks=sorted((r for r in rows if r[1]==0),key=lambda r:r[0])
        self.assertGreater(attacks[1][4],attacks[0][4])
        self.assertTrue(any(r[1]==2 for r in rows))
        self.assertAlmostEqual(attacks[-1][4],79)

    def test_stale_bar_has_no_friction(self):
        c = load_config('dobro')
        rows = note(0,.9,62,75,1,4)+note(.2,.9,64,75,1,5)
        a = render(rows,1.4,config=c,instrument='dobro')
        b = render(rows+bar(.4,[(0,62),(.4,67)],1,4),1.4,config=c,instrument='dobro')
        self.assertTrue(np.array_equal(a,b))

    def test_manifest_render(self):
        from soundbench.scripts import render_folk_strings
        from soundbench.scripts.model_dispatch import RENDERERS
        for inst in ('mandolin','dobro'):
            self.assertIn(inst,RENDERERS)
            with tempfile.TemporaryDirectory() as folder:
                root=Path(folder);midi=mido.MidiFile();t=mido.MidiTrack();midi.tracks.append(t)
                t.append(mido.MetaMessage('track_name',name=inst))
                t.append(mido.Message('note_on',note=76 if inst=='mandolin' else 62,velocity=75))
                t.append(mido.Message('note_off',note=76 if inst=='mandolin' else 62,time=480))
                midi.save(root/'score.mid')
                manifest=root/f'fixture.{inst}.json'
                manifest.write_text(json.dumps(dict(performed_midi=str(root/'score.mid'),score_midi=str(root/'score.mid'),
                    settings={},jobs=[dict(index=0,output=str(root/'stem.wav'))])))
                with patch.object(sys,'argv',['render',str(manifest)]),contextlib.redirect_stdout(io.StringIO()):
                    render_folk_strings.main()
                metrics=json.loads(manifest.with_suffix('.metrics.json').read_text())
                self.assertEqual(metrics['stems']['0']['model'],inst)
                self.assertTrue((root/'stem.wav').exists())
                self.assertLess(metrics['peak'],.98)

    def test_midi_contract_and_bindings(self):
        for inst in ('mandolin','dobro'):
            midi=mido.MidiFile();t=mido.MidiTrack();midi.tracks.append(t)
            t.append(mido.MetaMessage('track_name',name=inst))
            if inst=='mandolin':t.append(mido.Message('control_change',control=23,value=64))
            t.append(mido.Message('note_on',note=76 if inst=='mandolin' else 62,velocity=75))
            if inst=='dobro':t.append(mido.Message('pitchwheel',pitch=2048,time=240))
            t.append(mido.Message('note_off',note=76 if inst=='mandolin' else 62,time=480))
            settings={'instruments':{inst:{'model':inst}}}
            self.assertEqual(resolve_instruments(midi,settings)[0].model,inst)
            rows,seconds,assignment=schedule(midi,settings,[0],load_config(inst),inst)
            self.assertTrue(any(r[1]==1 for r in rows))
            if inst=='dobro':self.assertTrue(any(r[1]==2 and r[4]==65 for r in rows))
            else:self.assertGreater(sum(r[1]==0 for r in rows),5)
            self.assertEqual(len(assignment),1)


if __name__=='__main__':unittest.main()
