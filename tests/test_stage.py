#!/usr/bin/env python3
# /// script
# dependencies = ["mido>=1.3,<2"]
# ///
"""Continuous stage controls preserve score events and actual entry velocities."""
from pathlib import Path
import sys,tempfile,unittest
import mido
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1] / "soundbench"
sys.path.insert(0,str(ROOT/'scripts'))
from soundbench.scripts.stage import interpolate, controls, pitch_controls, volume_filter, bar_beat
from soundbench.scripts.perform import BarMap, perform_file

class StageTests(unittest.TestCase):
    def test_interpolation_and_validation(self):
        self.assertEqual(interpolate([(0,3),(4,7)],2),5)
        self.assertEqual(interpolate([(0,3),(4,7)],9),7)
        mid=mido.MidiFile();mid.tracks.append(mido.MidiTrack())
        bars=BarMap(mid)
        with self.assertRaises(ValueError):
            controls({'stage':{'mic':{'piano':[[1,6]]}}},'mic',[1],{1:'piano'},bars)
        with self.assertRaises(ValueError):
            controls({'stage':{'velocity':{'piano':[[2,50],[1,60]]}}},'velocity',[1],{1:'piano'},bars)
        retreat=controls({'stage':{'mic':{'piano':[[1,-18],[2,5]]}}},'mic',[1],{1:'piano'},bars)
        self.assertEqual(retreat[1][0][1],-18)
        fractional=controls({'stage':{'mic':{'piano':[[4,-10],[4,1.2,5],[6,5]]}}},'mic',[1],{1:'piano'},bars)
        self.assertEqual(bar_beat(bars,fractional[1][0][0]),(4,1))
        self.assertAlmostEqual(bar_beat(bars,fractional[1][1][0])[1],1.2)
        projection=controls({'stage':{'gain':{'piano':[[1,-12],[2,0]]}}},'gain',[1],{1:'piano'},bars)
        self.assertEqual(projection[1][0][1],-12)
        with self.assertRaises(ValueError):
            controls({'stage':{'gain':{'piano':[[1,1]]}}},'gain',[1],{1:'piano'},bars)
        with self.assertRaises(ValueError):
            controls({'stage':{'mic':{'piano':[[1,-25]]}}},'mic',[1],{1:'piano'},bars)
        self.assertIn('eval=frame',volume_filter([(0,5),(3,0)]))
        curve=pitch_controls({'stage':{'touch':{'piano':[[60,.5],[72,1]]}}},[1],{1:'piano'})
        self.assertEqual(interpolate(curve[1],66),.75)
        with self.assertRaises(ValueError):
            pitch_controls({'stage':{'touch':{'piano':[[60,1.1]]}}},[1],{1:'piano'})

    def test_actual_velocity_fades_after_adjustments(self):
        mid=mido.MidiFile(ticks_per_beat=480)
        tr=mido.MidiTrack();mid.tracks.append(tr)
        tr.append(mido.MetaMessage('track_name',name='piano'))
        for i in range(4):
            tr.append(mido.Message('note_on',note=60,velocity=80,time=0 if i==0 else 1440))
            tr.append(mido.Message('note_off',note=60,time=480))
        with tempfile.TemporaryDirectory(dir=ROOT/'build') as tmp:
            p=Path(tmp);mid.save(p/'score.mid')
            (p/'score.perform.toml').write_text('[humanize]\nonset_jitter_ms=0\nvelocity_jitter=0\n[tracks]\npiano={velocity_scale=0.75,velocity_offset=4}\n[stage.velocity]\npiano=[[1,48],[4,80]]\n')
            perform_file(p/'score.mid',p/'played.mid')
            velocities=[m.velocity for t in mido.MidiFile(p/'played.mid').tracks for m in t if m.type=='note_on' and m.velocity]
            self.assertEqual(velocities,[38,47,55,64])
            sidecar=p/'score.perform.toml'
            sidecar.write_text(sidecar.read_text().replace('velocity_jitter=0','velocity_jitter=4'))
            with patch('perform.random.Random.gauss',side_effect=lambda mean,sigma:8.0 if sigma else 0):
                perform_file(p/'score.mid',p/'played.mid')
            velocities=[m.velocity for t in mido.MidiFile(p/'played.mid').tracks for m in t if m.type=='note_on' and m.velocity]
            self.assertEqual(velocities,[42,52,62,72])
            sidecar.write_text(sidecar.read_text()+'\n[stage.touch]\npiano=[[60,0.5]]\n')
            with patch('perform.random.Random.gauss',side_effect=lambda mean,sigma:8.0 if sigma else 0):
                perform_file(p/'score.mid',p/'played.mid')
            velocities=[m.velocity for t in mido.MidiFile(p/'played.mid').tracks for m in t if m.type=='note_on' and m.velocity]
            self.assertEqual(velocities,[21,26,31,36])

if __name__=='__main__':unittest.main()
