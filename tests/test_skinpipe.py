#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy", "numba", "mido", "soundfile"]
# ///
"""Skinpipe contract: coupled audio, controls, stochastic identity and routing."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import mido
import numpy as np
ROOT=Path(__file__).resolve().parents[1] / "soundbench"
sys.path.insert(0,str(ROOT/'scripts'))
from soundbench.scripts.models.skinpipe import load_config, render, stream_seed, stochastic_lanes, bag_lane, validate
from soundbench.scripts.skinpipe_measure import note, cc, rms, span, db_ratio, cents, bloom_metrics, measure_all
from soundbench.scripts.render_skinpipe import selected_config, groups
from soundbench.scripts.instrumentation import resolve_instruments, Binding
from soundbench.scripts.model_dispatch import RENDERERS, render_model_stems
from soundbench.scripts.models import mallorca_winds
from soundbench.scripts.perform import perform_file


def midi_fixture():
    midi=mido.MidiFile(ticks_per_beat=480)
    midi.tracks.append(mido.MidiTrack([mido.MetaMessage('set_tempo',tempo=500000)]))
    for name,p in [('skinpipe:skinpipe',62),('skinthumb:skinthumb',50)]:
        midi.tracks.append(mido.MidiTrack([
            mido.MetaMessage('track_name',name=name),
            mido.Message('control_change',control=64,value=0),
            mido.Message('note_on',note=p,velocity=90),
            mido.Message('note_off',note=p,time=960)]))
    return midi


class Skinpipe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metrics=measure_all()
        out=ROOT/'build/skinpipe-probe';out.mkdir(parents=True,exist_ok=True)
        (out/'acceptance.json').write_text(json.dumps(cls.metrics,indent=2)+'\n')
        print(json.dumps(cls.metrics,indent=2))

    def test_pitch_and_feedback(self):
        self.assertLessEqual(max(abs(v) for v in self.metrics['pitch_cents'].values()),12)
        self.assertLessEqual(max(abs(v) for v in self.metrics['feedback_shift_cents'].values()),8)
        self.assertEqual(set(self.metrics['lock_in_cents']),{'58','61','63','65','67','68','70'})

    def test_bloom_and_balance(self):
        self.assertGreaterEqual(self.metrics['bloom_db'],10)
        self.assertLess(abs(self.metrics['skin_under_dry_db']),2)
        self.assertLessEqual(max(abs(self.metrics['pressure_course_energy_db_relative_cc96'][str(p)]) for p in range(90,109)),3)
        c=load_config();c['chamber_drive']=0
        y,parts=render([note()],5,c,details=True)
        self.assertEqual(np.max(abs(parts['courses'])),0)
        self.assertEqual(np.max(abs(parts['skin'])),0)

    def test_halo(self):
        self.assertGreaterEqual(self.metrics['halo_at_1s_db'],-30)
        self.assertGreaterEqual(self.metrics['e4_halo_deficit_db'],8)
        self.assertTrue(3<=self.metrics['t60_s']<=8)
        self.assertGreater(self.metrics['halo_course_harmonic_power_share'],.95)

    def test_damper(self):
        self.assertLessEqual(self.metrics['damper_40db_s'],.3)
        self.assertLessEqual(self.metrics['damper_at_300ms_db'],-40)
        self.assertLess(self.metrics['damper_click_peak_ratio'],1)

    def test_plucks(self):
        self.assertLessEqual(max(abs(v) for v in self.metrics['pluck_cents'].values()),5)
        with self.assertRaisesRegex(ValueError,'E4'):
            render([],1,plucks=[note(64,off=.1)])
        # Note-off does not kill the free course; bag off and bag on both work.
        free=render([],3,plucks=[note(50,off=.1)])
        self.assertGreater(rms(span(free,1,2)),1e-5)
        combined=render([note(62,off=2)],3,plucks=[note(50,time=.5,off=.6)])
        self.assertTrue(np.isfinite(combined).all())

    def test_stability_and_speed(self):
        self.assertEqual(self.metrics['stability_cases'],240)
        self.assertLess(self.metrics['worst_peak'],.9)
        self.assertLess(self.metrics['render_180s_wall_s'],180)

    def test_config_validation_and_overrides(self):
        c=load_config()
        for mutate in [lambda d:d.update(typo=1),lambda d:d.update(cut_s=0),
                       lambda d:d.update(seed=-1),lambda d:d.update(output_gain=float('nan')),
                       lambda d:d.update(courses=[50]*5),lambda d:d.update(courses=50),
                       lambda d:d.update(tuning_cents=0),lambda d:d['head'].update(typo=1),
                       lambda d:d['strings'].update(damper_t60=-1),lambda d:d.update(tuning_cents=[0]*11)]:
            d=copy.deepcopy(c);mutate(d)
            with self.assertRaises(ValueError):validate(d)
        with self.assertRaises(ValueError):render([],1,{})
        _,_,digest,_=selected_config({})
        alt,_,other,_=selected_config({'skinpipe_model':{'courses':[48,55,60,64,67]}})
        self.assertNotEqual(digest,other);self.assertEqual(alt['courses'][0],48)
        for options in [{'typo':1},{'courses':[50,57]}]:
            with self.assertRaises(ValueError):selected_config({'skinpipe_model':options})

    def test_streams_are_track_note_and_source_keyed(self):
        c=load_config();rows=[note(62,off=.5),note(62,time=.5,off=1)]
        a=stochastic_lanes(rows,48000,c,'player-A')
        b=stochastic_lanes(rows,48000,c,'player-B')
        self.assertFalse(np.array_equal(a[0],b[0]))
        self.assertFalse(np.array_equal(a[0][:24000],a[0][24000:]))
        self.assertFalse(np.array_equal(a[0],a[1]))
        self.assertTrue(np.array_equal(a[0],stochastic_lanes(rows,48000,c,'player-A')[0]))
        seeds={stream_seed(c['seed'],track,source,k,k*.5) for track in ['A','B']
               for source in ['reed','air','bag','pluck'] for k in range(2)}
        self.assertEqual(len(seeds),16)
        self.assertFalse(np.array_equal(bag_lane(rows,48000,c,'A'),bag_lane(rows,48000,c,'B')))
        y=render(rows,2,c,track_id='A',plucks=[note(50,off=.1),note(50,time=.5,off=.6)])
        self.assertTrue(np.array_equal(y,render(rows,2,c,track_id='A',plucks=[note(50,off=.1),note(50,time=.5,off=.6)])))
        self.assertFalse(np.array_equal(y,render(rows,2,c,track_id='B',plucks=[note(50,off=.1),note(50,time=.5,off=.6)])))
        midi=midi_fixture();midi.tracks[2].name=midi.tracks[1].name
        with self.assertRaisesRegex(ValueError,'identities must be unique'):
            groups(midi,[{'index':1},{'index':2}])

    def test_lean_affects_head_not_pitch(self):
        c=load_config();c['wobble_cents']=0
        a,ap=render([note()],4.5,c,details=True)
        b,bp=render([note()],4.5,c,[cc(74,127)],details=True)
        self.assertLess(abs(cents(a,62,c)-cents(b,62,c)),8)
        self.assertGreater(np.linalg.norm(ap['skin']-bp['skin']),.1)

    def test_bag_cut_rest_grace_and_invalid_notes(self):
        c=load_config()
        rows=[note(62,off=.4),note(64,time=.8,off=.84),note(62,time=.84,off=2)]
        y=render(rows,3,c)
        self.assertGreater(rms(span(y,.5,.7)),.01)
        for rows in [[note(56)], [note(72)], [note(),note()], [note(off=0)]]:
            with self.assertRaises(ValueError):render(rows,5)
        self.assertEqual(max(abs(render([],1))),0)
        # An explicit cut at the last note-off has the breath-cut ramp, not end sag.
        wc=dict(c,pressure_pitch_cents=180.,drone_pitch=50,wobble_cents=0)
        _,p,*_=mallorca_winds.plan([note(62,off=1)],2,wc,'xeremia',[cc(24,0,1)],
                                   compass=(57,71),cut_s=.09)
        self.assertEqual(p[round(1.1*48000)],0)
        _,p,*_=mallorca_winds.plan([note(62,off=1)],2,wc,'xeremia',
                                   compass=(57,71),cut_s=.09)
        self.assertGreater(p[round(1.1*48000)],0)

    def test_registry_inheritance_dispatch_and_cc64(self):
        midi=midi_fixture();settings={'instruments':{'skinpipe':{'model':'skinpipe'}}}
        bindings=resolve_instruments(midi,settings)
        self.assertEqual(bindings[1],bindings[2]);self.assertTrue(RENDERERS['skinpipe'][0].exists())
        ROOT.joinpath('build').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'build') as d:
            path=Path(d);mid=path/'test.mid';midi.save(mid)
            sidecar=path/'perform.toml';sidecar.write_text('[instruments]\nskinpipe = { model = "skinpipe" }\n')
            output=path/'performed.mid';perform_file(mid,output,sidecar)
            performed=mido.MidiFile(output)
            self.assertTrue(any(m.type=='control_change' and m.control==64 and m.value==0 for m in performed.tracks[1]))
            metrics=render_model_stems(mid,mid,settings,bindings,path,'test')
            self.assertEqual(metrics['2']['merged_into'],1)
            self.assertEqual(metrics['1']['track_indices'],[1,2])
            self.assertEqual(len(list(path.glob('*.raw.wav'))),1)
        with self.assertRaises(ValueError):resolve_instruments(midi,{'instruments':{'skinpipe':'skinpipe'}})

    def test_banjo_radius_extension_bit_identity(self):
        from soundbench.scripts.models.banjo import render as banjo_render, load_config as banjo_config
        from soundbench.scripts.paths import preset_path
        events=[[0,0,0,0,62,90,0,1,0,0,0],[9600,0,4,1,69,82,0,1,0,0,1]]
        y=banjo_render(events,2,2,banjo_config(preset_path('banjo', 'membrane')))
        self.assertEqual(hashlib.sha256(y.tobytes()).hexdigest(),
                         '81787ea34b32ce3c8721707a869072b4e7f240a60654ac42e2f30610ab6dfc2d')

    def test_wind_refactor_bit_identity(self):
        # Float64 wind output: independent seeds, raw reed and breath flutter,
        # without the skinpipe's opt-in compass/cut_s extension.
        expected={'flabiol':'ae49a04cd21dd2e8cd491d4777f1e65bdaa0609dfcc500a519e31637c793d246',
                  'xeremia':'116ca8819a635193b677efaf991d65b21f43b360121d39eac09568f90c914527'}
        for inst,p in [('flabiol',86),('xeremia',64)]:
            rows=[note(p,time=.1,off=.9),note(p+2,time=1.2,off=1.9,velocity=82)]
            controls=[cc(2,104,.65),cc(24,0,1),cc(24,127,1.2)]
            y=mallorca_winds.render(rows,3.5,mallorca_winds.load_config(inst),inst,controls)
            self.assertEqual(hashlib.sha256(y.tobytes()).hexdigest(),expected[inst])

if __name__=='__main__':unittest.main()
