#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["mido>=1.3,<2", "numpy>=1.26", "numba>=0.61", "scipy>=1.11"]
# ///
"""Bass tuning, articulation, release, shared-string and dispatch invariants."""
import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import mido
import numpy as np
from scipy.signal import periodogram
from soundbench.scripts.models.bass import load_config, render, note, assign_string, ROOT
from soundbench.scripts.render_bass import schedule
from soundbench.scripts.instrumentation import resolve_instruments


class BassTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = load_config()
        cls.c['excitation']['intonation_cents'] = 0

    def test_tuning(self):
        for preset in ('default','steel'):
            c = load_config(ROOT/f'presets/bass/{preset}.toml')
            c['excitation']['intonation_cents'] = 0
            for p in (28,33,38,43,48,55,64):
                s = assign_string(p,c)+1
                y = render(note(0,2,p,95,s,0),2.2,config=c)[0]
                f,a = periodogram(y[4800:72000],48000,nfft=1048576)
                f0 = 440*2**((p-69)/12)
                selected = (f > f0*.97)&(f < f0*1.03)
                est = f[selected][np.argmax(a[selected])]
                self.assertLess(abs(1200*np.log2(est/f0)),3.,(preset,p,est))

    def test_finite_and_no_clipping(self):
        for preset in ('default','steel'):
            c = load_config(ROOT/f'presets/bass/{preset}.toml')
            for kind in ('pizz','slap','slap-back','dead','ghost'):
                ev = sum((note(0,.8,p,127,s+1,s,kind=kind) for s,p in enumerate(c['strings']['open_midi'])),[])
                y = render(ev,2,config=c)
                self.assertTrue(np.all(np.isfinite(y)))
                self.assertLess(np.max(abs(y)),1)

    def test_release_and_final_tail(self):
        short = render(note(0,.4,33,100,3,0),2,config=self.c)[0]
        long = render(note(0,1.7,33,100,3,0),2,config=self.c)[0]
        self.assertLess(np.linalg.norm(short[48000:]),np.linalg.norm(long[48000:])*.01)
        self.assertLess(np.max(abs(short[-4800:])),1e-7)

    def test_seed_determinism(self):
        ev = note(0,1,40,95,2,0)
        a = render(ev,1.5,config=self.c)
        self.assertTrue(np.array_equal(a,render(ev,1.5,config=self.c)))
        other = copy.deepcopy(self.c); other['engine']['seed'] += 1
        self.assertFalse(np.array_equal(a,render(ev,1.5,config=other)))

    def test_stale_off_does_not_damp_replacement(self):
        ev = note(0,.6,33,90,3,0)+note(.4,1.2,35,95,3,1)
        a = render(ev,1.8,config=self.c)
        b = render([r for r in ev if not(r[1] == 1 and r[10] == 0)],1.8,config=self.c)
        self.assertTrue(np.array_equal(a,b))

    def test_slide_is_continuous_and_reaches_target(self):
        ev = note(0,1.8,33,100,3,0)
        ev += [[4800,2,2,0,38,0,24000,1,0,0,0]]
        y = render(ev,2,config=self.c)[0]
        f,a = periodogram(y[38400:76800],48000,nfft=1048576)
        f0 = 440*2**((38-69)/12)
        mask = (f > f0*.97)&(f < f0*1.03)
        est = f[mask][np.argmax(a[mask])]
        self.assertLess(abs(1200*np.log2(est/f0)),3)
        self.assertLess(np.max(abs(np.diff(y))),.25)

    def test_velocity_monotone(self):
        rms = [np.sqrt(np.mean(render(note(0,.3,38,v,2,0),.6,config=self.c)**2)) for v in range(1,128,4)]
        self.assertTrue(np.all(np.diff(rms) > 0))
        self.assertLess(max(np.diff(20*np.log10(rms[8:]))),2)

    def test_gestures_differ(self):
        ys = {k:render(note(0,1,33,100,3,0,kind=k),1.5,config=self.c) for k in ('pizz','slap','dead','ghost')}
        self.assertGreater(np.linalg.norm(ys['slap']-ys['pizz']),.1)
        self.assertLess(np.linalg.norm(ys['dead'][:,12000:]),np.linalg.norm(ys['pizz'][:,12000:])*.1)
        self.assertLess(np.linalg.norm(ys['ghost']),np.linalg.norm(ys['dead']))

    def test_slap_uses_physical_barrier(self):
        ev = note(0,.6,33,110,3,0,kind='slap')
        enabled = copy.deepcopy(self.c)
        disabled = copy.deepcopy(self.c)
        disabled['fingerboard']['clearance_m'] = .019
        a = render(ev,1.,config=enabled)
        b = render(ev,1.,config=disabled)
        self.assertGreater(np.linalg.norm(a-b),.05)
        # Collision impulse is deterministic, not a randomly selected sample.
        self.assertTrue(np.array_equal(a,render(ev,1.,config=enabled)))

    def test_slap_back_is_damped(self):
        a = render(note(0,1.,33,100,3,0,kind='slap'),1.5,config=self.c)
        b = render(note(0,1.,33,100,3,0,kind='slap-back'),1.5,config=self.c)
        self.assertLess(np.linalg.norm(b[:,12000:]),np.linalg.norm(a[:,12000:])*.1)
        settings = {'bass_articulation':[dict(track='bass',bar=1,kind='slap-back')]}
        midi = mido.MidiFile(); track=mido.MidiTrack(); midi.tracks.append(track)
        track.extend([mido.MetaMessage('track_name',name='bass'),mido.Message('note_on',note=33,velocity=100),mido.Message('note_off',note=33,time=480)])
        events,seconds,assignment = schedule(midi,midi,settings,[0],self.c)
        self.assertEqual(events[0][1],7)
        self.assertTrue(np.all(np.isfinite(render(events,seconds,config=self.c))))

    def test_flesh_contact_softens_leading_edge(self):
        short, long = copy.deepcopy(self.c), copy.deepcopy(self.c)
        short['excitation']['flesh_contact_ms'] = 1.5
        long['excitation']['flesh_contact_ms'] = 6.
        short['excitation']['noise_gain'] = long['excitation']['noise_gain'] = 0
        short['excitation']['attack_gain'] = long['excitation']['attack_gain'] = 0
        short['excitation']['release_noise_gain'] = long['excitation']['release_noise_gain'] = 0
        ev = note(0,.6,38,100,2,0)
        a = render(ev,1.,config=short)
        b = render(ev,1.,config=long)
        self.assertLess(np.linalg.norm(b[:,:48]),np.linalg.norm(a[:,:48]))

    def test_release_front_speaks_then_dies(self):
        base = copy.deepcopy(self.c)
        base['excitation'].update(attack_gain=0,release_noise_gain=0)
        bright = copy.deepcopy(base)
        bright['excitation'].update(attack_gain=1.5,release_noise_gain=.12,attack_t60=.25,attack_loss_slope=1.,release_noise_ms=4.,attack_velocity_power=0)
        ev = note(0,.9,33,95,3,0)
        old = render(ev,1.2,config=base)[0]
        new = render(ev,1.2,config=bright)[0]
        def energy(y):
            f,p = periodogram(y[:7200],48000)
            return np.sum(p[(f>=800)&(f<7000)])
        self.assertGreater(energy(new),energy(old)*3)
        self.assertLess(np.linalg.norm((new-old)[19200:38400]),np.linalg.norm(old[19200:38400])*.005)
        self.assertTrue(np.array_equal(new,render(ev,1.2,config=bright)[0]))

    def test_attack_hardness_tracks_velocity_without_layers(self):
        c=copy.deepcopy(self.c)
        c['excitation'].update(attack_gain=1.,release_noise_gain=.02,attack_velocity_power=4.)
        ratios=[]
        for velocity in (60,80,100):
            y=render(note(0,.6,33,velocity,3,0),1.,config=c)[0]
            f,p=periodogram(y[:7200],48000)
            ratios.append(np.sum(p[(f>800)&(f<4000)])/np.sum(p[(f>30)&(f<350)]))
        self.assertTrue(np.all(np.diff(ratios)>0))

    def test_release_noise_zero_has_no_contact_fallback(self):
        c=copy.deepcopy(self.c)
        c['excitation'].update(release_noise_gain=0,noise_gain=.1,attack_gain=0)
        events=note(0,.6,33,95,3,0)
        a=render(events,1.,config=c)[0]
        c['excitation']['noise_gain']=0
        self.assertTrue(np.array_equal(a,render(events,1.,config=c)[0]))

    def test_release_noise_tracks_plucking_point(self):
        c = copy.deepcopy(self.c)
        c['excitation'].update(attack_gain=0,release_noise_gain=.12,release_noise_ms=4.,noise_gain=0)
        # Above the highest string partial, only the contact burst remains.
        a = render(note(0,.6,28,100,4,0,position=.07),1.,config=c)[0]
        b = render(note(0,.6,28,100,4,0,position=.40),1.,config=c)[0]
        f,pa = periodogram(a[:2400],48000); _,pb = periodogram(b[:2400],48000)
        sel=(f>3000)&(f<7000)
        self.assertGreater(np.linalg.norm(pa[sel]-pb[sel]),np.linalg.norm(pa[sel])*.1)

    def test_empty_zero_duration(self):
        self.assertEqual(render([],0,owners=2,config=self.c).shape,(2,0))
        self.assertTrue(np.array_equal(render([],0.001,config=self.c),np.zeros((1,48))))

    def test_validation(self):
        with self.assertRaises(ValueError): assign_string(27,self.c)
        with self.assertRaises(ValueError): render([[0,0,4,0,40,90,0,0,0,0,0]],1,config=self.c)
        with self.assertRaises(ValueError): render(note(0,1,33,100,3,0,position=.9),1,config=self.c)
        with self.assertRaises(ValueError): render([],float('nan'),config=self.c)

    def test_manifest_schedule_and_cc64(self):
        midi = mido.MidiFile(); t=mido.MidiTrack(); midi.tracks.append(t)
        t.extend([mido.MetaMessage('track_name',name='bass'),mido.MetaMessage('set_tempo',tempo=500000),
                  mido.Message('note_on',note=33,velocity=90),
                  mido.Message('control_change',control=64,value=127,time=240),
                  mido.Message('note_off',note=33,time=240),
                  mido.Message('control_change',control=64,value=0,time=480)])
        settings={'instruments':{'bass':{'model':'bass'}},'bass_articulation':[dict(track='bass',bar=1,kind='slide',to=2,duration_beats=1)]}
        self.assertEqual(resolve_instruments(midi,settings)[0].model,'bass')
        ev,duration,assignment=schedule(midi,midi,settings,[0],self.c)
        self.assertEqual(assignment[0]['off_time_s'],1.)
        self.assertEqual([r[1] for r in ev[:2]],[0,2])
        self.assertEqual(len([r for r in ev if r[1] == 1]),1)
        self.assertTrue(np.all(np.isfinite(render(ev,duration,config=self.c))))
        settings['bass_articulation'][0]['duration_beats']=3
        with self.assertRaises(ValueError): schedule(midi,midi,settings,[0],self.c)


if __name__ == '__main__': unittest.main()
