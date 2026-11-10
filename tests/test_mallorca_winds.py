# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy", "numba", "mido", "soundfile"]
# ///
"""Audio-domain wind invariants, MIDI semantics and dispatch integration."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import unittest
import numpy as np
import mido
from soundbench.scripts.models.mallorca_winds import ROOT, load_config, render, plan, frequency, _reed
from soundbench.scripts.winds_diagnose import pitch
from soundbench.scripts.render_winds import midi_controls, selected_config, write_wav
from soundbench.scripts.instrumentation import resolve_instruments
from soundbench.scripts.model_dispatch import RENDERERS


def note(p=86, time=0, off=1, velocity=90):
    return dict(time=time, off=off, pitch=p, velocity=velocity)


def clean(inst):
    c = load_config(inst)
    c['drone_gain'] = c['wobble_cents'] = c['squawk_cents'] = c['sag_cents'] = c['tremor_cents'] = 0
    return c


class Winds(unittest.TestCase):
    def test_stem_float_and_single_pan(self):
        import tempfile
        import soundfile as sf
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'stem.wav'
            y = np.array([.25, -.5, 1e-8, 0.])
            write_wav(path, y, 48000)
            self.assertEqual(sf.info(path).subtype, 'FLOAT')
            stereo, sr = sf.read(path)
            self.assertEqual(sr, 48000)
            np.testing.assert_allclose(stereo, np.repeat(y[:, None], 2, axis=1), rtol=1e-7, atol=0)
            # Folding the stem to mono must preserve its amplitude before panning.
            np.testing.assert_allclose(stereo.mean(axis=1), y, rtol=1e-7, atol=0)

    def test_full_compass_tuning(self):
        errors = {}
        for inst, pitches in [('flabiol',range(74,99)), ('xeremia',range(59,73))]:
            c = clean(inst)
            errors[inst] = []
            for p in pitches:
                y = render([note(p, off=2.2)], 2.3, c, inst)
                actual, _ = pitch(y[:100800], 48000, frequency(p,c))
                cents = 1200*np.log2(actual/frequency(p,c))
                errors[inst].append(float(cents))
                self.assertLess(abs(cents), 9, (inst,p,cents))
                self.assertGreater(np.sqrt(np.mean(y[72000:96000]**2)), .01)
        print('compass max cents:', {k:max(map(abs,v)) for k,v in errors.items()})

    def test_drone_tuning(self):
        c = load_config('xeremia')
        errors = []
        for p in range(36,61):
            f = frequency(p,c)
            y = _reed(np.full(72000,f),np.ones(72000),np.random.default_rng(1).uniform(-.01,.01,72000),48000,.65,False,1.0,np.zeros(72000))
            actual,_ = pitch(y,48000,f)
            cents = 1200*np.log2(actual/f)
            errors.append(abs(cents))
            self.assertLess(abs(cents),3,(p,cents))
        print('drone max cents:',max(errors))

    def test_seed_stability_and_release(self):
        for inst,p in [('flabiol',86),('xeremia',64)]:
            c = load_config(inst)
            y = render([note(p)],2.5,c,inst)
            self.assertTrue(np.array_equal(y,render([note(p)],2.5,c,inst)))
            other = dict(c, seed=c['seed']+1)
            self.assertFalse(np.array_equal(y,render([note(p)],2.5,other,inst)))
            self.assertTrue(np.isfinite(y).all())
            self.assertLess(max(abs(y)),1)
            sustain = np.sqrt(np.mean(y[24000:40000]**2))
            self.assertGreater(np.sqrt(np.mean(y[48000:49000]**2)), sustain*.01)
            self.assertLess(np.sqrt(np.mean(y[-4800:]**2)),sustain*.001)

    def test_track_and_note_noise_are_independent(self):
        from unittest.mock import patch
        import soundbench.scripts.models.mallorca_winds as engine
        for inst, p in [('flabiol', 86), ('xeremia', 64)]:
            c = load_config(inst)
            c['squawk_noise'] = 0
            rows = [note(p, off=.2), note(p, time=.8, off=1.1)]
            original = engine._flute if inst == 'flabiol' else engine._reed
            target = '_flute' if inst == 'flabiol' else '_reed'
            def capture(notes, track):
                with patch.object(engine, target, wraps=original) as loop:
                    y = render(notes, 2, c, inst, track_id=track)
                    noise = loop.call_args_list[0].args[3 if inst == 'flabiol' else 2].copy()
                return y, noise
            a, na = capture(rows, 1)
            b, nb = capture(rows, 2)
            self.assertFalse(np.array_equal(a, b))
            self.assertFalse(np.array_equal(na, nb))
            _, edited = capture([note(p, time=.4, off=.6), rows[-1]], 1)
            np.testing.assert_array_equal(na[38400:], edited[38400:])
            np.testing.assert_array_equal(a, capture(rows, 1)[0])
        _, bag1, *_ = plan(rows=[note(64, off=4)], seconds=5, config=load_config('xeremia'), instrument='xeremia', track_id=1)
        _, bag2, *_ = plan(rows=[note(64, off=4)], seconds=5, config=load_config('xeremia'), instrument='xeremia', track_id=2)
        self.assertFalse(np.array_equal(bag1, bag2))

    def test_velocity_ramp(self):
        c = load_config('flabiol')
        rms = []
        for v in range(16,128,8):
            y = render([note(86, velocity=v)],1.5,c)
            rms.append(np.sqrt(np.mean(y[24000:40000]**2)))
            self.assertLess(np.max(np.abs(np.diff(y))), .25)
        self.assertTrue(np.all(np.diff(rms)>0),rms)
        self.assertLess(max(np.diff(20*np.log10(rms))),3)

    def test_proxy_presets_tuning_and_release(self):
        from soundbench.scripts.models.mallorca_winds import ROOT
        for inst,pitches in [('flabiol',[74,78,86,98]),('xeremia',[59,64,66,72])]:
            c=load_config(inst)
            c['drone_gain']=c['wobble_cents']=c['squawk_cents']=c['sag_cents']=0
            for p in pitches:
                y=render([note(p,off=1.5)],3,c,inst)
                actual,_=pitch(y[:60000],48000,frequency(p,c))
                self.assertLess(abs(1200*np.log2(actual/frequency(p,c))),9,(inst,p))
                self.assertLess(np.sqrt(np.mean(y[-4800:]**2)),1e-5)


    def test_xeremia_velocity_slews_without_boundary_step(self):
        c = clean('xeremia')
        rows = [note(64, off=.5, velocity=127), note(64, time=.5, off=1, velocity=1)]
        loud = plan(rows, 2, c, 'xeremia')[2]
        boundary = 24000
        target = .85+.15*(.35+.65/127)
        self.assertLess(abs(loud[boundary]-loud[boundary-1]), .001)
        self.assertAlmostEqual(loud[boundary-1], 1, delta=1e-6)
        self.assertAlmostEqual(loud[boundary+2400], target, delta=.001)
        steady = [dict(r, velocity=127) for r in rows]
        y = render(rows, 2, c, 'xeremia')
        baseline = render(steady, 2, c, 'xeremia')
        self.assertLess(abs(y[boundary]-baseline[boundary]), .0001)

    def test_bag_holds_through_rests_and_graces(self):
        c = load_config('xeremia')
        rows=[note(60,off=.4),note(67,time=.8,off=.84),note(64,time=.84,off=1.2)]
        f,p,loud,df,gate=plan(rows,2.5,c,'xeremia')
        self.assertTrue(np.all(p[20000:38000]>0))
        self.assertTrue(np.all(loud[20000:38000]>0))
        self.assertAlmostEqual(f[40000]/f[42000],frequency(67,c)/frequency(64,c),delta=.005)
        y=render(rows,2.5,c,'xeremia')
        self.assertGreater(np.sqrt(np.mean(y[22000:35000]**2)),.01)

    def test_controls(self):
        c=load_config('xeremia')
        c['tremor_cents']=0
        controls=[dict(time=0,control=22,value=0),dict(time=0,control=23,value=0),
                  dict(time=.5,control=20,value=0),dict(time=0,control=21,value=43)]
        f,p,_,df,g=plan([note(64)],2,c,'xeremia',controls)
        self.assertEqual(g[30000],0)
        self.assertAlmostEqual(df[24000]/f[24000],frequency(43,c)/frequency(64,c))
        self.assertLess(f[70000]/f[48000],1.005)
        controls=[dict(time=0,control=2,value=0)]
        y=render([note()],2,load_config(),controls=controls)
        self.assertEqual(np.max(abs(y)),0)
        y=render([note(64)],2,load_config('xeremia'),'xeremia',controls)
        self.assertEqual(np.max(abs(y)),0)
        # A release must not overwrite the following startup's envelope.
        c=load_config('xeremia')
        c['wobble_cents']=c['tremor_cents']=0
        controls=[dict(time=.5,control=24,value=0),dict(time=.55,control=24,value=127)]
        _,p,_,_,_=plan([note(64,off=1.5)],3,c,'xeremia',controls)
        self.assertAlmostEqual(p[round(1.4*48000)],1)

    def test_midi_sustain_ignored_and_final_release_required(self):
        midi=mido.MidiFile()
        track=mido.MidiTrack();midi.tracks.append(track)
        track.extend([mido.Message('note_on',note=86,velocity=90),
                      mido.Message('control_change',control=64,value=127),
                      mido.Message('note_off',note=86,time=480)])
        rows,controls=midi_controls(midi,0)
        self.assertAlmostEqual(rows[0]['off'],.5)
        track.pop()
        with self.assertRaises(ValueError):midi_controls(midi,0)

    def test_manifest_tracks_do_not_share_noise(self):
        import json
        import tempfile
        from unittest.mock import patch
        import soundfile as sf
        from soundbench.scripts.render_winds import main
        midi = mido.MidiFile()
        for name in ('first', 'second'):
            track = mido.MidiTrack([mido.MetaMessage('track_name', name=name),
                mido.Message('note_on', note=86, velocity=90),
                mido.Message('note_off', note=86, time=960)])
            midi.tracks.append(track)
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            source = d/'notes.mid'; midi.save(source)
            manifest = d/'manifest.json'
            manifest.write_text(json.dumps(dict(performed_midi=str(source),
                jobs=[dict(index=i, output=str(d/f'{i}.wav')) for i in range(2)])))
            with patch.object(sys, 'argv', ['render_winds', str(manifest)]):
                main('flabiol')
            a, _ = sf.read(d/'0.wav'); b, _ = sf.read(d/'1.wav')
            self.assertFalse(np.array_equal(a, b))

    def test_registry_and_config_provenance(self):
        midi=mido.MidiFile();t=mido.MidiTrack();midi.tracks.append(t)
        t.name='wind:wind';t.append(mido.Message('note_on',note=86,velocity=90))
        for inst in ['flabiol','xeremia']:
            b=resolve_instruments(midi,{'instruments':{'wind':{'model':inst}}})
            self.assertEqual(b[0].model,inst)
            self.assertTrue(RENDERERS[inst][0].exists())
            _,_,a,_=selected_config({},inst)
            _,_,b,_=selected_config({f'{inst}_model':{'output_gain':.5}},inst)
            self.assertNotEqual(a,b)

    def test_bag_pressure_wander_is_shared_and_seeded(self):
        c=load_config('xeremia');c['tremor_cents']=0;rows=[note(64,time=.3,off=12)]
        f,p,_,df,_=plan(rows,14,c,'xeremia')
        cents=1200*np.log2(f/frequency(64,c))
        drone_cents=1200*np.log2(df/frequency(48,c))
        self.assertLess(np.max(abs(cents[14400:]-drone_cents[14400:])),1e-9)
        span=slice(2*48000,11*48000)
        self.assertGreater(np.ptp(cents[span]),3)
        self.assertLess(np.ptp(cents[span]),10)
        self.assertGreater(np.ptp(p[span]),.01)
        self.assertTrue(np.array_equal(p,plan(rows,14,c,'xeremia')[1]))
        plain=clean('xeremia')
        controls=[dict(time=2,control=2,value=102)]
        f,_,_,df,_=plan(rows,14,plain,'xeremia',controls)
        self.assertAlmostEqual(1200*np.log2(f[144000]/frequency(64,plain)),11.25)
        self.assertAlmostEqual(1200*np.log2(df[144000]/frequency(48,plain)),11.25)

    def test_bag_tremor_flutters_both_pipes(self):
        from scipy.ndimage import uniform_filter1d
        c=load_config('xeremia');rows=[note(64,time=.3,off=12)]
        f,_,_,df,_=plan(rows,14,c,'xeremia')
        cents=1200*np.log2(f/frequency(64,c))[2*48000:11*48000]
        # Fast flutter: what is left after a one-second moving average.
        fast=cents-uniform_filter1d(cents,48000)
        self.assertAlmostEqual(np.std(fast)/c['tremor_cents'],1,delta=.35)
        drone=1200*np.log2(df/frequency(48,c))[2*48000:11*48000]
        self.assertLess(np.max(abs(cents-drone)),1e-9)
        c['tremor_cents']=0
        still=1200*np.log2(plan(rows,14,c,'xeremia')[0]/frequency(64,c))[2*48000:11*48000]
        self.assertLess(np.std(still-uniform_filter1d(still,48000)),.5)

    def test_flute_breath_flutter(self):
        from scipy.ndimage import uniform_filter1d
        c=load_config('flabiol');rows=[note(86,time=.1,off=8)]
        f=plan(rows,9,c,'flabiol')[0]
        cents=1200*np.log2(f/frequency(86,c))[48000:7*48000]
        self.assertAlmostEqual(np.std(cents-uniform_filter1d(cents,48000))/c['tremor_cents'],1,delta=.35)
        c['tremor_cents']=0
        self.assertLess(np.ptp(1200*np.log2(plan(rows,9,c,'flabiol')[0]/frequency(86,c))[48000:7*48000]),1e-9)

    def test_reed_noise_floor_and_radiation(self):
        from scipy.signal import stft, welch
        def measure(**changes):
            c=load_config('xeremia');c.update(changes)
            y=render([note(64,time=.3,off=3.5)],4.5,c,'xeremia')[round(1.2*48000):round(3.4*48000)]
            f,_,z=stft(y,48000,nperseg=9600,noverlap=4800)
            db=20*np.log10(abs(z)+1e-12)
            floors=[]
            for lo,hi in ((300,1000),(1000,2000),(2000,4000)):
                band=(f>=lo)&(f<hi)
                floors.append(np.median(np.percentile(db[band],99,axis=0)-np.percentile(db[band],20,axis=0)))
            hz,power=welch(y,48000,nperseg=4096)
            level=lambda lo,hi:10*np.log10(power[(hz>=lo)&(hz<hi)].sum()/power[(hz>=150)&(hz<2000)].sum())
            crest=20*np.log10(np.max(abs(y))/np.sqrt(np.mean(y*y)))
            return np.array(floors),level(2000,4000),np.sqrt(np.mean(y*y)),crest,level(4000,8000)
        # The reed formant is tested on its own below.
        raw=measure(formant_db=0);smooth=measure(reed_turbulence=0,formant_db=0);dull=measure(radiation=0,formant_db=0)
        # A recorded Galician gaita measures 51/42/35 dB from harmonic peaks to
        # the floor in 0.3–1/1–2/2–4 kHz; the turbulence-free reed measured 72/61/57.
        for got,low,high in zip(raw[0],(45,35,27),(62,50,42)):
            self.assertTrue(low<got<high,raw[0])
        self.assertTrue(np.all(smooth[0]>raw[0]+8),smooth[0]-raw[0])
        # Open-hole radiation lifts 2–4 kHz a few dB, at about the same
        # loudness and crest factor: reedier, not spikier.
        self.assertTrue(2<raw[1]-dull[1]<8,raw[1]-dull[1])
        self.assertLess(abs(20*np.log10(raw[2]/dull[2])),1.5)
        self.assertLess(raw[3]-dull[3],1.5)
        # The lift stays modest above 4 kHz to limit high-frequency hiss.
        self.assertLess(raw[4]-dull[4],5.5,raw[4]-dull[4])

    def test_reed_formant_gives_the_nasal_core_without_the_top(self):
        from scipy.signal import welch
        def bands(**changes):
            c=load_config('xeremia');c.update(changes)
            y=render([note(64,time=.3,off=3.5)],4.5,c,'xeremia')[round(1.2*48000):round(3.4*48000)]
            hz,power=welch(y,48000,nperseg=8192)
            level=lambda lo,hi:10*np.log10(power[(hz>=lo)&(hz<hi)].sum())
            crest=20*np.log10(np.max(abs(y))/np.sqrt(np.mean(y*y)))
            return level(2000,3000)-level(250,1000),level(4000,8000)-level(250,1000),np.sqrt(np.mean(y*y)),crest
        core,plain=bands(),bands(formant_db=0)
        # Recorded xeremiers sit 1–4 dB under their 250–1000 Hz level at 2–3 kHz.
        self.assertGreater(core[0]-plain[0],4)
        self.assertLess(core[1]-plain[1],2)
        self.assertLess(abs(20*np.log10(core[2]/plain[2])),1.5)
        self.assertLess(core[3]-plain[3],2.5)

    def test_turbulence_range_keeps_the_reed_driven(self):
        c=load_config('xeremia');c['reed_turbulence']=.6
        y=render([note(64,time=.3,off=3)],4,c,'xeremia')
        self.assertTrue(np.all(np.isfinite(y)))
        import tempfile,os
        with tempfile.TemporaryDirectory() as d:
            path=os.path.join(d,'x.toml')
            text=(ROOT/'presets/xeremia/default.toml').read_text()
            open(path,'w').write(text+'\nreed_turbulence = 0.7\n')
            with self.assertRaises(ValueError):
                load_config('xeremia',path)

    def test_bag_tremor_is_heard_in_the_chanter(self):
        def spread(tremor):
            c=load_config('xeremia');c.update(drone_gain=0,wobble_cents=0,tremor_cents=tremor)
            y=render([note(64,time=.3,off=9)],10,c,'xeremia')
            found=[pitch(y[round((t-.25)*48000):round((t+.25)*48000)],48000,frequency(64,c))[0] for t in np.arange(1.5,8.5,.1)]
            return np.std(1200*np.log2(np.array(found)/np.median(found)))
        self.assertGreater(spread(6),2*spread(0))

    def test_repeated_notes_get_a_cut(self):
        c=load_config('xeremia');c['tremor_cents']=c['wobble_cents']=0
        rows=[note(64,off=1.5),note(64,time=1.5,off=2)]
        f=plan(rows,3,c,'xeremia')[0]
        at=lambda t:f[round(t*48000)]
        # A whole-tone cut from above for 30 ms, then the repeated note.
        self.assertAlmostEqual(1200*np.log2(at(1.51)/at(1.55)),1200*np.log2(frequency(66,c)/frequency(64,c)),delta=1)
        self.assertAlmostEqual(at(1.55)/at(1.45),1,delta=1e-3)
        self.assertAlmostEqual(at(1.532)/at(1.45),1,delta=1e-3)
        # A 40 ms repeat keeps its second half at its own pitch.
        f=plan([note(64,off=1.5),note(64,time=1.5,off=1.54),note(65,time=1.54,off=2)],3,c,'xeremia')[0]
        self.assertGreater(at(1.51)/f[round(1.45*48000)],1.05)
        self.assertAlmostEqual(f[round(1.525*48000)]/f[round(1.45*48000)],1,delta=1e-3)
        # At the top of the compass the cut falls back below the compass edge.
        top=c['pitch_range'][1]
        f=plan([note(top,off=1.5),note(top,time=1.5,off=2)],3,c,'xeremia')[0]
        self.assertLess(f[round(1.51*48000)],f[round(1.45*48000)])
        c['repeat_cut_ms']=0
        f=plan(rows,3,c,'xeremia')[0]
        self.assertAlmostEqual(f[round(1.51*48000)]/f[round(1.45*48000)],1,delta=1e-3)

    def test_chanter_starts_on_its_first_note(self):
        c=load_config('xeremia');c['squawk_cents']=0
        f=plan([note(72,time=.3,off=2)],3,c,'xeremia')[0]
        self.assertAlmostEqual(f[0]/f[round(.31*48000)],1,delta=.01)

    def test_startup_rises_audibly_from_below(self):
        c=load_config('xeremia');rows=[note(64,time=.3,off=2)]
        f,p,_,_,_,transient=plan(rows,3.5,c,'xeremia',details=True)
        self.assertLess(1200*np.log2(f[round(.4*48000)]/frequency(64,c)),-100)
        self.assertLess(abs(1200*np.log2(f[round(1.1*48000)]/frequency(64,c))),3)
        self.assertGreater(p[round(.4*48000)],.8)
        self.assertGreater(transient[round(.4*48000)],.5)
        y=render(rows,3.5,c,'xeremia')
        self.assertGreater(np.sqrt(np.mean(y[round(.45*48000):round(.65*48000)]**2)),.005)

    def test_flute_breath_floor_and_tongued_chiff(self):
        c=load_config('flabiol')
        ratios=[]
        for p in [74,86,98]:
            rows=[note(p,off=2,velocity=80)]
            noisy=render(rows,2.3,c)
            dry=dict(c,breath_noise=0,chiff_noise=0)
            tone=render(rows,2.3,dry)
            span=slice(24000,72000)
            ratio=20*np.log10(np.sqrt(np.mean((noisy[span]-tone[span])**2)/np.mean(tone[span]**2)))
            ratios.append(ratio)
        self.assertTrue(-30 < ratios[0] < -25,ratios)
        self.assertTrue(-30 < ratios[1] < -25,ratios)
        self.assertTrue(-26 < ratios[2] < -22,ratios)
        self.assertTrue(np.all(np.diff(ratios)>0))
        rows=[note(86,off=.3),note(88,time=.3,off=.7)]
        *_,tongued=plan(rows,1.8,c,'flabiol',details=True)
        *_,slurred=plan(rows,1.8,c,'flabiol',[dict(time=0,control=68,value=127)],details=True)
        self.assertGreater(max(tongued[14400:16000]),.5)
        self.assertEqual(max(slurred[14400:16000]),0)

    def test_invalid_inputs(self):
        for rows in [[note(60)], [note(off=0)], [note(),note()]]:
            with self.assertRaises(ValueError):render(rows,2)
        with self.assertRaises(ValueError):
            render([note()],2,controls=[dict(time=0,control=21,value=127)])
        with self.assertRaises(ValueError):
            selected_config({'flabiol_model':{'output_gain':float('nan')}},'flabiol')
        self.assertEqual(max(abs(render([],1))),0)
        self.assertEqual(max(abs(render([],1,load_config('xeremia'),'xeremia'))),0)


if __name__=='__main__':unittest.main()
