#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["mido>=1.3,<2", "numpy>=1.26", "numba>=0.61", "scipy>=1.11", "soundfile>=0.12", "tomli-w>=1.0"]
# ///
import copy
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

import mido
import numpy as np
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1] / "soundbench"
sys.path.insert(0, str(ROOT/'scripts'))
from soundbench.scripts.models.guitar import load_config, configure, assign_string, render, note, strum
from soundbench.scripts.render_guitar import schedule
from soundbench.scripts.instrumentation import resolve_instruments
from soundbench.scripts.models.guitar_release import finish, sustain_metrics
from guitar_studies import steel_demo, nylon_demo


class GuitarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.configs = [load_config(ROOT/f'presets/guitar/{p}.toml') for p in ('default', 'nylon')]

    def test_tuning_all_strings_and_registers(self):
        errors = []
        for config in self.configs:
            for string in range(1, 7):
                for fret in (0, 7, 19):
                    pitch = config['strings']['open_midi'][string-1]+fret
                    y = render(note(0, 1.2, pitch, 100, string, 1), 1.4, config=config)[0]
                    n = 262144
                    segment = y[4800:48000]
                    spectrum = abs(np.fft.rfft(segment*np.hanning(len(segment)), n))
                    f = np.fft.rfftfreq(n, 1/48000)
                    target = 440*2**((pitch-69)/12)
                    selected = np.flatnonzero((f > target*.97) & (f < target*1.03))
                    k = selected[np.argmax(spectrum[selected])]
                    log = np.log(spectrum[k-1:k+2]+1e-30)
                    offset = .5*(log[0]-log[2])/(log[0]-2*log[1]+log[2])
                    measured = (k+offset)*48000/n
                    cents = 1200*np.log2(measured/target)
                    errors.append(float(cents))
                    self.assertLess(abs(cents), 5, (string, fret, cents))
        print(f'Guitar tuning: max {max(abs(x) for x in errors):.3f} cents across {len(errors)} probes', file=sys.stderr)

    def test_release_determinism_and_finite(self):
        for c in self.configs:
            events = finish(note(0, .6, 52, 127, 6, 123), .8)
            a = render(events, 2, config=c)
            self.assertTrue(np.array_equal(a, render(events, 2, config=c)))
            self.assertTrue(np.isfinite(a).all())
            self.assertLess(np.max(abs(a)), .98)
            self.assertGreater(np.sqrt(np.mean(a[:, :24000]**2)), 1e-5)
            self.assertLess(np.sqrt(np.mean(a[:, -4800:]**2)), 1e-4)
            self.assertLess(np.sqrt(np.mean(a[:, -4800:]**2)), .001*np.sqrt(np.mean(a[:, :24000]**2)))
            d = copy.deepcopy(c)
            d['excitation']['jitter'] = .2
            b = render(events, 2, config=d)
            d['engine']['seed'] += 1
            self.assertFalse(np.array_equal(b, render(events, 2, config=d)))

    def test_velocity_continuity(self):
        c = self.configs[0]
        energy = []
        for v in range(1, 128):
            y = render(note(0, .1, 64, v, 1, 1), .15, config=c)
            energy.append(np.sqrt(np.mean(y*y)))
        self.assertTrue(np.all(np.diff(energy) > 0))
        step = max(np.diff(20*np.log10(energy[39:])))
        self.assertLess(step, .5)
        print(f'Guitar velocity: max adjacent step {step:.3f} dB for velocities 40..127', file=sys.stderr)

    def test_strum_order_and_releases(self):
        c = self.configs[0]
        for direction in ('up', 'down'):
            rows = strum(0, .5, [64, 59, 55, 50, 45, 40], c,
                         strings=[1, 2, 3, 4, 5, 6], direction=direction, speed_ms=80, fingers=4, reach=6)
            starts = [r for r in rows if r[1] == 0]
            self.assertEqual(len(starts), 24)
            self.assertEqual(starts[0][2], 5 if direction == 'down' else 0)
            self.assertEqual(starts[5][0]-starts[0][0], 3840)
            self.assertEqual({r[10] for r in starts}, {r[10] for r in rows if r[1] == 1})
            y = render(rows, 2, config=c)
            self.assertTrue(np.isfinite(y).all())
            self.assertLess(np.max(abs(y)), .98)
        with self.assertRaises(ValueError):
            strum(0,.5,[64],c,strings=[1],fingers=1.5)

    def test_rasgueado_sweeps_do_not_collide_or_interleave(self):
        for speed in (10, 35, 80):
            for direction in ('up', 'down'):
                for count in (1, 6):
                    rows = strum(0, .5, [64, 59, 55, 50, 45, 40][:count], self.configs[0],
                                 strings=list(range(1, count+1)), direction=direction, speed_ms=speed, fingers=4, reach=count)
                    attacks = [r for r in rows if r[1] == 0]
                    self.assertEqual(len({(r[0], r[2]) for r in attacks}), len(attacks))
                    for finger in range(3):
                        sweep = attacks[finger*count:(finger+1)*count]
                        following = attacks[(finger+1)*count:(finger+2)*count]
                        self.assertLess(sweep[-1][0], following[0][0])
                        self.assertEqual([r[2] for r in sweep], [r[2] for r in following][::-1])

    def test_assignment_and_tunings(self):
        c = self.configs[0]
        self.assertEqual(assign_string(40, c), 5)
        with self.assertRaises(ValueError):
            assign_string(40, c, 1)
        self.assertEqual(configure(c, 'dadgad', 2)['strings']['open_midi'], [64, 59, 57, 52, 47, 40])
        self.assertEqual(c['strings']['open_midi'], [64, 59, 55, 50, 45, 40])
        with self.assertRaises(ValueError):
            configure(c, 'unknown')

    def fixture(self, pitches=(55, 59, 64)):
        midi = mido.MidiFile()
        track = mido.MidiTrack()
        midi.tracks.append(track)
        track.append(mido.MetaMessage('track_name', name='guitar'))
        for p in pitches:
            track.append(mido.Message('note_on', note=p, velocity=90))
        for i, p in enumerate(pitches):
            track.append(mido.Message('note_off', note=p, time=480 if i == 0 else 0))
        return midi

    def test_schedule_and_registry(self):
        midi = self.fixture()
        settings = {'instruments': {'guitar': {'model': 'guitar'}},
                    'guitar_articulation': [dict(track='guitar', bar=1, kind='rasgueado')]}
        self.assertEqual(resolve_instruments(midi, settings)[0].model, 'guitar')
        events, seconds, assignment = schedule(midi, midi, settings, [0], self.configs[1])
        self.assertEqual(sum(r[1] == 0 for r in events), 9)
        self.assertEqual(len(assignment), 3)
        self.assertGreater(seconds, .5)
        self.assertEqual(len({r[10] for r in events if r[1] == 0}), 9)

    def test_manifest_renderer(self):
        (ROOT/'build').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'build') as directory:
            d = Path(directory)
            midi = d/'guitar.mid'
            self.fixture().save(midi)
            for preset in ('default', 'nylon'):
                manifest = d/f'{preset}.json'
                output = d/f'{preset}.wav'
                manifest.write_text(json.dumps(dict(performed_midi=str(midi), score_midi=str(midi),
                    settings={'guitar_model': {'preset':preset}}, jobs=[dict(index=0,output=str(output))])))
                subprocess.run([sys.executable,str(ROOT/'scripts/render_guitar.py'),str(manifest)],check=True,stdout=subprocess.DEVNULL)
                self.assertTrue(output.exists())
                metrics = json.loads(manifest.with_suffix('.metrics.json').read_text())
                self.assertEqual(metrics['stems']['0']['model'], 'guitar')
                self.assertEqual(metrics['stems']['0']['parameters'], str(ROOT/f'presets/guitar/{preset}.toml'))
                self.assertLess(metrics['peak'], 1)

    def test_legato_validation_and_nylon_bend_rejection(self):
        midi = self.fixture((64,))
        settings = {'guitar_strings': {'guitar':1},
                    'guitar_articulation':[dict(track='guitar',bar=1,kind='hammer')]}
        with self.assertRaises(ValueError):
            schedule(midi,midi,settings,[0],self.configs[0])
        settings['guitar_articulation'][0] = dict(track='guitar',bar=1,kind='bend',to=.5,duration_beats=.25)
        with self.assertRaises(ValueError):
            schedule(midi,midi,settings,[0],self.configs[1])

    def test_stale_release_does_not_kill_destination(self):
        for c in self.configs:
            rows = note(0,.35,64,95,1,1)+note(.2,.8,67,95,1,2,kind=3)
            y = render(finish(rows,1.),1.5,config=c)[0]
            self.assertGreater(np.sqrt(np.mean(y[24000:28800]**2)), 1e-4)
            self.assertLess(np.sqrt(np.mean(y[-4800:]**2)), 1e-4)

    def test_open_ring_and_explicit_damp(self):
        for c in self.configs:
            rows = note(0,.1,64,95,1,1)
            a = render(rows,1.5,config=c)[0]
            b = render(finish(rows,.2),1.5,config=c)[0]
            self.assertGreater(np.sqrt(np.mean(a[24000:28800]**2)), .005)
            self.assertLess(np.sqrt(np.mean(b[24000:28800]**2)), 1e-4)
            # A MIDI off after a choke must not undo the choke.
            crows = finish(rows,.05)
            self.assertTrue(np.array_equal(render(crows,1.5,config=c),render([r for r in crows if r[1]!=1],1.5,config=c)))

    def test_seeded_slide_contact_noise(self):
        c = self.configs[0]
        sample = 4800
        token = next(t for t in range(100) if np.random.default_rng(np.random.SeedSequence([c['engine']['seed'],t,sample])).random() < .35)
        rows = note(0,.5,64,95,1,token)+[[sample,2,0,0,69,0,14400,1,0,0,token]]
        a = render(rows,1.,config=c)
        self.assertTrue(np.array_equal(a,render(rows,1.,config=c)))
        silent = copy.deepcopy(c)
        silent['excitation']['slide_noise'] = 0
        self.assertFalse(np.array_equal(a,render(rows,1.,config=silent)))

    def test_track_and_strum_let_ring(self):
        midi = self.fixture((65,))
        for value in (True,False):
            settings = {'tracks':{'guitar':{'let_ring':value}}}
            rows, seconds, _ = schedule(midi,midi,settings,[0],self.configs[0])
            self.assertEqual(next(r[8] for r in rows if r[1]==1), int(value))
        rows,_,_ = schedule(midi,midi,{'guitar_articulation':[dict(track='guitar',bar=1,kind='strum')]},[0],self.configs[0])
        self.assertEqual(next(r[8] for r in rows if r[1]==1),1)

    def test_let_ring_honours_harmony_release(self):
        midi = self.fixture((48,))
        track = midi.tracks[0]
        track.insert(1, mido.Message('control_change', control=64, value=127))
        track.append(mido.Message('control_change', control=64, value=0, time=480))
        settings = {'tracks': {'guitar': {'let_ring': True}}, 'guitar_strings': {'guitar': 5}}
        rows, _, _ = schedule(midi, midi, settings, [0], self.configs[0])
        release = next(r for r in rows if r[1] == 1)
        self.assertEqual(release[0], 48000)
        for kind in ('strum', 'rasgueado'):
            chord = self.fixture((56, 60, 65))
            chord.tracks[0].insert(1, mido.Message('control_change', control=64, value=127))
            chord.tracks[0].append(mido.Message('control_change', control=64, value=0, time=480))
            options = {'tracks': {'guitar': {'let_ring': True}},
                       'guitar_articulation': [dict(track='guitar', bar=1, kind=kind,
                                                   strings=[1, 2, 3], speed_ms=80, fingers=4)]}
            swept, _, _ = schedule(chord, chord, options, [0], self.configs[0])
            self.assertTrue(all(r[0] == 48000 for r in swept if r[1] == 1))
        for c in self.configs:
            ringing = note(0, 1, 48, 95, 5, 1, let_ring=True)
            ordinary = note(0, 1, 48, 95, 5, 1)
            y = render(ringing, 3, config=c)
            np.testing.assert_array_equal(y, render(ordinary, 3, config=c))
            self.assertGreater(np.sqrt(np.mean(y[:, 24000:40000]**2)), 1e-4)
            self.assertLess(np.sqrt(np.mean(y[:, -4800:]**2)), 1e-5)
            opened = note(0, .1, 45, 95, 5, 1, let_ring=True)
            self.assertGreater(np.sqrt(np.mean(render(opened, 2, config=c)[:, 48000:60000]**2)), 1e-4)

    def test_musical_sustain_gate(self):
        for c, demo in zip(self.configs,(steel_demo,nylon_demo)):
            rows, seconds = demo(c)
            y = render(finish(rows,24.8),seconds,config=c)[0]
            metrics = sustain_metrics(y)
            print(f'{demo.__name__}: {metrics}',file=sys.stderr)
            # A boom-chuck intentionally damps between strokes; do not erase it
            # to claim continuous sustain. Nylon fingerpicking has no dead gaps.
            if demo == nylon_demo:
                self.assertEqual(metrics['count'],0)
            attacks = sorted(r[0]/48000 for r in rows if r[1] in (0, 3, 4))
            quiet_windows = [(r[0]/48000, next((t for t in attacks if t > r[0]/48000+.05), seconds))
                             for r in rows if r[1] == 5]
            for a, b in metrics['dead_gaps']:
                self.assertTrue(any(start <= a and b <= end for start, end in quiet_windows), (a, b))


    def test_damping_and_restrike_are_click_free(self):
        # A choke, a finger release, a refret and a restrike change the string
        # without a step. Against the counterfactual (the old note and the new
        # pluck rendered apart), the change adds almost no energy above 8 kHz
        # in the 2 ms around the event. Clearing the string, or stepping the
        # whole period's loss in at once, scores 1 to 1e4 here.
        from scipy.signal import butter, sosfiltfilt
        high = lambda y: sosfiltfilt(butter(4, 8000, 'hp', fs=48000, output='sos'), y)
        for c in self.configs:
            lead = c['strings']['refret_ms']/1000
            for pitch, string in ((40, 6), (50, 4), (59, 2), (64, 1)):
                old, new = note(0, 2, pitch, 95, string, 1), note(.6, 2, pitch, 1, string, 2)
                moved = note(.6, 2, pitch+2, 1, string, 2)
                cases = {
                    'choke': (old+[[28800, 5, string-1, 0, 0, 0, 0, 0, 0, 0, 1]], [old], .6),
                    'release': (note(0, .6, pitch+2, 95, string, 1), [note(0, 2, pitch+2, 95, string, 1)], .6),
                    'restrike': (old+new, [old, new], .6),
                    'refret': (old+moved, [old, moved], .6-lead),
                }
                for name, (rows, parts, at) in cases.items():
                    apart = sum(render(r, 1, config=c)[0] for r in parts)
                    window = slice(round((at-.0005)*48000), round((at+.0015)*48000))
                    added = high(render(rows, 1, config=c)[0]-apart)[window]
                    # An 80 dB floor under the note keeps a dark, decayed string from dividing by zero.
                    reference = np.sum(high(apart)[window]**2)+1e-8*np.sum(apart[window]**2)
                    ratio = np.sum(added**2)/reference
                    self.assertLess(ratio, .3, (name, pitch, ratio))

    def test_extreme_damping_stays_finite(self):
        # A damp so fast its per-period gain underflows (T60 * f < 0.0093 on
        # low E) must not poison the render, nor a later note on another string.
        for c in self.configs:
            fast = copy.deepcopy(c)
            fast['strings']['choke_seconds'] = 1e-4
            rows = note(0, 1, 40, 100, 6, 1)+[[4800, 5, 5, 0, 0, 0, 0, 0, 0, 0, 1]]+note(.25, 1, 84, 100, 1, 2)
            y = render(rows, .5, config=fast)
            self.assertTrue(np.isfinite(y).all())
            self.assertGreater(np.sqrt(np.mean(y[0, 12000:]**2)), 1e-5)
        with self.assertRaises(ValueError):
            strum(0, .5, [64, 59], self.configs[0], strings=[1, 2], muted=1e-4)
        with self.assertRaises(ValueError):
            render([[0, 5, 0, 0, 0, 0, 0, 0, 0, 1e-4, 1]], .1, config=self.configs[0])

    def test_restrike_is_as_loud_as_a_fresh_pluck(self):
        # The pick contact spans less than a round trip even at the top frets,
        # and a pick lifts a choke at once, so the new pulse is never damped.
        for c in self.configs:
            for string, pitch in ((1, 76), (1, 81), (1, 84), (4, 53), (6, 43)):
                for choke in (.05, .455):
                    old = note(0, .05, pitch, 95, string, 1)+[[round(choke*48000), 5, string-1, 0, 0, 0, 0, 0, 0, 0, 1]]
                    fresh = render(note(.5, 1, pitch, 95, string, 2), 1, config=c)[0]
                    again = render(old+note(.5, 1, pitch, 95, string, 2), 1, config=c)[0]
                    window = slice(24240, 26400)
                    ratio = 20*np.log10(np.sqrt(np.mean(again[window]**2))/np.sqrt(np.mean(fresh[window]**2)))
                    self.assertLess(abs(ratio), .2, (string, pitch, choke, ratio))

    def test_restrike_adds_to_the_moving_string(self):
        for c in self.configs:
            base = note(0, 2, 64, 95, 1, 1)
            zero = note(.2, 2, 64, 0, 1, 2)
            whole = copy.deepcopy(c)
            whole['excitation']['restrike_keep'] = 1.
            np.testing.assert_allclose(render(base, 1, config=whole), render(base+zero, 1, config=whole), atol=1e-12, rtol=0)
            # The pick's contact takes some of the old motion: quieter, never louder.
            a, b = render(base, 1, config=c)[0], render(base+zero, 1, config=c)[0]
            after = slice(round(.25*48000), round(.6*48000))
            self.assertLess(np.sqrt(np.mean(b[after]**2)), np.sqrt(np.mean(a[after]**2)))
            self.assertGreater(np.sqrt(np.mean(b[after]**2)), .3*np.sqrt(np.mean(a[after]**2)))

    def test_refret_damps_the_old_pitch_before_the_pick(self):
        c = self.configs[0]
        rows = note(0, 2, 45, 95, 5, 1)+note(.6, 2, 48, 90, 5, 2)
        late = copy.deepcopy(c)
        late['strings']['refret_ms'] = 0.
        level = lambda y: np.sqrt(np.mean(y[round(.59*48000):round(.6*48000)]**2))
        damped, ringing = level(render(rows, 1, config=c)[0]), level(render(rows, 1, config=late)[0])
        self.assertLess(20*np.log10(damped/ringing), -6)

    def test_stroke_follows_the_hand(self):
        c = self.configs[0]
        chord = dict(pitches=[67, 59, 55, 50, 47, 43], strings=[1, 2, 3, 4, 5, 6])
        def spans(**kw):
            out = []
            for token in range(0, 600, 30):
                rows = strum(0, .5, chord['pitches'], c, strings=chord['strings'], token=token, **kw)
                ons = sorted(r[0] for r in rows if r[1] == 0)
                out.append((ons[-1]-ons[0])/48)
            return np.array(out)
        mf = spans(velocity=80, tempo_bpm=100)
        self.assertTrue(15 < np.median(mf) < 40)
        self.assertGreater(np.std(mf), 1)
        self.assertLess(np.median(spans(velocity=120, tempo_bpm=100)), np.median(mf))
        self.assertLess(np.median(spans(velocity=80, tempo_bpm=160)), np.median(mf))
        # Fewer strings, shorter stroke: the hand's speed is what the player sets.
        rows = strum(0, .5, [67, 59, 55], c, strings=[1, 2, 3], velocity=80, humanize=False)
        ons = sorted(r[0] for r in rows if r[1] == 0)
        full = strum(0, .5, chord['pitches'], c, strings=chord['strings'], velocity=80, humanize=False)
        full_ons = sorted(r[0] for r in full if r[1] == 0)
        self.assertAlmostEqual((ons[-1]-ons[0])/(full_ons[-1]-full_ons[0]), 2/5, delta=.02)
        # Downstrokes linger after the heavy strings and lose force across the stroke.
        gaps = np.diff(full_ons)
        self.assertGreater(gaps[0], gaps[-1])
        velocities = [r[5] for r in sorted(full, key=lambda r: r[0]) if r[1] == 0]
        self.assertGreater(velocities[0], velocities[-1])
        for token in range(20):
            self.assertEqual(strum(0, .5, chord['pitches'], c, strings=chord['strings'], token=token),
                             strum(0, .5, chord['pitches'], c, strings=chord['strings'], token=token))
        up = strum(0, .5, chord['pitches'], c, strings=chord['strings'], direction='up')
        self.assertIn(sum(r[1] == 0 for r in up), (3, 4))
        self.assertTrue(all(r[2] <= 3 for r in up))
        self.assertEqual(sorted(r[2] for r in up if r[1] == 0)[0], 0)

    def test_muted_strokes(self):
        c = self.configs[0]
        p, s = [67, 59, 55, 50], [1, 2, 3, 4]
        ring = render(strum(0, .5, p, c, strings=s, token=1), .6, config=c)[0]
        chuck = render(strum(0, .5, p, c, strings=s, token=1, muted=True), .6, config=c)[0]
        palm = render(strum(0, .5, p, c, strings=s, token=1, muted=.15), .6, config=c)[0]
        late = slice(round(.25*48000), round(.45*48000))
        level = lambda y: np.sqrt(np.mean(y[late]**2))
        self.assertLess(level(chuck), level(palm))
        self.assertLess(level(palm), .3*level(ring))
        with self.assertRaises(ValueError):
            strum(0, .5, p, c, strings=s, muted=3)
        chord = self.fixture((55, 59, 67))
        for art in (dict(kind='strum', mute=True), dict(kind='strum', mute=3), dict(kind='chuck', mute=.1),
                    dict(kind='pluck', reach=2), dict(kind='strum', reach=4)):
            with self.assertRaises(ValueError, msg=art):
                schedule(chord, chord, {'guitar_articulation': [dict(track='guitar', bar=1, **art)]}, [0], c)

    def test_a_slide_keeps_the_string_damped(self):
        # A slide moves the fretting finger; it does not lift a damp already on the string.
        for c in self.configs:
            held = note(0, 1, 52, 95, 4, 1)
            damp = [[9600, 5, 3, 0, 0, 0, 0, 0, 0, .04, 1]]
            slide = [[9840, 2, 3, 0, 54, 0, 2400, 1, 0, 0, 1]]
            late = slice(24000, 28800)
            level = lambda rows: np.sqrt(np.mean(render(rows, .6, config=c)[0][late]**2))
            # The floor is the open strings' sympathetic ring, which no damp on string 4 reaches.
            self.assertLess(level(held+damp+slide), 1.05*level(held+damp))
            self.assertLess(level(held+damp), 1e-2*level(held))

    def test_steel_partials_are_stretched_like_stiff_strings(self):
        def stretch(c, pitch=64, string=1, n=20):
            y = render(note(0, 2, pitch, 95, string, 1), 1, config=c)[0][2400:38400]
            spectrum = abs(np.fft.rfft(y*np.hanning(len(y)), 1 << 20))
            f = np.fft.rfftfreq(1 << 20, 1/48000)
            f0 = 440*2**((pitch-69)/12)
            peak = lambda k: f[np.argmax(spectrum*((f > k*f0*.99) & (f < k*f0*1.03)))]
            return 1200*np.log2(peak(n)/(n*peak(1)))
        steel, nylon = self.configs
        B = steel['strings']['inharmonicity'][0]
        expected = 1200*np.log2(np.sqrt(1+B*400)/np.sqrt(1+B))
        self.assertAlmostEqual(stretch(steel), expected, delta=.4*expected)
        self.assertLess(abs(stretch(nylon)), .6*expected)

    def test_bad_events(self):
        for events in ([[0, 0, 6, 0, 64, 90, 0, 0, 0, 0, 1]],
                       [[0, 0, 0, 0, 64, 128, 0, 0, 0, 0, 1]],
                       [[0, 0, 0, 0, 64, 90, 0, 0, 0, 0, 1.5]]):
            with self.assertRaises(ValueError):
                render(events, .1, config=self.configs[0])


if __name__ == '__main__':
    unittest.main()
