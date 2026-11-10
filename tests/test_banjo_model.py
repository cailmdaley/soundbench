#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["mido>=1.3,<2", "numpy>=1.26", "numba>=0.61", "scipy>=1.11"]
# ///
"""Audio-domain pitch, gesture, physical voice and numerical regression tests."""
import copy
import sys
import re
import tempfile
import unittest
import json
import hashlib
import wave
import subprocess
from pathlib import Path
import numpy as np
from scipy.signal import freqz, lfilter
from scipy.optimize import brentq
from scipy.special import jn_zeros, jv

ROOT = Path(__file__).resolve().parents[1] / "soundbench"


def package_build():
    """A scratch directory inside the package, where bundled-path labels apply."""
    (ROOT / "build").mkdir(exist_ok=True)
    return ROOT / "build"
sys.path.insert(0, str(ROOT / "scripts"))
from soundbench.scripts.paths import preset_bytes, preset_path
from soundbench.scripts.models.banjo import _coefficients, assign_string, load_config, membrane_modes, modal_coefficients, render as render_banjo

CONFIG = preset_path("banjo", "membrane")


def render(events, seconds, owners=1, config=None):
    """The banjo engine under its hand-set membrane preset unless told otherwise."""
    return render_banjo(events, seconds, owners, config or load_config(CONFIG))
from soundbench.scripts.render_model import assignment_statistics, schedule
from soundbench.scripts import instrumentation
import mido
from soundbench.scripts.audio_pitch import pitch_cents


def event(time, kind=0, string=0, pitch=62, token=0, duration=0, owner=0):
    return [round(time*48000), kind, string, owner, pitch, 90, duration*48000, 1, 0, 0, token]


def midi_tracks(*tracks):
    midi = mido.MidiFile(ticks_per_beat=480)
    midi.tracks.append(mido.MidiTrack([mido.MetaMessage('set_tempo', tempo=500000)]))
    for name, notes in tracks:
        rows = [(0, mido.MetaMessage('track_name', name=name))]
        for start, end, pitch in notes:
            rows.extend([(start, mido.Message('note_on', note=pitch, velocity=90)),
                         (end, mido.Message('note_off', note=pitch))])
        track, last = mido.MidiTrack(), 0
        for tick, message in sorted(rows, key=lambda r: r[0]):
            track.append(message.copy(time=tick-last))
            last = tick
        midi.tracks.append(track)
    return midi


class BanjoTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(CONFIG)

    def test_every_playable_pitch_within_three_cents(self):
        results = []
        for string, open_pitch in enumerate(self.config["strings"]["open_midi"]):
            for fret in range(self.config["strings"]["max_fret"]+1):
                pitch = open_pitch + fret
                y = render([event(0, string=string, pitch=pitch)], .65, config=self.config)[0]
                cents = pitch_cents(y[4800:24000], pitch)
                results.append(abs(cents))
                with self.subTest(string=string+1, fret=fret, cents=cents):
                    self.assertLess(abs(cents), 3)
        print(f"125 string/fret audio pitches: max error {max(results):.3f} cents")

    def test_finite_no_subnormals_and_deterministic(self):
        events = [event(i*.06, string=i%5, pitch=[62,59,55,50,67][i%5]+i%8, token=i) for i in range(80)]
        a, b = render(events, 8), render(events, 8)
        self.assertTrue(np.array_equal(a, b))
        self.assertTrue(np.isfinite(a).all())
        self.assertFalse(np.any((np.abs(a)>0) & (np.abs(a)<np.finfo(a.dtype).tiny)))
        c = copy.deepcopy(self.config)
        c["engine"]["seed"] += 1
        self.assertFalse(np.array_equal(a, render(events, 8, config=c)))

    def test_bend_reaches_target_and_slide_crosses_delay_integers(self):
        for semitones in (1, 5):
            y = render([event(0, pitch=62), event(.03, 2, pitch=62+semitones, duration=.25)], .8)[0]
            cents = pitch_cents(y[round(.36*48000):round(.62*48000)], 62+semitones)
            self.assertLess(abs(cents), 3)
            self.assertLess(np.max(np.abs(np.diff(y))), .2)

    def test_thiran_fraction_is_centered_and_smooth_at_integer_delays(self):
        def coefficients(pitch):
            integer, a, _ = _coefficients(pitch, 48000, .144, -.1, 2.2)
            w = 2*np.pi*440*2**((pitch-69)/12)/48000
            fraction = 2*np.arctan((1-a)/(1+a)*np.tan(w/2))/w
            return integer, a, fraction
        for pitch in np.linspace(50, 86, 1000):
            _, _, fraction = coefficients(pitch)
            self.assertGreaterEqual(fraction, .5-1e-10)
            self.assertLess(fraction, 1.5)
        pitch = brentq(lambda p: sum(coefficients(p)[::2])-200, 40, 80)
        left, right = coefficients(pitch-1e-5), coefficients(pitch+1e-5)
        self.assertEqual(left[0], right[0])
        self.assertLess(abs(left[1]-right[1]), .001)

    def test_voice_stealing_and_stale_note_off(self):
        c = copy.deepcopy(self.config)
        c["bridge"]["coupling"] = c["bridge"]["head_feedback"] = 0
        c["pickups"]["head"] = 0
        first = event(0, pitch=62)
        second = event(.25, pitch=67, token=1)
        a = render([first, second], .9, config=c)
        b = render([second], .9, config=c)
        self.assertTrue(np.array_equal(a[:,12000:], b[:,12000:]))
        stale = render([first, second, event(.3,1,token=0)], .9, config=c)
        self.assertTrue(np.array_equal(a, stale))

    def test_sympathetic_open_strings(self):
        c = copy.deepcopy(self.config)
        y = render([event(0, string=0, pitch=67, owner=1)], .6, 2, c)
        self.assertGreater(np.linalg.norm(y[0]), 0)
        ratio_db = 20*np.log10(np.linalg.norm(y[0,3840:19200])/np.linalg.norm(y[1,3840:19200]))
        self.assertGreater(ratio_db, -40)
        self.assertLess(ratio_db, -30)
        c["bridge"]["coupling"] = c["bridge"]["head_feedback"] = 0
        self.assertEqual(np.linalg.norm(render([event(0, pitch=67, owner=1)], .6, 2, c)[0]), 0)

    def test_hammer_pull_targets_and_choke(self):
        y = render([event(0, pitch=62), event(.25, 3, pitch=64, token=1),
                    event(.6, 4, pitch=62, token=2)], 1)[0]
        self.assertLess(abs(pitch_cents(y[16800:25920], 64)), 3)
        self.assertLess(abs(pitch_cents(y[33600:43200], 62)), 3)
        plain = render([event(0)], .8)[0]
        choked = render([event(0), event(.25, 5)], .8)[0]
        # A choke damps one string; the sympathetically excited strings keep ringing.
        self.assertLess(np.linalg.norm(choked[24000:]), np.linalg.norm(plain[24000:]) * .032)

    def test_gestures_use_global_tokens_and_do_not_choke_same_sample_retrigger(self):
        midi = midi_tracks(('banjolead', [(0, 480, 62), (480, 960, 64)]))
        settings = {'banjo_strings': {'banjolead': 1}, 'articulation': [
            {'track': 'banjolead', 'bar': 1, 'kind': 'bend', 'to': 1, 'duration_beats': .5},
            {'track': 'banjolead', 'bar': 1, 'beat': 2, 'kind': 'choke'}]}
        rows, _, _ = schedule(midi, midi, settings, [1], self.config)
        for row in rows:
            if row[1] in (2, 5):
                self.assertEqual(row[10], 1000000)
        c = copy.deepcopy(self.config)
        c['bridge']['coupling'] = c['bridge']['head_feedback'] = 0
        c['pickups']['head'] = 0
        first, second = event(0), event(.25, pitch=64, token=1)
        plain = render([first, second], .8, config=c)
        stale = render([first, second, event(.25, 5)], .8, config=c)
        self.assertTrue(np.array_equal(plain, stale))
        stale_bend = render([first, second, event(.3, 2, pitch=67)], .8, config=c)
        self.assertTrue(np.array_equal(plain, stale_bend))

    def test_hammer_source_is_same_string_not_midi_order(self):
        midi = midi_tracks(('banjolead', [(0, 480, 62), (0, 480, 55), (480, 960, 64)]))
        settings = {'articulation': [{'track': 'banjolead', 'bar': 1, 'beat': 2, 'kind': 'hammer'}]}
        _, _, assignment = schedule(midi, midi, settings, [1], self.config)
        self.assertEqual([r['string'] for r in assignment], [1, 3, 1])

    def test_score_positions_follow_performed_timing_and_pedal(self):
        def midi(first, second, pedal=False):
            result = mido.MidiFile(ticks_per_beat=480)
            result.tracks.append(mido.MidiTrack([mido.MetaMessage('set_tempo',tempo=500000)]))
            events = [(0,mido.MetaMessage('track_name',name='banjolead')),
                      (first,mido.Message('note_on',note=62,velocity=90)),
                      (first+120,mido.Message('note_off',note=62)),
                      (second,mido.Message('note_on',note=64,velocity=90)),
                      (second+120,mido.Message('note_off',note=64))]
            if pedal:
                events += [(0,mido.Message('control_change',control=64,value=127)),
                           (second+240,mido.Message('control_change',control=64,value=0))]
            track = mido.MidiTrack()
            last = 0
            for tick,msg in sorted(events,key=lambda r:r[0]):
                track.append(msg.copy(time=tick-last)); last=tick
            result.tracks.append(track)
            return result
        settings = {'articulation':[{'track':'banjolead','bar':1,'kind':'bend','to':1,'duration_beats':.5}]}
        rows,_,_ = schedule(midi(24,984,True),midi(0,480),settings,[1],self.config)
        bend = next(r for r in rows if r[1]==2)
        self.assertEqual(bend[0], 1200)
        self.assertEqual(bend[6], 24000)
        off = next(r for r in rows if r[1]==1)
        self.assertEqual(off[0], 61200)

    def test_pitch_estimator_measures_detuning(self):
        frequency = 440 * 2**((62-69)/12) * 2**(20/1200)
        sine = np.sin(2*np.pi*frequency*np.arange(19200)/48000)
        self.assertAlmostEqual(pitch_cents(sine,62), 20, delta=.2)

    def test_parameter_schema_rejects_typos_and_nonfinite_values(self):
        text = CONFIG.read_text()
        for broken in (text.replace('head_feedback', 'head_feedbak'),
                       text.replace('output_gain = 12.0', 'output_gain = nan'),
                       re.sub(r'(?m)^pick_ms = .*$', 'pick_ms = 0.0', text),
                       re.sub(r'(?m)^bridge_radius = .*$', 'bridge_radius = 1.5', text)):
            with tempfile.TemporaryDirectory(dir=package_build()) as directory:
                path = Path(directory)/'banjo.toml'
                path.write_text(broken)
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_renderer_accepts_parameter_file_and_records_its_hash(self):
        with tempfile.TemporaryDirectory(dir=package_build()) as directory:
            path = Path(directory)
            config_path = path/'instrument.toml'
            config_path.write_text(CONFIG.read_text().replace('output_gain = 12.0', 'output_gain = 6.0'))
            midi = path/'note.mid'
            midi_tracks(('banjo', [(0, 480, 62)])).save(midi)
            manifest = path/'render.json'
            manifest.write_text(json.dumps(dict(performed_midi=str(midi), score_midi=str(midi),
                                                settings={'banjo_model': {'preset': str(config_path)}},
                                                jobs=[dict(index=1, output=str(path/'note.wav'))])))
            subprocess.run([sys.executable, str(ROOT/'scripts/render_model.py'), str(manifest)], check=True, capture_output=True)
            metrics = json.loads(manifest.with_suffix('.metrics.json').read_text())
            self.assertEqual(metrics['stems']['1']['parameters'], str(config_path.resolve().relative_to(ROOT)))
            self.assertEqual(metrics['stems']['1']['parameters_sha256'], hashlib.sha256(preset_bytes(config_path)).hexdigest())
            self.assertTrue((path/'note.wav').is_file())

    def test_pluck_fingering_overrides_a_track_pin_without_changing_pitch(self):
        mid=midi_tracks(('banjolead',[(0,480,50),(480,960,61),(960,1440,50),(1440,1920,62)]))
        settings={'banjo_strings':{'banjolead':1},'articulation':[
            {'track':'banjolead','bar':1,'beat':1,'kind':'pluck','string':4},
            {'track':'banjolead','bar':1,'beat':2,'kind':'bend','string':2,'to':1,'duration_beats':1.5},
            {'track':'banjolead','bar':1,'beat':3,'kind':'pluck','string':4}]}
        _,_,assignments=schedule(mid,mid,settings,[1],self.config)
        self.assertEqual([a['string'] for a in assignments],[4,2,4,1])
        self.assertEqual([a['pitch'] for a in assignments],[50,61,50,62])

    def test_selected_gut_config_and_gain_override_provenance(self):
        config_path = preset_path('banjo')
        options = {'output_gain': 3.0}
        with tempfile.TemporaryDirectory(dir=package_build()) as directory:
            path=Path(directory); midi=path/'note.mid'
            midi_tracks(('banjo', [(0,480,62)])).save(midi)
            manifest=path/'render.json'
            manifest.write_text(json.dumps(dict(performed_midi=str(midi), score_midi=str(midi),
                                                settings={'banjo_model':{'preset':'default', **options}}, jobs=[dict(index=1,output=str(path/'note.wav'))])))
            subprocess.run([sys.executable,str(ROOT/'scripts/render_model.py'),str(manifest)],check=True,capture_output=True)
            stem=json.loads(manifest.with_suffix('.metrics.json').read_text())['stems']['1']
            self.assertEqual(stem['parameters'],'presets/banjo/default.toml')
            self.assertEqual(stem['overrides'],options)
            self.assertEqual(stem['parameters_sha256'],hashlib.sha256(preset_bytes(config_path)+json.dumps(options,sort_keys=True).encode()).hexdigest())
            self.assertTrue((path/'note.wav').is_file())
            with wave.open(str(path/'note.wav')) as w:
                quiet=np.frombuffer(w.readframes(w.getnframes()),dtype='<i4').copy()
            data=json.loads(manifest.read_text())
            data['settings']['banjo_model']['output_gain']=6.0
            manifest.write_text(json.dumps(data))
            subprocess.run([sys.executable,str(ROOT/'scripts/render_model.py'),str(manifest)],check=True,capture_output=True)
            with wave.open(str(path/'note.wav')) as w:
                louder=np.frombuffer(w.readframes(w.getnframes()),dtype='<i4').copy()
            np.testing.assert_allclose(louder.astype(float),2*quiet.astype(float),rtol=0,atol=1)

    def test_default_output_matches_recorded_audio_fingerprint(self):
        evidence=json.loads((ROOT/'references/banjo-fit/engine/default-regression.json').read_text())
        self.assertEqual(hashlib.sha256(CONFIG.read_bytes()).hexdigest(),evidence['config_sha256'])
        audio=render(evidence['events'],evidence['seconds'],config=self.config)
        self.assertEqual(hashlib.sha256(audio.tobytes()).hexdigest(),evidence['sample_sha256'])

    def test_modal_gain_reference_does_not_depend_on_bank_size(self):
        full = modal_coefficients(self.config)
        for count in (1, 8, 40):
            small = copy.deepcopy(self.config)
            for key in ('ratios', 'weights', 't60_seconds'):
                small['head'][key] = small['head'][key][:count]
            subset = modal_coefficients(small)
            for a, b in zip(full, subset):
                np.testing.assert_array_equal(a[:count], b[:count])
                self.assertEqual(a[-1], b[-1])
        self.assertAlmostEqual(full[-1][-1] / full[-1][0], self.config['pot']['weight'])

    def test_force_driven_modal_peak_and_energy_scale_with_t60(self):
        changed = copy.deepcopy(self.config)
        changed['head']['t60_seconds'][1] *= 2
        responses, energies = [], []
        doubled = copy.deepcopy(self.config)
        doubled['head']['t60_seconds'] = [t*2 for t in doubled['head']['t60_seconds']]
        doubled['pot']['t60_seconds'] *= 2
        for config, factor in ((self.config, 1), (doubled, 2)):
            c, r2, gain, _ = modal_coefficients(config)
            angle = 2*np.pi*config['head']['fundamental_hz']/48000
            _, response = freqz([gain[0]], [1, -c[0], r2[0]], worN=[angle])
            self.assertAlmostEqual(abs(response[0]), factor*.5, places=8)
        for config in (self.config, changed):
            c, r2, gain, _ = modal_coefficients(config)
            angle = 2*np.pi*config['head']['fundamental_hz']*config['head']['ratios'][1]/48000
            _, response = freqz([gain[1]], [1, -c[1], r2[1]], worN=[angle])
            responses.append(abs(response[0]))
            impulse = np.zeros(48000)
            impulse[0] = 1
            energies.append(np.sum(lfilter([gain[1]], [1, -c[1], r2[1]], impulse)**2))
        self.assertAlmostEqual(responses[1]/responses[0], 2, places=8)
        self.assertAlmostEqual(energies[1]/energies[0], 2, delta=.01)

    def test_bridge_drive_uses_normalized_shape_and_multipole_radiation(self):
        head = dict(fundamental_hz=80, bridge_radius=.65, max_hz=6000,
                    t60_fundamental=.12, damping_exponent=.5, weight_tilt=0)
        ratios, weights, _ = membrane_modes(head)
        j01, j11 = jn_zeros(0, 1)[0], jn_zeros(1, 1)[0]
        fundamental_drive = abs(jv(0, j01*.65)/jv(1, j01))
        ratio = j11/j01
        ka = 2*np.pi*80*ratio*.14/343
        expected = abs(jv(1, j11*.65)/jv(2, j11))*ka/np.sqrt(1+ka**2)/fundamental_drive
        i = np.argmin(np.abs(np.array(ratios)-ratio))
        self.assertAlmostEqual(weights[i], expected, places=10)
        self.assertEqual(weights[0], 1)

    def test_membrane_head_follows_bessel_zeros(self):
        ratios = self.config['head']['ratios']
        for expected in (1.0, 1.5933, 2.1355, 2.2954, 2.6531, 2.9173):
            self.assertTrue(any(abs(r - expected) < 1e-3 for r in ratios), expected)
        self.assertTrue(any(r * self.config['head']['fundamental_hz'] > 6000 for r in ratios))
        self.assertTrue(all(r * self.config['head']['fundamental_hz'] <= 12000 for r in ratios))

    def test_membrane_bounds_include_high_orders_and_radial_roots(self):
        head = dict(fundamental_hz=100, bridge_radius=.65, max_hz=6000,
                    t60_fundamental=.12, damping_exponent=.5, weight_tilt=0)
        ratios, weights, _ = membrane_modes(head)
        j01 = jn_zeros(0, 1)[0]
        for zero in (jn_zeros(45, 1)[0], jn_zeros(0, 50)[-1]):
            self.assertTrue(any(abs(r-zero/j01) < 1e-9 for r in ratios))
        # Modes below the knee retain exactly the same drive in a wider bank.
        wide = dict(head, max_hz=9000)
        wide_ratios, wide_weights, _ = membrane_modes(wide)
        for ratio, weight in zip(ratios, weights):
            if 5500 < ratio*100 < 6000:
                i = np.argmin(np.abs(np.array(wide_ratios)-ratio))
                self.assertAlmostEqual(weight, wide_weights[i], places=12)
            if ratio*100 > 11000:
                i = np.argmin(np.abs(np.array(wide_ratios)-ratio))
                self.assertLess(weight, wide_weights[i]*.25)

    def test_assignment_respects_ringing_across_tracks_and_steals_oldest(self):
        midi = midi_tracks(('banjolead', [(0, 2400, 62)]),
                           ('banjoroll', [(240*i, 2400, 62) for i in range(1, 5)]))
        events, _, assignment = schedule(midi, midi, {}, [1, 2], self.config)
        self.assertEqual([r['string'] for r in assignment], [1, 2, 3, 4, 1])
        stats = assignment_statistics(events)
        self.assertEqual(stats['cut_before_note_off'], 1)
        self.assertEqual(stats['median_lost_seconds'], 1.5)
        _, _, pinned = schedule(midi, midi, {'banjo_strings': {'banjoroll': 1}}, [1, 2], self.config)
        self.assertEqual([r['string'] for r in pinned], [1]*5)
        free = [True, False, False, False, False]
        self.assertEqual(assign_string(62, False, self.config, sounding=free,
                         started=[0]*5, last_frets=[0]*5), 1)
        # A one-fret advantage wins even against a moderate hand-position jump.
        self.assertEqual(assign_string(62, False, self.config, sounding=[False]*5,
                         started=[0]*5, last_frets=[4, 3, 7, 12, 0]), 0)

    def test_pedal_protects_released_note_and_hammer_keeps_actual_source_string(self):
        midi = midi_tracks(('otherbanjo', [(0, 1440, 62)]),
                           ('banjolead', [(120, 240, 62), (480, 960, 64)]))
        track = midi.tracks[2]
        track.insert(1, mido.Message('control_change', control=64, value=127))
        track.append(mido.Message('control_change', control=64, value=0, time=480))
        _, _, ordinary = schedule(midi, midi, {}, [1, 2], self.config)
        self.assertEqual([r['string'] for r in ordinary], [1, 2, 3])
        settings = {'articulation': [{'track': 'banjolead', 'bar': 1, 'beat': 2, 'kind': 'hammer'}]}
        _, _, legato = schedule(midi, midi, settings, [1, 2], self.config)
        self.assertEqual([r['string'] for r in legato], [1, 2, 2])
        settings['articulation'][0] = {'track': 'banjolead', 'bar': 1, 'beat': 2,
                                      'kind': 'slide', 'to': 1, 'duration_beats': .5, 'string': 4}
        rows, _, explicit = schedule(midi, midi, settings, [1, 2], self.config)
        self.assertEqual(explicit[-1]['string'], 4)
        self.assertEqual(next(r for r in rows if r[1] == 2)[2], 3)

    def test_pull_below_open_d_plans_source_on_b_string(self):
        midi = midi_tracks(('banjolead', [(0, 240, 62), (240, 480, 60), (480, 720, 59)]))
        settings = {'articulation': [{'track': 'banjolead', 'bar': 1, 'beat': 1.5, 'kind': 'pull'}]}
        _, _, assignment = schedule(midi, midi, settings, [1], self.config)
        self.assertEqual([a['string'] for a in assignment[:2]], [2, 2])
        self.assertEqual(assignment[1]['kind'], 4)
        self.assertFalse(any('dropped' in a for a in assignment))

    def test_assignment_and_model_binding(self):
        self.assertEqual(assign_string(67, True, self.config), 4)
        self.assertEqual(assign_string(67, False, self.config), 0)
        self.assertEqual(assign_string(55, False, self.config), 2)
        midi = mido.MidiFile()
        midi.tracks.append(mido.MidiTrack([mido.MetaMessage('track_name',name='banjo:lead'), mido.Message('note_on',note=62)]))
        midi.tracks.append(mido.MidiTrack([mido.MetaMessage('track_name',name='thumb:thumb'), mido.MetaMessage('instrument_name',name='banjo'), mido.Message('note_on',note=67)]))
        bindings = instrumentation.resolve_instruments(midi, {'instruments': {'banjo': {'model':'banjo'}}})
        self.assertEqual(bindings[1].model, 'banjo')
        self.assertEqual(bindings[1].instrument, 'banjo-thumb')
        midi.tracks[1] = mido.MidiTrack(m for m in midi.tracks[1] if m.type != 'instrument_name')
        self.assertEqual(instrumentation.resolve_instruments(midi, {'instruments':{'banjo':{'model':'banjo'}}})[1].model, 'banjo')
        with self.assertRaises(ValueError):
            instrumentation.resolve_instruments(midi, {'instruments':{'banjo':'banjo'}})

    def test_schema_and_hammer_pull(self):
        midi = mido.MidiFile()
        midi.tracks.append(mido.MidiTrack([mido.MetaMessage('time_signature',numerator=4,denominator=4)]))
        track = mido.MidiTrack([mido.MetaMessage('track_name',name='banjolead')])
        for pitch in (62,64,62):
            track.extend([mido.Message('note_on',note=pitch,velocity=90),mido.Message('note_off',note=pitch,time=480)])
        midi.tracks.append(track)
        settings = {'articulation':[{'track':'banjolead','bar':1,'beat':2,'kind':'hammer'},
                                    {'track':'banjolead','bar':1,'beat':3,'kind':'pull'}]}
        rows, seconds, assignment = schedule(midi,midi,settings,[1],self.config)
        self.assertEqual([a['kind'] for a in assignment],[0,3,4])
        self.assertEqual([a['string'] for a in assignment],[1,1,1])
        self.assertTrue(np.isfinite(render(rows,seconds)).all())
        with self.assertRaises(ValueError):
            schedule(midi,midi,{'articulation':[{'track':'wrong','bar':1,'kind':'bend'}]},[1],self.config)
        # a hammer whose source note is gone (here: a hammer up from a higher note) is plucked
        rising = {'articulation':[{'track':'banjolead','bar':1,'beat':3,'kind':'hammer'}]}
        _, _, assignment = schedule(midi,midi,rising,[1],self.config)
        self.assertEqual([a['kind'] for a in assignment],[0,0,0])
        self.assertEqual([a.get('dropped') for a in assignment],[None,None,'hammer'])


if __name__ == '__main__':
    unittest.main()
