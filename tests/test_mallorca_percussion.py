#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["mido>=1.3,<2", "numpy>=1.26", "numba>=0.61", "scipy>=1.11"]
# ///
"""Audio and routing invariants for the Mallorcan percussion engines."""
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
from scipy.signal import correlate, find_peaks, welch
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[1] / "soundbench"
sys.path.insert(0, str(ROOT / "scripts"))
from soundbench.scripts.models.mallorca_percussion import KEYS, hand_profile, load_config, modal_signal, render, tambori_modes, ximbomba_stroke
from soundbench.scripts.render_mallorca_percussion import gestures, selected_config
from soundbench.scripts.instrumentation import resolve_instruments
from soundbench.scripts.model_dispatch import render_model_stems


def note(pitch, velocity=80, off=.3, time=.05):
    return dict(time=time, off=off, pitch=pitch, velocity=velocity)


XIMBOMBAS = ("ximbomba", "ximbomba-fitted", "ximbomba-austria")


class PercussionTests(unittest.TestCase):
    def test_all_gestures_finite_and_released(self):
        for model, keys in KEYS.items():
            c = load_config(model)
            for pitch in keys:
                with self.subTest(model=model, pitch=pitch):
                    y = render([note(pitch)], 2, c)
                    self.assertTrue(np.all(np.isfinite(y)))
                    self.assertLess(np.max(abs(y)), 1)
                    self.assertGreater(np.max(abs(y)), 1e-6)
                    self.assertLess(np.max(abs(y[-2400:])), np.max(abs(y)) * 1e-4)
                    self.assertLess(np.max(abs(y[48000:50400])), np.max(abs(y)) * 1e-4)

    def test_determinism(self):
        for model in KEYS:
            c = load_config(model)
            rows = [note(KEYS[model][0]), note(KEYS[model][-1], time=.5, off=.8)]
            np.testing.assert_array_equal(render(rows, 2.3, c), render(rows, 2.3, c))

    def test_seed_changes_buzz_and_dancers(self):
        for model in ("tambori", "castanyoles"):
            c = load_config(model)
            if model == "castanyoles":
                c["body"].update(dancers=4, timing_ms=6, tuning_cents=12)
            a = render([note(KEYS[model][0])], 2, c)
            c["engine"]["seed"] += 1
            self.assertGreater(np.max(abs(a-render([note(KEYS[model][0])], 2, c))), 1e-8)

    def test_track_and_note_seeds_keep_edits_local(self):
        for model in KEYS:
            c = load_config(model)
            if model == 'castanyoles':
                c['body'].update(dancers=3, timing_ms=4, tuning_cents=8)
            first = note(KEYS[model][0], time=.05, off=.3)
            later = note(KEYS[model][0], time=.8, off=1.1)
            y = render([first, later], 2.8, c, track_id=1)
            baseline = render([first], 2.8, c, track_id=1)
            solo = render([later], 2.8, c, track_id=1)
            np.testing.assert_allclose(y-baseline, solo, atol=1e-12, rtol=1e-9)
            np.testing.assert_array_equal(y, render([first, later], 2.8, c, track_id=1))
            self.assertFalse(np.array_equal(y, render([first, later], 2.8, c, track_id=2)))
            changed = dict(first, velocity=45, off=.5)
            np.testing.assert_allclose(render([changed, later], 2.8, c, track_id=1)-
                render([changed], 2.8, c, track_id=1), solo, atol=1e-12, rtol=1e-9)

    def test_modal_tuning_in_cents(self):
        # Measure implemented modal poles, not an arbitrary MIDI pitch claim.
        f = 331.7
        force = np.zeros(96000)
        force[0] = 1
        y = modal_signal(force, [f], [.6], [1])
        spec = abs(np.fft.rfft(y, n=2**20))
        k = np.argmax(spec)
        measured = k*48000/2**20
        self.assertLess(abs(1200*np.log2(measured/f)), 1)
        for model in ("tambori", "castanyoles"):
            c = load_config(model)
            if model == "tambori":
                frequencies, times, weights, _ = tambori_modes(c)
                k = np.argmax(np.abs(weights))
                f = frequencies[k]
            else:
                f = c["body"]["frequencies"][0]
            y = modal_signal(force, [f], [.3], [1])
            measured = np.argmax(abs(np.fft.rfft(y, n=2**20)))*48000/2**20
            self.assertLess(abs(1200*np.log2(measured/f)), 1)

    def test_velocity_continuity_and_fixed_gain(self):
        for model in ("tambori", "castanyoles"):
            c = load_config(model)
            values = [np.sqrt(np.mean(render([note(KEYS[model][0], v)], 1.9, c)**2)) for v in range(1, 128)]
            self.assertTrue(np.all(np.diff(values) > 0))
            ratios = np.array(values[1:])/values[:-1]
            self.assertLess(max(ratios[30:]), 1.08)
            a = render([note(KEYS[model][0])], 2.1, c)
            b = render([note(KEYS[model][0]), note(KEYS[model][0], time=1, off=1.2)], 2.1, c)
            np.testing.assert_array_equal(a[:48000], b[:48000])

    def test_roll_obeys_length(self):
        for model, pitch in (("tambori", 40), ("castanyoles", 62)):
            c = load_config(model)
            short = render([note(pitch, off=.12)], 2, c)
            long = render([note(pitch, off=.5)], 2, c)
            self.assertGreater(np.sum(long[12000:24000]**2), np.sum(short[12000:24000]**2)*3)

    def test_mechanical_step_preserves_energy_and_physical_decay(self):
        from soundbench.scripts.models.mallorca_percussion import _mechanical_stepper
        sr = 48000
        for hz in (46., 72.4, 166., 500.):
            omega = 2*np.pi*hz
            mass = np.array([[.005]])
            stiffness = mass*omega**2
            for t60 in (np.inf, .144):
                damping = mass*2*np.log(1000)/t60
                vv, vq, response = _mechanical_stepper(mass, stiffness, damping, sr, np.ones(1))
                dt = 1/sr
                # The same q/v trapezoidal update used by the friction loop.
                update = np.array([[1+dt*vq[0,0]/2, dt*(1+vv[0,0])/2],
                                   [vq[0,0], vv[0,0]]])
                radius = max(abs(np.linalg.eigvals(update)))
                if np.isinf(t60):
                    self.assertAlmostEqual(radius, 1., delta=1e-12)
                    state = np.array([1., 0.])
                    state = np.linalg.matrix_power(update, 48000) @ state
                    energy = mass[0,0]*state[1]**2+stiffness[0,0]*state[0]**2
                    self.assertAlmostEqual(energy/stiffness[0,0], 1., delta=1e-9)
                else:
                    actual = -np.log(1000)/(sr*np.log(radius))
                    self.assertAlmostEqual(actual/t60, 1., delta=.002)

    def test_friction_drum_modes_load_the_head(self):
        from soundbench.scripts.models.mallorca_percussion import friction_drum_modes
        for name in XIMBOMBAS:
            b = load_config("ximbomba", ROOT/f"instruments/models/{name}.toml")["body"]
            m, k, damping, volume, drive, hz = friction_drum_modes(b)
            n = b["modes"]
            self.assertAlmostEqual(hz[0], b["fundamental_hz"], delta=.01)
            self.assertEqual(len(hz), n)
            self.assertTrue(np.all(np.diff(hz) > 0))
            # The cane's mass couples every mode; the bare head would be diagonal.
            self.assertTrue(np.all(m[:n, :n] > 0))
            self.assertAlmostEqual(volume[0], 1.)
            # The hole's air plug radiates but the cane does not drive it.
            self.assertEqual((volume[n], drive[n]), (1., 0.))
            if b["pot_coupling"] > 0:
                # Air sloshing through the hole cancels the head well below pot_hz.
                w = 2*np.pi*b["pot_hz"]/8
                h = np.linalg.solve(k-w*w*m+1j*w*damping, drive)
                self.assertLess(abs(volume @ h), .5*abs(volume[:n] @ np.linalg.solve(
                    k[:n, :n]-w*w*m[:n, :n]+1j*w*damping[:n, :n], drive[:n])))

    def test_pressure_direction_and_duration(self):
        c = load_config("ximbomba")
        a = render([dict(note(48), pressure=.7)], 2, c)
        b = render([dict(note(49), pressure=.7)], 2, c)
        self.assertGreater(np.linalg.norm(a+b)/np.linalg.norm(a), .3)
        quiet = render([dict(note(48), pressure=0)], 2, c)
        self.assertEqual(float(np.max(abs(quiet))), 0)
        long = render([dict(note(48, off=.7), pressure=.7)], 2.3, c)
        self.assertGreater(np.sum(long[18000:30000]**2), np.sum(a[18000:30000]**2)*2)
        changing = render([dict(note(48, off=.7), pressure=.7, pressure_events=[(.2, 0)])], 2.3, c)
        self.assertLess(np.sum(changing[18000:30000]**2), np.sum(long[18000:30000]**2))

    def test_ximbomba_stick_slip_groans_at_its_fundamental(self):
        # A friction drum speaks as a pitched groan; a train of separate slip
        # clicks is the crackle this engine replaced.
        for filename in XIMBOMBAS:
            c = load_config("ximbomba", ROOT/f"instruments/models/{filename}.toml")
            y, d = ximbomba_stroke(note(48, off=.7, time=.05), c, diagnostics=True)
            held = round(.65*48000)
            expected, grips = hand_profile(c["body"], .65, 48000)
            # An up stroke speeds up along its length by ±direction_asymmetry.
            expected = expected*(1+c["body"]["direction_asymmetry"]*(2*np.arange(held)/held-1))
            hand = np.abs(d["hand_velocity"][:held])
            self.assertGreater(np.corrcoef(hand, expected)[0, 1], .9999)
            self.assertEqual(np.max(abs(d["hand_velocity"][held:])), 0)
            self.assertEqual(np.count_nonzero(d["slip_velocity"][held:]), 0)
            # Stick and slip alternate: neither regime holds the whole stroke.
            middle = d["slip_velocity"][round(.25*held):round(.75*held)]
            self.assertTrue(.1 < np.mean(middle > 0) < .9)
            period = 48000/c["body"]["fundamental_hz"]
            lags = np.arange(round(period*.7), round(period*1.7))
            strength, ratio = [], []
            for a in range(round(.25*held), round(.75*held)-2400, 2400):
                frame = y[a:a+2400]-y[a:a+2400].mean()
                ac = correlate(frame, frame, method="fft")[len(frame)-1:]
                lag = lags[np.argmax(ac[lags])]
                strength.append(ac[lag]/ac[0]); ratio.append(lag/period)
            # 50 ms frames repeat at about the head's period: sticking lengthens
            # a cycle, and a hoarse drum can flip toward a higher mode. Measured
            # medians are 0.90–1.10 of the head's period across both presets.
            self.assertGreater(np.median(strength), .25)
            self.assertTrue(.85 < np.median(ratio) < 1.25, ratio)
            z = y[round(.25*held):round(.75*held)]
            self.assertLess(np.max(abs(z))/np.sqrt(np.mean(z*z)), 5)
            # Most energy sits in the groan's harmonics, below 1 kHz.
            hz, power = welch(z, 48000, nperseg=4096)
            self.assertGreater(power[hz < 1000].sum()/power.sum(), .9)
            bins = np.array_split(y[:held], 10)
            power = np.array([np.mean(x*x) for x in bins])
            if grips:
                # A rubbing hand keeps the groan up to the turn.
                self.assertLess(np.max(power[1:9])/np.min(power[1:9]), 10)
            else:
                # A lifting hand speaks early, as players dig in, and falls away by the stroke's end.
                self.assertGreater(np.argmax(power), 0)
                self.assertLess(np.argmax(power), 5)
                self.assertGreater(np.max(power), power[9]*2)
            silent, sd = ximbomba_stroke(dict(note(48), pressure=0), c, diagnostics=True)
            self.assertEqual(np.max(abs(silent)), 0)
            self.assertEqual(np.count_nonzero(sd["slip_velocity"]), 0)
            c["engine"]["seed"] += 1
            changed = ximbomba_stroke(note(48, off=.7, time=.05), c)
            self.assertGreater(np.linalg.norm(y-changed)/np.linalg.norm(y), .1)

    def test_ximbomba_runs_join_on_the_latest_off(self):
        from soundbench.scripts.models.mallorca_percussion import ximbomba_chains
        long, short, after = note(48, off=2., time=0.), note(49, off=.7, time=.5), note(48, off=1., time=.8)
        self.assertEqual(len(ximbomba_chains([long, short, after])), 1)
        self.assertEqual(len(ximbomba_chains([short, note(48, off=1., time=.8)])), 2)

    def test_older_ximbomba_presets_keep_loading(self):
        old = (ROOT/"instruments/models/ximbomba-austria.toml").read_text()
        old = "\n".join(line for line in old.splitlines() if not line.startswith(("pot_", "stroke_peak")))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"old.toml"
            path.write_text(old)
            b = load_config("ximbomba", path)["body"]
            # A closed pot and the symmetric lifting bell, as before either existed.
            self.assertEqual((b["pot_coupling"], b["stroke_peak"]), (0., .5))
            path.write_text(old.replace("roughness", "roughnes"))
            with self.assertRaisesRegex(ValueError, r"missing \['roughness'\], unknown \['roughnes'\]"):
                load_config("ximbomba", path)

    def test_ximbomba_legato_strokes_share_one_head(self):
        # A Mallorcan ximbomba carries one groan across the hand's turns.
        p = .88
        rows = [dict(note(48+i % 2, 80, time=.05+i*p, off=.05+(i+1)*p), pressure=.7) for i in range(6)]
        for name in XIMBOMBAS:
            c = load_config("ximbomba", ROOT/f"instruments/models/{name}.toml")
            joined = render(rows, 6.5, c)
            apart = sum(render([r], 6.5, c) for r in rows)
            self.assertGreater(np.linalg.norm(joined-apart)/np.linalg.norm(apart), .3)
            if "turn_s" in c["body"]:
                z = joined[round(.5*48000):round(5.2*48000)]
                frames = z[:len(z)//480*480].reshape(-1, 480)
                level = 10*np.log10(np.mean(frames**2, axis=1)+1e-20)
                # The recording drops 10 dB below its median in 2–9% of 10 ms frames.
                self.assertLess(np.mean(level < np.median(level)-10), .1, name)

    def test_ximbomba_glosa_has_no_crackle(self):
        beat = 60/54
        rows = [dict(note(48+i % 2, [85, 62, 70][i % 3], time=.2+i*beat, off=.85+i*beat), pressure=[.8, .7, .75][i % 3])
                for i in range(6)]
        for name in XIMBOMBAS:
            c = load_config("ximbomba", ROOT/f"instruments/models/{name}.toml")
            # Roughness is seeded: the groan must hold for any hand, not one lucky seed.
            for seed in (1729, 1730, 1731, 1732):
                c["engine"]["seed"] = seed
                y = render(rows, 8.5, c)
                frames = y[:len(y)//480*480].reshape(-1, 480)
                rms = np.sqrt(np.mean(frames**2, axis=1))
                loud = frames[rms > rms.max()*.1]
                # On this gate the crackling slip-pulse engine measured 20 dB crest, and
                # recorded friction drums (Austrian, Spanish, Mallorcan) 10–14 dB.
                crest = 20*np.log10(np.max(abs(loud))/np.sqrt(np.mean(loud**2)))
                self.assertLess(crest, 17, (name, seed))

    def test_ximbomba_pressure_and_duration_headroom(self):
        for name in XIMBOMBAS:
            c = load_config("ximbomba", ROOT/f"instruments/models/{name}.toml")
            for seed in (1729, 1730):
                c["engine"]["seed"] = seed
                for pressure in (.1, .5, .7, 1.):
                    for duration in (.25, .65):
                        for pitch in (48, 49):
                            y = render([dict(note(pitch, 127, off=.05+duration), pressure=pressure)], duration+1.6, c)
                            self.assertLess(np.max(abs(y)), 1)
                            self.assertTrue(np.all(np.isfinite(y)))

    def test_tambori_low_body_is_coupled_not_band_limited(self):
        c = load_config("tambori")
        f, t, w, _ = tambori_modes(c)
        low = f < 170
        self.assertTrue(np.any(low))
        self.assertGreater(np.max(abs(w[low])), .005)
        bare = copy.deepcopy(c)
        bare["body"]["shell_volume"] = 0
        bf, bt, bw, _ = tambori_modes(bare)
        self.assertLess(np.max(abs(bw[bf < 170])), 1e-10)
        # The low shell responds only through its volume coupling to the heads.
        self.assertGreater(np.linalg.norm(render([note(36)], 2, c)-render([note(36)], 2, bare)), .01)

    def test_fitted_profiles_finite_released_and_velocity_continuous(self):
        for model in ("castanyoles", "ximbomba"):
            c = load_config(model, ROOT/f"instruments/models/{model}-fitted.toml")
            for pitch in KEYS[model]:
                y = render([note(pitch, 127)], 2, c)
                self.assertTrue(np.all(np.isfinite(y)))
                self.assertLess(np.max(abs(y)), 1)
                self.assertLess(np.max(abs(y[-2400:])), np.max(abs(y))*1e-4)
                self.assertLess(np.max(abs(y[48000:50400])), np.max(abs(y))*1e-4)
            levels = []
            for velocity in range(1, 128):
                y = render([note(KEYS[model][0], velocity)], 2, c)
                levels.append(np.sqrt(np.mean(y*y)))
            self.assertTrue(np.all(np.diff(levels) > 0))
            self.assertLess(max(np.array(levels[32:])/levels[31:-1]), 1.08)


    def test_dancers_headroom(self):
        c = load_config("castanyoles", ROOT/"instruments/models/castanyoles-fitted.toml")
        c["body"].update(dancers=6, timing_ms=8, tuning_cents=16)
        y = render([note(64, 127)], 2, c)
        self.assertLess(np.max(abs(y)), 1)
        self.assertGreater(np.max(abs(y)), .05)

    def test_invalid_gestures_and_options(self):
        for model in KEYS:
            c = load_config(model)
            for row in (note(100), note(KEYS[model][0], velocity=0), note(KEYS[model][0], off=.01)):
                with self.assertRaises(ValueError):
                    render([row], 2, c)
        for options in ({"dancers": 0}, {"dancers": 3.5}, {"output_gain": -1}, {"bogus": 1}):
            with self.assertRaises(ValueError):
                selected_config("castanyoles", {"castanyoles_model": options})

    def fixture(self):
        midi = mido.MidiFile()
        midi.tracks.append(mido.MidiTrack([mido.MetaMessage("set_tempo", tempo=500000)]))
        for ch, (model, pitch) in enumerate((("tambori", 36), ("castanyoles", 60), ("ximbomba", 48))):
            midi.tracks.append(mido.MidiTrack([
                mido.MetaMessage("track_name", name=model),
                mido.Message("control_change", channel=ch, control=2, value=64),
                mido.Message("note_on", channel=ch, note=pitch, velocity=70, time=48),
                mido.Message("control_change", channel=ch, control=2, value=100, time=48),
                mido.Message("note_off", channel=ch, note=pitch, time=192),
                mido.MetaMessage("end_of_track", time=480)]))
        return midi

    def test_cc2_and_missing_release(self):
        midi = self.fixture()
        row = gestures(midi, 3)[0]
        self.assertAlmostEqual(row["pressure"], 64/127)
        self.assertAlmostEqual(row["pressure_events"][0][1], 100/127)
        midi.tracks[3] = mido.MidiTrack(m for m in midi.tracks[3] if m.type != "note_off")
        with self.assertRaises(ValueError):
            gestures(midi, 3)

    def test_manifest_tracks_do_not_share_buzz(self):
        from unittest.mock import patch
        from scipy.io import wavfile
        from soundbench.scripts.render_mallorca_percussion import main
        midi = mido.MidiFile()
        for name in ('first', 'second'):
            midi.tracks.append(mido.MidiTrack([mido.MetaMessage('track_name', name=name),
                mido.Message('note_on', note=36, velocity=90),
                mido.Message('note_off', note=36, time=480)]))
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            source = d/'notes.mid'; midi.save(source)
            manifest = d/'manifest.json'
            manifest.write_text(json.dumps(dict(performed_midi=str(source),
                settings={'instruments': {name: {'model': 'tambori'} for name in ('first', 'second')}},
                jobs=[dict(index=i, output=str(d/f'{i}.wav')) for i in range(2)])))
            with patch.object(sys, 'argv', ['render_percussion', str(manifest)]):
                main()
            _, a = wavfile.read(d/'0.wav'); _, b = wavfile.read(d/'1.wav')
            self.assertFalse(np.array_equal(a, b))

    def test_dispatch_real_manifest(self):
        midi = self.fixture()
        settings = dict(instruments={m: dict(model=m) for m in KEYS})
        bindings = resolve_instruments(midi, settings)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            path = d/"probe.mid"
            midi.save(path)
            stems = render_model_stems(path, path, settings, bindings, d, "probe")
            self.assertEqual(len(stems), 3)
            for model in KEYS:
                self.assertTrue((d/f"probe.{model}.metrics.json").exists())
            self.assertEqual(len(list(d.glob("*.raw.wav"))), 3)


if __name__ == "__main__":
    unittest.main()
