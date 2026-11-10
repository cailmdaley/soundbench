#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["mido>=1.3,<2", "numpy>=1.26", "numba>=0.61", "scipy>=1.11"]
# ///
"""Drum-machine circuits, clock and build-path invariants."""
import math
from pathlib import Path
import sys
import tempfile
import unittest

import mido
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "soundbench"
sys.path.insert(0, str(ROOT / "scripts"))
from soundbench.scripts.models.drum_machine import KEYMAP, load_config, render, unit_tolerances
from soundbench.scripts.machine_time import swing_tick, timing
from soundbench.scripts.instrumentation import STEREO_INSTRUMENTS, resolve_instruments
from soundbench.scripts.model_dispatch import RENDERERS
from soundbench.scripts.perform import perform_file
from soundbench.scripts.render_drum_machine import gestures, selected_config

SR = 48000


def hit(p=36, v=96, time=.01, off=.2):
    return dict(pitch=p, velocity=v, time=time, off=off)


def mono(rows, machine="tr808", seconds=4., config=None, **kw):
    return render(rows, seconds, machine, config or load_config(machine), **kw).mean(axis=1)


class Voices(unittest.TestCase):
    def test_every_key_finite_audible_and_decayed(self):
        for machine, keys in KEYMAP.items():
            c = load_config(machine)
            for k in keys:
                with self.subTest(machine=machine, key=k):
                    y = render([hit(k)], 5., machine, c)
                    self.assertTrue(np.isfinite(y).all())
                    self.assertLess(np.max(abs(y)), 1)
                    self.assertGreater(np.max(abs(y)), 1e-3)
                    self.assertLess(np.max(abs(y[-4800:])), 1e-4)

    def test_missing_keys_and_releases_are_errors(self):
        with self.assertRaises(ValueError):
            render([hit(51)], 4., "tr808")
        with self.assertRaises(ValueError):
            render([hit(75)], 4., "tr909")
        with self.assertRaises(ValueError):
            render([hit(36, off=.01)], 4., "tr808")
        midi = mido.MidiFile()
        track = mido.MidiTrack([mido.Message("note_on", note=36, velocity=90, channel=9)])
        midi.tracks.append(track)
        with self.assertRaises(ValueError):
            gestures(midi, 0)

    def test_onsets_land_on_their_sample(self):
        for machine, key in (("tr808", 75), ("tr808", 36), ("tr909", 36), ("linn", 36)):
            t = .123456
            y = mono([hit(key, time=t, off=t+.1)], machine)
            first = np.flatnonzero(abs(y) > 1e-9)[0]
            self.assertLessEqual(abs(first-round(t*SR)), 1, (machine, key))

    def test_seed_moves_noise_and_phases_but_unit_moves_components(self):
        c = load_config("tr808")
        rows = [hit(42), hit(38, time=.5, off=.6)]
        y = mono(rows, config=c)
        np.testing.assert_array_equal(y, mono(rows, config=c))
        other = load_config("tr808", overrides={"engine": {"seed": 1}})
        self.assertGreater(np.max(abs(y-mono(rows, config=other))), 1e-4)
        self.assertEqual(unit_tolerances(c, "tr808"), unit_tolerances(other, "tr808"))
        box = load_config("tr808", overrides={"engine": {"unit": "studio-b"}})
        self.assertNotEqual(unit_tolerances(c, "tr808")["bd"], unit_tolerances(box, "tr808")["bd"])
        for name, tol in unit_tolerances(box, "tr808").items():
            if name != "squares":
                self.assertLessEqual(abs(tol["freq"]-1), .02+1e-12)
                self.assertLessEqual(abs(tol["decay"]-1), .05+1e-12)


class Circuits(unittest.TestCase):
    def consecutive_xcorr(self, machine):
        rows = [hit(42, time=.25*i+.01, off=.25*i+.1) for i in range(32)]
        y = mono(rows, machine, seconds=12.)
        segs = [y[round((.25*i+.01)*SR):][:2400] for i in range(32)]
        return max(float(np.dot(a, b)/np.sqrt(np.dot(a, a)*np.dot(b, b))) for a, b in zip(segs, segs[1:]))

    def test_808_metal_never_repeats_and_909_rom_always_does(self):
        self.assertLess(self.consecutive_xcorr("tr808"), .98)
        self.assertGreater(self.consecutive_xcorr("tr909"), .999)

    def test_accent_deepens_blip_and_harmonics(self):
        out = {}
        for v in (96, 127):
            y = mono([hit(36, v)], config=load_config("tr808", overrides={"voices": {"bd": {"click": 0.}}}))[round(.01*SR):]
            zc = np.flatnonzero(np.diff(np.signbit(y)) & (np.arange(len(y)-1) > 2))
            early = SR/(2*(zc[1]-zc[0]))
            late = SR/(2*np.mean(np.diff(zc[10:20])))
            sp = np.abs(np.fft.rfft(y[round(.05*SR):round(.45*SR)]*np.hanning(round(.4*SR)), 1 << 17))
            fr = np.fft.rfftfreq(1 << 17, 1/SR)

            def amp(h):
                k = np.argmin(abs(fr-h*late))
                return sp[k-4:k+5].max()
            out[v] = (early/late-1, amp(2)/amp(1))
        self.assertGreater(out[96][0], .15)
        self.assertGreater(out[127][0], out[96][0]*1.2)
        self.assertGreater(out[127][1], out[96][1]*2)

    def test_knob_sweeps_are_monotonic(self):
        def t60(voice, key, x):
            y = mono([hit(key)], config=load_config("tr808", overrides={"voices": {voice: {"decay": x}}}))
            env = np.sqrt(np.convolve(y*y, np.ones(480)/480, "same"))
            db = 20*np.log10(env/env.max()+1e-12)
            return np.flatnonzero(db > -40)[-1]
        for voice, key in (("bd", 36), ("oh", 46), ("cy", 49)):
            values = [t60(voice, key, x) for x in np.linspace(0, 1, 5)]
            self.assertTrue(np.all(np.diff(values) > 0), (voice, values))

        def snappy(x):
            y = mono([hit(38)], config=load_config("tr808", overrides={"voices": {"sd": {"snappy": x}}}))
            p = np.abs(np.fft.rfft(y[:round(.15*SR)]))**2
            f = np.fft.rfftfreq(round(.15*SR), 1/SR)
            return p[f > 1500].sum()/p[f < 600].sum()
        values = [snappy(x) for x in np.linspace(.1, 1, 4)]
        self.assertTrue(np.all(np.diff(values) > 0), values)

    def test_closed_hat_chokes_open_hat_without_click(self):
        c = load_config("tr808", overrides={"voices": {"ch": {"level": 0.}}})
        oh = mono([hit(46)], config=c)
        both = mono([hit(46), hit(42, time=.13, off=.2)], config=c)
        k = round(.13*SR)
        before = np.sqrt(np.mean(oh[k-480:k]**2))
        after = np.sqrt(np.mean(both[k+round(.015*SR)-240:k+round(.015*SR)+240]**2))
        self.assertLess(20*math.log10(after/before), -40)
        self.assertLess(np.max(abs(np.diff(both[k-48:k+480]))), 4*np.max(abs(np.diff(oh[k-480:k]))))

    def test_hat_tone_darkens_only_the_top(self):
        def bands(tone):
            y = mono([hit(42)], config=load_config("tr808", overrides={"voices": {"ch": {"tone": tone}}}))
            p = np.abs(np.fft.rfft(y[:SR//4]))**2
            f = np.fft.rfftfreq(SR//4, 1/SR)
            return p[(f > 6000) & (f < 10000)].sum(), p[f > 10000].sum()
        hardware, dark = bands(1.), bands(.7)
        self.assertLess(dark[1], hardware[1]*.4)
        self.assertGreater(dark[0], hardware[0]*.7)

    def test_bass_mode_is_in_tune_and_legato_glides_without_a_new_pulse(self):
        c = load_config("tr808")
        for midi in (26, 38, 45):
            y = mono([dict(pitch=midi, velocity=96, time=.01, off=1.5)], config=c, bass=True, seconds=5.)
            seg = y[round(.3*SR):round(1.2*SR)]*np.hanning(round(.9*SR))
            sp = np.abs(np.fft.rfft(seg, 1 << 20))
            f = np.fft.rfftfreq(1 << 20, 1/SR)
            target = 440*2**((midi-69)/12)
            band = (f > target*.8) & (f < target*1.2)
            k = np.argmax(sp*band)
            a, b, g = np.log(sp[k-1:k+2])
            fk = (k+.5*(a-g)/(a-2*b+g))*SR/(1 << 20)
            self.assertLess(abs(1200*math.log2(fk/target)), 2, midi)
        with self.assertRaises(ValueError):
            render([dict(pitch=72, velocity=96, time=0, off=1)], 4., "tr808", c, bass=True)
        legato = mono([dict(pitch=38, velocity=96, time=.01, off=.8), dict(pitch=43, velocity=96, time=.6, off=1.4)], config=c, bass=True, seconds=5.)
        alone = mono([dict(pitch=38, velocity=96, time=.01, off=.8)], config=c, bass=True)
        k = round(.6*SR)
        # No second pulse: the legato note's onset adds no burst of energy over the ringing note.
        self.assertLess(np.max(abs(legato[k:k+480])), 1.05*np.max(abs(alone[k-480:k])))

    def test_retrigger_adds_to_the_ringing_808_but_cuts_the_909(self):
        rows = [hit(36), hit(36, time=.135, off=.3)]
        two = mono(rows)
        sum_808 = mono([rows[0]])+mono([rows[1]])
        self.assertGreater(np.max(abs(two-sum_808)), 1e-3)  # the shared state and clipper interact
        y9 = mono(rows, "tr909")
        first = mono([rows[0]], "tr909")
        k = round(.135*SR)
        # After the second hit's noise click, only the second hit sounds.
        w = slice(k+960, k+4800)
        self.assertLess(np.sqrt(np.mean((y9[w]-mono([rows[1]], "tr909")[w])**2)), .02*np.sqrt(np.mean(first[w]**2)))


class Levels(unittest.TestCase):
    def test_no_key_clips_at_any_velocity(self):
        for machine, keys in KEYMAP.items():
            c = load_config(machine)
            for k in keys:
                for v in (1, 50, 127):
                    self.assertLess(np.max(abs(render([hit(k, v)], 5., machine, c))), .9, (machine, k, v))
        for p in (24, 36, 48, 60):
            self.assertLess(np.max(abs(render([dict(pitch=p, velocity=127, time=.01, off=1.)], 5., "tr808", bass=True))), .9, p)

    def test_level_rises_with_velocity_and_bass_is_flat_across_pitch(self):
        def rms(rows, machine="tr808", **kw):
            y = mono(rows, machine, seconds=5., **kw)
            return float(np.sqrt(np.mean(y**2)))
        for machine, keys in (("tr808", (36, 38, 42, 39)), ("tr909", (36, 38, 42)), ("linn", (36, 38))):
            for k in keys:
                levels = [rms([hit(k, v)], machine) for v in (30, 60, 96, 112, 127)]
                self.assertTrue(np.all(np.diff(levels) > 0), (machine, k, levels))
        bass = [rms([dict(pitch=p, velocity=96, time=.01, off=1.)], bass=True) for p in (24, 33, 45, 60)]
        self.assertLess(20*math.log10(max(bass)/min(bass)), 4, bass)
        accent = [rms([dict(pitch=38, velocity=v, time=.01, off=1.)], bass=True) for v in (60, 96, 127)]
        self.assertTrue(np.all(np.diff(accent) > 0), accent)

    def test_clap_and_rom_tails_are_not_grit(self):
        for machine in ("tr808", "tr909", "linn"):
            y = mono([hit(39)], machine)
            p = np.abs(np.fft.rfft(y))**2
            f = np.fft.rfftfreq(len(y), 1/SR)
            self.assertLess(p[f > 10000].sum()/p.sum(), .15, machine)
        for key in (46, 49, 51):
            y = mono([hit(key)], "tr909", seconds=5.)[round(.3*SR):round(.8*SR)]
            live = y[abs(y) > 0]
            self.assertGreater(len(live)/len(y), .95, key)  # a decaying ROM tail is not truncated to zeros

    def test_rides_set_knobs_by_score_bar(self):
        from soundbench.scripts.render_drum_machine import apply_rides
        from soundbench.scripts.models.drum_machine import voice_of
        tpb = 480
        mid = mido.MidiFile(ticks_per_beat=tpb)
        tr = mido.MidiTrack()
        for i in range(4):
            tr.append(mido.Message("note_on", note=38, velocity=90, time=0 if i == 0 else 4*tpb-60))
            tr.append(mido.Message("note_off", note=38, velocity=0, time=60))
        mid.tracks.append(tr)
        rows = [dict(pitch=38, velocity=90, time=2.*i, off=2.*i+.1) for i in range(4)]
        c = load_config("tr808")
        out = apply_rides(rows, [dict(voice="bass", knob="drive", bars=[2, 3], value=.35)], mid, 0,
                          lambda p: voice_of("tr808", p, c, True), c["voices"])
        self.assertEqual([r.get("knobs", {}).get("drive") for r in out], [None, .35, .35, None])
        y = mono(out, bass=True, seconds=12.)
        self.assertGreater(np.max(abs(y-mono(rows, bass=True, seconds=12.))), 1e-3)


class Clock(unittest.TestCase):
    def test_swing_tick(self):
        tpb = 960
        self.assertEqual(swing_tick(240, tpb, .5, 16), 240)
        self.assertAlmostEqual(swing_tick(240, tpb, .58, 16), .58*480)
        self.assertAlmostEqual(swing_tick(480, tpb, .58, 16), 480)
        self.assertAlmostEqual(swing_tick(480, tpb, .66, 8), .66*960)
        xs = [swing_tick(t, tpb, .62, 16) for t in range(0, 3840, 7)]
        self.assertTrue(np.all(np.diff(xs) > 0))
        self.assertEqual(timing({"tr808_model": {"swing": "linn58"}}, "tr808-bass")[0], .58)
        with self.assertRaises(ValueError):
            timing({"tr909_model": {"swing": .9}}, "tr909")

    def test_performed_machine_is_exact_swung_and_unjittered(self):
        tpb = 480
        mid = mido.MidiFile(ticks_per_beat=tpb)
        tempo = mido.MidiTrack([mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(100), time=0)])
        mid.tracks.append(tempo)
        for name, note, channel in (("kit", 42, 9), ("piano", 60, 0)):
            tr = mido.MidiTrack([mido.MetaMessage("track_name", name=name)])
            for i in range(16):
                tr.append(mido.Message("note_on", note=note, velocity=90, channel=channel, time=0 if i == 0 else tpb//4-60))
                tr.append(mido.Message("note_off", note=note, velocity=0, channel=channel, time=60))
            mid.tracks.append(tr)
        with tempfile.TemporaryDirectory() as d:
            src, out, side = Path(d)/"a.mid", Path(d)/"b.mid", Path(d)/"a.perform.toml"
            mid.save(src)
            side.write_text('[instruments]\nkit = { model = "tr808" }\npiano = "grand"\n[tr808_model]\nswing = 0.6\n'
                            '[humanize]\nonset_jitter_ms = 12\nvelocity_jitter = 8\n')
            perform_file(src, out, side)
            performed = mido.MidiFile(out)

        def onsets(track):
            tick, rows = 0, []
            for m in track:
                tick += m.time
                if m.type == "note_on" and m.velocity:
                    rows.append((tick, m.velocity))
            return rows
        kit = onsets(performed.tracks[1])
        expected = [round(swing_tick(i*tpb//4, tpb, .6, 16)) for i in range(16)]
        self.assertEqual([t for t, _ in kit], expected)
        self.assertEqual({v for _, v in kit}, {90})
        piano = onsets(performed.tracks[2])
        self.assertNotEqual([t for t, _ in piano], [i*tpb//4 for i in range(16)])
        tick, offs = 0, []
        for m in performed.tracks[1]:
            tick += m.time
            if m.type == "note_off" or (m.type == "note_on" and not m.velocity):
                offs.append(tick)
        self.assertEqual(offs, [round(swing_tick(i*tpb//4+60, tpb, .6, 16)) for i in range(16)])


class BuildPath(unittest.TestCase):
    def test_registered_stereo_models_and_sidecar_options(self):
        for model in ("tr808", "tr808-bass", "tr909", "linn"):
            self.assertTrue(RENDERERS[model][0].exists())
            self.assertIn(model, STEREO_INSTRUMENTS)
        mid = mido.MidiFile()
        tr = mido.MidiTrack([mido.MetaMessage("track_name", name="kit"), mido.Message("note_on", note=36, velocity=90),
                             mido.Message("note_off", note=36, time=100)])
        mid.tracks.append(tr)
        bindings = resolve_instruments(mid, {"instruments": {"kit": {"model": "tr909"}}})
        self.assertEqual(bindings[0].model, "tr909")
        c, _, _ = selected_config({"tr808_model": {"accent": .9, "knobs": {"bd": {"decay": .2}}, "swing": "linn62"}}, "tr808")
        self.assertEqual((c["engine"]["accent"], c["voices"]["bd"]["decay"]), (.9, .2))
        for bad in ({"tr808_model": {"bogus": 1}}, {"tr808_model": {"knobs": {"bd": {"bogus": 1.}}}},
                    {"tr808_model": {"knobs": {"bd": {"decay": 2.}}}}):
            with self.assertRaises(ValueError):
                selected_config(bad, "tr808")


if __name__ == "__main__":
    unittest.main()
