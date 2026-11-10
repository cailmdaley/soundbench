#!/usr/bin/env python3
# /// script
# dependencies = ["mido>=1.3,<2"]
# ///
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import mido

ROOT = Path(__file__).resolve().parents[1] / "soundbench"
TEST_TEMP_DIR = ROOT / "build" / "test-tmp"
TEST_TEMP_DIR.mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(TEST_TEMP_DIR)
sys.path.insert(0, str(ROOT / "scripts"))
from soundbench.scripts import perform


def write_midi(
    path: Path, with_pedal: bool = True, tempo_change: bool = False
) -> None:
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    conductor = mido.MidiTrack()
    conductor.append(mido.MetaMessage("track_name", name="conductor", time=0))
    conductor.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))
    conductor.append(mido.MetaMessage(
        "time_signature", numerator=3, denominator=4, time=0
    ))
    if tempo_change:
        conductor.append(mido.MetaMessage("set_tempo", tempo=1_000_000, time=720))
    conductor.append(mido.MetaMessage("end_of_track", time=0))
    midi.tracks.append(conductor)

    events = [
        (0, 0, mido.MetaMessage("track_name", name="upper:", time=0)),
        (0, 1, mido.Message("note_on", channel=0, note=60, velocity=80, time=0)),
        (0, 2, mido.Message("note_on", channel=0, note=64, velocity=80, time=0)),
        (480, 3, mido.Message("note_off", channel=0, note=60, velocity=0, time=0)),
        (480, 4, mido.Message("note_off", channel=0, note=64, velocity=0, time=0)),
    ]
    if with_pedal:
        events.extend([
            (0, 3, mido.Message("control_change", channel=0, control=64, value=127, time=0)),
            (480, 0, mido.Message("control_change", channel=0, control=64, value=0, time=0)),
            (480, 1, mido.Message("control_change", channel=0, control=64, value=127, time=0)),
            (960, 0, mido.Message("control_change", channel=0, control=64, value=0, time=0)),
        ])
    events.extend([
        (480, 5, mido.Message("note_on", channel=0, note=55, velocity=72, time=0)),
        (480, 6, mido.Message("note_on", channel=0, note=67, velocity=72, time=0)),
        (960, 2, mido.Message("note_off", channel=0, note=55, velocity=0, time=0)),
        (960, 3, mido.Message("note_off", channel=0, note=67, velocity=0, time=0)),
        (1440, 0, mido.MetaMessage("end_of_track", time=0)),
    ])
    events.sort(key=lambda item: (item[0], item[1]))
    track = mido.MidiTrack()
    last_tick = 0
    for tick, _, message in events:
        message.time = tick - last_tick
        track.append(message)
        last_tick = tick
    midi.tracks.append(track)
    midi.save(path)


def write_retrigger_midi(path: Path) -> None:
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    conductor = mido.MidiTrack([
        mido.MetaMessage("set_tempo", tempo=500_000, time=0),
        mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0),
        mido.MetaMessage("end_of_track", time=0),
    ])
    notes = mido.MidiTrack([
        mido.MetaMessage("track_name", name="upper:", time=0),
        mido.Message("note_on", channel=0, note=60, velocity=80, time=0),
        mido.Message("note_off", channel=0, note=60, velocity=0, time=240),
        mido.Message("note_on", channel=0, note=60, velocity=80, time=0),
        mido.Message("note_off", channel=0, note=60, velocity=0, time=240),
        mido.MetaMessage("end_of_track", time=0),
    ])
    midi.tracks.extend([conductor, notes])
    midi.save(path)


def write_boom_chuck_midi(path: Path) -> None:
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    midi.tracks.append(mido.MidiTrack([
        mido.MetaMessage("set_tempo", tempo=500_000, time=0),
        mido.MetaMessage("time_signature", numerator=2, denominator=4, time=0),
        mido.MetaMessage("end_of_track", time=1920),
    ]))
    events = [(0, 0, mido.MetaMessage("track_name", name="lower:"))]
    for start, pitches, duration in (
        (0, (48,), 240),
        (300, (55, 60, 64), 120),
        (960, (50,), 240),
        (1260, (60, 64), 120),
    ):
        for pitch in pitches:
            events.extend([
                (start, 1, mido.Message("note_on", channel=0, note=pitch, velocity=80)),
                (start + duration, 0, mido.Message("note_off", channel=0, note=pitch)),
            ])
    track = mido.MidiTrack()
    last = 0
    for tick, _, message in sorted(events, key=lambda event: (event[0], event[1])):
        track.append(message.copy(time=tick - last))
        last = tick
    track.append(mido.MetaMessage("end_of_track", time=1920 - last))
    midi.tracks.append(track)
    midi.save(path)


def write_nested_chord_midi(path: Path) -> None:
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    midi.tracks.append(mido.MidiTrack([
        mido.MetaMessage("set_tempo", tempo=500_000, time=0),
        mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0),
        mido.MetaMessage("end_of_track", time=1920),
    ]))
    events = [(0, 0, mido.MetaMessage("track_name", name="upper:"))]
    for start, pitches in ((0, (60, 64, 67)), (480, (60, 64)), (960, (57, 60, 64))):
        for pitch in pitches:
            events.extend([
                (start, 1, mido.Message("note_on", channel=0, note=pitch, velocity=80)),
                (start + 360, 0, mido.Message("note_off", channel=0, note=pitch)),
            ])
    track = mido.MidiTrack()
    last = 0
    for tick, _, message in sorted(events, key=lambda event: (event[0], event[1])):
        track.append(message.copy(time=tick - last))
        last = tick
    track.append(mido.MetaMessage("end_of_track", time=1920 - last))
    midi.tracks.append(track)
    midi.save(path)


def absolute_messages(path: Path):
    midi = mido.MidiFile(path)
    for track_index, track in enumerate(midi.tracks):
        tick = 0
        for message in track:
            tick += message.time
            yield track_index, tick, message


class PerformTests(unittest.TestCase):
    def test_final_piano_pedal_release_preserves_notes(self):
        midi = mido.MidiFile(ticks_per_beat=480)
        for name in ('piano', 'bass'):
            track = mido.MidiTrack([
                mido.MetaMessage('track_name', name=name),
                mido.Message('control_change', control=64, value=127),
                mido.Message('note_on', note=60, velocity=60),
                mido.Message('note_off', note=60, time=960),
                mido.Message('control_change', control=64, value=0),
                mido.MetaMessage('end_of_track'),
            ])
            midi.tracks.append(track)
        perform.final_pedal_release(midi, [0], 3)
        def times(track, kind):
            tick, result = 0, []
            for msg in track:
                tick += msg.time
                if msg.type == kind:
                    result.append(tick)
            return result
        self.assertEqual(times(midi.tracks[0], 'note_off'), [960])
        self.assertEqual(times(midi.tracks[0], 'control_change'), [0, 3840])
        self.assertEqual(times(midi.tracks[1], 'control_change'), [0, 960])
        for value in (-1, float('nan'), 11):
            with self.assertRaises(ValueError):
                perform.final_pedal_release(midi, [0], value)

    def test_bar_map_uses_time_signature_and_one_based_positions(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "score.mid"
            write_midi(path)
            bars = perform.BarMap(mido.MidiFile(path))
            self.assertEqual(bars.bar_start(2), 1440)
            self.assertEqual(bars.position(2, 3), 2400)
            self.assertEqual(bars.bar_at(1440), 2)

    def test_repeatable_render_preserves_notes_and_delays_pedal_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "score.mid"
            sidecar = root / "score.perform.toml"
            output_a = root / "a.mid"
            output_b = root / "b.mid"
            write_midi(source)
            sidecar.write_text(
                'seed = 12\n'
                'rit = [{ start_beat = 1, end_beat = 4, factor = 0.9 }]\n'
                'fermata = [{ bar = 2, beat = 2, hold = 0.8 }]\n'
                '[groove]\nsubdivision = 8\nvelocity = [4, -2]\n'
                'timing_ms = [0, 10]\ntracks = ["upper"]\n'
                '[[groove_sections]]\nstart_bar = 2\nend_bar = 2\n'
                'timing_ms = [0, 20]\n'
                '[humanize]\n'
                'onset_jitter_ms = 6\nvelocity_jitter = 2\n'
                'bass_roll_ms = 2\nmax_chord_spread_ms = 15\n'
                '[tracks]\nupper = { velocity_scale = 1.1, velocity_offset = 2 }\n'
                '[pedal]\nauto = true\nchange_delay_ms = 30\n'
            )
            summary = perform.perform_file(source, output_a, sidecar)
            perform.perform_file(source, output_b, sidecar)
            self.assertEqual(output_a.read_bytes(), output_b.read_bytes())
            self.assertEqual(summary["notes_in"], 4)
            self.assertEqual(summary["notes_out"], 4)
            self.assertTrue(summary["pedal_in_score"])
            self.assertFalse(summary["auto_pedal_added"])

            notes = {}
            pedals = []
            for track_index, tick, message in absolute_messages(output_a):
                if message.type == "note_on" and message.velocity:
                    notes.setdefault((tick, message.note), []).append(message.velocity)
                if message.type == "control_change" and message.control == 64:
                    pedals.append((tick, message.value))
            first_chord_ticks = [tick for (tick, pitch) in notes if pitch in (60, 64)]
            self.assertLessEqual((max(first_chord_ticks) - min(first_chord_ticks)) * 500 / 480, 15.1)
            self.assertEqual(sum(1 for _, value in pedals if value >= 64), 2)
            # The CC64 off/on change follows the new chord's bass note by about 30/90 ms.
            bass_tick = min(tick for tick, pitch in notes if pitch == 55)
            pedal_off_tick = next(tick for tick, value in pedals if value < 64 and tick > 0)
            pedal_on_tick = next(tick for tick, value in pedals if value >= 64 and tick > 0)
            self.assertAlmostEqual((pedal_off_tick - bass_tick) * 500 / 480, 30, delta=2)
            self.assertAlmostEqual((pedal_on_tick - bass_tick) * 500 / 480, 90, delta=2)

    def test_groove_repeats_by_grid_position_and_sections_override(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "score.mid"
            write_midi(path)
            midi = mido.MidiFile(path)
            bars = perform.BarMap(midi)
            events = perform._event_positions(midi)
            note_tracks = perform._note_track_indices(events)
            names = perform._track_names(events)
            settings = {
                "groove": {
                    "subdivision": 8,
                    "velocity": [12, -6],
                    "timing_ms": [0, 45],
                    "tracks": ["upper"],
                },
                "groove_sections": [{
                    "start_bar": 1,
                    "end_bar": 1,
                    "timing_ms": [0, 25],
                }],
            }
            on_grid = perform.Event(
                1, 0, 240, mido.Message("note_on", channel=0, note=64, velocity=80)
            )
            off_grid = perform.Event(
                1, 1, 120, mido.Message("note_on", channel=0, note=64, velocity=80)
            )
            later_bar = perform.Event(
                1, 2, 1680, mido.Message("note_on", channel=0, note=64, velocity=80)
            )
            self.assertEqual(perform._groove_offsets(on_grid, settings, bars, note_tracks, names), (25, -6))
            self.assertEqual(perform._groove_offsets(off_grid, settings, bars, note_tracks, names), (0, 0))
            self.assertEqual(perform._groove_offsets(later_bar, settings, bars, note_tracks, names), (45, -6))

    def test_grooves_layer_per_track(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "score.mid"
            midi = mido.MidiFile(type=1, ticks_per_beat=480)
            midi.tracks.append(mido.MidiTrack([
                mido.MetaMessage("time_signature", numerator=2, denominator=4, time=0),
                mido.MetaMessage("end_of_track", time=0),
            ]))
            for name, note in (("upper:", 72), ("lower:", 48)):
                midi.tracks.append(mido.MidiTrack([
                    mido.MetaMessage("track_name", name=name, time=0),
                    mido.Message("note_on", channel=0, note=note, velocity=80, time=0),
                    mido.Message("note_off", channel=0, note=note, velocity=0, time=240),
                    mido.MetaMessage("end_of_track", time=0),
                ]))
            midi.save(path)
            midi = mido.MidiFile(path)
            bars = perform.BarMap(midi)
            events = perform._event_positions(midi)
            note_tracks = perform._note_track_indices(events)
            names = perform._track_names(events)
            settings = {
                "groove": {"subdivision": 8, "velocity": [12, -6], "tracks": ["upper"]},
                "groove_sections": [{
                    "start_bar": 1, "end_bar": 4, "subdivision": 8,
                    "velocity": [7, -5], "tracks": ["lower"],
                }],
            }
            upper = perform.Event(1, 0, 0, mido.Message("note_on", channel=0, note=64, velocity=80))
            lower = perform.Event(2, 0, 0, mido.Message("note_on", channel=1, note=48, velocity=80))
            self.assertEqual(perform._groove_offsets(upper, settings, bars, note_tracks, names), (0, 12))
            self.assertEqual(perform._groove_offsets(lower, settings, bars, note_tracks, names), (0, 7))

    def test_chord_spread_stays_bounded_after_selective_groove(self):
        first = perform.Event(
            1, 0, 240, mido.Message("note_on", channel=0, note=48, velocity=80)
        )
        second = perform.Event(
            2, 0, 240, mido.Message("note_on", channel=1, note=72, velocity=80)
        )
        offsets = {id(first): 24.0, id(second): 0.0}
        perform._clamp_chord_spread([first, second], offsets, 15.0)
        self.assertLessEqual(max(offsets.values()) - min(offsets.values()), 15.0)

    def test_retrigger_release_never_follows_the_next_onset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "retrigger.mid"
            sidecar = root / "retrigger.perform.toml"
            output = root / "performed.mid"
            write_retrigger_midi(source)
            sidecar.write_text("seed = 0\n[humanize]\nvelocity_jitter = 0\n")
            perform.perform_file(source, output, sidecar)
            onsets = []
            releases = []
            for _, tick, message in absolute_messages(output):
                if message.type == "note_on" and message.velocity:
                    onsets.append(tick)
                elif message.type == "note_off" or (
                    message.type == "note_on" and message.velocity == 0
                ):
                    releases.append(tick)
            self.assertEqual(len(onsets), 2)
            self.assertEqual(len(releases), 2)
            self.assertLessEqual(releases[0], onsets[1])

    def test_auto_pedal_uses_first_onset_and_configured_change_delay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "no-pedal.mid"
            sidecar = root / "no-pedal.perform.toml"
            output = root / "performed.mid"
            write_midi(source, with_pedal=False)
            sidecar.write_text(
                "[humanize]\nonset_jitter_ms = 0\nvelocity_jitter = 0\n"
                "[pedal]\nauto = true\nchange_delay_ms = 10\n"
            )
            summary = perform.perform_file(source, output, sidecar)
            self.assertTrue(summary["auto_pedal_added"])
            onsets = [
                tick for _, tick, message in absolute_messages(output)
                if message.type == "note_on" and message.velocity
            ]
            pedal_ons = [
                tick for _, tick, message in absolute_messages(output)
                if message.type == "control_change" and message.control == 64
                and message.value >= 64
            ]
            self.assertEqual(min(onsets), 0)
            self.assertAlmostEqual((min(pedal_ons) - min(onsets)) * 500 / 480, 30, delta=1.1)

    def test_fermata_hold_spans_tempo_changes_in_seconds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "tempo-change.mid"
            sidecar = root / "tempo-change.perform.toml"
            output = root / "performed.mid"
            write_midi(source, tempo_change=True)
            sidecar.write_text("fermata = [{ bar = 1, beat = 2, hold = 1, width_beats = 2 }]\n")
            perform.perform_file(source, output, sidecar)
            source_midi = mido.MidiFile(source)
            output_midi = mido.MidiFile(output)
            source_tempo = perform.TempoMap(source_midi)
            output_tempo = perform.TempoMap(output_midi)
            bars = perform.BarMap(source_midi)
            start = bars.position(1, 2)
            end = start + 2 * bars.beat_length_at(start)
            warper = perform._build_warper(
                {"fermata": [{"bar": 1, "beat": 2, "hold": 1, "width_beats": 2}]},
                bars,
                source_tempo,
            )
            start_out = round(warper.apply(start))
            end_out = round(warper.apply(end))
            extra_seconds = (
                output_tempo.seconds_between(start_out, end_out)
                - source_tempo.seconds_between(start, end)
            )
            self.assertAlmostEqual(extra_seconds, 1.0, delta=0.02)

    def test_auto_pedal_releases_after_nested_chords_before_a_new_harmony(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "nested-chords.mid"
            sidecar = root / "nested-chords.perform.toml"
            output = root / "performed.mid"
            write_nested_chord_midi(source)
            sidecar.write_text(
                "[humanize]\nonset_jitter_ms = 0\nvelocity_jitter = 0\n"
                "[pedal]\nauto = true\nchange_delay_ms = 10\n"
            )
            perform.perform_file(source, output, sidecar)
            pedal = [
                (tick, message.value) for track, tick, message in absolute_messages(output)
                if track == 1 and message.type == "control_change" and message.control == 64
            ]
        self.assertIn((959, 0), pedal)
        self.assertIn((989, 127), pedal)

    def test_auto_pedal_only_adds_when_score_has_no_cc64(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "no-pedal.mid"
            sidecar = root / "no-pedal.perform.toml"
            output = root / "performed.mid"
            write_midi(source, with_pedal=False)
            sidecar.write_text("[pedal]\nauto = true\n")
            summary = perform.perform_file(source, output, sidecar)
            self.assertTrue(summary["auto_pedal_added"])
            self.assertGreaterEqual(sum(
                1 for _, _, message in absolute_messages(output)
                if message.type == "control_change" and message.control == 64
            ), 2)

    def test_shared_pedal_bridges_short_rests_and_ignores_revoicing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "boom-chuck.mid"
            sidecar = root / "boom-chuck.perform.toml"
            output = root / "performed.mid"
            write_boom_chuck_midi(source)
            sidecar.write_text(
                "[humanize]\nonset_jitter_ms = 0\nvelocity_jitter = 0\n"
                "[pedal]\nauto = true\nchange_delay_ms = 10\n"
            )
            perform.perform_file(source, output, sidecar)
            pedal = [
                (tick, message.value) for track, tick, message in absolute_messages(output)
                if track == 1 and message.type == "control_change" and message.control == 64
            ]
            self.assertEqual([value for _, value in pedal], [127, 0, 127, 0])
            bars = perform.BarMap(mido.MidiFile(source))
            self.assertEqual([bars.bar_at(tick) for tick, value in pedal if value >= 64], [1, 2])
            self.assertLess(pedal[0][0], 240)
            self.assertGreater(pedal[1][0], 300)
            self.assertLess(pedal[2][0], 1260)
            self.assertGreater(pedal[3][0], 1260)

    def test_smooth_warper_is_monotonic_and_adds_fermata_duration(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "score.mid"
            write_midi(path)
            midi = mido.MidiFile(path)
            tempo = perform.TempoMap(midi)
            bars = perform.BarMap(midi)
            settings = {
                "rit": [{"start_bar": 1, "end_bar": 2, "factor": 0.8}],
                "fermata": [{"bar": 1, "beat": 2, "hold": 1.0}],
            }
            warper = perform._build_warper(settings, bars, tempo)
            points = [0, 240, 480, 720, 960, 1440, 1920]
            warped = [warper.apply(point) for point in points]
            self.assertEqual(warped, sorted(warped))
            self.assertGreater(warped[-1] - points[-1], 900)
            self.assertLess(abs(warper.apply(0)), 1e-9)


if __name__ == "__main__":
    unittest.main()
