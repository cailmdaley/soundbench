#!/usr/bin/env python3
# /// script
# dependencies = ["mido>=1.3,<2", "matplotlib>=3.8", "numpy>=1.26"]
# ///
"""Tests for the room: coupled timekeepers, nights, feel and the room report."""
from __future__ import annotations

import hashlib
import math
import random
import statistics
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
from soundbench.scripts import perform  # noqa: E402
from soundbench.scripts import room  # noqa: E402
from soundbench.scripts import room_report  # noqa: E402

PPQ = 480


def band_midi(path: Path, bars: int = 8, sixteenths: bool = False) -> None:
    """Two players at 120 bpm in 4/4: `lead` (upper) and `follow` (lower).

    The lead plays quarter notes (sixteenths when asked); the follower plays
    quarter-note chords an octave and a fifth lower.
    """
    midi = mido.MidiFile(type=1, ticks_per_beat=PPQ)
    midi.tracks.append(mido.MidiTrack([
        mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0),
        mido.MetaMessage("set_tempo", tempo=500_000, time=0),
    ]))
    step = PPQ // 4 if sixteenths else PPQ
    for name, pitches, length in (("upper:", [72], step), ("lower:", [48, 55], PPQ)):
        events = []
        for start in range(0, bars * 4 * PPQ, length):
            for k, pitch in enumerate(pitches):
                velocity = 70 + (start // length) % 5
                events.append((start, 1, k, mido.Message("note_on", note=pitch, velocity=velocity)))
                events.append((start + length - 10, 0, k, mido.Message("note_off", note=pitch, velocity=0)))
        events.sort(key=lambda e: e[:3])
        track = mido.MidiTrack([mido.MetaMessage("track_name", name=name, time=0)])
        last = 0
        for tick, _, _, message in events:
            message.time = tick - last
            track.append(message)
            last = tick
        midi.tracks.append(track)
    midi.save(path)


BASE_SIDECAR = """seed = 7
rit = [{ start_bar = 6, end_bar = 8, factor = 0.85 }]
[humanize]
onset_jitter_ms = 4
velocity_jitter = 3
[expression]
melody = "upper"
metric = [2, 0, 1, -1]
[[expression.arch]]
bars = 4
velocity = 4
tempo = 0.03
"""

ROOM = """
[room]
lead = "lead"
[room.players]
lead = "upper"
follow = "lower"
"""

# The fixed performance of band_midi() under BASE_SIDECAR: a sidecar without a room
# must keep producing exactly these bytes.
ROOMLESS_SHA256 = "8f1fa72278dd95ca5505b9a4f2a1d254da23aa84e454cfb3aa1638dc7e01bda8"


def perform_with(sidecar_text: str, night=None, sixteenths=False, bars=8) -> bytes:
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        band_midi(directory / "score.midi", bars, sixteenths)
        (directory / "score.perform.toml").write_text(sidecar_text)
        perform.perform_file(directory / "score.midi", directory / "out.mid",
                             directory / "score.perform.toml", night)
        return (directory / "out.mid").read_bytes()


def onsets(data: bytes, track: int) -> list[tuple[int, int, int]]:
    with tempfile.NamedTemporaryFile(suffix=".mid") as handle:
        handle.write(data)
        handle.flush()
        midi = mido.MidiFile(handle.name)
    tick, result = 0, []
    for message in midi.tracks[track]:
        tick += message.time
        if message.type == "note_on" and message.velocity:
            result.append((tick, message.note, message.velocity))
    return result


class NightTests(unittest.TestCase):
    def test_a_sidecar_without_a_room_performs_byte_identically(self):
        data = perform_with(BASE_SIDECAR)
        self.assertEqual(hashlib.sha256(data).hexdigest(), ROOMLESS_SHA256)

    def test_the_same_night_is_byte_identical(self):
        self.assertEqual(perform_with(BASE_SIDECAR + ROOM, 3), perform_with(BASE_SIDECAR + ROOM, 3))
        self.assertEqual(perform_with(BASE_SIDECAR, 3), perform_with(BASE_SIDECAR, 3))

    def test_different_nights_differ(self):
        nights = {perform_with(BASE_SIDECAR + ROOM, n) for n in (1, 2, 3)}
        self.assertEqual(len(nights), 3)
        self.assertNotEqual(perform_with(BASE_SIDECAR, 1), perform_with(BASE_SIDECAR, 2))
        self.assertNotEqual(perform_with(BASE_SIDECAR + ROOM), perform_with(BASE_SIDECAR + ROOM, 1))

    def test_night_seeds_are_derived_from_the_base(self):
        self.assertEqual(room.night_seed(1892, None), 1892)
        self.assertNotEqual(room.night_seed(1892, 1), room.night_seed(1892, 2))
        self.assertNotEqual(room.night_seed(1892, 1), room.night_seed(1893, 1))
        self.assertEqual(room.night_seed(1892, 1), room.night_seed(1892, 1))


def two_players(follow_alpha: float, lead_alpha: float = 0.0, **params) -> room.Room:
    values = {**room.DEFAULTS, "follow": follow_alpha, "lead_listens": lead_alpha, **params}
    return room.Room({"lead": [1], "follow": [2]}, {1: "lead", 2: "follow"}, "lead", [], [],
                     values, {}, ["lead", "follow"])


def run(the_room: room.Room, beats: int, seed: int, plays=None):
    target = [0.5] * beats
    plays = plays or {p: [True] * (beats + 1) for p in the_room.order}
    return room.simulate(the_room, plays, target, [1 + n // 4 for n in range(beats)], random.Random(seed))


def corr(xs, ys):
    return statistics.correlation(xs, ys)


class TimekeeperTests(unittest.TestCase):
    def test_the_follower_corrects_toward_the_leader(self):
        onset, _, _ = run(two_players(0.4), 1500, 11)
        lead, follow = onset["lead"], onset["follow"]
        asynchrony = [f - l for f, l in zip(follow, lead)]
        follow_interval = [b - a for a, b in zip(follow, follow[1:])]
        lead_interval = [b - a for a, b in zip(lead, lead[1:])]
        # a follower behind the leader shortens its next interval; the leader does not answer
        self.assertLess(corr(asynchrony[:-1], follow_interval), -0.3)
        self.assertGreater(corr(asynchrony[:-1], lead_interval), -0.1)
        # the follower's asynchrony carries the leader's previous interval
        self.assertLess(corr(asynchrony[1:], lead_interval), -0.4)
        uncoupled, _, _ = run(two_players(0.0), 1500, 11)
        loose = [f - l for f, l in zip(uncoupled["follow"], uncoupled["lead"])]
        loose_lead = [b - a for a, b in zip(uncoupled["lead"], uncoupled["lead"][1:])]
        self.assertGreater(corr(loose[1:], loose_lead), -0.2)

    def test_drift_is_bounded_and_coupling_pulls_a_wanderer_back(self):
        loud = dict(tightness=0.5, drift=0.01, drift_return=0.0, drift_bound=0.03, beta=0.0)
        _, _, period = run(two_players(0.3, 0.07, **loud), 3000, 5)
        for series in period.values():
            self.assertLessEqual(max(abs(p) for p in series), 0.03 + 1e-12)
            self.assertGreater(max(abs(p) for p in series), 0.02)  # it does wander
        coupled, _, _ = run(two_players(0.3, 0.07, **loud), 3000, 5)
        apart, _, _ = run(two_players(0.0, 0.0, **loud), 3000, 5)
        spread = lambda o: max(abs(f - l) for f, l in zip(o["follow"], o["lead"]))
        self.assertLess(spread(coupled), 0.15)
        self.assertGreater(spread(apart), 0.5)

    def test_a_silent_player_is_not_heard(self):
        plays = {"lead": [False] * 401, "follow": [True] * 401}
        onset, _, _ = run(two_players(0.5, 0.0), 400, 3, plays)
        # the follower hears nothing, so it free-runs like an uncoupled player
        free, _, _ = run(two_players(0.0, 0.0), 400, 3, plays)
        self.assertEqual(onset["follow"], free["follow"])


class ReportTests(unittest.TestCase):
    def test_room_report_finds_who_follows_whom(self):
        sidecar = """seed = 3
[humanize]
velocity_jitter = 0
[room]
lead = "lead"
follow = 0.5
lead_listens = 0.0
timekeeper = 0.02
[room.players]
lead = "upper"
follow = "lower"
"""
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            band_midi(directory / "song.midi", bars=120)
            (directory / "song.perform.toml").write_text(sidecar)
            settings = room_report.tomllib.loads(sidecar)
            performed = directory / "song.performed.mid"
            perform.perform_file(directory / "song.midi", performed, directory / "song.perform.toml", 1)
            result = room_report.analyse(performed, directory / "song.midi", settings)
            room_report.plot(result, directory / "song.room.png", "test")
            self.assertTrue((directory / "song.room.png").stat().st_size > 10_000)
        (pair,) = result["pairs"]
        self.assertEqual(pair["pair"], ["lead", "follow"])
        # the follower's next interval answers the lead's; the lead ignores the follower,
        # so the reverse correlation holds only their shared slow tempo drift
        self.assertGreater(pair["r_a_to_b"], 0.3)
        self.assertGreater(pair["r_a_to_b"] - pair["r_b_to_a"], 0.15)
        self.assertLess(pair["async_sd_ms"], 40)
        for cv in result["ioi_cv"].values():
            self.assertTrue(0.005 < cv < 0.05)


class FeelTests(unittest.TestCase):
    def test_feel_places_and_accents_grid_positions(self):
        sidecar = """seed = 3
[humanize]
velocity_jitter = 0
[room]
motor_ms = 0
timekeeper = 0
drift = 0
contagion = 0
[room.players]
lead = "upper"
follow = "lower"
[room.feel.lead]
subdivision = 16
timing_ms = [-10, 0, 10, 0]
velocity = [8, 0, -8, 0]
night_ms = 0
night_velocity = 0
night_scale = 0
wander_ms = 0
wander_velocity = 0
"""
        sidecar = sidecar.replace('[room]', '[room]\nfeel_span_ms = 20')
        plain = onsets(perform_with(sidecar.split("[room.feel.lead]")[0], 1, sixteenths=True), 1)
        felt = onsets(perform_with(sidecar, 1, sixteenths=True), 1)
        self.assertEqual(len(plain), len(felt))
        ms_per_tick = 500 / PPQ
        for k, ((t0, _, v0), (t1, _, v1)) in enumerate(zip(plain, felt)):
            if t0 < PPQ:
                continue  # the first beat can be clamped at zero
            self.assertAlmostEqual((t1 - t0) * ms_per_tick, [-10, 0, 10, 0][k % 4], delta=1.1)
            self.assertEqual(v1 - v0, [8, 0, -8, 0][k % 4])

    def test_built_in_feels_vary_by_night_within_bounds(self):
        for name in ("scruggs", "boomchuck", "bass-push", "straight"):
            profile = room.FEELS[name]
            nights = [room.Feel(profile, random.Random(n)) for n in range(40)]
            for feel in nights:
                for mean, value in zip(profile["timing_ms"], feel.timing):
                    slack = abs(mean) * profile["night_scale"] + 2 * profile["night_ms"] + 1e-9
                    self.assertLessEqual(abs(value - mean), slack)
            self.assertGreater(len({tuple(f.timing) for f in nights}), 30)
        scruggs = room.Feel({**room.FEELS["scruggs"], "night_ms": 0, "night_velocity": 0,
                             "night_scale": 0, "wander_ms": 0, "wander_velocity": 0}, random.Random(1))
        thumbs, middles = [0, 3, 6], [2, 5, 7]
        self.assertTrue(all(scruggs.velocity[t] > 0 for t in thumbs))
        self.assertTrue(all(scruggs.velocity[m] < -5 for m in middles))
        self.assertTrue(all(scruggs.timing[m] > 0 for m in middles[:2]))
        self.assertTrue(all(scruggs.timing[t] < 0 for t in thumbs))

    def test_within_night_wander_is_slow_and_bounded(self):
        feel = room.Feel(room.FEELS["scruggs"], random.Random(4))
        drifts = [feel._wander(bar)[0][0] for bar in range(1, 400)]
        self.assertLessEqual(max(abs(d) for d in drifts), 2 * room.FEELS["scruggs"]["wander_ms"] + 1e-9)
        steps = [abs(b - a) for a, b in zip(drifts, drifts[1:])]
        self.assertLess(statistics.mean(steps), 0.6 * statistics.pstdev(drifts) * math.sqrt(2))


class DynamicsTests(unittest.TestCase):
    def test_accompaniment_yields_under_the_lead_and_answers_its_gaps(self):
        midi = mido.MidiFile(type=1, ticks_per_beat=PPQ)
        midi.tracks.append(mido.MidiTrack([mido.MetaMessage("set_tempo", tempo=500_000, time=0)]))
        # lead: one-beat notes on the first beat of bars 1, 2 and 3
        lead = mido.MidiTrack([mido.MetaMessage("track_name", name="upper:", time=0)])
        last = 0
        for start in (0, 4 * PPQ, 8 * PPQ):
            lead.append(mido.Message("note_on", note=72, velocity=70, time=start - last))
            lead.append(mido.Message("note_off", note=72, velocity=0, time=PPQ))
            last = start + PPQ
        accompaniment = mido.MidiTrack([mido.MetaMessage("track_name", name="lower:", time=0)])
        for _ in range(12):
            accompaniment.append(mido.Message("note_on", note=67, velocity=60, time=0))
            accompaniment.append(mido.Message("note_off", note=67, velocity=0, time=PPQ))
        midi.tracks.extend([lead, accompaniment])
        sidecar = """[humanize]
velocity_jitter = 0
[room]
contagion = 0
[room.players]
lead = "upper"
follow = "lower"
"""
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            midi.save(directory / "s.midi")
            (directory / "s.perform.toml").write_text(sidecar)
            perform.perform_file(directory / "s.midi", directory / "o.mid", directory / "s.perform.toml")
            data = (directory / "o.mid").read_bytes()
        velocities = [v for _, _, v in onsets(data, 2)]
        self.assertEqual(velocities[:5], [
            60 - 4,   # under the lead's fresh note, a fourth away
            60 + 5,   # a beat after the lead's note, with three beats until the next: answer
            62,       # tapering: 60 + 2.5, rounded to even
            60,       # tapered away
            60 - 4,   # the lead is back
        ])

if __name__ == "__main__":
    unittest.main()
