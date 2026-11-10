#!/usr/bin/env python3
# /// script
# dependencies = ["mido>=1.3,<2"]
# ///
"""Humanize a LilyPond MIDI while preserving its musical events."""

from __future__ import annotations

import argparse
import bisect
import math
import random
import statistics
import sys
import tomllib
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mido

from soundbench.scripts import expression
from soundbench.scripts import instrumentation
from soundbench.scripts import machine_time
from soundbench.scripts import plucked
from soundbench.scripts import room
from soundbench.scripts import sustain
from soundbench.scripts import stage


@dataclass
class Event:
    track: int
    order: int
    tick: int
    message: mido.Message | mido.MetaMessage


class TempoMap:
    def __init__(self, midi: mido.MidiFile):
        events: list[tuple[int, int, int]] = []
        for track_index, track in enumerate(midi.tracks):
            tick = 0
            for order, message in enumerate(track):
                tick += message.time
                if message.type == "set_tempo":
                    events.append((tick, track_index * 1_000_000 + order, message.tempo))
        events.sort()
        self.ticks = [0]
        self.tempos = [500_000]
        for tick, _, tempo in events:
            if tick == self.ticks[-1]:
                self.tempos[-1] = tempo
            else:
                self.ticks.append(tick)
                self.tempos.append(tempo)
        self.ppq = midi.ticks_per_beat
        self.cumulative_seconds = [0.0]
        for index in range(1, len(self.ticks)):
            elapsed_ticks = self.ticks[index] - self.ticks[index - 1]
            self.cumulative_seconds.append(
                self.cumulative_seconds[-1]
                + elapsed_ticks * self.tempos[index - 1] / (self.ppq * 1_000_000)
            )

    def at(self, tick: float) -> int:
        return self.tempos[bisect.bisect_right(self.ticks, tick) - 1]

    def seconds_at(self, tick: float) -> float:
        index = bisect.bisect_right(self.ticks, tick) - 1
        return (
            self.cumulative_seconds[index]
            + (tick - self.ticks[index]) * self.tempos[index] / (self.ppq * 1_000_000)
        )

    def seconds_between(self, start: float, end: float) -> float:
        return self.seconds_at(end) - self.seconds_at(start)

    def ticks_for_ms(self, milliseconds: float, tick: float) -> float:
        return milliseconds * self.ppq * 1000.0 / self.at(tick)


class BarMap:
    """Map one-based bar and quarter-beat positions to MIDI ticks."""

    def __init__(self, midi: mido.MidiFile):
        events: list[tuple[int, int, int, int]] = []
        for track_index, track in enumerate(midi.tracks):
            tick = 0
            for order, message in enumerate(track):
                tick += message.time
                if message.type == "time_signature":
                    events.append(
                        (tick, track_index * 1_000_000 + order,
                         message.numerator, message.denominator)
                    )
        events.sort()
        self.ppq = midi.ticks_per_beat
        self.signatures = [(0, 4, 4)]
        for tick, _, numerator, denominator in events:
            if tick == 0:
                self.signatures[0] = (0, numerator, denominator)
            elif tick == self.signatures[-1][0]:
                self.signatures[-1] = (tick, numerator, denominator)
            else:
                self.signatures.append((tick, numerator, denominator))
        self.starts = [0]
        self.generated = 0
        self.signature_index = 0
        self.numerator = self.signatures[0][1]
        self.denominator = self.signatures[0][2]

    def _advance_signature(self, tick: int) -> None:
        while (self.signature_index + 1 < len(self.signatures)
               and self.signatures[self.signature_index + 1][0] <= tick):
            self.signature_index += 1
            _, self.numerator, self.denominator = self.signatures[self.signature_index]

    def ensure_bar(self, bar: int) -> None:
        while len(self.starts) < bar:
            current = self.starts[-1]
            next_boundary = current + round(
                self.numerator * self.ppq * 4 / self.denominator
            )
            next_signature = None
            if self.signature_index + 1 < len(self.signatures):
                candidate = self.signatures[self.signature_index + 1][0]
                if candidate <= next_boundary:
                    next_signature = candidate
            if next_signature is not None:
                next_start = next_signature
                self._advance_signature(next_start)
            else:
                next_start = next_boundary
                self._advance_signature(next_start)
            if next_start <= current:
                raise ValueError("time signatures do not advance the bar map")
            self.starts.append(next_start)

    def bar_start(self, bar: int) -> int:
        if bar < 1:
            raise ValueError("bar numbers start at 1")
        self.ensure_bar(bar)
        return self.starts[bar - 1]

    def position(self, bar: int, beat: float = 1.0) -> int:
        start = self.bar_start(bar)
        signature_index = bisect.bisect_right(
            [item[0] for item in self.signatures], start
        ) - 1
        _, numerator, denominator = self.signatures[signature_index]
        beat_ticks = self.ppq * 4.0 / denominator
        offset = (beat - 1.0) * beat_ticks
        if beat < 1.0 or offset >= (numerator * beat_ticks):
            raise ValueError(f"beat {beat:g} is outside bar {bar}")
        return round(start + offset)

    def beat_position(self, beat: float) -> int:
        if beat < 1.0:
            raise ValueError("absolute beat numbers start at 1")
        return round((beat - 1.0) * self.ppq)

    def bar_at(self, tick: int) -> int:
        while self.starts[-1] <= tick:
            self.ensure_bar(len(self.starts) + 1)
        return max(1, bisect.bisect_right(self.starts, tick))

    def beat_length_at(self, tick: int) -> int:
        signature_index = bisect.bisect_right(
            [item[0] for item in self.signatures], tick
        ) - 1
        _, _, denominator = self.signatures[signature_index]
        return round(self.ppq * 4 / denominator)


class SmoothCurve:
    """A sampled, monotonic time displacement in MIDI ticks."""

    def __init__(self, start: float, end: float, rate):
        self.start = start
        self.end = end
        samples = max(128, min(16_384, math.ceil((end - start) / 120.0)))
        self.x = [start + (end - start) * i / samples for i in range(samples + 1)]
        self.y = [0.0]
        previous_rate = rate(self.x[0])
        for left, right in zip(self.x, self.x[1:]):
            next_rate = rate(right)
            self.y.append(self.y[-1] + (previous_rate + next_rate) * (right - left) / 2)
            previous_rate = next_rate

    def value(self, tick: float) -> float:
        if tick <= self.start:
            return 0.0
        if tick >= self.end:
            return self.y[-1]
        index = bisect.bisect_right(self.x, tick) - 1
        fraction = (tick - self.x[index]) / (self.x[index + 1] - self.x[index])
        return self.y[index] + fraction * (self.y[index + 1] - self.y[index])


def smoothstep(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value * value * (3.0 - 2.0 * value)


def arch(value: float) -> float:
    return smoothstep(2.0 * value) if value <= 0.5 else smoothstep(2.0 * (1.0 - value))


class TimeWarper:
    def __init__(self, curves: list[SmoothCurve]):
        self.curves = curves

    def apply(self, tick: float) -> float:
        return tick + sum(curve.value(tick) for curve in self.curves)


def _event_positions(midi: mido.MidiFile) -> list[list[Event]]:
    result: list[list[Event]] = []
    for track_index, track in enumerate(midi.tracks):
        tick = 0
        events = []
        for order, message in enumerate(track):
            tick += message.time
            if message.type != "end_of_track":
                events.append(Event(track_index, order, tick, message.copy(time=0)))
        result.append(events)
    return result


def _track_names(events_by_track: list[list[Event]]) -> dict[int, str]:
    names = {}
    for track_index, events in enumerate(events_by_track):
        for event in events:
            if event.message.type == "track_name":
                names[track_index] = event.message.name
                break
    return names


def _note_track_indices(events_by_track: list[list[Event]]) -> list[int]:
    return [
        index for index, events in enumerate(events_by_track)
        if any(e.message.type == "note_on" and e.message.velocity > 0 for e in events)
    ]


def _resolve_track_index(
    selector: Any, note_tracks: list[int], names: dict[int, str]
) -> int:
    matches = instrumentation.matching_tracks(selector, note_tracks, names)
    if len(matches) != 1:
        raise ValueError(f"track name {selector!r} matched {len(matches)} MIDI tracks")
    return matches[0]


def _track_adjustments(
    settings: dict[str, Any], events_by_track: list[list[Event]]
) -> dict[int, tuple[float, float]]:
    note_tracks = _note_track_indices(events_by_track)
    names = _track_names(events_by_track)
    adjustments = {index: (1.0, 0.0) for index in note_tracks}
    for selector, options in settings.get("tracks", {}).items():
        if not isinstance(options, dict):
            raise ValueError(f"track setting {selector!r} must be an inline table")
        matches = instrumentation.matching_tracks(selector, note_tracks, names)
        if not matches:
            raise ValueError(f"track {selector!r} matched no note-bearing MIDI tracks")
        scale = float(options.get("velocity_scale", 1.0))
        offset = float(options.get("velocity_offset", 0.0))
        if not math.isfinite(scale) or scale < 0 or not math.isfinite(offset):
            raise ValueError(f"invalid velocity adjustment for track {selector!r}")
        for track_index in matches:
            adjustments[track_index] = (scale, offset)
    return adjustments


def _groove_options(
    settings: dict[str, Any],
    bar: int,
    track: int,
    note_tracks: list[int],
    names: dict[int, str],
) -> dict[str, Any] | None:
    """The groove that applies to one track in one bar.

    Grooves are per-track layers: the base `[groove]` covers every bar, and each
    `[[groove_sections]]` entry covers its bar range. A layer applies to the tracks
    it names (all tracks when it names none). Later layers override earlier ones
    for the tracks they apply to, inheriting the base's keys other than `tracks`.
    """
    def applies(layer: dict[str, Any]) -> bool:
        selectors = layer.get("tracks")
        if selectors is None:
            return True
        if not isinstance(selectors, list):
            raise ValueError("groove.tracks must be an array of track names or ordinals")
        return track in {_resolve_track_index(item, note_tracks, names) for item in selectors}

    base = settings.get("groove")
    base = base if isinstance(base, dict) else {}
    inherited = {key: value for key, value in base.items() if key != "tracks"}
    options = dict(base) if base and applies(base) else None
    for section in settings.get("groove_sections", []):
        start = int(section.get("start_bar", 0))
        end = int(section.get("end_bar", 0))
        if start < 1 or end < start:
            raise ValueError("groove section needs positive start_bar and end_bar >= start_bar")
        if start <= bar <= end and applies(section):
            options = {**(options if options is not None else inherited),
                       **{key: value for key, value in section.items()
                          if key not in ("start_bar", "end_bar")}}
    return options


def _groove_offsets(
    event: Event,
    settings: dict[str, Any],
    bars: BarMap,
    note_tracks: list[int],
    names: dict[int, str],
) -> tuple[float, float]:
    bar = bars.bar_at(event.tick)
    options = _groove_options(settings, bar, event.track, note_tracks, names)
    if options is None:
        return 0.0, 0.0
    subdivision = int(options.get("subdivision", 0))
    if subdivision <= 0:
        raise ValueError("groove.subdivision must be a positive note-value denominator")
    grid_ticks = bars.ppq * 4.0 / subdivision
    relative = event.tick - bars.bar_start(bar)
    position = round(relative / grid_ticks)
    if position < 0 or abs(relative - position * grid_ticks) > 1.0:
        return 0.0, 0.0
    velocity = options.get("velocity", [])
    timing = options.get("timing_ms", [])
    for key, values in (("velocity", velocity), ("timing_ms", timing)):
        if not isinstance(values, list):
            raise ValueError(f"groove.{key} must be an array")
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError(f"groove.{key} values must be finite")
    velocity_offset = float(velocity[position % len(velocity)]) if velocity else 0.0
    timing_offset = float(timing[position % len(timing)]) if timing else 0.0
    return timing_offset, velocity_offset


def _build_warper(
    settings: dict[str, Any], bars: BarMap, tempo: TempoMap
) -> TimeWarper:
    curves = []
    for index, event in enumerate(settings.get("rit", []), 1):
        if not isinstance(event, dict):
            raise ValueError(f"rit event {index} must be a table")
        if "start_bar" in event or "end_bar" in event:
            if "start_bar" not in event or "end_bar" not in event:
                raise ValueError("rit needs both start_bar and end_bar")
            start = bars.bar_start(int(event["start_bar"]))
            end = bars.bar_start(int(event["end_bar"]))
        elif "start_beat" in event or "end_beat" in event:
            if "start_beat" not in event or "end_beat" not in event:
                raise ValueError("rit needs both start_beat and end_beat")
            start = bars.beat_position(float(event["start_beat"]))
            end = bars.beat_position(float(event["end_beat"]))
        else:
            raise ValueError("rit needs start_bar/end_bar or start_beat/end_beat")
        factor = float(event.get("factor", 1.0))
        if not 0 < factor < math.inf or end <= start:
            raise ValueError(f"rit event {index} needs a positive factor and interval")

        def rate(tick: float, a=start, b=end, f=factor) -> float:
            envelope = arch((tick - a) / (b - a))
            speed = 1.0 + (f - 1.0) * envelope
            return 1.0 / speed - 1.0

        curves.append(SmoothCurve(start, end, rate))

    for index, event in enumerate(settings.get("fermata", []), 1):
        if not isinstance(event, dict) or "bar" not in event:
            raise ValueError(f"fermata event {index} needs a bar")
        start = bars.position(int(event["bar"]), float(event.get("beat", 1)))
        hold = float(event.get("hold", 0))
        width_beats = float(event.get("width_beats", 1.0))
        if hold < 0 or width_beats <= 0 or not math.isfinite(hold + width_beats):
            raise ValueError(f"fermata event {index} needs nonnegative hold and positive width")
        end = start + bars.beat_length_at(start) * width_beats
        duration_seconds = tempo.seconds_between(start, end)
        start_seconds = tempo.seconds_at(start)

        def rate(tick: float, a=start_seconds, duration=duration_seconds, amount=hold) -> float:
            u = min(1.0, max(0.0, (tempo.seconds_at(tick) - a) / duration))
            return amount * 6.0 * u * (1.0 - u) / duration

        curves.append(SmoothCurve(start, end, rate))
    return TimeWarper(curves)


def _pair_notes(events_by_track: list[list[Event]]):
    pairs: dict[int, int] = {}
    onset_events: list[Event] = []
    note_ons_by_track: dict[int, list[Event]] = defaultdict(list)
    for events in events_by_track:
        queues: dict[tuple[int, int], deque[Event]] = defaultdict(deque)
        for event in events:
            message = event.message
            if message.type == "note_on" and message.velocity > 0:
                queues[(message.channel, message.note)].append(event)
                onset_events.append(event)
                note_ons_by_track[event.track].append(event)
            elif message.type == "note_off" or (
                message.type == "note_on" and message.velocity == 0
            ):
                queue = queues[(message.channel, message.note)]
                if queue:
                    onset = queue.popleft()
                    pairs[id(onset)] = id(event)
                    pairs[id(event)] = id(onset)
    return pairs, onset_events, note_ons_by_track


def _protect_retriggers(
    events_by_track: list[list[Event]],
    pairs: dict[int, int],
    output_ticks: dict[int, int],
) -> None:
    for events in events_by_track:
        grouped: dict[tuple[int, int], list[Event]] = defaultdict(list)
        for event in events:
            if event.message.type == "note_on" and event.message.velocity > 0:
                grouped[(event.message.channel, event.message.note)].append(event)
        for retriggers in grouped.values():
            retriggers.sort(key=lambda event: (event.tick, event.order))
            for previous, current in zip(retriggers, retriggers[1:]):
                current_tick = max(output_ticks[id(current)], output_ticks[id(previous)] + 1)
                output_ticks[id(current)] = current_tick
                previous_off = pairs.get(id(previous))
                if previous_off is not None:
                    output_ticks[previous_off] = min(
                        output_ticks[previous_off], current_tick
                    )


def _chord_jitters(
    onset_events: list[Event],
    settings: dict[str, Any],
    tempo: TempoMap,
    rng: random.Random,
) -> dict[int, float]:
    humanize = settings.get("humanize", {})
    sigma = float(humanize.get("onset_jitter_ms", 6.0))
    roll = float(humanize.get("bass_roll_ms", 2.0))
    spread = float(humanize.get("max_chord_spread_ms", 15.0))
    if sigma < 0 or roll < 0 or spread < 0 or not all(map(math.isfinite, (sigma, roll, spread))):
        raise ValueError("humanize timing values must be finite and nonnegative")
    grouped: dict[int, list[Event]] = defaultdict(list)
    for event in onset_events:
        grouped[event.tick].append(event)
    result: dict[int, float] = {}
    for tick, chord in grouped.items():
        pitches = [event.message.note for event in chord]
        low, high = min(pitches), max(pitches)
        width = max(1, high - low)
        offsets = [
            rng.gauss(0.0, sigma) + (event.message.note - low) / width * roll
            for event in chord
        ]
        center = statistics.mean(offsets)
        half_spread = spread / 2.0
        offsets = [min(center + half_spread, max(center - half_spread, value)) for value in offsets]
        for event, offset in zip(chord, offsets):
            result[id(event)] = offset
    return result


def _clamp_chord_spread(
    onset_events: list[Event], offsets_ms: dict[int, float], max_spread_ms: float,
    player_of: dict[int, str] | None = None,
) -> None:
    """Bound each chord's spread: a chord is a tick's onsets, per room player when given."""
    if max_spread_ms < 0 or not math.isfinite(max_spread_ms):
        raise ValueError("humanize.max_chord_spread_ms must be finite and nonnegative")
    grouped: dict[Any, list[Event]] = defaultdict(list)
    for event in onset_events:
        key = event.tick if player_of is None else (player_of.get(event.track), event.tick)
        grouped[key].append(event)
    for chord in grouped.values():
        center = statistics.mean(offsets_ms[id(event)] for event in chord)
        lower = center - max_spread_ms / 2.0
        upper = center + max_spread_ms / 2.0
        for event in chord:
            offsets_ms[id(event)] = min(upper, max(lower, offsets_ms[id(event)]))


def _pedal_change_anchors(
    events_by_track: list[list[Event]],
    onset_events: list[Event],
    onset_jitters: dict[int, float],
    settings: dict[str, Any],
    ppq: int,
    player_of: dict[int, str] | None = None,
) -> dict[int, tuple[int, float]]:
    delay_ms = float(settings.get("pedal", {}).get("change_delay_ms", 30.0))
    if delay_ms < 0 or not math.isfinite(delay_ms):
        raise ValueError("pedal.change_delay_ms must be finite and nonnegative")
    def owner(event):
        if player_of is not None and event.track in player_of:
            return ("player", player_of[event.track])
        return ("channel", event.message.channel)

    note_groups = defaultdict(list)
    for event in onset_events:
        note_groups[(owner(event), event.tick)].append(event)
    changes: dict[int, tuple[int, float]] = {}
    for events in events_by_track:
        cc_by_channel: dict[int, list[Event]] = defaultdict(list)
        for event in events:
            if event.message.type == "control_change" and event.message.control == 64:
                cc_by_channel[event.message.channel].append(event)
        for channel_events in cc_by_channel.values():
            channel_events.sort(key=lambda e: (e.tick, e.order))
            for off_index, off in enumerate(channel_events[:-1]):
                if off.message.value >= 64:
                    continue
                on = channel_events[off_index + 1]
                if on.message.value < 64:
                    continue
                delta = on.tick - off.tick
                if delta < 0 or delta > max(1.0, ppq * 0.001):
                    continue
                anchor = on.tick
                candidates = [
                    note for (note_owner, note_tick), notes in note_groups.items()
                    if note_owner == owner(on)
                    and abs(note_tick - anchor) <= max(1.0, ppq * 0.001)
                    for note in notes
                ]
                if candidates:
                    bass = min(candidates, key=lambda e: e.message.note)
                    anchor = bass.tick
                    anchor_jitter_ms = onset_jitters.get(id(bass), 0.0)
                else:
                    anchor_jitter_ms = 0.0
                changes[id(off)] = (anchor, anchor_jitter_ms + delay_ms)
                changes[id(on)] = (anchor, anchor_jitter_ms + 3.0 * delay_ms)
    return changes


def _performed_position(tick, anchors, warper, *, preserve_warp=False):
    """Interpolate a score position, optionally retaining the warp between anchors."""
    j = bisect.bisect_right([a for a, _ in anchors], tick)
    if j == 0:
        return warper.apply(tick)
    if j == len(anchors):
        a, b = anchors[-1]
        return b + warper.apply(tick) - warper.apply(a)
    a, b = anchors[j - 1]
    c, d = anchors[j]
    fraction = (tick - a) / (c - a)
    if preserve_warp:
        return warper.apply(tick) + (1 - fraction) * (b - warper.apply(a)) + fraction * (d - warper.apply(c))
    return b + (d - b) * (tick - a) / (c - a)


def _room_pedal_players(events_by_track, onsets, player_of):
    """Resolve controller-only tracks by an unambiguous staff name or MIDI channel."""
    result = player_of.copy()
    names = _track_names(events_by_track)
    note_tracks = {e.track for e in onsets}
    for index, events in enumerate(events_by_track):
        pedals = [e for e in events if e.message.type == "control_change" and e.message.control == 64]
        if not pedals or index in result or index in note_tracks:
            continue
        components = set(names.get(index, "").split(":")) - {""}
        named = {player_of[n.track] for n in onsets if n.track in player_of
                 and components.intersection(names.get(n.track, "").split(":"))}
        channels = {e.message.channel for e in pedals}
        # An unseated track is a distinct possible owner, not evidence for a room player.
        heard = {player_of.get(n.track, ("track", n.track)) for n in onsets
                 if n.message.channel in channels}
        candidates = named if named else heard
        if len(candidates) > 1 and named:
            candidates = named.intersection(heard)
        owner = next(iter(candidates), None)
        if len(candidates) == 1 and owner in player_of.values():
            result[index] = owner
        else:
            warnings.warn(f"CC64 track {names.get(index, index)!r} has no unambiguous room player; "
                          "using the score clock", RuntimeWarning, stacklevel=2)
    return result


def _room_pedal_ticks(events_by_track, onsets, output_ticks, pedal_anchors,
                      player_of, settings, tempo, warper):
    """Place written CC64 on bass-led onset clocks, excluding release articulation."""
    notes = defaultdict(list)
    for note in onsets:
        if note.track in player_of:
            notes[player_of[note.track]].append(note)
    clocks = {}
    for player, player_notes in notes.items():
        # One pair of hands shares a bass-led onset anchor across its voices.
        anchors = {}
        for note in sorted(player_notes, key=lambda n: n.message.note):
            anchors.setdefault(note.tick, output_ticks[id(note)])
        clocks[player] = sorted(anchors.items())
    delay_ms = float(settings.get("pedal", {}).get("change_delay_ms", 30.0))
    for events in events_by_track:
        for event in events:
            message = event.message
            if (event.track not in player_of or message.type != "control_change"
                    or message.control != 64):
                continue
            anchors = clocks.get(player_of[event.track])
            if not anchors:
                warnings.warn(f"CC64 player {player_of[event.track]!r} has no note onsets; "
                              "using the score clock", RuntimeWarning, stacklevel=2)
                output_ticks[id(event)] = max(0, round(warper.apply(event.tick)))
                continue
            anchor = pedal_anchors.get(id(event), (event.tick, 0.0))[0]
            at = _performed_position(anchor, anchors, warper, preserve_warp=True)
            if anchor < anchors[0][0]:
                a, b = anchors[0]
                at += b - warper.apply(a)
            if id(event) in pedal_anchors:
                delay = delay_ms * (3 if message.value >= 64 else 1)
                at += tempo.ticks_for_ms(delay, anchor)
            output_ticks[id(event)] = max(0, round(at))


def _auto_pedal_events(
    onset_events, harmony_onsets, pairs, event_ticks, output_ticks, bindings,
    settings, names, bars, tempo, warper, score_end,
) -> list[tuple[int, int, mido.Message]]:
    """One rest-aware pedal per piano instrument, mirrored to its voice channels."""
    delay_ms = float(settings.get("pedal", {}).get("change_delay_ms", 30.0))
    if delay_ms < 0 or not math.isfinite(delay_ms):
        raise ValueError("pedal.change_delay_ms must be finite and nonnegative")
    instruments = defaultdict(list)
    for note in onset_events:
        instruments[bindings[note.track].instrument].append(note)
    changes = sustain.harmony_ticks(settings, harmony_onsets, bindings, bars, names, score_end)
    performed = sustain.performed_boundaries(changes, harmony_onsets, output_ticks, warper)
    generated = []
    for notes in instruments.values():
        windows = sustain.activity_windows(notes, pairs, event_ticks, bars, score_end,
                                           bridge_short_rests=True)
        stream = []
        for start, end in windows:
            first = min(output_ticks[id(n)] for n in notes if n.tick == start)
            releases = [output_ticks[pairs[id(n)]] for n in notes
                        if id(n) in pairs and event_ticks[pairs[id(n)]] == end]
            last = max(releases, default=round(warper.apply(end)))
            boundaries = [(start, first), *[(t, performed[t]) for t in changes
                                            if start < t < end and first < performed[t] < last],
                          (end, last)]
            for index, ((score_tick, left), (_, right)) in enumerate(zip(boundaries, boundaries[1:])):
                # Harmonic releases precede new notes; rest releases follow note-offs.
                off = right-1 if index < len(boundaries)-2 else right
                if off <= left:
                    continue
                on = min(off-1, left + round(tempo.ticks_for_ms(3 * delay_ms, score_tick)))
                stream.extend([(on, 127), (off, 0)])
        destinations = sorted({(n.track, n.message.channel) for n in notes})
        for track, channel in destinations:
            generated.extend((track, tick, mido.Message(
                "control_change", channel=channel, control=64, value=value
            )) for tick, value in stream)
    return generated


def expressive_time(
    midi: mido.MidiFile, settings: dict[str, Any], events_by_track: list[list[Event]] | None = None
) -> tuple[TempoMap, BarMap, TimeWarper]:
    """The score's tempo map, bar map and shared expressive warp (rit, fermata, arches)."""
    events_by_track = events_by_track if events_by_track is not None else _event_positions(midi)
    tempo = TempoMap(midi)
    bars = BarMap(midi)
    maximum_tick = max((event.tick for track in events_by_track for event in track), default=0)
    max_bar = max(1, math.ceil(maximum_tick / max(1, midi.ticks_per_beat * 4)) + 2)
    for event in settings.get("rit", []):
        if isinstance(event, dict):
            max_bar = max(max_bar, int(event.get("end_bar", 1)) + 1)
    for event in settings.get("fermata", []):
        if isinstance(event, dict):
            max_bar = max(max_bar, int(event.get("bar", 1)) + 1)
    bars.ensure_bar(max_bar)
    warper = _build_warper(settings, bars, tempo)
    arch_rate = expression.arch_tempo_rate(settings, bars, maximum_tick)
    if arch_rate is not None:
        warper.curves.append(SmoothCurve(0, bars.bar_start(max_bar), arch_rate))
    return tempo, bars, warper


def final_pedal_release(midi, piano_tracks, seconds, player_tracks=None):
    """Delay a final lift near its player's end, or the music's end without a room."""
    if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or not 0 <= seconds <= 10:
        raise ValueError("pedal.final_release_s must be in [0, 10]")
    if not seconds:
        return
    tempo = TempoMap(midi)
    absolute = []
    track_ends = []
    for track in midi.tracks:
        tick, rows, last_note = 0, [], 0
        for message in track:
            tick += message.time
            rows.append((tick, message))
            if message.type == "note_off" or (message.type == "note_on" and not message.velocity):
                last_note = max(last_note, tick)
        absolute.append(rows)
        track_ends.append(last_note)
    for index in piano_tracks:
        group = range(len(midi.tracks)) if player_tracks is None else player_tracks.get(index, [index])
        last_note = max((track_ends[i] for i in group), default=0)
        rows = absolute[index]
        releases = [(tick, message) for tick, message in rows
                    if message.type == "note_off" or (message.type == "note_on" and not message.velocity)]
        pedals = [(tick, message) for tick, message in rows
                  if message.type == "control_change" and message.control == 64]
        if not pedals or (not releases and player_tracks is None):
            continue
        tick, lift = pedals[-1]
        if lift.value >= 64:
            continue
        if player_tracks is None and abs(tempo.seconds_between(releases[-1][0], last_note)) > .25:
            continue
        if abs(tempo.seconds_between(tick, last_note)) > .25:
            continue
        delayed = tick + round(tempo.ticks_for_ms(seconds * 1000, tick))
        rows = [(delayed if message is lift else time, message) for time, message in rows
                if message.type != "end_of_track"]
        rows.sort(key=lambda row: row[0])
        track = mido.MidiTrack()
        previous = 0
        for time, message in rows:
            track.append(message.copy(time=time-previous))
            previous = time
        track.append(mido.MetaMessage("end_of_track", time=0))
        midi.tracks[index] = track


def attack_latencies(options):
    """Measured instrument attack references for the room's heard pulse."""
    result = {track: float(values.get('attack_latency_ms', 0)) for track, values in options.items()}
    if any(not math.isfinite(value) or value < 0 for value in result.values()):
        raise ValueError('tracks.attack_latency_ms must be finite and nonnegative')
    return result


def onset_styles(onsets, offsets_ms, settings, bars, note_tracks, names, options, bindings, player_of):
    """Apply groove, chord bounds and strokes, retaining per-player translations."""
    result = offsets_ms.copy()
    for event in onsets:
        result[id(event)] += _groove_offsets(event, settings, bars, note_tracks, names)[0]
    _clamp_chord_spread(onsets, result,
                        float(settings.get('humanize', {}).get('max_chord_spread_ms', 15)), player_of)
    plucked.strum_offsets(onsets, result, options, bindings, bars)
    return result


def perform_file(
    input_path: Path,
    output_path: Path,
    sidecar_path: Path | None = None,
    night: int | None = None,
) -> dict[str, Any]:
    """Perform one score MIDI. `night` selects a night of the sidecar's seed.

    Onset timing is built in this order: the room's coupled timekeepers and feel
    (players in `[room]`) or the generic chord jitter (other tracks); per-track
    groove; the chord-spread clamp; strums; the pedal anchors; the melody lead.
    """
    midi = mido.MidiFile(input_path)
    settings: dict[str, Any] = {}
    if sidecar_path is None:
        inferred = input_path.with_suffix(".perform.toml")
        sidecar_path = inferred if inferred.exists() else None
    if sidecar_path is not None and sidecar_path.exists():
        with sidecar_path.open("rb") as handle:
            settings = tomllib.load(handle)

    events_by_track = _event_positions(midi)
    note_tracks = _note_track_indices(events_by_track)
    names = _track_names(events_by_track)
    bindings = instrumentation.resolve_instruments(midi, settings)
    options = plucked.track_options(settings, note_tracks, names)
    # Sustain belongs to the instrument, not to LilyPond's GM program number.
    for index, binding in bindings.items():
        if binding.model != "skinpipe" and (binding.instrument not in instrumentation.PIANOS or options[index].get("let_ring", False)):
            events_by_track[index] = [e for e in events_by_track[index]
                                      if not (e.message.type == "control_change" and e.message.control == 64)]
    tempo, bars, warper = expressive_time(midi, settings, events_by_track)
    adjustments = _track_adjustments(settings, events_by_track)
    pairs, onset_events, _ = _pair_notes(events_by_track)
    expr_velocity, expr_onset_ms, expr_release_ms = expression.expression_offsets(
        events_by_track, pairs, onset_events, settings, bars,
        lambda selector: _resolve_track_index(selector, note_tracks, names),
    )
    base_seed = int(settings.get("seed", 1729))
    seed = room.night_seed(base_seed, night)
    rng = random.Random(seed)
    the_room = room.parse(settings, note_tracks, names)
    score_end = max(sum(m.time for m in track) for track in midi.tracks)
    room_velocity: dict[int, float] = {}
    if the_room is None:
        onset_jitters = _chord_jitters(onset_events, settings, tempo, rng)
    else:
        onset_jitters = _chord_jitters(
            [e for e in onset_events if e.track not in the_room.player_of], settings, tempo, rng)
        latencies = attack_latencies(options)

        def performed_bias(offsets):
            styled = onset_styles(onset_events, offsets, settings, bars, note_tracks, names,
                                  options, bindings, the_room.player_of)
            return {id(note): styled[id(note)] + expr_onset_ms.get(id(note), 0.0) + latencies[note.track]
                    for note in onset_events}

        room_offsets, room_velocity, room_night = room.timing(
            the_room, onset_events, bars, tempo, warper, score_end, seed, compose=performed_bias)
        onset_jitters.update(room_offsets)
    groove_velocity = {id(event): _groove_offsets(event, settings, bars, note_tracks, names)[1]
                       for event in onset_events}
    onset_jitters = onset_styles(onset_events, onset_jitters, settings, bars, note_tracks, names,
                                options, bindings, None if the_room is None else the_room.player_of)
    pedal_players = None if the_room is None else _room_pedal_players(
        events_by_track, onset_events, the_room.player_of)
    pedal_anchors = _pedal_change_anchors(
        events_by_track, onset_events, onset_jitters, settings, midi.ticks_per_beat,
        pedal_players,
    )
    # the melody lead is added after the chord-spread clamp, which would otherwise undo it
    for event in onset_events:
        onset_jitters[id(event)] += expr_onset_ms.get(id(event), 0.0)

    tick_shifts: dict[int, float] = {}
    release_ticks = {id(e): e.tick for track in events_by_track for e in track}
    for event in onset_events:
        jitter_ms = onset_jitters[id(event)]
        tick_shifts[id(event)] = tempo.ticks_for_ms(jitter_ms, event.tick)
        partner = pairs.get(id(event))
        if partner is not None:
            release = tempo.ticks_for_ms(expr_release_ms.get(partner, 0.0), event.tick)
            if release < 0:  # a breath never takes more than 60% of the note
                release = max(release, -0.6 * (release_ticks[partner] - event.tick))
            tick_shifts[partner] = tick_shifts[id(event)] + release

    humanize = settings.get("humanize", {})
    velocity_sigma = float(humanize.get("velocity_jitter", 4.0))
    if velocity_sigma < 0 or not math.isfinite(velocity_sigma):
        raise ValueError("humanize.velocity_jitter must be finite and nonnegative")
    auto_pedal = bool(settings.get("pedal", {}).get("auto", False))
    score_has_pedal = any(
        event.message.type == "control_change" and event.message.control == 64
        for track in events_by_track for event in track
    )
    scored_pianos = {bindings[index].instrument for index, es in enumerate(events_by_track)
                     if index in bindings and bindings[index].instrument in instrumentation.PIANOS
                     for e in es if e.message.type == "control_change" and e.message.control == 64}
    auto_onsets = [e for e in onset_events
                   if bindings[e.track].instrument in instrumentation.PIANOS
                   and not options[e.track].get("let_ring", False)
                   and bindings[e.track].instrument not in scored_pianos]
    add_auto_pedal = auto_pedal and bool(auto_onsets)
    # A drum machine keeps its own clock: the score's tempo map plus composed swing,
    # never the room, groove, expression or humanize jitter.
    machines = {i: machine_time.timing(settings, b.model) for i, b in bindings.items()
                if b.model in machine_time.MACHINES}
    # A humanized machine still fires a step's voices together: one draw per step, shared.
    machine_rng = random.Random(seed + 808)
    step_ms: dict[int, float] = {}
    machine_ms: dict[int, float] = {}
    for event in sorted(onset_events, key=lambda e: e.tick):
        if event.track in machines and machines[event.track][2]:
            if event.tick not in step_ms:
                step_ms[event.tick] = machine_rng.gauss(0.0, 1.0)
            machine_ms[id(event)] = step_ms[event.tick]*machines[event.track][2]
            if pairs.get(id(event)) is not None:
                machine_ms[pairs[id(event)]] = machine_ms[id(event)]
    output_ticks: dict[int, int] = {}
    for track_index, events in enumerate(events_by_track):
        for event in events:
            if track_index in machines:
                swing, unit, _ = machines[track_index]
                tick = warper.apply(machine_time.swing_tick(event.tick, midi.ticks_per_beat, swing, unit))
                output_ticks[id(event)] = max(0, round(tick + tempo.ticks_for_ms(machine_ms.get(id(event), 0.0), event.tick)))
                continue
            base_tick = warper.apply(event.tick)
            if id(event) in pedal_anchors:
                anchor, delay_ms = pedal_anchors[id(event)]
                base_tick = warper.apply(anchor) + tempo.ticks_for_ms(delay_ms, anchor)
            else:
                base_tick += tick_shifts.get(id(event), 0.0)
            output_ticks[id(event)] = max(0, round(base_tick))
    if the_room is not None:
        intended = {}
        for event in onset_events:
            scale, offset = adjustments.get(event.track, (1.0, 0.0))
            intended[id(event)] = (event.message.velocity * scale + offset
                                   + groove_velocity.get(id(event), 0.0)
                                   + expr_velocity.get(id(event), 0.0)
                                   + room_velocity.get(id(event), 0.0))
        for key, value in room.dynamics(the_room, onset_events, pairs, release_ticks,
                                        intended, bars, room_night).items():
            room_velocity[key] = room_velocity.get(key, 0.0) + value
    if add_auto_pedal:
        generated_pedals = _auto_pedal_events(
            auto_onsets, onset_events, pairs, release_ticks, output_ticks, bindings,
            settings, names, bars, tempo, warper, score_end,
        )
    else:
        generated_pedals = []
    generated_pedals.extend(plucked.let_ring(
        onset_events, pairs, release_ticks, output_ticks, options, bindings,
        settings, bars, warper, names,
        score_end,
    ))
    # Score-addressed controllers follow the room's score/performance clock.
    for entry in settings.get("controls", []):
        if set(entry) - {"track", "bar", "beat", "control", "value"}:
            raise ValueError("unknown score controller field")
        index = _resolve_track_index(entry["track"], note_tracks, names)
        bar, beat = entry.get("bar", 1), entry.get("beat", 1)
        if type(bar) is not int or bar < 1 or not math.isfinite(beat) or beat < 1:
            raise ValueError("controller position must be positive")
        for key in ("control", "value"):
            if type(entry[key]) is not int or not 0 <= entry[key] <= 127:
                raise ValueError("controller number/value must be MIDI integers")
        tick = bars.position(bar, beat)
        candidates = [e for e in onset_events if e.track == index]
        channel = candidates[0].message.channel
        anchors = sorted((e.tick, output_ticks[id(e)]) for e in candidates)
        at = _performed_position(tick, anchors, warper, preserve_warp=the_room is not None)
        generated_pedals.append((index, max(0, round(at)), mido.Message(
            "control_change", channel=channel, control=entry["control"], value=entry["value"])))
    _protect_retriggers(events_by_track, pairs, output_ticks)
    # An onset cannot move past its own release even for very short strummed notes.
    for event in onset_events:
        partner = pairs.get(id(event))
        if partner is not None:
            output_ticks[partner] = max(output_ticks[partner], output_ticks[id(event)] + 1)

    if the_room is not None:
        _room_pedal_ticks(events_by_track, onset_events, output_ticks, pedal_anchors,
                          pedal_players, settings, tempo, warper)

    output = mido.MidiFile(type=midi.type, ticks_per_beat=midi.ticks_per_beat)
    velocity_controls = stage.controls(settings, 'velocity', note_tracks, names, bars)
    touch_controls = stage.pitch_controls(settings, note_tracks, names)
    note_count = 0
    pitches: list[int] = []
    for track_index, events in enumerate(events_by_track):
        transformed: list[tuple[int, int, int, mido.Message | mido.MetaMessage]] = []
        for event in events:
            message = event.message.copy(time=0)
            out_tick = output_ticks[id(event)]
            if message.type == "note_on" and message.velocity > 0:
                scale, offset = adjustments.get(track_index, (1.0, 0.0))
                # Draw for every note so adding a machine never reshuffles the other tracks' jitter.
                jitter = rng.gauss(0.0, velocity_sigma)
                variation = 0.0 if track_index in machines else (
                    groove_velocity.get(id(event), 0.0) + expr_velocity.get(id(event), 0.0)
                    + room_velocity.get(id(event), 0.0) + jitter)
                target = stage.interpolate(velocity_controls[track_index], event.tick) if track_index in velocity_controls else None
                if target is not None:
                    # A pianissimo touch carries proportionately small expressive deviations.
                    variation *= min(1, target/64)
                velocity = message.velocity*scale + offset + variation
                if target is not None:
                    velocity *= target / event.message.velocity
                if track_index in touch_controls:
                    velocity *= stage.interpolate(touch_controls[track_index], message.note)
                message.velocity = min(127, max(1, round(velocity)))
                note_count += 1
                pitches.append(message.note)
            priority = 0 if message.type == "note_off" or (
                message.type == "note_on" and message.velocity == 0
            ) else 1 if message.type == "note_on" else 0
            transformed.append((out_tick, priority, event.order, message))
        for extra_index, (target_track, out_tick, message) in enumerate(generated_pedals):
            if target_track == track_index:
                priority = -1 if message.control != 64 else (0 if message.value < 64 else 1)
                transformed.append((out_tick, priority, 1_000_000 + extra_index, message))
        transformed.sort(key=lambda item: (item[0], item[1], item[2]))
        new_track = mido.MidiTrack()
        last_tick = 0
        for tick, _, _, message in transformed:
            message.time = tick - last_tick
            new_track.append(message)
            last_tick = tick
        new_track.append(mido.MetaMessage("end_of_track", time=0))
        output.tracks.append(new_track)

    piano_tracks = {i for i, binding in bindings.items()
                    if binding.instrument in instrumentation.PIANOS}
    player_tracks = None
    if pedal_players is not None:
        piano_players = {player for i, player in pedal_players.items() if i in piano_tracks}
        piano_tracks.update(i for i, player in pedal_players.items()
                            if i not in bindings and player in piano_players)
        player_tracks = {i: [j for j, owner in pedal_players.items() if owner == player]
                         for i, player in pedal_players.items()}
        source_ends = {i: max((e.tick for e in es if e.message.type == "note_off"
                              or (e.message.type == "note_on" and not e.message.velocity)), default=0)
                       for i, es in enumerate(events_by_track)}
        # Eligibility belongs to the score; lift alignment belongs to the player's clock.
        # A player still sounding within 0.25 s of the score's last release counts as
        # sounding at the end, the same tolerance the no-room path uses.
        ensemble_end = max(source_ends.values(), default=0)
        score_tempo = TempoMap(midi)
        piano_tracks = {i for i in piano_tracks
                        if score_tempo.seconds_between(
                            max(source_ends[j] for j in player_tracks.get(i, [i])), ensemble_end) <= .25}
    final_pedal_release(output, sorted(piano_tracks),
                        settings.get("pedal", {}).get("final_release_s", 0), player_tracks)
    output.save(output_path)
    source_note_count = sum(
        1 for event in onset_events
        if event.message.type == "note_on" and event.message.velocity > 0
    )
    return {
        "notes_in": source_note_count,
        "notes_out": note_count,
        "pitch_min": min(pitches) if pitches else None,
        "pitch_max": max(pitches) if pitches else None,
        "seed": seed,
        "base_seed": base_seed,
        "night": night,
        "auto_pedal_added": add_auto_pedal,
        "pedal_in_score": score_has_pedal,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="LilyPond MIDI file")
    parser.add_argument("output", type=Path, help="performed MIDI output")
    parser.add_argument("--sidecar", type=Path, help="optional performance TOML")
    parser.add_argument("--night", type=int, help="perform this night of the sidecar's seed")
    args = parser.parse_args(argv)
    try:
        summary = perform_file(args.input, args.output, args.sidecar, args.night)
    except (OSError, ValueError, KeyError, tomllib.TOMLDecodeError) as error:
        print(f"perform.py: {error}", file=sys.stderr)
        return 2
    print(
        f"Performed {args.input} -> {args.output}: "
        f"{summary['notes_in']} notes, seed {summary['seed']}"
        + (f" (night {summary['night']} of {summary['base_seed']})" if summary["night"] is not None else "")
        + ", "
        f"pitch {summary['pitch_min']}..{summary['pitch_max']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
