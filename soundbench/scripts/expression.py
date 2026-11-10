"""Structured expression for the performance pass: voicing, phrasing, articulation.

Random jitter makes a render sloppy, not human. What a pianist adds is structured:
the tune sings over the accompaniment and sounds a little ahead of it; phrases
swell toward their high point and relax at the cadence, pressing the tempo forward
as they grow and easing it as they close (Todd's phrase-arch model, nested over
phrase and sub-phrase); higher melody notes are played a little louder; legato
notes overlap slightly. This module computes those deviations from the score MIDI.

Sidecar keys, all under `[expression]` and all optional:

  melody = "upper"            track that carries the tune (name or 1-based number)
  melody_line = "top"         "top" or "bottom": which sounding line of that track is the tune
  drone = [79]                pitches never treated as the tune (e.g. a banjo's fifth string)
  melody_boost = 10           velocity added to tune notes
  inner_cut = 8               velocity removed from other notes in the tune's track
  accomp_cut = 8              velocity removed from other tracks' notes, except their bass
  bass_boost = 0              velocity added to the lowest sounding note of other tracks
  melody_lead_ms = 20         tune onsets come this much before the rest
  high_loud = 0.15            velocity per semitone above the phrase's mean tune pitch
  high_loud_cap = 4           ... capped at this many velocity units either way
  legato_ms = 30              notes held this far into the following onset of the same track
  legato = "melody"           "melody", "all" or "none": which notes get the overlap
  metric = [3, 0, -1]         velocity by beat within the bar (cycles if the bar is longer)
  breath_ms = [40, 110]       the tune's last note before a phrase end releases this early:
                              [sub-phrase ends, phrase ends] (the first two arch levels)
  [[expression.arch]]         one table per nesting level, outermost first, e.g.
    bars = 8                  phrase length in bars (phrases start at `offset_bars + 1`)
    offset_bars = 0
    peak = 0.6                where in the phrase the high point falls (0..1)
    velocity = [-4, 8, -8]    velocity offset at the start, the peak and the end
                              (a single number v means [-v, v, -v])
    tempo = [0.97, 1.04, 0.92]  tempo multiplier at the start, the peak and the end
                              (a single number t means [1-t, 1+t, 1-t])

The defaults and the suggested ranges follow the KTH performance rules and the
voicing and melody-lead literature (melody leads of 15-25 ms; phrase arches that
end quieter and slower than they begin).
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


def _shape(u: float, peak: float) -> float:
    """-1 at the phrase ends, +1 at the peak, smooth (half-sine on each side)."""
    return 2.0 * _rise(u, peak) - 1.0


def _rise(u: float, peak: float) -> float:
    """0 at the phrase ends, 1 at the peak (half-sine on each side)."""
    u = min(1.0, max(0.0, u))
    if u <= peak:
        return math.sin(0.5 * math.pi * (u / peak)) if peak > 0 else 1.0
    return math.cos(0.5 * math.pi * ((u - peak) / (1.0 - peak))) if peak < 1 else 1.0


def _three(value, symmetric) -> tuple[float, float, float]:
    """A [start, peak, end] triple; a single number means `symmetric(number)`."""
    if isinstance(value, (list, tuple)):
        if len(value) != 3:
            raise ValueError("an arch value list needs [start, peak, end]")
        return tuple(float(x) for x in value)
    return symmetric(float(value))


def _arch_value(u: float, peak: float, triple: tuple[float, float, float]) -> float:
    start, top, end = triple
    r = _rise(u, peak)
    return (start if u <= peak else end) + (top - (start if u <= peak else end)) * r


def _levels(settings: dict[str, Any]) -> list[dict[str, Any]]:
    levels = settings.get("expression", {}).get("arch", [])
    if isinstance(levels, dict):
        levels = [levels]
    return [level for level in levels if int(level.get("bars", 0)) > 0]


def _phrase_position(level: dict[str, Any], bars, tick: float) -> tuple[float, int, int]:
    """(u in 0..1, start tick, end tick) of the phrase containing `tick`."""
    length = int(level["bars"])
    offset = int(level.get("offset_bars", 0))
    bar = bars.bar_at(int(tick))
    index = max(0, (bar - 1 - offset) // length)
    first = offset + index * length + 1
    start = bars.bar_start(max(1, first))
    end = bars.bar_start(max(1, first) + length)
    return (tick - start) / max(1, end - start), start, end


def arch_tempo_rate(settings: dict[str, Any], bars, last_tick: int):
    """A tick-rate function for SmoothCurve: summed phrase-arch tempo deviations."""
    levels = [(level, _three(level["tempo"], lambda t: (1 - t, 1 + t, 1 - t)))
              for level in _levels(settings) if "tempo" in level]
    levels = [(level, triple) for level, triple in levels if triple != (1.0, 1.0, 1.0)]
    if not levels:
        return None

    def rate(tick: float) -> float:
        speed = 1.0
        for level, triple in levels:
            u, _, _ = _phrase_position(level, bars, tick)
            speed *= _arch_value(u, float(level.get("peak", 0.6)), triple)
        return 1.0 / speed - 1.0

    return rate


def expression_offsets(events_by_track, pairs, onset_events, settings, bars, resolve_track):
    """Per-event deviations: velocity offsets, onset shifts (ms) and note-off extensions (ms).

    `resolve_track(selector)` maps a track name or number to its track index.
    Returns three dicts keyed by id(event): velocity, onset_ms (onsets), release_ms (note-offs).
    """
    ex = settings.get("expression", {})
    velocity: dict[int, float] = defaultdict(float)
    onset_ms: dict[int, float] = defaultdict(float)
    release_ms: dict[int, float] = defaultdict(float)
    if not ex:
        return velocity, onset_ms, release_ms

    melody_track = resolve_track(ex["melody"]) if "melody" in ex else None
    bottom = ex.get("melody_line", "top") == "bottom"
    drone = set(int(p) for p in ex.get("drone", []))
    by_id = {id(e): e for track in events_by_track for e in track}

    # notes per track as (on_tick, off_tick, pitch, onset_event)
    notes: dict[int, list] = defaultdict(list)
    for e in onset_events:
        off = by_id.get(pairs.get(id(e)))
        notes[e.track].append((e.tick, off.tick if off else e.tick + 1, e.message.note, e))

    def sounding(track, tick):
        return [n for n in notes[track] if n[0] <= tick < n[1]]

    tune: set[int] = set()
    if melody_track is not None:
        for on, off, pitch, e in notes[melody_track]:
            if pitch in drone:
                continue
            others = [n[2] for n in sounding(melody_track, on) if n[2] not in drone]
            if (not bottom and pitch >= max(others, default=pitch)) or \
               (bottom and pitch <= min(others, default=pitch)):
                tune.add(id(e))

    boost = float(ex.get("melody_boost", 10))
    inner = float(ex.get("inner_cut", 8))
    accomp = float(ex.get("accomp_cut", 8))
    bass = float(ex.get("bass_boost", 0))
    lead = float(ex.get("melody_lead_ms", 20))
    high_loud = float(ex.get("high_loud", 0.15))
    high_cap = float(ex.get("high_loud_cap", 4))
    metric = [float(v) for v in ex.get("metric", [])]
    levels = _levels(settings)

    # mean tune pitch per top-level phrase, for the high-loud rule
    tune_pitch = defaultdict(list)
    top = levels[0] if levels else None
    for e in onset_events:
        if id(e) in tune and top:
            _, start, _ = _phrase_position(top, bars, e.tick)
            tune_pitch[start].append(e.message.note)
    mean_pitch = {k: sum(v) / len(v) for k, v in tune_pitch.items()}

    for e in onset_events:
        v = 0.0
        if id(e) in tune:
            v += boost
            onset_ms[id(e)] -= lead
            if top and high_loud:
                _, start, _ = _phrase_position(top, bars, e.tick)
                lift = high_loud * (e.message.note - mean_pitch.get(start, e.message.note))
                v += max(-high_cap, min(high_cap, lift))
        elif e.track == melody_track:
            v -= inner
        else:
            lowest = min((n[2] for n in sounding(e.track, e.tick)), default=e.message.note)
            v += bass if e.message.note <= lowest else -accomp
        for level in levels:
            u, _, _ = _phrase_position(level, bars, e.tick)
            triple = _three(level.get("velocity", 0), lambda x: (-x, x, -x))
            v += _arch_value(u, float(level.get("peak", 0.6)), triple)
        if metric:
            beat_len = max(1, bars.beat_length_at(e.tick))
            beat = int((e.tick - bars.bar_start(bars.bar_at(e.tick))) // beat_len)
            v += metric[beat % len(metric)]
        velocity[id(e)] = v

    legato = float(ex.get("legato_ms", 0))
    which = ex.get("legato", "melody")
    if legato and which != "none":
        for track, track_notes in notes.items():
            onsets = sorted({n[0] for n in track_notes})
            for on, off, pitch, e in track_notes:
                if which == "melody" and id(e) not in tune:
                    continue
                # extend only when another onset in this track follows close behind the release
                i = next((k for k, t in enumerate(onsets) if t >= off), None)
                if i is not None and onsets[i] - off <= max(1, bars.beat_length_at(off)) // 8:
                    off_event = pairs.get(id(e))
                    if off_event is not None:
                        release_ms[off_event] += legato
    # breathe: the tune's last note before a phrase end releases early (negative extension)
    breath = [float(x) for x in ex.get("breath_ms", [])]
    for depth, level in enumerate(levels[:2]):
        # the outermost level takes the phrase-end breath, the next the sub-phrase breath
        amount = (breath[-1] if depth == 0 else breath[0]) if breath else 0.0
        if not amount or melody_track is None:
            continue
        ends = set()
        for on, off, pitch, e in notes[melody_track]:
            _, _, end = _phrase_position(level, bars, on)
            ends.add(end)
        for end in ends:
            last = max((n for n in notes[melody_track] if id(n[3]) in tune and n[0] < end <= n[1] + 1),
                       key=lambda n: n[0], default=None)
            if last is None:
                last = max((n for n in notes[melody_track] if id(n[3]) in tune and n[1] <= end),
                           key=lambda n: n[1], default=None)
            if last is not None and (off_event := pairs.get(id(last[3]))) is not None:
                release_ms[off_event] = min(release_ms[off_event], 0.0) - amount
    return velocity, onset_ms, release_ms
