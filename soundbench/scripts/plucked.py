"""Strummed onsets and harmony-aware plucked-string sustain (duck-typed MIDI Events)."""
from collections import defaultdict
import bisect
import math
import mido
from soundbench.scripts.instrumentation import CC64_SUSTAIN, matching_tracks
from soundbench.scripts import sustain


def track_options(settings, note_tracks, names):
    options = {i: {} for i in note_tracks}
    for selector, values in settings.get("tracks", {}).items():
        if not isinstance(values, dict):
            raise ValueError(f"track {selector!r} needs a table")
        matches = matching_tracks(selector, note_tracks, names)
        if not matches:
            raise ValueError(f"track {selector!r} matched no MIDI tracks")
        if "let_ring" in values and not isinstance(values["let_ring"], bool):
            raise ValueError("let_ring must be true or false")
        for i in matches:
            options[i].update(values)
    return options


def strum_offsets(onsets, offsets_ms, options, bindings, bars):
    chords = defaultdict(list)
    for note in onsets:
        chords[(note.track, note.tick)].append(note)
    for (track, tick), notes in chords.items():
        spec = options[track].get("strum", False)
        if spec is False or spec is None:
            continue
        if spec is True:
            spec = {}
        if not isinstance(spec, dict):
            raise ValueError("strum must be a boolean or inline table")
        default = 12 if bindings[track].instrument.startswith("banjo") else 18
        spread = float(spec.get("spread_ms", default))
        direction = spec.get("direction", "alternate")
        pattern = spec.get("pattern")
        if not math.isfinite(spread) or not 0 <= spread <= 200:
            raise ValueError("strum.spread_ms must be between 0 and 200")
        relative = tick-bars.bar_start(bars.bar_at(tick))
        beat_index = relative // bars.beat_length_at(tick)
        if pattern is not None:
            if not isinstance(pattern, list) or not pattern or any(p not in ("down", "up") for p in pattern):
                raise ValueError("strum.pattern must be a nonempty array of 'down'/'up'")
            direction = pattern[beat_index % len(pattern)]
        elif direction == "alternate":
            eighth_index = relative // (bars.ppq / 2)
            direction = "down" if eighth_index % 2 == 0 else "up"
        if direction not in ("down", "up"):
            raise ValueError("strum.direction must be down, up, or alternate")
        if len(notes) < 2:
            continue
        notes.sort(key=lambda n: n.message.note, reverse=direction == "up")
        # Shared jitter preserves stroke order; melody lead can precede it independently.
        center = sum(offsets_ms[id(n)] for n in notes) / len(notes)
        for rank, note in enumerate(notes):
            offsets_ms[id(note)] = center + rank * spread / (len(notes)-1)


def let_ring(onsets, pairs, event_ticks, output_ticks, options, bindings, settings, bars, warper, names, score_end):
    """Sustain within harmony/rest windows; return (track, tick, CC64) triples."""
    changes = sustain.harmony_ticks(settings, onsets, bindings, bars, names, score_end)
    performed = sustain.performed_boundaries(changes, onsets, output_ticks, warper)
    generated = []
    for track, binding in bindings.items():
        if not options[track].get("let_ring", False):
            continue
        notes = [n for n in onsets if n.track == track]
        if not notes:
            continue
        windows = sustain.activity_windows(notes, pairs, event_ticks, bars, score_end)
        regions = []
        for start, end in windows:
            first = min(output_ticks[id(n)] for n in notes if n.tick == start)
            releases = [output_ticks[pairs[id(n)]] for n in notes
                        if id(n) in pairs and event_ticks[pairs[id(n)]] == end]
            last = max(releases, default=round(warper.apply(end)))
            boundaries = sorted({first, last, *[performed[t] for t in changes
                                                if start < t < end and first < performed[t] < last]})
            regions.extend(zip(boundaries, boundaries[1:]))
        if binding.instrument in CC64_SUSTAIN:
            for channel in sorted({n.message.channel for n in notes}):
                for start, end in regions:
                    generated.extend([
                        (track, start, mido.Message("control_change", channel=channel, control=64, value=127)),
                        (track, max(start + 1, end-1), mido.Message("control_change", channel=channel, control=64, value=0)),
                    ])
        else:
            boundaries = sorted({end for _, end in regions})
            same_pitch = defaultdict(list)
            for note in notes:
                same_pitch[(note.message.channel, note.message.note)].append(output_ticks[id(note)])
            for ticks in same_pitch.values():
                ticks.sort()
            for note in notes:
                off = pairs.get(id(note))
                if off is None:
                    continue
                tick = output_ticks[id(note)]
                next_change = next((b for b in boundaries if b > tick), output_ticks[off])
                ticks = same_pitch[(note.message.channel, note.message.note)]
                index = bisect.bisect_right(ticks, tick)
                retrigger = ticks[index] if index < len(ticks) else next_change
                output_ticks[off] = min(retrigger, max(output_ticks[off], next_change))
    return generated
