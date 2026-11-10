"""Shared harmonic boundaries and rest-aware sustain windows in score time."""
from bisect import bisect_right
from collections import defaultdict

from soundbench.scripts.instrumentation import PIANOS, matching_tracks


def harmony_ticks(settings, onsets, bindings, bars, names, score_end):
    """Release every bar and at explicit or detected within-bar harmony changes.

    Bar lines guard intervals where a selected harmony voice rests. Guitar/piano
    chords supplement selected changes, except between neighboring selected onsets
    no more than a bar apart; nested pitch-class sets accumulate until a real change.
    """
    changes = {bars.bar_start(b) for b in range(1, bars.bar_at(score_end) + 1)}
    pedal = settings.get("pedal", {})
    explicit = pedal.get("harmony_changes")
    if explicit is not None:
        if not isinstance(explicit, list):
            raise ValueError("pedal.harmony_changes must be an array of bar/beat tables")
        if any(not isinstance(p, dict) or "bar" not in p for p in explicit):
            raise ValueError("each harmony change needs a bar and optional beat")
        changes.update(bars.position(int(p["bar"]), float(p.get("beat", 1))) for p in explicit)
        return sorted(changes)
    selectors = pedal.get("harmony_tracks")
    note_tracks = list(bindings)
    tracks = {i for i, b in bindings.items() if b.instrument == "bass"}
    if selectors is not None:
        if not isinstance(selectors, list):
            raise ValueError("pedal.harmony_tracks must be an array")
        tracks = set()
        for selector in selectors:
            matches = matching_tracks(selector, note_tracks, names)
            if not matches:
                raise ValueError(f"harmony track {selector!r} matched no note tracks")
            tracks.update(matches)
    melody = settings.get("expression", {}).get("melody")
    melodic_tracks = set(matching_tracks(melody, note_tracks, names)) if melody is not None else set()
    groups = defaultdict(lambda: defaultdict(list))
    for note in onsets:
        if note.track in tracks or (note.track not in melodic_tracks
                                   and bindings[note.track].instrument in ("guitar", *PIANOS)):
            groups[note.tick][note.track].append(note.message.note)
    selected_onsets = sorted({note.tick for note in onsets if note.track in tracks})
    previous_by_track: dict[int, frozenset[int]] = {}
    for tick, notes_by_track in sorted(groups.items()):
        for track, notes in notes_by_track.items():
            if track not in tracks and len(notes) < 2:
                continue
            signature = frozenset(p % 12 for p in notes)
            previous = previous_by_track.get(track)
            if previous is None:
                previous_by_track[track] = signature
                continue
            if signature <= previous or previous <= signature:
                previous_by_track[track] = previous | signature
                continue
            right_index = bisect_right(selected_onsets, tick)
            between_selected_onsets = (
                track not in tracks
                and 0 < right_index < len(selected_onsets)
                and selected_onsets[right_index - 1] < tick < selected_onsets[right_index]
                and bars.bar_at(selected_onsets[right_index])
                - bars.bar_at(selected_onsets[right_index - 1]) <= 1
            )
            if not between_selected_onsets:
                changes.add(tick)
                previous_by_track[track] = signature
    return sorted(changes)


def activity_windows(notes, pairs, event_ticks, bars, score_end, bridge_short_rests=True):
    """Union note spans, bridging gaps shorter than a beat when enabled."""
    spans = sorted((n.tick, event_ticks.get(pairs.get(id(n)), n.tick + 1)) for n in notes)
    windows = []
    for start, end in spans:
        gap_limit = bars.beat_length_at(start) if bridge_short_rests else 0
        if windows and (start <= windows[-1][1] or start - windows[-1][1] < gap_limit):
            windows[-1] = (windows[-1][0], max(end, windows[-1][1]))
        else:
            windows.append((start, end))
    if windows and bridge_short_rests and score_end - windows[-1][1] < bars.beat_length_at(windows[-1][1]):
        windows[-1] = (windows[-1][0], score_end)
    return windows


def performed_boundaries(changes, onsets, output_ticks, warper):
    """Lift before the earliest humanized note of a new harmonic onset."""
    boundaries = {t: max(0, round(warper.apply(t))) for t in changes}
    for note in onsets:
        if note.tick in boundaries:
            boundaries[note.tick] = min(boundaries[note.tick], output_ticks[id(note)])
    return boundaries
