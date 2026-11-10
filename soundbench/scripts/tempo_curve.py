# /// script
# dependencies = ["mido>=1.3,<2"]
# ///
"""Per-bar performed duration relative to score duration, including meter/tempo changes.

Notes are matched by (track, pitch, occurrence), so humanized chord reordering
and another player's repeated pitches don't confuse the correspondence.
1.00 means in tempo; 1.10 means the bar took 10% longer than written.
Silent boundaries are interpolated between matched score/performance onsets.
The optional quarters-per-bar and score-bpm arguments select a constant grid.
"""
import argparse
import bisect
from collections import Counter, defaultdict
import statistics

import mido
from soundbench.scripts.perform import BarMap, TempoMap


def onsets(midi):
    result = {}
    for index, track in enumerate(midi.tracks):
        seen, tick = Counter(), 0
        for message in track:
            tick += message.time
            if message.type in ("note_on", "note_off"):
                kind = "on" if message.type == "note_on" and message.velocity else "off"
                occurrence = (message.note, kind)
                key = (index, message.note, kind, seen[occurrence])
                seen[occurrence] += 1
                result[key] = tick
    return result


def ratios(score, performed, quarters=None, bpm=None):
    st, pt = TempoMap(score), TempoMap(performed)
    so, po = onsets(score), onsets(performed)
    matched = defaultdict(list)
    for key, tick in so.items():
        if key in po:
            matched[st.seconds_at(tick)].append(pt.seconds_at(po[key]))
    if not matched:
        return []
    # Multiple players on the same beat contribute one median time anchor.
    anchors = sorted((s, statistics.median(times)) for s, times in matched.items())
    if anchors[0][0] > 0:
        anchors.insert(0, (0.0, 0.0))
    xs = [s for s, _ in anchors]
    def mapped(s):
        index = bisect.bisect_right(xs, s)
        if index == 0:
            return s
        if index == len(anchors):
            x, y = anchors[-1]
            return y + s - x
        x0, y0 = anchors[index - 1]; x1, y1 = anchors[index]
        return y0 + (y1 - y0) * (s - x0) / (x1 - x0)
    end_tick = max(sum(m.time for m in track) for track in score.tracks)
    boundaries = [0.0]
    if quarters is not None:
        length = quarters * 60 / bpm
        while boundaries[-1] + length < st.seconds_at(end_tick):
            boundaries.append(boundaries[-1] + length)
    else:
        bars = BarMap(score)
        bar = 2
        while bars.bar_start(bar) < end_tick:
            boundaries.append(st.seconds_at(bars.bar_start(bar)))
            bar += 1
    boundaries.append(st.seconds_at(end_tick))
    return [(mapped(b) - mapped(a)) / (b - a) for a, b in zip(boundaries, boundaries[1:]) if b > a]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("score")
    parser.add_argument("performed")
    parser.add_argument("quarters_per_bar", type=float, nargs="?")
    parser.add_argument("score_bpm", type=float, nargs="?")
    args = parser.parse_args()
    if (args.quarters_per_bar is None) != (args.score_bpm is None):
        parser.error("the constant-grid quarters-per-bar and score-bpm arguments go together")
    if args.quarters_per_bar is not None and min(args.quarters_per_bar, args.score_bpm) <= 0:
        parser.error("quarters-per-bar and score-bpm must be positive")
    values = ratios(mido.MidiFile(args.score), mido.MidiFile(args.performed), args.quarters_per_bar, args.score_bpm)
    for i in range(0, len(values), 16):
        print(f"  bars {i + 1:2d}-{min(i + 16, len(values)):2d}: " + " ".join(f"{r:.2f}" for r in values[i:i + 16]))
    if not values:
        print("no note onsets to match")


if __name__ == "__main__":
    main()
