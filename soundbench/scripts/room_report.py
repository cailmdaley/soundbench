#!/usr/bin/env python3
# /// script
# dependencies = ["mido>=1.3,<2", "matplotlib>=3.8", "numpy>=1.26"]
# ///
"""How the players in a performed MIDI listened to each other.

    uv run scripts/room_report.py build/<project>/<piece>/<piece>.performed.mid

Each performed note is matched to its score note (the sibling `<piece>.midi`, or
--score), so every player's onset on every score beat it strikes has a time in
seconds. Players come from the sidecar's [room.players] (found through the build's
qa.json, or --sidecar); without a room each note track is its own player.

Every onset is measured against the score's shared expressive target (the beat
where the sidecar's tempo map, rit, fermata and arches put it), so what remains is
what the players did. Printed: per player pair, the asynchrony (first minus
second) mean and SD over the beats both strike, and the cross-correlations of their
interval deviations (performed minus target beat interval). r(a->b) = corr(a's
interval n, b's interval n+1) is positive when b follows a; r lag0 is the same-beat
correlation. Also printed: each player's tempo curve (quarter bpm per bar) and the
coefficient of variation of its intervals relative to the target intervals. The
PNG shows the tempo curves, each player's displacement from the target and from
the rest of the band, and the lag-1 matrix.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tomllib
from collections import defaultdict, deque
from pathlib import Path

import mido

sys.path.insert(0, str(Path(__file__).resolve().parent))
from soundbench.scripts import perform  # noqa: E402
from soundbench.scripts import room  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _onsets(midi: mido.MidiFile) -> dict[int, list[tuple[int, int, int]]]:
    """Per track: (tick, channel, pitch) of every note-on, in file order."""
    result = defaultdict(list)
    for index, track in enumerate(midi.tracks):
        tick = 0
        for message in track:
            tick += message.time
            if message.type == "note_on" and message.velocity > 0:
                result[index].append((tick, message.channel, message.note))
    return result


def _players(score: mido.MidiFile, settings: dict) -> dict[str, list[int]]:
    events = perform._event_positions(score)
    note_tracks = perform._note_track_indices(events)
    names = perform._track_names(events)
    the_room = room.parse(settings, note_tracks, names) if settings.get("room") else None
    if the_room is not None:
        return the_room.players
    return {names.get(i, str(i)).split(":")[0] or str(i): [i] for i in note_tracks}


def _corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3:
        return math.nan
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else math.nan


def _mean_sd(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    mean = sum(values) / len(values)
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1)) if len(values) > 1 else 0.0
    return mean, sd


def analyse(performed: Path, score: Path, settings: dict,
            bar_range: tuple[int, int] | None = None) -> dict:
    played = mido.MidiFile(performed)
    written = mido.MidiFile(score)
    players = _players(written, settings)
    bars = perform.BarMap(written)
    score_end = max(sum(m.time for m in track) for track in written.tracks)
    beats = room.beat_grid(bars, score_end)
    beat_of = {tick: n for n, tick in enumerate(beats)}
    clock = perform.TempoMap(played)

    performed_by_key = defaultdict(deque)
    for track, notes in _onsets(played).items():
        for tick, channel, pitch in sorted(notes, key=lambda n: n[0]):
            performed_by_key[(track, channel, pitch)].append(tick)
    times: dict[str, dict[int, list[float]]] = {p: defaultdict(list) for p in players}
    track_player = {t: p for p, tracks in players.items() for t in tracks}
    for track, notes in _onsets(written).items():
        for tick, channel, pitch in sorted(notes, key=lambda n: n[0]):
            queue = performed_by_key[(track, channel, pitch)]
            if not queue:
                raise ValueError(f"performed MIDI is missing a note of track {track} pitch {pitch}")
            out = queue.popleft()
            who = track_player.get(track)
            if who is not None and tick in beat_of:
                times[who][beat_of[tick]].append(clock.seconds_at(out))
    first, last = bar_range or (1, 1 << 30)
    onset = {p: {n: sum(v) / len(v) for n, v in by_beat.items()
                 if first <= bars.bar_at(beats[n]) <= last}
             for p, by_beat in times.items()}

    # displacement from the score's shared expressive target, and interval deviations
    _, _, warper = perform.expressive_time(written, settings)
    target = [clock.seconds_at(warper.apply(b)) for b in beats]
    shift = {p: {n: 1000.0 * (t - target[n]) for n, t in onset[p].items()} for p in players}
    step = {p: {n: shift[p][n + 1] - shift[p][n] for n in shift[p] if n + 1 in shift[p]}
            for p in players}

    pairs = []
    names = list(players)
    for a_index, a in enumerate(names):
        for b in names[a_index + 1:]:
            shared = sorted(set(onset[a]) & set(onset[b]))
            mean, sd = _mean_sd([shift[a][n] - shift[b][n] for n in shared])
            ab = [n for n in step[a] if n + 1 in step[b]]
            ba = [n for n in step[b] if n + 1 in step[a]]
            same = [n for n in step[a] if n in step[b]]
            pairs.append({
                "pair": [a, b], "beats": len(shared), "async_mean_ms": mean, "async_sd_ms": sd,
                "r_a_to_b": _corr([step[a][n] for n in ab], [step[b][n + 1] for n in ab]),
                "r_b_to_a": _corr([step[b][n] for n in ba], [step[a][n + 1] for n in ba]),
                "r_lag0": _corr([step[a][n] for n in same], [step[b][n] for n in same]),
            })

    tempo = {}
    cv = {}
    for p in players:
        per_bar = defaultdict(lambda: [0.0, 0.0])
        ratios = []
        for n in step[p]:
            interval = onset[p][n + 1] - onset[p][n]
            per_bar[bars.bar_at(beats[n])][0] += (beats[n + 1] - beats[n]) / bars.ppq
            per_bar[bars.bar_at(beats[n])][1] += interval
            ratios.append(interval / (target[n + 1] - target[n]))
        tempo[p] = {bar: 60.0 * quarters / seconds for bar, (quarters, seconds) in per_bar.items()}
        mean, sd = _mean_sd(ratios)
        cv[p] = sd / mean if ratios else math.nan

    from_band = {}
    for p in players:
        series = {}
        for n, t in onset[p].items():
            others = [onset[q][n] for q in players if q != p and n in onset[q]]
            if others:
                series[n] = 1000.0 * (t - sum(others) / len(others))
        from_band[p] = series
    return {"players": names, "pairs": pairs, "tempo": tempo, "ioi_cv": cv,
            "from_band": from_band, "shift": shift, "beats": beats,
            "beat_x": [bars.bar_at(b) + (b - bars.bar_start(bars.bar_at(b)))
                       / (bars.bar_start(bars.bar_at(b) + 1) - bars.bar_start(bars.bar_at(b)))
                       for b in beats]}


def plot(result: dict, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    names = result["players"]
    x = result["beat_x"]
    fig, axes = plt.subplots(4, 1, figsize=(11, 14), gridspec_kw={"height_ratios": [1, 1, 1, 1.1]})
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for k, p in enumerate(names):
        color = colors[k % len(colors)]
        curve = sorted(result["tempo"][p].items())
        if curve:
            axes[0].plot(*zip(*curve), label=p, color=color, lw=1.3)
        shift = sorted(result["shift"][p].items())
        if shift:
            axes[1].plot([x[n] for n, _ in shift], [v for _, v in shift], ".-", ms=2, lw=0.6,
                         color=color, label=p)
        band = sorted(result["from_band"][p].items())
        if band:
            axes[2].plot([x[n] for n, _ in band], [v for _, v in band], ".", ms=2.5,
                         color=color, label=p)
    axes[0].set(xlabel="bar", ylabel="quarter bpm", title="each player's tempo, bar by bar")
    axes[1].set(xlabel="bar", ylabel="ms early (-) / late (+)",
                title="each player's onsets against the score's shared expressive target")
    axes[2].set(xlabel="bar", ylabel="ms early (-) / late (+)",
                title="each player's asynchrony from the mean of the others on shared beats")
    for ax in axes[:3]:
        ax.axhline(0, color="0.6", lw=0.5) if ax is not axes[0] else None
        ax.legend(ncol=len(names), fontsize=8, markerscale=3)

    ax = axes[3]
    matrix = np.full((len(names), len(names)), np.nan)
    for pair in result["pairs"]:
        a, b = (names.index(n) for n in pair["pair"])
        matrix[a, b] = pair["r_a_to_b"]
        matrix[b, a] = pair["r_b_to_a"]
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    ax.set_xticks(range(len(names)), names)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("follower: interval n+1")
    ax.set_ylabel("leader: interval n")
    for (i, j), value in np.ndenumerate(matrix):
        if not np.isnan(value):
            ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("lag-1 cross-correlation of interval deviations (who follows whom)")
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _sidecar(performed: Path, explicit: Path | None) -> dict:
    path = explicit
    if path is None:
        qa = performed.parent / "qa.json"
        if qa.is_file():
            recorded = json.loads(qa.read_text()).get("sidecar")
            path = ROOT / recorded if recorded else None
    if path is None or not path.is_file():
        return {}
    with path.open("rb") as handle:
        return tomllib.load(handle)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("performed", type=Path)
    parser.add_argument("--score", type=Path, help="score MIDI (default: the sibling <piece>.midi)")
    parser.add_argument("--sidecar", type=Path, help="sidecar with [room.players]")
    parser.add_argument("--png", type=Path, help="figure path (default: <piece>.room.png beside the MIDI)")
    parser.add_argument("--bars", help="only these bars, e.g. 21-36 (one lead's passage)")
    args = parser.parse_args(argv)
    stem = args.performed.name.removesuffix(".mid").removesuffix(".performed")
    score = args.score or args.performed.with_name(f"{stem}.midi")
    try:
        bar_range = tuple(int(b) for b in args.bars.split("-")) if args.bars else None
        result = analyse(args.performed, score, _sidecar(args.performed, args.sidecar), bar_range)
    except (OSError, ValueError) as error:
        print(f"room_report.py: {error}", file=sys.stderr)
        return 2
    print(f"{'pair':22} {'beats':>5} {'async ms':>15} {'r(a->b)':>8} {'r(b->a)':>8} {'r lag0':>7}")
    for pair in result["pairs"]:
        a, b = pair["pair"]
        print(f"{a + ' - ' + b:22} {pair['beats']:5d} "
              f"{pair['async_mean_ms']:+6.1f} ± {pair['async_sd_ms']:5.1f} "
              f"{pair['r_a_to_b']:+8.2f} {pair['r_b_to_a']:+8.2f} {pair['r_lag0']:+7.2f}")
    print("\nIOI CV against the target intervals: " + ", ".join(
        f"{p} {100 * cv:.1f}%" for p, cv in result["ioi_cv"].items()))
    print("\ntempo (quarter bpm) by bar:")
    names = result["players"]
    print("bar  " + " ".join(f"{p:>7}" for p in names))
    all_bars = sorted({bar for p in names for bar in result["tempo"][p]})
    for bar in all_bars:
        print(f"{bar:3d}  " + " ".join(
            f"{result['tempo'][p][bar]:7.1f}" if bar in result["tempo"][p] else f"{'':>7}" for p in names))
    png = args.png or args.performed.with_name(
        f"{stem}.room{'.bars' + args.bars if args.bars else ''}.png")
    plot(result, png, stem)
    print(f"\nfigure: {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
