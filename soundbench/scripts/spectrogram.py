#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["librosa>=0.10", "matplotlib>=3.8", "numpy", "mido>=1.3", "soundfile"]
# ///
"""See a render: log-frequency spectrogram, loudness contour and piano roll, plus gap metrics.

This is the composer's substitute for listening. The spectrogram is A-weighted by default,
so brightness approximates perceived loudness rather than raw energy. It shows what a render actually does in
time and pitch — note onsets, how notes release (natural decay versus abrupt cutoff),
reverb tails, the shape of the dynamics — and flags silences that sound choppy.

Usage:
  uv run scripts/spectrogram.py AUDIO [--midi PERFORMED.mid] [--start S] [--end S]
                                      [--out PNG] [--title TEXT]
Prints metrics to stdout and writes a PNG (default: next to AUDIO, `.spec.png`).
"""
import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mido
import numpy as np

SR = 22050
HOP = 256


def load(path, start, end):
    y, sr = librosa.load(path, sr=SR, mono=True, offset=start or 0.0,
                         duration=None if end is None else end - (start or 0.0))
    return y, sr


def cqt_db(y, sr, weighted=True):
    """Log-frequency spectrogram in dB, 3 bins per semitone over the piano (A0..C8).

    With `weighted`, each bin gets A-weighting, so brightness approximates perceived
    loudness: a bass note needs far more energy than a treble note to sound as loud,
    and the unweighted picture makes basses look louder than they sound.
    """
    fmin = librosa.note_to_hz("A0")
    C = librosa.cqt(y, sr=sr, hop_length=HOP, fmin=fmin, n_bins=88 * 3, bins_per_octave=36)
    S = librosa.amplitude_to_db(np.abs(C), ref=np.max)
    if weighted:
        freqs = librosa.cqt_frequencies(n_bins=88 * 3, fmin=fmin, bins_per_octave=36)
        S = S + librosa.A_weighting(freqs)[:, None]
        S = S - S.max()
    return S


def loudness_db(y, sr, win_s=0.05, relative=True):
    n = int(win_s * sr)
    frames = librosa.util.frame(np.pad(y, (0, n)), frame_length=n, hop_length=HOP)
    rms = np.sqrt(np.mean(frames ** 2, axis=0)) + 1e-9
    return 20 * np.log10(rms / rms.max() if relative else rms)


def silences(db, sr, threshold=-45.0, min_s=0.12):
    """Stretches below `threshold` dB (relative to the loudest moment) longer than `min_s`."""
    t = np.arange(len(db)) * HOP / sr
    quiet = db < threshold
    out, i = [], 0
    while i < len(quiet):
        if quiet[i]:
            j = i
            while j < len(quiet) and quiet[j]:
                j += 1
            if (j - i) * HOP / sr >= min_s:
                out.append((t[i], t[min(j, len(t) - 1)]))
            i = j
        else:
            i += 1
    return out


def midi_notes(path):
    """(start_s, end_s, pitch, velocity) with sustain-pedal state ignored; plus pedal spans."""
    t, notes, pedal = 0.0, [], []
    on = defaultdict(deque)
    down = {}
    for msg in mido.MidiFile(path):
        t += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            on[(msg.channel, msg.note)].append((t, msg.velocity))
        elif msg.type in ("note_off", "note_on") and on[(msg.channel, msg.note)]:
            s, v = on[(msg.channel, msg.note)].popleft()
            notes.append((s, t, msg.note, v))
        elif msg.type == "control_change" and msg.control == 64:
            if msg.value >= 64 and msg.channel not in down:
                down[msg.channel] = t
            elif msg.value < 64 and msg.channel in down:
                pedal.append((down.pop(msg.channel), t))
    pedal.extend((start, t) for start in down.values())
    return notes, pedal


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio")
    ap.add_argument("--midi")
    ap.add_argument("--start", type=float)
    ap.add_argument("--end", type=float)
    ap.add_argument("--out")
    ap.add_argument("--title")
    ap.add_argument("--unweighted", action="store_true", help="raw energy, no A-weighting")
    ap.add_argument("--stems", nargs="*", metavar="WAV", help="stack stem RMS contours; no paths reads qa.json beside AUDIO")
    ap.add_argument("--midi-offset", type=float, default=0, help="shift MIDI and stems against audio, e.g. 0.3 for master lead-in")
    a = ap.parse_args()
    y, sr = load(a.audio, a.start, a.end)
    t0 = a.start or 0.0
    S = cqt_db(y, sr, weighted=not a.unweighted)
    L = loudness_db(y, sr)
    tl = t0 + np.arange(len(L)) * HOP / sr
    dur = len(y) / sr

    # metrics
    body = L[(tl > t0 + 0.5) & (tl < t0 + dur - 3.0)] if dur > 5 else L
    p10, p50, p90 = np.percentile(body, [10, 50, 90])
    gaps = [(t0 + s, t0 + e) for s, e in silences(L, sr) if t0 + s > t0 + 0.3 and t0 + e < t0 + dur - 1.0]
    print(f"duration {dur:.1f} s   loudness (dB re max) p10 {p10:.1f}  p50 {p50:.1f}  p90 {p90:.1f}  "
          f"spread p90-p10 {p90 - p10:.1f} dB")
    print(f"internal silences below -45 dB longer than 120 ms: {len(gaps)}"
          + ("".join(f"\n  {s:7.2f}-{e:7.2f} s ({e - s:.2f} s)" for s, e in gaps[:20])))

    notes = pedal = None
    if a.midi:
        notes, pedal = midi_notes(a.midi)
        notes = [(s+a.midi_offset, e+a.midi_offset, p, v) for s,e,p,v in notes]
        pedal = [(s+a.midi_offset, e+a.midi_offset) for s,e in pedal]
        notes = [n for n in notes if n[1] > t0 and (a.end is None or n[0] < a.end)]
        if notes:
            vel = np.array([n[3] for n in notes])
            print(f"midi notes {len(notes)}  velocity mean {vel.mean():.0f} sd {vel.std():.1f} "
                  f"range {vel.min()}-{vel.max()}  pedal spans {len(pedal)}")

    stems = []
    if a.stems is not None:
        if a.stems:
            stems = [(Path(p).stem, Path(p)) for p in a.stems]
        else:
            qa_path = Path(a.audio).parent / "qa.json"
            qa = json.loads(qa_path.read_text())
            stems = [(s['track_name'] + ' / ' + s['instrument'], qa_path.parent / s['wav'])
                     for s in qa.get('stems', {}).values()]
        if not stems:
            ap.error("--stems found no stems; pass WAV paths or an ensemble build with qa.json")
    rows = (3 if notes else 2) + len(stems)
    fig, axes = plt.subplots(rows, 1, figsize=(18, 10 + 1.5 * len(stems)), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1] + ([2] if notes else []) + [1]*len(stems)})
    ax = axes[0]
    ext = [t0, t0 + S.shape[1] * HOP / sr, librosa.note_to_midi("A0") - 0.5, librosa.note_to_midi("A0") + 88 - 0.5]
    ax.imshow(S, origin="lower", aspect="auto", extent=ext, cmap="magma", vmin=-80, vmax=0,
              interpolation="nearest")
    ax.set_ylabel("pitch (MIDI)")
    for m in range(24, 109, 12):
        ax.axhline(m - 0.5, color="white", lw=0.3, alpha=0.3)
    ax.set_title((a.title or Path(a.audio).name) + ("" if a.unweighted else "  (A-weighted)"))
    ax = axes[1]
    ax.plot(tl, L, lw=0.6, color="k")
    ax.axhline(-45, color="r", lw=0.5, ls="--")
    for s, e in gaps:
        ax.axvspan(s, e, color="r", alpha=0.25)
    ax.set_ylim(-80, 2)
    ax.set_ylabel("loudness dB")
    if notes:
        ax = axes[2]
        for s, e, p, v in notes:
            ax.plot([s, e], [p, p], lw=2.2, color=plt.cm.viridis(v / 127), solid_capstyle="butt")
        for s, e in pedal or []:
            if e > t0 and (a.end is None or s < a.end):
                ax.axvspan(s, e, ymin=0, ymax=0.04, color="tab:red", alpha=0.6)
        ax.set_ylabel("MIDI (colour = velocity)\nred strip = pedal")
    for index, (name, path) in enumerate(stems):
        # Stems are untrimmed/unmastered and retain the MIDI time origin.
        stem_y, stem_sr = load(path, None, None)
        contour = loudness_db(stem_y, stem_sr, relative=False)
        times = np.arange(len(contour)) * HOP / stem_sr + a.midi_offset
        ax = axes[(3 if notes else 2) + index]
        ax.plot(times, contour, color=plt.cm.tab10(index % 10), lw=.7)
        ax.set_ylim(-80, 0)
        ax.set_ylabel(name + "\nRMS dBFS", fontsize=8)
        print(f"stem {name}: max 50 ms RMS {contour.max():.1f} dBFS (unmastered)")
    axes[-1].set_xlabel("time (s)")
    axes[-1].set_xlim(t0, t0 + dur)
    out = a.out or str(Path(a.audio).with_suffix(".spec.png"))
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
