#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=1.26", "scipy>=1.11", "soundfile", "matplotlib>=3.8", "mosqito"]
# ///
"""Harsh moments in singing: the voice's "ah my ears" measured against real singers.

A listener flinches at moments, not at a long-term average, so every measure
is taken on 400 ms windows (100 ms hop) of sung audio and summarised by its
95th percentile and its 99th (a max would favour short renders). Sung windows are those whose level is within
30 dB of the recording's loud (p95) level. The sung part of each signal is
calibrated to 70 dB SPL as a whole, so a loud moment is louder than 70 dB and
a soft one quieter, as they would play.

- presence_db: 2-5 kHz power over 200-1500 Hz, the ear-canal band against the
  body of the voice.
- glare_db: the strongest 1/6-octave band between 2 and 5 kHz above a line
  fitted through the bands an octave either side (excluding its neighbours): one partial or
  resonance standing out of the spectrum, heard as a whistle or a squeal.
- purity_db: the strongest 2-5 kHz spectral line over the band's median, at
  6 Hz resolution. Real voices fill the space between their high partials with
  noise (jitter smears them, turbulence adds to them); clean synthetic partials
  stand alone there and whine.
- sharpness_acum: DIN 45692 sharpness of each window (mosqito, free field).
- whistle: the share of loud windows (within 6 dB of the loud level, at
  least the loudest 20) whose strongest harmonic, summed along a tracked f0,
  holds more than half the power and lies above 500 Hz: an exposed,
  near-pure high tone. Voice v2's high holler, a hum held around C5-A5, is
  the case. Below about B4 no harmonic can count, so low voices are not
  gated by it.

Whistle is a candidate measure, pending a human-calibrated listening ladder:
it separates the harsh v2 holler from a mix a listener found comfortable,
and its bar is a placeholder in that gap.
Glare and purity are reported only; they did not separate the two.

  uv run scripts/voice_harshness.py measure LABEL=FILE [LABEL=FILE ...] --out DIR
  uv run scripts/voice_harshness.py gate FILE --against DIR/harshness.json [--group real]

The gate: whistle at most WHISTLE_BAR, and a presence and sharpness p95 no
higher than the brightest real calibration singer's p95.

  uv run scripts/voice_harshness.py calibrate references/voice-3/harshness/sources.toml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

SR = 48000
WINDOW_S, HOP_S = 0.4, 0.1
MEASURES = ("presence_db", "glare_db", "purity_db", "sharpness_acum")
GATED = ("presence_db", "sharpness_acum")
WHISTLE_BAR = 0.3        # a placeholder between the liked mix (0.19) and the holler (0.95), pending human calibration
MIN_LOUD = 20


def load(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ar", str(SR), "-ac", "1", "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).astype(float)


def windows(y):
    """(start sample, window) for the sung windows of a signal."""
    w, h = round(WINDOW_S * SR), round(HOP_S * SR)
    starts = np.arange(0, max(len(y) - w, 0) + 1, h)
    level = np.array([10 * np.log10(np.mean(y[s:s + w] ** 2) + 1e-20) for s in starts])
    keep = level > np.percentile(level, 95) - 30
    return starts[keep], w


def sixth_octave_db(f, p):
    centres = 1000 * 2 ** (np.arange(-6, 23) / 6)           # 500 Hz .. 12.7 kHz
    edges = centres[:, None] * 2 ** (np.array([-1, 1]) / 12)
    levels = np.array([p[(f >= lo) & (f < hi)].sum() for lo, hi in edges])
    return centres, 10 * np.log10(levels + 1e-30)


def window_measures(x):
    from scipy.signal import welch
    f, p = welch(x, SR, nperseg=4096, noverlap=3072)
    body = p[(f >= 200) & (f < 1500)].sum() + 1e-30
    presence = 10 * np.log10(p[(f >= 2000) & (f < 5000)].sum() / body + 1e-30)
    centres, db = sixth_octave_db(f, p)
    glare = -np.inf
    octaves = np.log2(centres)
    for i in np.flatnonzero((centres >= 2000) & (centres <= 5000)):
        # Against the spectrum's own slope through the octave either side.
        around = [j for j in range(len(centres)) if 1 < abs(j - i) <= 6]
        fit = np.polyfit(octaves[around], db[around], 1)
        glare = max(glare, db[i] - np.polyval(fit, octaves[i]))
    # Purity: the strongest 2-5 kHz bin over the band's median bin, at 6 Hz
    # resolution. Real voices fill the space between high partials with
    # noise; clean synthetic partials there stand alone and whine.
    f, p = welch(x, SR, nperseg=8192, noverlap=6144)
    band = p[(f >= 2000) & (f < 5000)]
    purity = 10 * np.log10(band.max() / (np.median(band) + 1e-30) + 1e-30)
    share, freq = partial_share(x)
    return presence, glare, purity, share, freq


def track_f0(frame, lo=70.0, hi=1200.0):
    """f0 of a short frame by normalised autocorrelation (the shortest lag
    within 15% of the best, against sub-octave errors); nan when unvoiced."""
    frame = frame - frame.mean()
    n = len(frame)
    r = np.fft.irfft(np.abs(np.fft.rfft(frame, 2 * n)) ** 2)[:n]
    if r[0] <= 0:
        return np.nan
    r = r / r[0]
    a, b = int(SR / hi), int(SR / lo)
    seg = r[a:b]
    best = seg.max()
    if best < 0.5:
        return np.nan
    peaks = np.flatnonzero((seg[1:-1] > seg[:-2]) & (seg[1:-1] >= seg[2:]) & (seg[1:-1] > 0.85 * best)) + 1
    lag = a + (peaks[0] if len(peaks) else int(np.argmax(seg)))
    return SR / lag


def partial_share(x, frame=2048, hop=1024, nfft=8192):
    """The strongest harmonic's share of a window's power, and its frequency.

    f0 is tracked on 43 ms frames, and each harmonic's energy is summed along
    the track (within 3% or two bins of k*f0), so vibrato does not smear a
    partial out of a fixed bin and a strong formant-boosted harmonic is
    weighed as the harmonic it is."""
    window = np.hanning(frame)
    f = np.fft.rfftfreq(nfft, 1 / SR)
    energy = {}
    total = 0.0
    f0s = []
    for start in range(0, len(x) - frame + 1, hop):
        seg = x[start:start + frame]
        spec = np.abs(np.fft.rfft(seg * window, nfft)) ** 2
        total += spec.sum()
        f0 = track_f0(seg)
        if not np.isfinite(f0):
            continue
        f0s.append(f0)
        for k in range(1, int(8000 / f0) + 1):
            half = max(2 * SR / nfft, 0.03 * k * f0)
            energy[k] = energy.get(k, 0.0) + spec[(f > k * f0 - half) & (f < k * f0 + half)].sum()
    if not energy or total <= 0:
        return 0.0, 0.0
    k = max(energy, key=energy.get)
    return float(energy[k] / total), float(k * np.median(f0s))


def measure(y):
    from mosqito.sq_metrics import sharpness_din_st
    starts, w = windows(y)
    sung = np.concatenate([y[s:s + w] for s in starts])
    pascal = 2e-5 * 10 ** (70 / 20) / np.sqrt(np.mean(sung ** 2))
    rows = [window_measures(y[s:s + w]) for s in starts]
    presence = np.array([r[0] for r in rows])
    glare = np.array([r[1] for r in rows])
    purity = np.array([r[2] for r in rows])
    level = np.array([10 * np.log10(np.mean(y[s:s + w] ** 2) + 1e-20) for s in starts])
    # Loud windows: within 6 dB of the loud level, and never fewer than the
    # loudest MIN_LOUD (or all), so a short render's fraction is not one or two windows.
    loud = level >= np.percentile(level, 95) - 6
    if loud.sum() < MIN_LOUD:
        loud = level >= np.sort(level)[-min(MIN_LOUD, len(level))]
    whistle = float(np.mean([(r[3] > 0.5) and (r[4] > 500) for r, l in zip(rows, loud) if l]))
    # Sharpness per window, one at a time (mosqito's segmenter holds them all at once).
    sharp = np.array([float(sharpness_din_st(y[s:s + w] * pascal, SR, field_type="free")) for s in starts])
    out = {"windows": len(starts), "sung_seconds": len(starts) * HOP_S, "whistle": whistle,
           "loud_windows": int(loud.sum()),
           "partial_share_median": float(np.median([r[3] for r, l in zip(rows, loud) if l]))}
    for name, v in (("presence_db", presence), ("glare_db", glare), ("purity_db", purity), ("sharpness_acum", sharp)):
        v = v[np.isfinite(v)]
        out[name] = {"median": float(np.median(v)), "p95": float(np.percentile(v, 95)), "p99": float(np.percentile(v, 99))}
    out["series"] = {"time_s": (starts / SR).round(2).tolist(), "presence_db": presence.round(2).tolist(),
                     "glare_db": glare.round(2).tolist(), "purity_db": purity.round(2).tolist(),
                     "sharpness_acum": sharp.round(3).tolist()}
    return out


def bar(results):
    """The gate's limits: a p95 brightness no higher than the brightest real
    singer's p95, whistle at most WHISTLE_BAR."""
    real = [v for v in results.values() if v.get("group") == "real"]
    return {**{m: max(v[m]["p95"] for v in real) for m in GATED}, "whistle": WHISTLE_BAR}


def verdict(result, limits):
    fails = [f"{m}.p95 {result[m]['p95']:.2f} > {limits[m]:.2f}" for m in GATED if result[m]["p95"] > limits[m]]
    if result["whistle"] > limits["whistle"]:
        fails.append(f"whistle {result['whistle']:.2f} > {limits['whistle']:.2f}")
    return {"pass": not fails, "fails": fails}


def figure(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = list(results)
    colours = {"real": "#3a7d44", "holdout": "#7fb08a", "model": "#3b5ba5", "heard-harsh": "#b8402e",
               "heard-liked": "#c79a1e"}
    fig, axes = plt.subplots(1, 4, figsize=(19, 0.42 * len(labels) + 1.6), sharey=True)
    for ax, m in zip(axes, MEASURES):
        data = [results[k]["series"][m] for k in labels]
        parts = ax.boxplot(data, vert=False, whis=(5, 95), showfliers=False, patch_artist=True)
        for patch, k in zip(parts["boxes"], labels):
            patch.set_facecolor(colours.get(results[k].get("group"), "#999"))
            patch.set_alpha(0.6)
        for i, k in enumerate(labels):
            ax.plot(results[k][m]["p99"], i + 1, "k|", ms=10)
        limit = bar(results).get(m) if any(v.get("group") == "real" for v in results.values()) else None
        if limit is not None:
            ax.axvline(limit, color="#3a7d44", ls="--", lw=1)
        ax.set_title(m.replace("_", " ") + "  (box p25-75, whisker p5-p95, | p99)")
    axes[0].set_yticks(range(1, len(labels) + 1), [f"{k}  (whistle {results[k]['whistle']:.2f})" for k in labels])
    fig.text(0.01, 0.005, "green real singers (pale: held out of the bar), red negative listening controls, "
                          "gold what he liked, blue models; "
                          "dashed: the brightness ceiling (presence and sharpness are gated, with whistle <= %.1f)" % WHISTLE_BAR, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


ROOT = Path(__file__).resolve().parents[1]


def resolve(path):
    """A repo-relative path, here or in the main checkout (a worktree's build/ is its own)."""
    here = ROOT / path
    if here.exists():
        return here
    common = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--git-common-dir"], capture_output=True,
                            text=True).stdout.strip()
    main = (ROOT / common).resolve().parent / path
    if main.exists():
        return main
    raise FileNotFoundError(f"{path}: fetch it with the recipe in the sources file")


def calibrate(manifest, out):
    import tomllib
    sources = tomllib.loads(Path(manifest).read_text())["source"]
    results = {}
    for src in sources:
        r = measure(load(resolve(src["path"])))
        r.update(group=src["group"], path=src["path"], note=src.get("note", ""))
        results[src["label"]] = r
        print(f"{src['label']:24} {src['group']:12} " +
              "  ".join(f"{m.split('_')[0]} {r[m]['p95']:6.2f}/{r[m]['p99']:6.2f}" for m in MEASURES) +
              f"  whistle {r['whistle']:.2f}", flush=True)
    limits = bar(results)
    for v in results.values():
        v["verdict"] = verdict(v, limits)
    out.mkdir(parents=True, exist_ok=True)
    figure(results, out / "harshness.png")
    (out / "harshness.series.json").write_text(json.dumps({k: v.pop("series") for k, v in results.items()}) + "\n")
    summary = {g: [k for k, v in results.items() if v["group"] == g and not v["verdict"]["pass"]]
               for g in ("real", "holdout", "heard-harsh", "heard-v2", "heard-liked", "context")}
    (out / "harshness.json").write_text(json.dumps({"bar": limits, "failing": summary, "results": results},
                                                   indent=1) + "\n")
    for g, names in summary.items():
        n = sum(v["group"] == g for v in results.values())
        print(f"{g:12} {n - len(names)}/{n} pass" + (f"  (fail: {', '.join(names)})" if names else ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("measure")
    m.add_argument("items", nargs="+", help="LABEL=FILE or GROUP:LABEL=FILE")
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--merge", action="store_true", help="add to an existing harshness.json")
    cal = sub.add_parser("calibrate")
    cal.add_argument("manifest", type=Path)
    cal.add_argument("--out", type=Path, default=None)
    g = sub.add_parser("gate")
    g.add_argument("files", nargs="+", type=Path)
    g.add_argument("--against", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "calibrate":
        calibrate(args.manifest, args.out or args.manifest.parent)
        return
    if args.command == "measure":
        args.out.mkdir(parents=True, exist_ok=True)
        path, series_path = args.out / "harshness.json", args.out / "harshness.series.json"
        results = {}
        if args.merge and path.exists():
            results = json.loads(path.read_text())["results"]
            series = json.loads(series_path.read_text()) if series_path.exists() else {}
            for k, v in results.items():
                v["series"] = series.get(k, {m: [v[m]["median"]] for m in MEASURES})
        for item in args.items:
            label, file = item.split("=", 1)
            group, _, label = label.rpartition(":")
            r = measure(load(file))
            r.update(group=group or "model", file=str(file))
            results[label] = r
            print(f"{label:34} " + "  ".join(f"{k} p95 {r[k]['p95']:6.2f} p99 {r[k]['p99']:6.2f}" for k in MEASURES),
                  flush=True)
        summary = {"bar": bar(results)} if any(v["group"] == "real" for v in results.values()) else {}
        for k, v in results.items():
            if summary:
                v["verdict"] = verdict(v, summary["bar"])
        figure(results, args.out / "harshness.png")
        series_path.write_text(json.dumps({k: v.pop("series") for k, v in results.items()}) + "\n")
        path.write_text(json.dumps({**summary, "results": results}, indent=1) + "\n")
    else:
        limits = json.loads(args.against.read_text())["bar"]
        failed = False
        for file in args.files:
            r = measure(load(file))
            v = verdict(r, limits)
            failed |= not v["pass"]
            print(f"{file.name}: {'PASS' if v['pass'] else 'FAIL'}  " +
                  "  ".join(f"{k} p95 {r[k]['p95']:.2f} p99 {r[k]['p99']:.2f}" for k in MEASURES) +
                  ("" if v["pass"] else "\n  " + "; ".join(v["fails"])))
        raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
