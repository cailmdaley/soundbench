import { FFT, hann, nextPow2 } from "./fft.ts";
import { Image, inferno } from "./image.ts";

export interface SpectrogramOptions {
  title?: string;
  maxFreq?: number; // Hz
  windowMs?: number; // analysis window (5 = wideband/formants, 30 = narrowband/harmonics)
  hopMs?: number;
  dynamicRange?: number; // dB
  width?: number; // px of the plot area (time axis resampled to fit)
  height?: number; // px of the spectrogram area
  overlays?: { points: [number, number][]; color: [number, number, number] }[]; // (time s, freq Hz)
  marks?: { t: number; label: string }[];
}

/** Render waveform + spectrogram PNG with labelled axes. */
export function renderSpectrogram(path: string, x: ArrayLike<number>, sr: number, opts: SpectrogramOptions = {}): void {
  const maxFreq = Math.min(opts.maxFreq ?? 8000, sr / 2);
  const winMs = opts.windowMs ?? 6;
  const hopMs = opts.hopMs ?? 2;
  const dr = opts.dynamicRange ?? 75;
  const win = Math.max(16, Math.round((winMs / 1000) * sr));
  const hop = Math.max(1, Math.round((hopMs / 1000) * sr));
  const nfft = nextPow2(Math.max(win * 2, 512));
  const fft = new FFT(nfft);
  const w = hann(win);
  const frames = Math.max(1, Math.floor((x.length - win) / hop) + 1);
  const bins = Math.floor((maxFreq / sr) * nfft);
  const spec: Float64Array[] = [];
  let maxDb = -Infinity;
  const re = new Float64Array(nfft);
  const im = new Float64Array(nfft);
  for (let f = 0; f < frames; f++) {
    re.fill(0);
    im.fill(0);
    const off = f * hop;
    for (let i = 0; i < win; i++) re[i] = (x[off + i] ?? 0) * w[i];
    fft.transform(re, im);
    const col = new Float64Array(bins);
    for (let k = 0; k < bins; k++) {
      const v = 10 * Math.log10(re[k] * re[k] + im[k] * im[k] + 1e-20);
      col[k] = v;
      if (v > maxDb) maxDb = v;
    }
    spec.push(col);
  }
  const plotW = opts.width ?? Math.min(1600, Math.max(400, frames));
  const plotH = opts.height ?? 360;
  const waveH = 70;
  const left = 46, top = 22, gap = 6, bottom = 22, right = 10;
  const img = new Image(left + plotW + right, top + waveH + gap + plotH + bottom, [14, 14, 20]);
  if (opts.title) img.text(left, 6, opts.title, [230, 230, 230]);
  // waveform (min/max per column)
  const spc = x.length / plotW;
  let peak = 1e-9;
  for (let i = 0; i < x.length; i++) peak = Math.max(peak, Math.abs(x[i]));
  img.rect(left, top, plotW, waveH, [24, 24, 32]);
  for (let px = 0; px < plotW; px++) {
    let mn = Infinity, mx = -Infinity;
    const a = Math.floor(px * spc), b = Math.min(x.length, Math.floor((px + 1) * spc) + 1);
    for (let i = a; i < b; i++) {
      if (x[i] < mn) mn = x[i];
      if (x[i] > mx) mx = x[i];
    }
    if (!Number.isFinite(mn)) continue;
    const y0 = top + waveH / 2 - (mx / peak) * (waveH / 2 - 2);
    const y1 = top + waveH / 2 - (mn / peak) * (waveH / 2 - 2);
    img.line(left + px, y0, left + px, y1, [120, 200, 255]);
  }
  img.text(left + plotW - 90, top + 2, `PK ${peak.toFixed(3)}`, [150, 150, 170]);
  // spectrogram
  const sy = top + waveH + gap;
  for (let px = 0; px < plotW; px++) {
    const f = Math.min(frames - 1, Math.floor((px / plotW) * frames));
    const col = spec[f];
    for (let py = 0; py < plotH; py++) {
      const k = Math.min(bins - 1, Math.floor(((plotH - 1 - py) / plotH) * bins));
      const t = (col[k] - (maxDb - dr)) / dr;
      img.set(left + px, sy + py, inferno(t));
    }
  }
  // freq axis
  const step = maxFreq > 10000 ? 2000 : maxFreq > 4000 ? 1000 : 500;
  for (let fr = 0; fr <= maxFreq; fr += step) {
    const py = sy + plotH - 1 - (fr / maxFreq) * (plotH - 1);
    img.line(left - 4, py, left - 1, py, [200, 200, 200]);
    img.text(2, py - 3, fr >= 1000 ? `${(fr / 1000).toFixed(fr % 1000 ? 1 : 0)}K` : `${fr}`, [200, 200, 200]);
    for (let px = 0; px < plotW; px += 6) img.set(left + px, py, [255, 255, 255], 0.18);
  }
  // time axis
  const dur = x.length / sr;
  const tStep = dur > 6 ? 1 : dur > 2 ? 0.5 : dur > 0.8 ? 0.2 : 0.1;
  for (let t = 0; t <= dur + 1e-9; t += tStep) {
    const px = left + (t / dur) * plotW;
    img.line(px, sy + plotH, px, sy + plotH + 4, [200, 200, 200]);
    img.text(px - 8, sy + plotH + 7, t.toFixed(tStep < 0.5 ? 1 : 1), [200, 200, 200]);
  }
  for (const ov of opts.overlays ?? []) {
    for (const [t, fr] of ov.points) {
      if (fr <= 0 || fr > maxFreq) continue;
      const px = left + (t / dur) * plotW;
      const py = sy + plotH - 1 - (fr / maxFreq) * (plotH - 1);
      img.rect(px - 1, py - 1, 3, 3, ov.color);
    }
  }
  for (const m of opts.marks ?? []) {
    const px = left + (m.t / dur) * plotW;
    for (let py = sy; py < sy + plotH; py += 3) img.set(px, py, [255, 255, 255], 0.5);
    img.text(px + 2, sy + 2, m.label, [255, 255, 255]);
  }
  img.savePng(path);
}

/** Simple line plot of one or more series (e.g. glottal flow). */
export function renderPlot(
  path: string,
  series: { y: ArrayLike<number>; color: [number, number, number]; label?: string }[],
  opts: { title?: string; width?: number; height?: number; xLabel?: string; xMax?: number; logY?: boolean; yMin?: number; yMax?: number } = {},
): void {
  const W = opts.width ?? 900, H = opts.height ?? 300;
  const left = 60, top = 22, right = 10, bottom = 24;
  const img = new Image(W, H, [14, 14, 20]);
  if (opts.title) img.text(left, 6, opts.title, [230, 230, 230]);
  let mn = opts.yMin ?? Infinity, mx = opts.yMax ?? -Infinity;
  if (opts.yMin === undefined || opts.yMax === undefined) {
    for (const s of series) for (let i = 0; i < s.y.length; i++) {
      const v = s.y[i];
      if (!Number.isFinite(v)) continue;
      if (opts.yMin === undefined) mn = Math.min(mn, v);
      if (opts.yMax === undefined) mx = Math.max(mx, v);
    }
  }
  if (mx - mn < 1e-12) { mx += 1; mn -= 1; }
  const pw = W - left - right, ph = H - top - bottom;
  img.rect(left, top, pw, ph, [24, 24, 32]);
  for (let k = 0; k <= 4; k++) {
    const v = mn + ((mx - mn) * k) / 4;
    const py = top + ph - (k / 4) * ph;
    img.text(2, py - 3, v.toPrecision(3), [180, 180, 190]);
    for (let px = 0; px < pw; px += 5) img.set(left + px, py, [255, 255, 255], 0.15);
  }
  if (opts.xMax !== undefined) {
    for (let k = 0; k <= 5; k++) {
      const px = left + (k / 5) * pw;
      img.text(px - 10, top + ph + 6, ((opts.xMax * k) / 5).toPrecision(3), [180, 180, 190]);
    }
  }
  let ly = top + 4;
  for (const s of series) {
    const n = s.y.length;
    let prev: [number, number] | null = null;
    for (let i = 0; i < n; i++) {
      const v = s.y[i];
      if (!Number.isFinite(v)) { prev = null; continue; }
      const px = left + (i / Math.max(1, n - 1)) * pw;
      const py = top + ph - ((v - mn) / (mx - mn)) * ph;
      if (prev) img.line(prev[0], prev[1], px, py, s.color);
      prev = [px, py];
    }
    if (s.label) { img.text(left + pw - 150, ly, s.label, s.color); ly += 10; }
  }
  img.savePng(path);
}
