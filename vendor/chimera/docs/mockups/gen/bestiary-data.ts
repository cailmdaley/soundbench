/**
 * Engine side of the Bestiary mockup: runs a creature call offline through CreatureVoice (the same
 * runtime the worklet uses), captures telemetry mid-call, and builds drawable geometry exactly the
 * way src/ui/components/CreatureView.tsx does (Catmull-Rom smoothing, tube centrelines, widths).
 */
import { CreatureVoice, type CreatureTelemetry } from "../../../src/engine/creatures/voice.ts";
import { CREATURES, creatureById } from "../../../src/engine/creatures/index.ts";
import type { CreatureDef } from "../../../src/engine/creatures/types.ts";
import { FFT, hann } from "../../../src/engine/analysis/fft.ts";
import { pressureLevel } from "../../../src/ui/colors.ts";

export { CREATURES, creatureById, pressureLevel };
export type { CreatureDef, CreatureTelemetry };
export type P = [number, number];

/** Catmull-Rom through the points; closed shapes wrap around (as CreatureView.smooth). */
export function smooth(pts: P[], per: number, closed: boolean): P[] {
  const n = pts.length;
  if (n < 3) return pts.slice();
  const at = (i: number) => (closed ? pts[(i + n) % n] : pts[Math.max(0, Math.min(n - 1, i))]);
  const out: P[] = [];
  const segs = closed ? n : n - 1;
  for (let i = 0; i < segs; i++) {
    const p0 = at(i - 1), p1 = at(i), p2 = at(i + 1), p3 = at(i + 2);
    for (let k = 0; k < per; k++) {
      const t = k / per, t2 = t * t, t3 = t2 * t;
      const f = (a: number, b: number, c: number, d: number) => 0.5 * (2 * b + (-a + c) * t + (2 * a - 5 * b + 4 * c - d) * t2 + (-a + 3 * b - 3 * c + d) * t3);
      out.push([f(p0[0], p1[0], p2[0], p3[0]), f(p0[1], p1[1], p2[1], p3[1])]);
    }
  }
  if (closed) out.push(out[0]);
  else out.push(pts[n - 1]);
  return out;
}

export interface TubeGeom { tube: string; label?: string; inView: boolean; pts: P[]; cum: number[]; length: number }
export interface CreatureGeom { silhouette: P[]; parts: { shade: number; pts: P[] }[]; tubes: TubeGeom[] }

export function geometry(def: CreatureDef): CreatureGeom {
  const v = def.visual;
  return {
    silhouette: smooth(v.silhouette, 8, true),
    parts: (v.parts ?? []).map((p) => ({ shade: p.shade, pts: smooth(p.pts, 6, true) })),
    tubes: v.tubes.map((t) => {
      const pts = smooth(t.path, 6, false);
      const cum = [0];
      for (let i = 1; i < pts.length; i++) cum.push(cum[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
      return { tube: t.tube, label: t.label, inView: t.inView !== false, pts, cum, length: cum[cum.length - 1] };
    }),
  };
}

export function pointAt(t: TubeGeom, s: number): P {
  let i = 1;
  while (i < t.cum.length - 1 && t.cum[i] < s) i++;
  const u = Math.max(0, Math.min(1, (s - t.cum[i - 1]) / Math.max(1e-9, t.cum[i] - t.cum[i - 1])));
  return [t.pts[i - 1][0] + (t.pts[i][0] - t.pts[i - 1][0]) * u, t.pts[i - 1][1] + (t.pts[i][1] - t.pts[i - 1][1]) * u];
}

/** unit tangent at arc length s */
export function tangentAt(t: TubeGeom, s: number): P {
  const e = Math.max(1e-3, t.length * 0.01);
  const a = pointAt(t, Math.max(0, s - e)), b = pointAt(t, Math.min(t.length, s + e));
  const dx = b[0] - a[0], dy = b[1] - a[1], L = Math.hypot(dx, dy) || 1;
  return [dx / L, dy / L];
}

export interface CallRun {
  def: CreatureDef;
  callId: string;
  tStar: number;
  duration: number;
  tele: CreatureTelemetry;
  /** whole call, output rate */
  out: Float32Array;
  fs: number;
  /** param values sampled every ~10 ms over the call (for traces) */
  track: { t: number; params: Record<string, number>; gate: number }[];
  /** vibration envelope per source: peak of sourceState() over the 120 ms before tStar */
  sourcePeak: Record<string, number>;
}

/** Play one call from the start and capture telemetry averaged over [tStar − avg, tStar]. */
export function runCall(def: CreatureDef, callId: string, tStar: number, { fs = 48000, avg = 0.03, tail = 0.25 } = {}): CallRun {
  const voice = new CreatureVoice(def, fs);
  const call = def.calls.find((c) => c.id === callId)!;
  const duration = call.keys[call.keys.length - 1].t + 0.05;
  voice.playCall(callId);
  const total = Math.round(fs * (duration + tail));
  const out = new Float32Array(total);
  const B = 128;
  const iReset = Math.round(((tStar - avg) * fs) / B) * B;
  const iRead = Math.round((tStar * fs) / B) * B;
  let tele: CreatureTelemetry | null = null;
  const track: CallRun["track"] = [];
  let nextTrack = 0;
  const sourcePeak: Record<string, number> = {};
  for (let i = 0; i < total; i += B) {
    // render sample by sample near tStar so the source envelope is seen at audio rate
    const near = i >= iRead - Math.round(0.12 * fs) && i <= iRead;
    if (near) {
      for (let j = 0; j < Math.min(B, total - i); j++) {
        voice.render(out, i + j, 1);
        const st = voice.sim.sourceState?.() ?? {};
        for (const [k, v] of Object.entries(st)) sourcePeak[k] = Math.max(sourcePeak[k] ?? 0, v);
      }
    } else voice.render(out, i, Math.min(B, total - i));
    const t = (i + B) / fs;
    if (t >= nextTrack) {
      track.push({ t, params: { ...voice.value }, gate: voice.gate });
      nextTrack += 0.01;
    }
    if (i === iReset) voice.telemetry();
    if (i === iRead) tele = voice.telemetry();
  }
  return { def, callId, tStar, duration, tele: tele ?? voice.telemetry(), out, fs, track, sourcePeak };
}

/** Short-time spectra (dB, 0 = loudest) on a log-frequency grid, for a waterfall. */
export function waterfall(x: Float32Array, fs: number, { fMin = 60, fMax = 8000, slices = 40, win = 2048, bins = 160, t0 = 0, t1 = x.length / fs } = {}) {
  const fft = new FFT(win);
  const w = hann(win);
  const re = new Float64Array(win), im = new Float64Array(win);
  const freqs = Array.from({ length: bins }, (_, i) => fMin * Math.pow(fMax / fMin, i / (bins - 1)));
  const out: { t: number; db: number[] }[] = [];
  let peak = -Infinity;
  for (let s = 0; s < slices; s++) {
    const t = t0 + ((t1 - t0) * s) / (slices - 1);
    const c = Math.round(t * fs);
    re.fill(0);
    im.fill(0);
    for (let i = 0; i < win; i++) re[i] = (x[c - win / 2 + i] ?? 0) * w[i];
    fft.transform(re, im);
    const mag = (k: number) => 10 * Math.log10(re[k] * re[k] + im[k] * im[k] + 1e-30);
    const db = freqs.map((f, i) => {
      const fa = i ? Math.sqrt(freqs[i - 1] * f) : f, fb = i < bins - 1 ? Math.sqrt(freqs[i + 1] * f) : f;
      const ka = (fa / fs) * win, kb = (fb / fs) * win;
      if (kb - ka >= 1) {
        // a cell wider than a bin: max over the bins it covers (keeps narrow harmonics visible)
        let m = -Infinity;
        for (let k = Math.max(1, Math.floor(ka)); k <= Math.ceil(kb) && k < win / 2; k++) m = Math.max(m, mag(k));
        return m;
      }
      // narrower than a bin: interpolate between neighbouring bins (no stair steps)
      const kf = (f / fs) * win, k0 = Math.max(1, Math.floor(kf)), u = kf - k0;
      return mag(k0) * (1 - u) + mag(Math.min(win / 2 - 1, k0 + 1)) * u;
    });
    for (const v of db) peak = Math.max(peak, v);
    out.push({ t, db });
  }
  for (const s of out) s.db = s.db.map((v) => v - peak);
  return { freqs, slices: out };
}
