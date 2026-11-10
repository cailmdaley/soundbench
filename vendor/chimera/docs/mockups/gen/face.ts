/**
 * Engine side of the mockups: turns real CHIMERA engine state into drawable data.
 *
 *   buildFaceData(model, artic, tele, { ghosts })  → everything faceScreen() in lib.ts draws
 *   spectrumData(out, fs, f0, model, artic)        → harmonics + tract envelope for spectrumScreen()
 *   tubeData(model, tele)                          → per-section areas + pressure for tubeScreen()
 *
 * All geometry is in body cm (y up), built exactly like src/ui/render/anatomy.ts builds it.
 * Import engine modules from REPO (absolute paths) in mode scripts.
 */
import { VoiceModel } from "../../../src/ui/analysis.ts";
import { NEUTRAL, type HumanArtic, type HumanFrame } from "../../../src/engine/anatomy/human.ts";
import { catmullRom, rotateAbout, dist, type Vec } from "../../../src/engine/anatomy/geom.ts";
import { FFT, hann } from "../../../src/engine/analysis/fft.ts";
import { pressureLevel } from "../../../src/ui/colors.ts";
import type { VoiceTelemetry } from "../../../src/engine/voice/humanVoice.ts";
import targets from "../../../src/engine/phonetics/vowelTargets.generated.json";

export const REPO = new URL("../../..", import.meta.url).pathname.replace(/\/$/, "");
export type Pt = [number, number];
export const VOWEL_TARGETS = targets as unknown as Record<string, Record<string, { artic: Partial<HumanArtic>; formants: number[] }>>;
export const vowelArtic = (body: "male" | "female" | "child", ipa: string): HumanArtic => ({ ...NEUTRAL, ...VOWEL_TARGETS[body][ipa].artic });
export const lerpArtic = (a: HumanArtic, b: HumanArtic, u: number): HumanArtic => {
  const o = { ...a };
  for (const k of Object.keys(a) as (keyof HumanArtic)[]) o[k] = (a[k] as number) + ((b[k] as number) - (a[k] as number)) * u;
  return o;
};
export { VoiceModel };

const pts = (v: Vec[]): Pt[] => v.map((p) => [p.x, p.y]);
const close = (v: Pt[]): Pt[] => [...v, v[0]];
const ellipse = (c: Vec, rx: number, ry: number, rot: number, n = 48): Pt[] => {
  const o: Pt[] = [];
  const cr = Math.cos(-rot), sr = Math.sin(-rot); // canvas rotation is in screen space (y down)
  for (let i = 0; i <= n; i++) {
    const a = (i / n) * Math.PI * 2;
    const x = Math.cos(a) * rx, y = Math.sin(a) * ry;
    o.push([c.x + x * cr - y * sr, c.y + x * sr + y * cr]);
  }
  return o;
};
const roundRect = (p0: Vec, p1: Vec, r: number): Pt[] => {
  const x0 = Math.min(p0.x, p1.x), x1 = Math.max(p0.x, p1.x), y0 = Math.min(p0.y, p1.y), y1 = Math.max(p0.y, p1.y);
  const rr = Math.min(r, (x1 - x0) / 2, (y1 - y0) / 2);
  const o: Pt[] = [];
  const corner = (cx: number, cy: number, a0: number) => {
    for (let i = 0; i <= 6; i++) {
      const a = a0 + (i / 6) * (Math.PI / 2);
      o.push([cx + Math.cos(a) * rr, cy + Math.sin(a) * rr]);
    }
  };
  corner(x1 - rr, y1 - rr, 0);
  corner(x0 + rr, y1 - rr, Math.PI / 2);
  corner(x0 + rr, y0 + rr, Math.PI);
  corner(x1 - rr, y0 + rr, (3 * Math.PI) / 2);
  o.push(o[0]);
  return o;
};

export interface FaceData {
  head: Record<string, any>;
  ghosts: { tongue: Pt[]; jaw: { lower: Pt[]; upper: Pt[] }; lips: Pt[][] }[];
  grid: { inner: Pt; outer: Pt; s: number; area: number; level: number }[];
  streaks: { a: Pt; b: Pt; w: number }[];
  /** turbulence: noise-source sparks downstream of the constriction (empty when quiet) */
  sparks: { p: Pt; r: number; a: number }[];
  caliper: { inner: Pt; outer: Pt; area: number } | null;
  handles: Record<"tongue" | "tip" | "jaw" | "lips" | "velum" | "larynx", Pt>;
  formants: number[];
  tractLength: number;
  lipArea: number;
  velumPort: number;
  artic: HumanArtic;
}

/**
 * Midsagittal drawing data for one moment.
 * artic: the articulation to draw (for speech, VoiceModel.articFromArray(tele.artic)).
 * tele: telemetry from HumanVoice.telemetry() for the same moment (pressure, flow, noise).
 * ghosts: earlier articulations, oldest first, drawn as fading afterglow (tongue, jaw, lips).
 */
export function buildFaceData(model: VoiceModel, artic: HumanArtic, tele: VoiceTelemetry, opts: { ghosts?: HumanArtic[]; seed?: number } = {}): FaceData {
  const t = model.airway.tract;
  const k = t.scale;
  const P = (x: number, y: number) => t.P(x, y);
  const condyle = t.P(-2.5, 10.5);
  const jawFnFor = (a: HumanArtic) => {
    const jawRef = t.P(6.95, 7.75);
    const phi = (-Math.max(-0.2, Math.min(3, a.jaw)) * k) / dist(condyle, jawRef);
    return (x: number, y: number) => rotateAbout(t.P(x, y), condyle, phi);
  };
  const jawBoneFor = (fr: HumanFrame) => {
    const jb = fr.jawBone.filter((p) => p.x > 0.8 * k);
    return { lower: pts(jb), upper: pts(jb.map((p) => ({ x: p.x, y: p.y + 0.75 * k }))) };
  };
  // ghosts first: VoiceModel.frame() caches only one key
  const ghosts = (opts.ghosts ?? []).map((a) => {
    const fr = model.frame(a);
    return { tongue: close(pts(fr.tongue)), lips: [close(pts(fr.upperLip)), close(pts(fr.lowerLip))], jaw: jawBoneFor(fr) };
  });
  const f = model.frame(artic);
  const J = jawFnFor(artic);
  const lar = artic.larynx * k;
  const ph = model.airway.body.pharynx;
  const B = (x: number, y: number): Vec => ({ x: x * k, y: (y < 7 ? 7 - (7 - y) * ph : y) * k });

  const ulOut = f.upperLip.slice(7);
  const llOut = f.lowerLip.slice(7);
  const ulEdge = f.upperLip[6];
  const llEdge = f.lowerLip[6];
  const neckFront = catmullRom([B(2.3, -4.6), B(2.38, -1.5), B(2.52, 0.4), B(3.0, 1.7), J(5.3, 3.0), J(7.25, 3.35), J(7.95, 4.6), J(7.55, 5.7)], 4);
  const face = catmullRom([ulOut[ulOut.length - 1], B(7.05, 10.6), B(7.75, 10.78), B(8.75, 11.05), B(9.25, 11.65), B(8.95, 12.6), B(8.35, 13.9), B(7.7, 15.0), B(7.95, 16.1), B(7.9, 18.1), B(6.9, 20.5)], 4);
  const cranium = catmullRom([B(6.9, 20.5), B(3.0, 22.9), B(-2.2, 22.7), B(-6.9, 19.6), B(-8.7, 15.4), B(-8.3, 11.9), B(-7.4, 8.9), B(-6.8, 4.8), B(-6.6, 0), B(-6.6, -4.6)], 4);
  const inset = catmullRom([B(1.85, -4.6), B(1.9, -0.8), B(2.3, 1.6), J(5.2, 3.6), J(6.9, 3.9), J(7.4, 4.9), J(7.2, 5.9), B(6.6, 9.9), B(7.6, 10.9), B(8.55, 11.5), B(8.1, 13.4), B(7.25, 15.2), B(7.45, 18.0), B(3.0, 22.2), B(-2.0, 22.0), B(-6.3, 19.2), B(-7.9, 15.3), B(-7.6, 12.0), B(-6.7, 8.7), B(-6.1, 4.6), B(-6.0, -4.6)], 3);
  const skull = catmullRom([B(7.0, 16.2), B(7.3, 18.5), B(6.0, 20.7), B(3.0, 22.1), B(-2.0, 22.0), B(-6.3, 19.2), B(-7.9, 15.3), B(-7.4, 12.6), B(-4.6, 12.2), B(-2.2, 13.3), B(1.0, 13.7), B(4.1, 14.1), B(6.2, 14.9)], 4);
  const brain = catmullRom([B(6.55, 16.3), B(6.75, 18.4), B(5.5, 20.1), B(2.8, 21.4), B(-1.9, 21.3), B(-5.8, 18.8), B(-7.1, 15.4), B(-6.7, 13.1), B(-4.4, 12.8), B(-2.1, 13.9), B(1.0, 14.3), B(4.0, 14.7), B(5.9, 15.4)], 4);
  const sulcus = catmullRom([B(5.4, 19.4), B(2.4, 20.5), B(-1.4, 20.4), B(-4.8, 18.4)], 4);
  const wallX = (y: number) => -1.2 - 0.025 * y;
  const vertebrae: Pt[][] = [];
  for (let i = 0; i < 9; i++) {
    const y0 = -4.4 + i * 1.75;
    if (y0 > 10.5) break;
    const x0 = wallX(y0) - 2.25;
    vertebrae.push(roundRect(P(x0, y0 + 1.4), P(x0 + 1.6, y0), 0.32 * k));
  }
  const nasalAir = catmullRom([B(0.3, 11.2), B(1.1, 10.25), B(2.7, 10.75), B(4.6, 10.85), B(6.4, 10.65), B(7.6, 10.85), B(8.45, 11.3), B(8.0, 12.2), B(6.7, 13.4), B(4.4, 13.9), B(1.8, 13.6), B(0.2, 12.6)], 4);
  const palRaw: [number, number][] = [[1.0, 9.6], [1.8, 9.92], [2.7, 10.18], [3.6, 10.3], [4.4, 10.25], [5.1, 10.05], [5.7, 9.72], [6.15, 9.36], [6.55, 9.15], [7.0, 9.6]];
  const palOral = palRaw.map(([x, y]) => t.P(x, y));
  const g = f.glottis;
  const hy = { x: P(1.6, 0).x, y: P(0, 2.9).y + 0.6 * lar };
  const rings: Pt[][] = [];
  for (let i = 0; i < 4; i++) {
    const y = g.y - (0.9 + i * 1.0) * k;
    if (y < -4.4) break;
    rings.push(pts([{ x: g.x + 0.8 * k, y }, { x: g.x + 1.05 * k, y: y - 0.05 }, { x: g.x + 1.05 * k, y: y - 0.5 * k }, { x: g.x + 0.82 * k, y: y - 0.45 * k }, { x: g.x + 0.8 * k, y }]));
  }
  const wallAtGlottis = { x: wallX(g.y / k) * k, y: g.y };
  const origin = J(6.0, 5.45);
  const tc = f.tongueCenter;
  const fibres: Pt[][] = [];
  for (let i = 0; i <= 16; i++) {
    const ang = Math.PI * (0.18 + (0.9 * i) / 16);
    const end = { x: tc.x + Math.cos(ang) * f.tongueRadius * 1.25, y: tc.y + Math.sin(ang) * f.tongueRadius * 1.25 };
    const ctrl = { x: tc.x, y: (tc.y + origin.y) / 2 };
    const c: Pt[] = [];
    for (let j = 0; j <= 16; j++) {
      const u = j / 16;
      c.push([(1 - u) * (1 - u) * origin.x + 2 * (1 - u) * u * ctrl.x + u * u * end.x, (1 - u) * (1 - u) * origin.y + 2 * (1 - u) * u * ctrl.y + u * u * end.y]);
    }
    fibres.push(c);
  }
  const head = {
    skinLower: pts([...neckFront, ...llOut.slice().reverse()]),
    skinUpper: pts([...ulOut, ...face.slice(1), ...cranium.slice(1)]),
    inset: pts(inset), skull: close(pts(skull)), brain: close(pts(brain)), sulcus: pts(sulcus),
    eye: ellipse(B(6.5, 15.4), 1.1 * k, 0.95 * k, 0), lens: ellipse(B(7.35, 15.4), 0.18 * k, 0.42 * k, 0, 24),
    vertebrae,
    cordA: pts([P(-5.3, -4.6), P(-4.5, -4.6), P(-4.5, 11.8), P(-5.1, 12.4), P(-5.6, 11.6)]),
    cordB: pts([P(-5.1, -4.6), P(-4.7, -4.6), P(-4.72, 11.4), P(-5.15, 11.6)]),
    nasal: close(pts(nasalAir)),
    turbinates: ([[3.0, 11.45, 2.1, 0.32], [3.5, 12.45, 1.9, 0.3], [4.0, 13.3, 1.45, 0.22]] as const).map(([cx, cy, rx, ry]) => ellipse(B(cx, cy), rx * k, ry * k, -0.04)),
    palOral: pts(palOral), palNasal: pts(palOral.map((p) => ({ x: p.x, y: p.y + 0.55 * k }))),
    jaw: jawBoneFor(f),
    hyoid: ellipse(hy, 0.5 * k, 0.28 * k, 0.3, 32),
    thyroid: pts([{ x: g.x + 0.75 * k, y: g.y - 1.6 * k }, { x: g.x + 1.75 * k, y: g.y - 0.6 * k }, { x: g.x + 1.95 * k, y: g.y + 1.2 * k }, { x: g.x + 1.45 * k, y: g.y + 1.9 * k }, { x: g.x + 1.15 * k, y: g.y + 1.1 * k }, { x: g.x + 1.3 * k, y: g.y - 0.4 * k }, { x: g.x + 0.75 * k, y: g.y - 1.6 * k }]),
    tracheaL: pts([{ x: g.x - 0.65 * k, y: g.y }, { x: g.x - 0.75 * k, y: -4.8 }]),
    tracheaR: pts([{ x: g.x + 0.75 * k, y: g.y }, { x: g.x + 0.95 * k, y: -4.8 }]),
    rings,
    outerWall: pts(f.outer.filter((p) => p.y > g.y || p.x > 0)),
    glottisWall: pts([wallAtGlottis, f.outer.find((p) => p.y > g.y) ?? wallAtGlottis]),
    innerWall: pts(f.inner), tongue: close(pts(f.tongue)), fibres, velum: close(pts(f.velum)),
    upperLip: close(pts(f.upperLip)), lowerLip: close(pts(f.lowerLip)),
    upperTeeth: close(pts(f.upperTeeth)), lowerTeeth: close(pts([...f.lowerTeeth, J(6.8, 6.0), J(6.4, 6.2)])),
    epiglottis: pts(f.epiglottis),
    foldL: pts([{ x: g.x - 0.62 * k, y: g.y + 0.05 }, { x: g.x - 0.1, y: g.y + 0.12 * k }, { x: g.x - 0.62 * k, y: g.y + 0.32 * k }]),
    foldR: pts([{ x: g.x + 0.72 * k, y: g.y + 0.05 }, { x: g.x + 0.12, y: g.y + 0.12 * k }, { x: g.x + 0.72 * k, y: g.y + 0.32 * k }]),
    glottis: [g.x, g.y] as Pt,
  };

  // pressure per cross-section line, airflow streaks, turbulence sparks
  const nodeS = model.nodeS(tele.phLen, tele.orLen);
  const nearest = (vals: ArrayLike<number>, s: number) => {
    let best = 0, bd = Infinity;
    for (let i = 0; i < nodeS.length; i++) {
      const d = Math.abs(nodeS[i] - s);
      if (d < bd) { bd = d; best = i; }
    }
    return vals[best] ?? 0;
  };
  const grid = f.grid.map((gs) => ({ inner: [gs.inner.x, gs.inner.y] as Pt, outer: [gs.outer.x, gs.outer.y] as Pt, s: gs.s, area: gs.area, level: pressureLevel(nearest(tele.pressureRms, gs.s)) }));
  const gridPoint = (s: number, u: number): Vec | null => {
    const gg = f.grid;
    if (gg.length < 2 || s <= gg[0].s || s >= gg[gg.length - 1].s) return null;
    let lo = 0, hi = gg.length - 1;
    while (hi - lo > 1) {
      const m = (lo + hi) >> 1;
      if (gg[m].s <= s) lo = m;
      else hi = m;
    }
    const q = (s - gg[lo].s) / Math.max(1e-6, gg[hi].s - gg[lo].s);
    const ix = gg[lo].inner.x + (gg[hi].inner.x - gg[lo].inner.x) * q, iy = gg[lo].inner.y + (gg[hi].inner.y - gg[lo].inner.y) * q;
    const ox = gg[lo].outer.x + (gg[hi].outer.x - gg[lo].outer.x) * q, oy = gg[lo].outer.y + (gg[hi].outer.y - gg[lo].outer.y) * q;
    const w = 0.5 + 0.5 * u * 0.82;
    return { x: ix + (ox - ix) * w, y: iy + (oy - iy) * w };
  };
  const areasAll = new Float32Array(tele.phAreas.length + tele.orAreas.length);
  areasAll.set(tele.phAreas, 0);
  areasAll.set(tele.orAreas, tele.phAreas.length);
  let seed = opts.seed ?? 7;
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  const streaks: FaceData["streaks"] = [];
  for (let i = 0; i < 120; i++) {
    const s = 0.6 + rnd() * (f.lipS - 1.0);
    const u = rnd() * 2 - 1;
    const U = nearest(tele.flow, s);
    const A = Math.max(0.05, nearest(areasAll, s));
    const vis = Math.sign(U / A) * 2.2 * Math.log1p(Math.abs(U / A) / 30);
    if (Math.abs(vis) < 0.05) continue;
    const len = Math.min(1.6, Math.abs(vis) * 0.22);
    const p0 = gridPoint(s, u), p1 = gridPoint(s + len * Math.sign(vis), u);
    if (!p0 || !p1) continue;
    streaks.push({ a: [p0.x, p0.y], b: [p1.x, p1.y], w: Math.min(1, Math.abs(vis) / 6) });
  }
  const sparks: FaceData["sparks"] = [];
  if (tele.noiseLevel > 1 && tele.noiseNode >= 0) {
    const s = model.branchS(tele.noiseNode, tele.phLen, tele.orLen);
    const intensity = Math.min(1, Math.log10(tele.noiseLevel) / 3.2);
    if (s >= 0 && intensity > 0.05) {
      for (let i = 0; i < 34; i++) {
        const p = gridPoint(s + 0.2 + rnd() * 1.2, rnd() * 2 - 1);
        if (p) sparks.push({ p: [p.x, p.y], r: 0.6 + rnd() * 1.6, a: 0.15 + 0.7 * intensity * rnd() });
      }
    }
  }
  let mi = -1, ma = Infinity;
  for (let i = 2; i < f.grid.length - 1; i++) if (f.grid[i].area < ma) { ma = f.grid[i].area; mi = i; }
  const caliper = mi >= 0 && ma < 1.2 ? { inner: [f.grid[mi].inner.x, f.grid[mi].inner.y] as Pt, outer: [f.grid[mi].outer.x, f.grid[mi].outer.y] as Pt, area: ma } : null;
  const lipsH = { x: (f.upperLip[6].x + f.lowerLip[6].x) / 2 + 0.15, y: (f.upperLip[6].y + f.lowerLip[6].y) / 2 };
  const jp = J(7.5, 4.9);
  const vp = f.velum[Math.floor(f.velum.length / 2)];
  const handles = {
    tongue: [f.tongueCenter.x, f.tongueCenter.y] as Pt, tip: [f.tip.x, f.tip.y] as Pt, jaw: [jp.x, jp.y] as Pt,
    lips: [lipsH.x, lipsH.y] as Pt, velum: [vp.x, vp.y] as Pt, larynx: [g.x + 1.2 * k, g.y] as Pt,
  };
  const tf = model.transfer(artic);
  return { head, ghosts, grid, streaks, sparks, caliper, handles, formants: tf.formants.slice(0, 4), tractLength: f.length, lipArea: f.lipArea, velumPort: f.velumPort, artic };
}

/** Harmonic peaks of a rendered signal + the tract's transfer function shifted onto them (dB, 0 = loudest). */
export function spectrumData(out: Float32Array, fs: number, f0: number, model: VoiceModel, artic: HumanArtic, N = 8192) {
  const seg = out.subarray(out.length - N);
  const fft = new FFT(N);
  const w = hann(N);
  const re = new Float64Array(N), im = new Float64Array(N);
  for (let i = 0; i < N; i++) re[i] = seg[i] * w[i];
  fft.transform(re, im);
  const magDb = (hz: number) => {
    const bin = Math.round((hz / fs) * N);
    return 10 * Math.log10(re[bin] * re[bin] + im[bin] * im[bin] + 1e-30);
  };
  const harmonics: { f: number; db: number }[] = [];
  for (let h = 1; h * f0 < 8000; h++) {
    let best = -Infinity, bf = h * f0;
    for (let df = -f0 * 0.3; df <= f0 * 0.3; df += fs / N) {
      const v = magDb(h * f0 + df);
      if (v > best) { best = v; bf = h * f0 + df; }
    }
    harmonics.push({ f: bf, db: best });
  }
  const hMax = Math.max(...harmonics.map((h) => h.db));
  for (const h of harmonics) h.db -= hMax;
  const tf = model.transfer(artic);
  const tfDb = Array.from(tf.db);
  const tfAt = (hz: number) => {
    const fr = tf.freqs;
    let i = 1;
    while (i < fr.length - 1 && fr[i] < hz) i++;
    const q = (hz - fr[i - 1]) / (fr[i] - fr[i - 1]);
    return tfDb[i - 1] + (tfDb[i] - tfDb[i - 1]) * q;
  };
  const offs = harmonics.slice(0, 20).map((h) => h.db - tfAt(h.f)).sort((a, b) => a - b);
  const off = offs[Math.floor(offs.length * 0.7)];
  return { harmonics, envelope: Array.from(tf.freqs).map((fr, i) => [fr, tfDb[i] + off] as Pt), formants: tf.formants.slice(0, 4) };
}

/** Per-section tube areas and pressure levels (0..1) for tubeScreen(). */
export function tubeData(model: VoiceModel, tele: VoiceTelemetry) {
  return { ph: Array.from(tele.phAreas), or: Array.from(tele.orAreas), phLen: tele.phLen, orLen: tele.orLen, rms: Array.from(tele.pressureRms).map((r) => pressureLevel(r)), nasalLen: 11.4 * model.airway.body.scale, velumPort: tele.velumPort };
}

/** Raw pressure → 0..1 display level, same mapping the app uses. */
export { pressureLevel };
