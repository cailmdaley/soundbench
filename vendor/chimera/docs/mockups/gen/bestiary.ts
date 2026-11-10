/**
 * BESTIARY mode mockup: a creature call run offline through the real CreatureVoice runtime,
 * drawn as a vector display. → out/bestiary.html
 * Usage: bun gen/bestiary.ts [creature] [call] [t] [name]
 */
import { runCall, geometry, pointAt, tangentAt, waterfall, CREATURES, creatureById, pressureLevel, type P, type CreatureDef, type TubeGeom } from "./bestiary-data.ts";
import { C, topBar, screen, glow, line, text, circle, rect, dPath, n1, heat, chip, encoderScreen, writePage, textWidth, type Rect, type Enc } from "./lib.ts";

const id = process.argv[2] ?? "songbird";
const callId = process.argv[3] ?? "flute";
const tStar = Number(process.argv[4] ?? 0.85);
const outName = process.argv[5] ?? "bestiary";
const def = creatureById(id)!;
const run = runCall(def, callId, tStar);
const tele = run.tele;
const geo = geometry(def);

// ---------------------------------------------------------------- per-creature slot mapping (OP-1)
type SlotKey = { key: string; label: string; color: string; unit: string; fmt: (v: number) => string; glyph: "source" | "lungs" | "route" | "tube"; ref?: string };
const PARAM = (k: string) => def.params.find((p) => p.key === k)!;
const pct = (v: number) => String(Math.round(v * 100));
const SLOTS: Record<string, SlotKey[]> = {
  elephant: [
    { key: "lipPitch", label: "VALVE", color: C.green, unit: "HZ", fmt: (v) => String(Math.round(v)), glyph: "source", ref: "lips" },
    { key: "pitch", label: "FOLDS", color: C.red, unit: "HZ", fmt: (v) => v.toFixed(0), glyph: "source", ref: "glottis" },
    { key: "pressure", label: "BREATH", color: C.pink, unit: "CMH2O", fmt: (v) => v.toFixed(0), glyph: "lungs" },
    { key: "trunk", label: "TRUNK", color: C.white, unit: "%", fmt: pct, glyph: "route", ref: "trunk" },
  ],
  songbird: [
    { key: "pitch", label: "LEFT", color: C.green, unit: "HZ", fmt: (v) => String(Math.round(v)), glyph: "source", ref: "left" },
    { key: "interval", label: "RIGHT", color: C.red, unit: "ST", fmt: (v) => (v >= 0 ? "+" : "") + v.toFixed(1), glyph: "source", ref: "right" },
    { key: "pressure", label: "AIR SAC", color: C.pink, unit: "KPA", fmt: (v) => v.toFixed(2), glyph: "lungs" },
    { key: "gape", label: "BEAK", color: C.white, unit: "%", fmt: pct, glyph: "tube", ref: "beak" },
  ],
  seal: [
    { key: "pitch", label: "PITCH", color: C.green, unit: "HZ", fmt: (v) => String(Math.round(v)), glyph: "source", ref: "glottis" },
    { key: "pulseRate", label: "PULSE", color: C.red, unit: "HZ", fmt: (v) => v.toFixed(0), glyph: "lungs" },
    { key: "pressure", label: "BREATH", color: C.pink, unit: "CMH2O", fmt: (v) => v.toFixed(0), glyph: "lungs" },
    { key: "mouth", label: "MOUTH", color: C.white, unit: "%", fmt: pct, glyph: "tube", ref: "oral" },
  ],
  alligator: [
    { key: "pitch", label: "PITCH", color: C.green, unit: "HZ", fmt: (v) => v.toFixed(0), glyph: "source", ref: "glottis" },
    { key: "chaos", label: "CHAOS", color: C.red, unit: "%", fmt: pct, glyph: "source", ref: "glottis" },
    { key: "pressure", label: "BREATH", color: C.pink, unit: "CMH2O", fmt: (v) => v.toFixed(0), glyph: "lungs" },
    { key: "nostrils", label: "NOSE", color: C.white, unit: "%", fmt: pct, glyph: "tube", ref: "nasal" },
  ],
};
const slots = SLOTS[id];
const fracOf = (k: string, v: number) => {
  const p = PARAM(k);
  return p.log ? Math.log(v / p.min) / Math.log(p.max / p.min) : (v - p.min) / (p.max - p.min);
};
const srcColor = (key: string) => slots.find((s) => s.glyph === "source" && s.ref === key)?.color ?? C.amber;
const tubeColor = (tube: string) => slots.find((s) => s.glyph === "tube" && s.ref === tube)?.color ?? null;

// display levels: the app's mapping (pressureLevel of 0.6 × rms), gained so this call's loudest section reads near full
const rawLv: Record<string, number[]> = {};
let maxLv = 0.02;
for (const [name, d] of Object.entries(tele.tubes)) {
  rawLv[name] = Array.from(d.rms).map((r) => pressureLevel(r * 0.6));
  for (const v of rawLv[name]) maxLv = Math.max(maxLv, v);
}
const gainLv = Math.min(4, 0.9 / maxLv);
const lvl = (name: string, k: number) => Math.min(1, (rawLv[name]?.[k] ?? 0) * gainLv);

// ---------------------------------------------------------------- creature screen
const LEFT: Rect = { x: 16, y: 52, w: 700, h: 832 };
const parts: string[] = [topBar("BESTIARY", null)];
{
  const S = LEFT;
  parts.push(screen(S, "CREATURE", "AIRWAY / LIVE", true));
  const v = def.visual;
  const xs = geo.silhouette.map((p) => p[0]), ys = geo.silhouette.map((p) => p[1]);
  const bb = { x0: Math.min(...xs), x1: Math.max(...xs), y0: Math.min(...ys), y1: Math.max(...ys) };
  const vw = (bb.x1 - bb.x0) * 1.06, vh = (bb.y1 - bb.y0) * 1.1;
  const area = { x: S.x + 20, y: S.y + 150, w: S.w - 40, h: S.h - 150 - 214 };
  const sc = Math.min(area.w / vw, area.h / vh);
  const ox = area.x + area.w / 2 - ((bb.x0 + bb.x1) / 2) * sc;
  const oy = area.y + area.h / 2 + ((bb.y0 + bb.y1) / 2) * sc;
  const X = (x: number) => ox + x * sc, Y = (y: number) => oy - y * sc;
  const Pp = (q: P): P => [X(q[0]), Y(q[1])];
  const pl = (q: P[], closePath = false) => dPath(q.map(Pp), closePath);
  const q: string[] = [];
  // body
  q.push(line(pl(geo.silhouette, true), C.blue, { w: 1.7, hot: true }));
  for (const part of geo.parts) q.push(line(pl(part.pts, true), part.shade < 0.3 ? C.blue : C.blueMid, { w: 1.1 }));
  // airway: walls as double lines, width from the simulation, rungs lit by pressure
  const exaggerate = id === "songbird" ? 1.15 : (v.tubeScale ?? 1.6);
  const labels: { x: number; y: number; s: string; color: string; anchor?: "start" | "end" }[] = [];
  for (const tg of geo.tubes) {
    const d = tele.tubes[tg.tube];
    if (!d) continue;
    const n = d.areas.length;
    const wallCol = tubeColor(tg.tube) ?? C.blue;
    const left: P[] = [], right: P[] = [], rungs: { a: P; b: P; lv: number }[] = [];
    for (let k = 0; k <= n; k++) {
      const s = (k / n) * tg.length;
      const kk = Math.min(n - 1, Math.max(0, k === n ? n - 1 : k));
      const half = Math.sqrt(Math.max(0, d.areas[kk]) / Math.PI) * exaggerate;
      const c = pointAt(tg, s), t = tangentAt(tg, s), nn: P = [-t[1], t[0]];
      left.push([c[0] + nn[0] * half, c[1] + nn[1] * half]);
      right.push([c[0] - nn[0] * half, c[1] - nn[1] * half]);
    }
    for (let k = 0; k < n; k++) {
      const s = ((k + 0.5) / n) * tg.length;
      const half = Math.sqrt(Math.max(0, d.areas[k]) / Math.PI) * exaggerate;
      const c = pointAt(tg, s), t = tangentAt(tg, s), nn: P = [-t[1], t[0]];
      rungs.push({ a: [c[0] + nn[0] * half, c[1] + nn[1] * half], b: [c[0] - nn[0] * half, c[1] - nn[1] * half], lv: lvl(tg.tube, k) });
    }
    for (const r of rungs) q.push(line(dPath([Pp(r.a), Pp(r.b)]), heat(Math.pow(r.lv, 0.85)), { w: 1.1, o: 0.3 + 0.7 * r.lv }));
    q.push(line(pl(left), wallCol, { w: 1.25, hot: wallCol !== C.blue }), line(pl(right), wallCol, { w: 1.25, hot: wallCol !== C.blue }));
    if (tg.label && tg.inView) {
      const m = pointAt(tg, tg.length * 0.5), t = tangentAt(tg, tg.length * 0.5);
      const off = Math.sqrt(Math.max(...Array.from(d.areas)) / Math.PI) * exaggerate * sc + 12;
      const nx = -t[1], ny = t[0];
      const side = ny >= 0 ? 1 : -1; // put labels above the tube when we can
      const lx = X(m[0]) + nx * side * off, ly = Y(m[1]) - ny * side * off;
      labels.push({ x: lx, y: ly, s: tg.label.toUpperCase(), color: tubeColor(tg.tube) ?? C.blueMid, anchor: nx * side < -0.2 ? "end" : "start" });
    }
  }
  // sound sources: ring sized by the vibration envelope, ripples, in the slot colour of its control
  const srcs = v.sources ?? [];
  srcs.forEach((s, si) => {
    const pk = Math.max(0, Math.min(1, run.sourcePeak[s.key] ?? 0));
    const col = srcColor(s.key);
    const [x, y] = Pp(s.at);
    const r0 = srcs.length > 1 ? 3 + 4 * pk : 4 + 7 * pk;
    // two voices side by side: each voice ripples away from the other (left voice down, right up)
    const other = srcs.length > 1 ? Pp(srcs[1 - si].at) : null;
    const away = other ? Math.atan2(y - other[1], x - other[0]) : 0;
    for (let j = 0; j < 4 && pk > 0.04; j++) {
      const rr = r0 + 6 + j * 7, o = (0.75 - j * 0.17) * Math.min(1, pk * 1.4 + 0.3);
      if (!other) q.push(circle(x, y, rr, col, 1, o));
      else {
        const a0 = away - 1.25, a1 = away + 1.25;
        q.push(line(`M${n1(x + Math.cos(a0) * rr)} ${n1(y + Math.sin(a0) * rr)}A${n1(rr)} ${n1(rr)} 0 0 1 ${n1(x + Math.cos(a1) * rr)} ${n1(y + Math.sin(a1) * rr)}`, col, { w: 1.1, o }));
      }
    }
    q.push(circle(x, y, r0, col, 1.6), circle(x, y, 1.6, col, 1, 1, col));
    // label on the side the ripples open to: above-left for the upper voice, below-right for the lower
    if (other && Math.sin(away) < 0) labels.push({ x: x - 26, y: y - 44, s: s.label.toUpperCase(), color: col, anchor: "end" });
    else labels.push({ x: x + 44, y: other ? y + 44 : y + 20, s: s.label.toUpperCase(), color: col });
  });
  // the other slot glyphs: lungs (breath), routing (trunk)
  for (const sl of slots) {
    const val = tele.params[sl.key];
    if (sl.glyph === "lungs") {
      const first = geo.tubes.find((t) => t.tube === "trachea" || t.tube.startsWith("bronchus"))!;
      const base = first.tube.startsWith("bronchus") ? pointAt(first, 0) : pointAt(first, 0);
      const [x, y] = Pp(base);
      const u = fracOf(sl.key, val);
      const rx = 9 + 6 * u, ry = 13 + 8 * u;
      const lx = x - rx * 1.6, ly = y + 6;
      q.push(line(`M${n1(x)} ${n1(y)}L${n1(lx)} ${n1(ly - 4)}`, sl.color, { w: 1.1, o: 0.8 }));
      q.push(`<ellipse cx="${n1(lx - rx * 0.45)}" cy="${n1(ly + ry * 0.9)}" rx="${n1(rx * 0.5)}" ry="${n1(ry)}" stroke="${sl.color}" stroke-width="1.4"/>`);
      q.push(`<ellipse cx="${n1(lx + rx * 0.45)}" cy="${n1(ly + ry * 0.9)}" rx="${n1(rx * 0.5)}" ry="${n1(ry)}" stroke="${sl.color}" stroke-width="1.4"/>`);
      labels.push({ x: lx - rx - 8, y: ly + ry * 2.4 + 12, s: sl.label === "AIR SAC" ? "AIR SAC" : "LUNGS", color: sl.color });
    }
    if (sl.glyph === "route") {
      const tg = geo.tubes.find((t) => t.tube === sl.ref)!;
      const s0 = pointAt(tg, 0), t0 = tangentAt(tg, 0);
      const [x, y] = Pp(s0);
      const dx = t0[0], dy = -t0[1];
      const a = 16;
      q.push(line(`M${n1(x - dx * 4 - dy * 9)} ${n1(y - dy * 4 + dx * 9)}L${n1(x + dx * a)} ${n1(y + dy * a)}L${n1(x - dx * 4 + dy * 9)} ${n1(y - dy * 4 - dx * 9)}`, sl.color, { w: 1.5, hot: true }));
      labels.push({ x: x - 34, y: y - 22, s: "TO TRUNK", color: sl.color });
    }
  }
  // labels, nudged apart
  const placed: { x0: number; x1: number; y0: number; y1: number }[] = [];
  for (const l of labels) {
    const w = textWidth(l.s, 10, 0.18);
    const lx = l.anchor === "end" ? l.x - w : l.x;
    let x = Math.min(S.x + S.w - 14 - w, Math.max(S.x + 14, lx)), y = l.y;
    for (let guard = 0; guard < 10; guard++) {
      const hit = placed.find((r) => x < r.x1 + 6 && x + w > r.x0 - 6 && y - 10 < r.y1 && y + 3 > r.y0);
      if (!hit) break;
      y = hit.y1 + 13;
    }
    placed.push({ x0: x, x1: x + w, y0: y - 10, y1: y + 3 });
    q.push(text(x, y, l.s, { size: 10, color: l.color, ls: 0.18 }));
  }
  {
    const unit = id === "elephant" ? 50 : id === "songbird" ? 1 : 10; // cm
    const bx = S.x + 30, byy = S.y + S.h - 196, len = unit * sc;
    q.push(line(`M${n1(bx)} ${n1(byy - 4)}L${n1(bx)} ${n1(byy + 4)}M${n1(bx)} ${n1(byy)}L${n1(bx + len)} ${n1(byy)}M${n1(bx + len)} ${n1(byy - 4)}L${n1(bx + len)} ${n1(byy + 4)}`, C.blueMid, { w: 1.1 }));
    q.push(text(bx + len + 10, byy + 4, `${unit} CM`, { size: 10, color: C.blueMid, ls: 0.16 }));
  }
  parts.push(`<clipPath id="crclip"><rect x="${S.x + 1}" y="${S.y + 30}" width="${S.w - 2}" height="${S.h - 31}"/></clipPath>`);
  parts.push(`<g clip-path="url(#crclip)">${glow(q.join(""))}</g>`);
  // map title (like the film's base names), latin name, call state
  const r: string[] = [];
  const words = def.name.toUpperCase().split(" ");
  const lines = words.length > 1 ? [words.slice(0, -1).join(" "), words[words.length - 1]] : words;
  lines.forEach((ln, i) => r.push(text(S.x + 30, S.y + 66 + i * 32, ln, { size: 27, color: C.blue, ls: 0.12 })));
  r.push(text(S.x + 32, S.y + 66 + lines.length * 32 - 6, def.latin.toUpperCase(), { size: 10, color: C.blueMid, ls: 0.2 }));
  const call = def.calls.find((c) => c.id === callId)!;
  const fx = S.x + S.w - 18;
  r.push(text(fx, S.y + 64, call.name.toUpperCase(), { size: 15, color: C.amber, anchor: "end", ls: 0.12 }));
  r.push(text(fx, S.y + 86, `${tStar.toFixed(2)} / ${run.duration.toFixed(2)} S`, { size: 11, color: C.amber, anchor: "end", ls: 0.12, o: 0.75 }));
  // progress through the call
  const bx0 = fx - 150, bx1 = fx, by = S.y + 98;
  r.push(line(`M${n1(bx0)} ${n1(by)}L${n1(bx1)} ${n1(by)}`, C.frame, { w: 1 }));
  r.push(line(`M${n1(bx0)} ${n1(by)}L${n1(bx0 + (bx1 - bx0) * (tStar / run.duration))} ${n1(by)}`, C.amber, { w: 1.6, hot: true }));
  // four slots, OP-1 style, along the bottom
  const sy = S.y + S.h - 176;
  const cellW = (S.w - 60) / 2;
  slots.forEach((s, i) => {
    const cx = S.x + 30 + (i % 2) * cellW, cy = sy + Math.floor(i / 2) * 88;
    const val = tele.params[s.key];
    const vs = s.fmt(val);
    r.push(circle(cx + 6, cy + 4, 4, s.color, 1.2, 1, s.color));
    r.push(text(cx + 16, cy + 8, s.label, { size: 11, color: s.color, ls: 0.2, o: 0.85 }));
    r.push(text(cx, cy + 54, vs, { size: 40, color: s.color, weight: "m", ls: 0.04 }));
    r.push(text(cx + textWidth(vs, 40, 0.04) - vs.length * 40 * 0.08 + 8, cy + 54, s.unit, { size: 11, color: s.color, ls: 0.16, o: 0.7 }));
  });
  parts.push(glow(r.join("")));
}

// ---------------------------------------------------------------- right column
const R = { gallery: { x: 728, y: 52, w: 696, h: 180 }, fall: { x: 728, y: 244, w: 696, h: 236 }, calls: { x: 728, y: 492, w: 284, h: 216 }, enc: { x: 1024, y: 492, w: 400, h: 216 }, air: { x: 728, y: 720, w: 696, h: 164 } };

// gallery: every creature as a small line drawing
{
  const S = R.gallery;
  parts.push(screen(S, "CREATURES", `${CREATURES.length} SPECIMENS`));
  const q: string[] = [];
  const cw = (S.w - 28) / CREATURES.length;
  CREATURES.forEach((c, i) => {
    const g2 = geometry(c);
    const vv = c.visual.view;
    const bx = S.x + 14 + i * cw + 8, by = S.y + 38, bw = cw - 16, bh = S.h - 38 - 40;
    const s2 = Math.min(bw / (vv.x1 - vv.x0), bh / (vv.y1 - vv.y0));
    const ox = bx + bw / 2 - ((vv.x0 + vv.x1) / 2) * s2, oy = by + bh / 2 + ((vv.y0 + vv.y1) / 2) * s2;
    const T = (p: P): P => [ox + p[0] * s2, oy - p[1] * s2];
    const sel = c.id === id;
    if (sel) q.push(rect(bx - 6, by - 6, bw + 12, S.h - 38 - 6, C.blue, 1.2));
    q.push(line(dPath(g2.silhouette.map(T), true), sel ? C.blue : C.blueMid, { w: sel ? 1.3 : 1, hot: sel }));
    for (const tg of g2.tubes) q.push(line(dPath(tg.pts.map(T)), C.amber, { w: 1.1, o: sel ? 0.9 : 0.5 }));
    q.push(text(bx + bw / 2, S.y + S.h - 16, c.name.toUpperCase().replace("AFRICAN ", "").replace("AMERICAN ", "").replace("WEDDELL ", ""), { size: 10, color: sel ? C.blue : C.blueMid, anchor: "middle", ls: 0.18 }));
  });
  parts.push(glow(q.join("")));
}

// spectrogram as a Fairlight-style waterfall of the whole call
{
  const S = R.fall;
  const band = id === "songbird" ? { fMin: 600, fMax: 10000, win: 1024 } : id === "alligator" || (id === "elephant" && callId === "rumble") ? { fMin: 20, fMax: 2000, win: 8192 } : { fMin: 100, fMax: 8000, win: 2048 };
  const wf = waterfall(run.out, run.fs, { ...band, slices: 30, bins: 140, t0: 0.02, t1: run.duration });
  parts.push(screen(S, "SPECTROGRAM", "WATERFALL OF THE CALL"));
  const q: string[] = [];
  const x0 = S.x + 46, x1 = S.x + S.w - 150, yBase = S.y + S.h - 34;
  const depthX = 118, depthY = 104, ampH = 62;
  const nS = wf.slices.length;
  const fx = (i: number) => x0 + (i / (wf.freqs.length - 1)) * (x1 - x0);
  const now = wf.slices.reduce((b, s, i) => (Math.abs(s.t - tStar) < Math.abs(wf.slices[b].t - tStar) ? i : b), 0);
  // back to front: later slices at the back
  for (let i = nS - 1; i >= 0; i--) {
    const u = i / (nS - 1);
    const dx = u * depthX, dy = -u * depthY;
    const pts: P[] = wf.slices[i].db.map((db, j) => [fx(j) + dx, yBase + dy - Math.max(0, (db + 60) / 60) * ampH]);
    const poly = [[fx(0) + dx, yBase + dy] as P, ...pts, [fx(wf.freqs.length - 1) + dx, yBase + dy] as P];
    const isNow = i === now;
    const col = isNow ? C.white : C.amber;
    const o = isNow ? 1 : 0.35 + 0.6 * (1 - u);
    q.push(`<path d="${dPath(poly, true)}" fill="#000" stroke="${col}" stroke-width="${isNow ? 1.5 : 1}" opacity="${n1(o)}"/>`);
  }
  // axes: frequency along the front, time into the depth
  q.push(line(`M${n1(x0)} ${n1(yBase)}L${n1(x1)} ${n1(yBase)}`, C.blueDim, { w: 1 }));
  q.push(line(`M${n1(x1)} ${n1(yBase)}L${n1(x1 + depthX)} ${n1(yBase - depthY)}`, C.blueDim, { w: 1, dash: "2 4" }));
  const lf = (f: number) => x0 + (Math.log(f / band.fMin) / Math.log(band.fMax / band.fMin)) * (x1 - x0);
  for (const f of [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]) {
    if (f < band.fMin * 1.05 || f > band.fMax * 0.98) continue;
    q.push(line(`M${n1(lf(f))} ${n1(yBase)}L${n1(lf(f))} ${n1(yBase + 4)}`, C.blueDim, { w: 1 }));
    q.push(text(lf(f), yBase + 16, f >= 1000 ? `${f / 1000}K` : String(f), { size: 9, color: C.blueMid, anchor: "middle", ls: 0.1 }));
  }
  q.push(text(x1 + 10, yBase + 4, "0 S", { size: 9, color: C.blueMid, ls: 0.1 }));
  q.push(text(x1 + depthX + 8, yBase - depthY + 4, `${run.duration.toFixed(1)} S`, { size: 9, color: C.blueMid, ls: 0.1 }));
  const nu = now / (nS - 1);
  q.push(text(x1 + nu * depthX + 12, yBase - nu * depthY + 4, `NOW ${tStar.toFixed(2)}`, { size: 9, color: C.white, ls: 0.1 }));
  parts.push(glow(q.join("")));
}

// calls: framed chips, the playing one lit, with its progress
{
  const S = R.calls;
  parts.push(screen(S, "CALLS"));
  const q: string[] = [];
  let y = S.y + 40;
  for (const c of def.calls) {
    const lit = c.id === callId;
    const { svg, w } = chip(S.x + 14, y, c.name.toUpperCase(), { lit, color: C.amber, size: 11 });
    q.push(svg);
    if (lit) {
      const bx0 = S.x + 14 + w + 10, bx1 = S.x + S.w - 16;
      q.push(line(`M${n1(bx0)} ${n1(y + 11)}L${n1(bx1)} ${n1(y + 11)}`, C.frame, { w: 1 }));
      q.push(line(`M${n1(bx0)} ${n1(y + 11)}L${n1(bx0 + (bx1 - bx0) * (tStar / run.duration))} ${n1(y + 11)}`, C.amber, { w: 1.6, hot: true }));
    }
    y += 32;
  }
  q.push(chip(S.x + 14, y + 4, "SUSTAIN", { size: 11 }).svg);
  parts.push(glow(q.join("")));
}

// four encoders: the same params and colours as the slots on the left
{
  const encs: Enc[] = slots.map((s) => ({ label: s.label, value: s.fmt(tele.params[s.key]), unit: s.unit, frac: Math.max(0, Math.min(1, fracOf(s.key, tele.params[s.key]))), color: s.color }));
  parts.push(encoderScreen(R.enc, "CONTROLS", encs, 1, Math.ceil(def.params.length / 4)));
}

// airway unrolled: every tube straight, rings lit by pressure, shared diameter scale
{
  const S = R.air;
  parts.push(screen(S, "AIRWAY", "EACH TUBE UNROLLED"));
  const q: string[] = [];
  const lanes = def.visual.tubes.filter((t) => tele.tubes[t.tube]);
  const a2d = (A: number) => 2 * Math.sqrt(Math.max(0, A) / Math.PI);
  let dMax = 0.01;
  for (const l of lanes) for (const a of tele.tubes[l.tube].areas) dMax = Math.max(dMax, a2d(a));
  const top = S.y + 38, laneH = (S.h - 38 - 10) / lanes.length;
  const x0 = S.x + 120, x1 = S.x + S.w - 18;
  lanes.forEach((l, li) => {
    const d = tele.tubes[l.tube];
    const y = top + laneH * (li + 0.5);
    const col = tubeColor(l.tube) ?? C.blue;
    q.push(text(S.x + 16, y + 4, (l.label ?? l.tube).toUpperCase(), { size: 10, color: tubeColor(l.tube) ?? C.blueMid, ls: 0.16 }));
    const n = d.areas.length;
    const dx = (x1 - x0) / n;
    const hs = (k: number) => Math.max(0.8, (a2d(d.areas[k]) / dMax) * (laneH - 6)) / 2;
    const topL: P[] = [], botL: P[] = [];
    for (let k = 0; k < n; k++) {
      const cx = x0 + (k + 0.5) * dx;
      topL.push([cx, y - hs(k)]);
      botL.push([cx, y + hs(k)]);
      const lv = lvl(l.tube, k);
      q.push(`<ellipse cx="${n1(cx)}" cy="${n1(y)}" rx="${n1(Math.max(0.8, Math.min(dx * 0.45, hs(k) * 0.3)))}" ry="${n1(hs(k))}" stroke="${heat(Math.pow(lv, 0.85))}" stroke-width="1" opacity="${n1(0.3 + 0.7 * lv)}"/>`);
    }
    q.push(line(dPath(topL), col, { w: 1.1 }), line(dPath(botL), col, { w: 1.1 }));
  });
  parts.push(glow(q.join("")));
}

const path = await writePage(outName, `CHIMERA Bestiary mockup: ${def.name}`, parts, `Bestiary: ${def.name}, ${def.calls.find((c) => c.id === callId)?.name} call at ${tStar.toFixed(2)} s, run offline through CreatureVoice.`);
console.log("wrote", path, "level gain", gainLv.toFixed(2), "source peaks", JSON.stringify(run.sourcePeak));
