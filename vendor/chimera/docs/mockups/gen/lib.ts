/**
 * Drawing side of the CHIMERA vector-display mockups (1440 × 900 design, scales to the window).
 *
 * Look: WarGames NORAD screens. Thin glowing lines on pure black, no fills, crisp glow.
 * Colour roles:
 *   C.blue (+blueMid/blueDim/frame)  the body and structure: outlines, grids, titles
 *   C.amber (+ heat())               sound: pressure, flow, spectra, waveforms
 *   C.green, C.red, C.pink, C.white  the four control slots (OP-1 style). Each screen maps its own
 *                                    four controls onto these colours and draws what they move in them.
 * Text: Hershey single-stroke caps (class "l" light, "m" medium). IPA: ipaGlyph().
 *
 * Usage in a mode script:
 *   const parts: string[] = [topBar("SPEAK"), screen(...), faceScreen(...), ...];
 *   await writePage("speak", "CHIMERA Speak mockup", parts);   // → ../out/speak.html
 */
import type { FaceData, Pt } from "./face.ts";
export type { Pt };

export const W = 1440, H = 900;
const HERE = new URL(".", import.meta.url).pathname;
const FONTS = new URL("./fonts/", import.meta.url).pathname;
export const OUT = HERE + "../";

export interface Rect { x: number; y: number; w: number; h: number }
/** The standard grid: tall left screen + right column (the Play layout). Modes may re-cut the right column. */
export const GRID = {
  left: { x: 16, y: 52, w: 700, h: 832 },
  right: { x: 728, y: 52, w: 696, h: 832 },
  gap: 12,
};

// ---------------------------------------------------------------- palette (sampled from the film)
export const C = {
  blue: "#4fb6f2",
  blueMid: "#3487c4",
  blueDim: "#1f5f8c",
  frame: "#133954",
  green: "#5be381",
  red: "#ff5d40",
  pink: "#ff73cf",
  white: "#eef6f8",
  amber: "#ffb340",
};
export const SLOTS = [C.green, C.red, C.pink, C.white] as const;
export const mix = (hex: string, to: string, u: number) => {
  const a = parseInt(hex.slice(1), 16), b = parseInt(to.slice(1), 16);
  const ch = (s: number) => Math.round(((a >> s) & 255) * (1 - u) + ((b >> s) & 255) * u);
  return "#" + [16, 8, 0].map((s) => ch(s).toString(16).padStart(2, "0")).join("");
};
export const core = (hex: string) => mix(hex, "#ffffff", 0.55);
/** sound heat: dim amber → amber → pale yellow → white */
export const heat = (lvl: number) => {
  const stops: [number, string][] = [[0, "#3a2408"], [0.35, "#c9822a"], [0.6, "#ffb340"], [0.82, "#ffe39a"], [1, "#fff8ec"]];
  const v = Math.max(0, Math.min(1, lvl));
  for (let i = 1; i < stops.length; i++) if (v <= stops[i][0]) return mix(stops[i - 1][1], stops[i][1], (v - stops[i - 1][0]) / (stops[i][0] - stops[i - 1][0]));
  return stops[stops.length - 1][1];
};

// ---------------------------------------------------------------- svg helpers
export const n1 = (v: number) => (Math.round(v * 10) / 10).toString();
export const simplify = (p: Pt[], eps: number): Pt[] => {
  if (p.length < 3) return p;
  const keep = new Uint8Array(p.length);
  keep[0] = keep[p.length - 1] = 1;
  const stack: [number, number][] = [[0, p.length - 1]];
  while (stack.length) {
    const [a, b] = stack.pop()!;
    let md = -1, mi = -1;
    const [ax, ay] = p[a], [bx, by] = p[b];
    const dx = bx - ax, dy = by - ay, L = Math.hypot(dx, dy);
    for (let i = a + 1; i < b; i++) {
      // closed shapes have coincident ends: fall back to the distance from the end point
      const d = L < 1e-6 ? Math.hypot(p[i][0] - ax, p[i][1] - ay) : Math.abs((p[i][0] - ax) * dy - (p[i][1] - ay) * dx) / L;
      if (d > md) { md = d; mi = i; }
    }
    if (md > eps) { keep[mi] = 1; stack.push([a, mi], [mi, b]); }
  }
  return p.filter((_, i) => keep[i]);
};
/** Chaikin corner cutting (closed shapes), for soft tissue the engine gives as coarse polygons */
export const chaikin = (p: Pt[], it = 2): Pt[] => {
  let q = p[0][0] === p[p.length - 1][0] && p[0][1] === p[p.length - 1][1] ? p.slice(0, -1) : p.slice();
  for (let k = 0; k < it; k++) {
    const o: Pt[] = [];
    for (let i = 0; i < q.length; i++) {
      const a = q[i], b = q[(i + 1) % q.length];
      o.push([a[0] * 0.75 + b[0] * 0.25, a[1] * 0.75 + b[1] * 0.25], [a[0] * 0.25 + b[0] * 0.75, a[1] * 0.25 + b[1] * 0.75]);
    }
    q = o;
  }
  return [...q, q[0]];
};
/** the tongue outline without its closing chord */
export const openTongue = (p: Pt[]) => p.slice(0, -1);
export const dPath = (p: Pt[], closePath = false) => {
  const q = simplify(p, 0.18);
  return q.map((v, i) => `${i ? "L" : "M"}${n1(v[0])} ${n1(v[1])}`).join("") + (closePath ? "Z" : "");
};
export interface LineOpts { w?: number; o?: number; hot?: boolean; dash?: string; extra?: string }
/** a stroked path; hot adds the whitish core of a bright beam */
export const line = (d: string, color: string, { w = 1.2, o = 1, hot = false, dash, extra = "" }: LineOpts = {}) =>
  `<path d="${d}" stroke="${color}" stroke-width="${w}"${o < 1 ? ` opacity="${o}"` : ""}${dash ? ` stroke-dasharray="${dash}"` : ""} ${extra}/>` +
  (hot ? `<path d="${d}" stroke="${core(color)}" stroke-width="${n1(w * 0.42)}"${o < 1 ? ` opacity="${o}"` : ""}${dash ? ` stroke-dasharray="${dash}"` : ""}/>` : "");
export interface TextOpts { size?: number; color?: string; anchor?: "start" | "middle" | "end"; weight?: "l" | "m"; ls?: number; o?: number }
export const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;");
/** Hershey text. Approximate advance per char ≈ size × (0.62 + ls); caps read best. */
export const text = (x: number, y: number, s: string, { size = 11, color = C.blue, anchor = "start", weight = "l", ls = 0.14, o = 1 }: TextOpts = {}) =>
  `<text x="${n1(x)}" y="${n1(y)}" font-size="${size}" fill="${color}" text-anchor="${anchor}" letter-spacing="${n1(size * ls)}" class="${weight}"${o < 1 ? ` opacity="${o}"` : ""}>${esc(s)}</text>`;
export const textWidth = (s: string, size: number, ls = 0.14) => s.length * size * (0.62 + ls);
export const circle = (x: number, y: number, r: number, color: string, w = 1.2, o = 1, fill = "none") => `<circle cx="${n1(x)}" cy="${n1(y)}" r="${n1(r)}" stroke="${color}" stroke-width="${w}" fill="${fill}"${o < 1 ? ` opacity="${o}"` : ""}/>`;
export const rect = (x: number, y: number, w: number, h: number, color: string, sw = 1, o = 1, fill = "none") => `<rect x="${n1(x)}" y="${n1(y)}" width="${n1(w)}" height="${n1(h)}" stroke="${color}" stroke-width="${sw}" fill="${fill}"${o < 1 ? ` opacity="${o}"` : ""}/>`;
/** group with the glow filter and line defaults; wrap every screen's drawing in one */
export const glow = (inner: string) => `<g filter="url(#glow)" fill="none" stroke-linejoin="round" stroke-linecap="round">${inner}</g>`;

/** handle: ring with crosshair ticks, in a slot colour */
export const handle = (x: number, y: number, color: string, r = 7, o = 1) => {
  const t = 4;
  const d = `M${n1(x - r - t)} ${n1(y)}L${n1(x - r + 2)} ${n1(y)}M${n1(x + r - 2)} ${n1(y)}L${n1(x + r + t)} ${n1(y)}M${n1(x)} ${n1(y - r - t)}L${n1(x)} ${n1(y - r + 2)}M${n1(x)} ${n1(y + r - 2)}L${n1(x)} ${n1(y + r + t)}`;
  return circle(x, y, r, color, 1.3, o) + line(d, color, { w: 1.2, o });
};

/** screen: black panel, thin frame, stroke-font title (+ dimmer subtitle) */
export const screen = (S: Rect, title: string, sub = "", focus = false) =>
  rect(S.x + 0.5, S.y + 0.5, S.w - 1, S.h - 1, focus ? C.blueDim : C.frame, 1, 1, "#000") +
  glow(text(S.x + 14, S.y + 22, title, { size: 12, color: C.blue, ls: 0.2 }) +
    (sub ? text(S.x + 14 + title.length * 12 * 0.86 + 14, S.y + 22, sub, { size: 10, color: C.blueMid, ls: 0.16 }) : ""));

/** a framed text button / chip: lit = bright frame + text, else dim */
export const chip = (x: number, y: number, label: string, { lit = false, color = C.blue, size = 11, padX = 10, h = 22 }: { lit?: boolean; color?: string; size?: number; padX?: number; h?: number } = {}) => {
  const w = textWidth(label, size, 0.16) + padX * 2;
  return { w, svg: rect(x, y, w, h, lit ? color : C.frame, lit ? 1.2 : 1) + text(x + w / 2, y + h / 2 + size * 0.36, label, { size, color: lit ? color : C.blueMid, anchor: "middle", ls: 0.16 }) };
};

// ---------------------------------------------------------------- single-stroke IPA
// Latin letters come from the Hershey font; the rest are built from font glyphs or short strokes.
export const ipaGlyph = (x: number, y: number, ipa: string, size: number, color: string): string => {
  const xh = size * 0.5;
  const sw = Math.max(0.9, size * 0.075);
  const st = (d: string) => `<path d="${d}" stroke="${color}" stroke-width="${n1(sw)}" fill="none"/>`;
  const font = (s: string, tr = "") => `<text x="${n1(x)}" y="${n1(y)}" font-size="${size}" fill="${color}" text-anchor="middle" class="l"${tr ? ` transform="${tr}"` : ""}>${esc(s)}</text>`;
  const cy = y - xh / 2;
  const rot = `rotate(180 ${n1(x)} ${n1(cy)})`;
  const flip = `translate(${n1(2 * x)} 0) scale(-1 1)`;
  const bowl = (cx: number, r: number) => `M${n1(cx + r)} ${n1(cy)}A${n1(r)} ${n1(xh / 2)} 0 1 0 ${n1(cx - r)} ${n1(cy)}A${n1(r)} ${n1(xh / 2)} 0 1 0 ${n1(cx + r)} ${n1(cy)}`;
  const r = xh * 0.48;
  switch (ipa) {
    case "ɑ": return st(bowl(x - 0.06 * size, r) + `M${n1(x - 0.06 * size + r)} ${n1(y - xh)}L${n1(x - 0.06 * size + r)} ${n1(y)}`);
    case "ɒ": return st(bowl(x + 0.06 * size, r) + `M${n1(x + 0.06 * size - r)} ${n1(y - xh)}L${n1(x + 0.06 * size - r)} ${n1(y)}`);
    case "ɪ": return st(`M${n1(x)} ${n1(y - xh)}L${n1(x)} ${n1(y)}M${n1(x - xh * 0.3)} ${n1(y - xh)}L${n1(x + xh * 0.3)} ${n1(y - xh)}M${n1(x - xh * 0.3)} ${n1(y)}L${n1(x + xh * 0.3)} ${n1(y)}`);
    case "ɛ": return st(`M${n1(x + r * 0.9)} ${n1(y - xh * 0.92)}Q${n1(x - r * 1.1)} ${n1(y - xh * 1.08)} ${n1(x - r * 0.25)} ${n1(cy)}Q${n1(x - r * 1.3)} ${n1(y + xh * 0.1)} ${n1(x + r * 0.95)} ${n1(y - xh * 0.06)}`);
    case "ɝ": return st(`M${n1(x - r * 0.9)} ${n1(y - xh * 0.92)}Q${n1(x + r * 1.1)} ${n1(y - xh * 1.08)} ${n1(x + r * 0.25)} ${n1(cy)}Q${n1(x + r * 1.3)} ${n1(y + xh * 0.1)} ${n1(x - r * 0.95)} ${n1(y - xh * 0.06)}M${n1(x + r * 0.55)} ${n1(cy - xh * 0.1)}Q${n1(x + r * 1.4)} ${n1(cy - xh * 0.5)} ${n1(x + r * 1.5)} ${n1(cy - xh * 0.15)}`);
    case "ʊ": return st(`M${n1(x - r * 1.15)} ${n1(y - xh)}Q${n1(x - r * 0.5)} ${n1(y - xh * 0.9)} ${n1(x - r * 0.75)} ${n1(cy)}Q${n1(x - r)} ${n1(y)} ${n1(x)} ${n1(y)}Q${n1(x + r)} ${n1(y)} ${n1(x + r * 0.75)} ${n1(cy)}Q${n1(x + r * 0.5)} ${n1(y - xh * 0.9)} ${n1(x + r * 1.15)} ${n1(y - xh)}`);
    case "ʌ": return st(`M${n1(x - r)} ${n1(y)}L${n1(x)} ${n1(y - xh)}L${n1(x + r)} ${n1(y)}`);
    case "ɤ": return st(`M${n1(x - r)} ${n1(y - xh)}L${n1(x)} ${n1(y - xh * 0.1)}L${n1(x + r)} ${n1(y - xh)}M${n1(x)} ${n1(y - xh * 0.1)}Q${n1(x - r * 0.5)} ${n1(y + xh * 0.4)} ${n1(x)} ${n1(y + xh * 0.45)}Q${n1(x + r * 0.5)} ${n1(y + xh * 0.4)} ${n1(x)} ${n1(y - xh * 0.1)}`);
    case "ɔ": return font("c", flip);
    case "ə": return font("e", rot);
    case "ɐ": return font("a", rot);
    case "ɯ": return font("m", rot);
    case "ɹ": return font("r", rot);
    case "ʎ": return font("y", rot);
    case "ɨ": return font("i") + st(`M${n1(x - r * 0.8)} ${n1(cy)}L${n1(x + r * 0.8)} ${n1(cy)}`);
    case "ø": return font("o") + st(`M${n1(x - r * 1.1)} ${n1(y + xh * 0.1)}L${n1(x + r * 1.1)} ${n1(y - xh * 1.1)}`);
    case "θ": return st(`M${n1(x + r)} ${n1(y - xh)}A${n1(r)} ${n1(xh)} 0 1 0 ${n1(x - r)} ${n1(y - xh)}A${n1(r)} ${n1(xh)} 0 1 0 ${n1(x + r)} ${n1(y - xh)}M${n1(x - r)} ${n1(y - xh)}L${n1(x + r)} ${n1(y - xh)}`);
    case "ʃ": return st(`M${n1(x + r * 0.9)} ${n1(y - xh * 1.9)}Q${n1(x)} ${n1(y - xh * 2.2)} ${n1(x)} ${n1(y - xh)}L${n1(x)} ${n1(y + xh * 0.3)}Q${n1(x)} ${n1(y + xh * 0.95)} ${n1(x - r * 0.9)} ${n1(y + xh * 0.7)}`);
    case "ŋ": return st(`M${n1(x - r)} ${n1(y)}L${n1(x - r)} ${n1(y - xh)}M${n1(x - r)} ${n1(y - xh * 0.6)}Q${n1(x - r * 0.2)} ${n1(y - xh * 1.1)} ${n1(x + r * 0.6)} ${n1(y - xh * 0.9)}Q${n1(x + r)} ${n1(y - xh * 0.75)} ${n1(x + r)} ${n1(y - xh * 0.4)}L${n1(x + r)} ${n1(y + xh * 0.4)}Q${n1(x + r)} ${n1(y + xh * 0.85)} ${n1(x + r * 0.3)} ${n1(y + xh * 0.8)}`);
    case "æ": return `<text x="${n1(x)}" y="${n1(y)}" font-size="${size}" fill="${color}" text-anchor="middle" letter-spacing="${n1(-size * 0.18)}" class="l">ae</text>`;
    case "œ": return `<text x="${n1(x)}" y="${n1(y)}" font-size="${size}" fill="${color}" text-anchor="middle" letter-spacing="${n1(-size * 0.18)}" class="l">oe</text>`;
    default:
      // anything else: the Hershey glyph if ASCII, else a thin outline of the fallback glyph (reads as vector lettering)
      if (/^[\x20-\x7e]+$/.test(ipa)) return font(ipa);
      return `<text x="${n1(x)}" y="${n1(y)}" font-size="${size}" fill="none" stroke="${color}" stroke-width="${n1(Math.max(0.5, size * 0.035))}" text-anchor="middle" class="ipa">${esc(ipa)}</text>`;
  }
};

// ---------------------------------------------------------------- top bar
export type Mode = "PLAY" | "SPEAK" | "SING" | "BESTIARY" | "BEATBOX";
export function topBar(active: Mode, body: "MALE" | "FEMALE" | "CHILD" | null = "MALE"): string {
  const y = 32;
  const p: string[] = [];
  // the wordmark and mode tabs link the mockup pages together, so they click through like the app
  p.push(`<a href="index.html">${text(18, y + 1, "CHIMERA", { size: 22, color: C.blue, weight: "m", ls: 0.34 })}<rect x="14" y="12" width="166" height="30" fill="transparent" stroke="none"/></a>`);
  p.push(text(196, y, "VOICE FROM A SIMULATED AIRWAY", { size: 10, color: C.blueDim, ls: 0.18 }));
  let x = 520;
  (["PLAY", "SPEAK", "SING", "BESTIARY", "BEATBOX"] as Mode[]).forEach((t) => {
    const w = t.length * 12 * 0.86 + 22;
    const tab = (t === active ? rect(x, 15, w, 25, C.blue, 1.2) : "") + text(x + w / 2, y, t, { size: 12, color: t === active ? C.blue : C.blueDim, anchor: "middle", ls: 0.18 });
    p.push(`<a href="${t.toLowerCase()}.html">${tab}<rect x="${n1(x)}" y="15" width="${n1(w)}" height="25" fill="transparent" stroke="none"/></a>`);
    x += w + 8;
  });
  if (body) {
    x = 1016;
    ["MALE", "FEMALE", "CHILD"].forEach((t) => {
      const w = t.length * 11 * 0.86 + 20;
      if (t === body) p.push(rect(x, 16, w, 23, C.blueMid, 1));
      p.push(text(x + w / 2, y - 0.5, t, { size: 11, color: t === body ? C.blue : C.blueDim, anchor: "middle", ls: 0.16 }));
      x += w + 6;
    });
  }
  p.push(text(1268, y, "ENGINE", { size: 10, color: C.blueDim, ls: 0.16 }));
  for (let i = 0; i < 8; i++) p.push(rect(1318 + i * 6, 22, 3, 12, i < 3 ? C.blue : C.frame, 1, 1, i < 3 ? C.blue : "none"));
  p.push(text(1378, y, "VOL", { size: 10, color: C.blueDim, ls: 0.16 }));
  for (let i = 0; i < 6; i++) p.push(rect(1404 + i * 4, 34 - (i + 1) * 2, 2, (i + 1) * 2, i < 4 ? C.blue : C.frame, 0.8, 1, i < 4 ? C.blue : "none"));
  return `<g filter="url(#glow)">${p.join("")}</g>`;
}

// ---------------------------------------------------------------- OP-1 style encoder
export interface Enc { label: string; value: string; unit: string; frac: number; color: string }
/** dial centred at (cx, cy); label 42 px above, big value 80 px below, unit 100 px below */
export function encoder(cx: number, cy: number, e: Enc, R = 30): string {
  const p: string[] = [];
  const a0 = (225 * Math.PI) / 180, span = (270 * Math.PI) / 180;
  const ang = (u: number) => a0 - u * span;
  const at = (u: number, rr: number): Pt => [cx + Math.cos(ang(u)) * rr, cy - Math.sin(ang(u)) * rr];
  for (let tk = 0; tk <= 10; tk++) {
    const [ax, ay] = at(tk / 10, R + 5), [bx, by] = at(tk / 10, R + 9);
    p.push(line(`M${n1(ax)} ${n1(ay)}L${n1(bx)} ${n1(by)}`, e.color, { w: 1, o: 0.35 }));
  }
  p.push(circle(cx, cy, R, e.color, 1.1, 0.55));
  if (e.frac > 0.002) {
    const [sx0, sy0] = at(0, R), [sx1, sy1] = at(Math.min(1, e.frac), R);
    p.push(line(`M${n1(sx0)} ${n1(sy0)}A${R} ${R} 0 ${e.frac * 270 > 180 ? 1 : 0} 1 ${n1(sx1)} ${n1(sy1)}`, e.color, { w: 2.2, hot: true }));
  }
  const [qx, qy] = at(Math.min(1, e.frac), R - 6);
  p.push(line(`M${n1(cx)} ${n1(cy)}L${n1(qx)} ${n1(qy)}`, e.color, { w: 1.6, hot: true }));
  p.push(circle(cx, cy, 2.5, e.color, 1, 1, e.color));
  p.push(text(cx, cy - 42, e.label, { size: 10, color: e.color, anchor: "middle", ls: 0.2, o: 0.85 }));
  p.push(text(cx, cy + 80, e.value, { size: e.value.length > 4 ? 22 : 30, color: e.color, anchor: "middle", weight: "m", ls: 0.02 }));
  p.push(text(cx, cy + 100, e.unit, { size: 9, color: e.color, anchor: "middle", ls: 0.16, o: 0.7 }));
  return p.join("");
}
/** four encoders in a row inside S (title row reserved), page indicator top right */
export function encoderScreen(S: Rect, title: string, encs: Enc[], page = 1, pages = 2, sub = ""): string {
  const p: string[] = [screen(S, title, sub)];
  const inner: string[] = [];
  if (pages > 1) for (let i = 0; i < pages; i++) {
    const x = S.x + S.w - 16 - (pages - i) * 16;
    inner.push(rect(x, S.y + 12, 12, 12, i + 1 === page ? C.blue : C.frame, 1), text(x + 6, S.y + 22, String(i + 1), { size: 9, color: i + 1 === page ? C.blue : C.blueDim, anchor: "middle", ls: 0 }));
  }
  const step = S.w / encs.length;
  encs.forEach((e, i) => inner.push(encoder(S.x + step * (i + 0.5), S.y + 92, e)));
  p.push(glow(inner.join("")));
  return p.join("");
}

// ---------------------------------------------------------------- keys
/** keyboard from C<oct0> for nWhite white keys; active = MIDI note lit in `color` */
export function keysScreen(S: Rect, title: string, sub: string, active: number | null, { oct0 = 2, nWhite = 15, color = C.green, above = new Map<number, string>() }: { oct0?: number; nWhite?: number; color?: string; above?: Map<number, string> } = {}): string {
  const p: string[] = [screen(S, title, sub)];
  const inner: string[] = [];
  const kx0 = S.x + 18, kx1 = S.x + S.w - 18, ky0 = S.y + 38, ky1 = S.y + S.h - 16;
  const kw = (kx1 - kx0) / nWhite;
  const WHITE = [0, 2, 4, 5, 7, 9, 11];
  const names = ["C", "D", "E", "F", "G", "A", "B"];
  const letters = ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"];
  const midiOfWhite = (i: number) => 12 * (oct0 + 1 + Math.floor(i / 7)) + WHITE[i % 7];
  for (let i = 0; i < nWhite; i++) {
    const x = kx0 + i * kw;
    const m = midiOfWhite(i);
    const on = m === active;
    inner.push(rect(x + 1.5, ky0, kw - 3, ky1 - ky0, on ? color : C.blueDim, on ? 1.5 : 1, 1, on ? `${color}1f` : "none"));
    if (on) inner.push(text(x + kw / 2, ky1 - 12, `${names[i % 7]}${oct0 + Math.floor(i / 7)}`, { size: 11, color, anchor: "middle", ls: 0.12 }));
    else if (i % 7 === 0) inner.push(text(x + kw / 2, ky1 - 12, `C${oct0 + Math.floor(i / 7)}`, { size: 9, color: C.blueDim, anchor: "middle", ls: 0.12 }));
    if (i < letters.length && !on) inner.push(text(x + kw / 2, ky1 - 30, letters[i], { size: 8, color: C.blueDim, anchor: "middle", o: 0.6 }));
    const lab = above.get(m);
    if (lab) inner.push(text(x + kw / 2, ky0 - 6, lab, { size: 10, color, anchor: "middle", ls: 0.1 }));
  }
  for (let i = 0; i < nWhite - 1; i++) {
    const nm = names[i % 7];
    if (nm === "E" || nm === "B") continue;
    const m = midiOfWhite(i) + 1;
    const on = m === active;
    const x = kx0 + (i + 1) * kw - kw * 0.3;
    inner.push(rect(x, ky0, kw * 0.6, (ky1 - ky0) * 0.58, on ? color : C.blueMid, on ? 1.5 : 1.1, 1, "#000"));
  }
  p.push(glow(inner.join("")));
  return p.join("");
}

// ---------------------------------------------------------------- the face
export interface Slot { label: string; value: string; unit?: string; color: string; ipa?: boolean }
export interface FaceOpts {
  title?: string;
  sub?: string;
  /** four OP-1 slot readouts under the chin (2 × 2) */
  slots?: Slot[];
  /** amber readouts top right (formants) */
  formants?: number[] | null;
  /** extra small lines under the formants */
  info?: string[];
  /** the articulator being dragged: gets the big ring */
  active?: "tongue" | "jaw" | "lips" | "velum" | null;
  /** slot colour per articulator (the four that have slots) */
  colors?: { tongue: string; jaw: string; lips: string; velum: string };
  /** drawn on top, in screen px, with the face transform available */
  extra?: (X: (x: number) => number, Y: (y: number) => number) => string;
}
/** The midsagittal face screen (tall). Returns svg; body→screen transform via opts.extra. */
export function faceScreen(S: Rect, F: FaceData, opts: FaceOpts = {}): string {
  const col = opts.colors ?? { tongue: C.green, jaw: C.red, lips: C.pink, velum: C.white };
  const p: string[] = [screen(S, opts.title ?? "VOCAL TRACT", opts.sub ?? "ADULT MALE / MIDSAGITTAL", true)];
  const view = { x0: -7.5, x1: 10.25, y0: -4.55, y1: 19.9 };
  const padT = 40, padB = 8;
  const sc = Math.min((S.w - 24) / (view.x1 - view.x0), (S.h - padT - padB) / (view.y1 - view.y0));
  const ox = S.x + 12 - view.x0 * sc;
  const oy = S.y + padT + view.y1 * sc;
  const X = (x: number) => ox + x * sc, Y = (y: number) => oy - y * sc;
  const P = (q: Pt): Pt => [X(q[0]), Y(q[1])];
  const pl = (q: Pt[], closePath = false) => dPath(q.map(P), closePath);
  const h = F.head;
  const id = `fc${Math.round(S.x)}_${Math.round(S.y)}`;
  p.push(`<clipPath id="${id}"><rect x="${S.x + 1}" y="${S.y + 30}" width="${S.w - 2}" height="${S.h - 31}"/></clipPath>`);
  p.push(`<clipPath id="${id}t"><path d="${pl(h.tongue, true)}"/></clipPath>`);
  const q: string[] = [];
  q.push(line(pl(h.inset), C.blueDim, { w: 0.9, o: 0.9 }));
  q.push(line(pl(h.skull), C.blueDim, { w: 0.9 }), line(pl(h.brain), C.blueMid, { w: 1.1 }), line(pl(h.sulcus), C.blueDim, { w: 0.9 }));
  q.push(line(pl(h.eye), C.blueMid, { w: 1.1 }), line(pl(h.lens), C.blueDim, { w: 0.9 }));
  for (const v of h.vertebrae) q.push(line(pl(v, true), C.blueDim, { w: 1 }));
  q.push(line(pl(h.cordA), C.blueDim, { w: 0.8, o: 0.8 }), line(pl(h.cordB), C.blueDim, { w: 0.8, o: 0.6 }));
  q.push(line(pl(h.nasal, true), C.blueMid, { w: 1 }));
  for (const tb of h.turbinates) q.push(line(pl(tb, true), C.blueDim, { w: 0.9 }));
  q.push(line(pl(h.palOral), C.blueMid, { w: 1.1 }), line(pl(h.palNasal), C.blueMid, { w: 1.1 }));
  q.push(line(pl(h.hyoid, true), C.blueDim, { w: 1 }), line(pl(h.thyroid, true), C.blueDim, { w: 1 }));
  q.push(line(pl(h.tracheaL), C.blueMid, { w: 1 }), line(pl(h.tracheaR), C.blueMid, { w: 1 }));
  for (const r of h.rings) q.push(line(pl(r, true), C.blueDim, { w: 0.9 }));
  // sound pressure: every cross-section line of the airway lit by its acoustic energy
  for (const g of F.grid) {
    const lv = Math.pow(g.level, 0.85);
    q.push(line(dPath([P(g.inner), P(g.outer)]), heat(lv), { w: 1.15, o: 0.35 + 0.65 * lv }));
  }
  for (const s of F.streaks) q.push(line(dPath([P(s.a), P(s.b)]), "#ffe2a8", { w: 1, o: 0.18 + 0.5 * s.w }));
  // afterglow
  const nG = F.ghosts.length;
  F.ghosts.forEach((gh, i) => {
    const o = 0.1 + 0.2 * ((i + 1) / Math.max(1, nG));
    q.push(line(pl(openTongue(gh.tongue)), col.tongue, { w: 1, o }), line(pl(gh.jaw.lower), col.jaw, { w: 1, o: o * 0.9 }));
    for (const lp of gh.lips) q.push(line(pl(chaikin(lp), true), col.lips, { w: 1, o: o * 0.9 }));
  });
  q.push(line(pl(h.skinLower), C.blue, { w: 1.7, hot: true }), line(pl(h.skinUpper), C.blue, { w: 1.7, hot: true }));
  q.push(line(pl(h.outerWall), C.blue, { w: 1.4, hot: true }), line(pl(h.innerWall), C.blue, { w: 1.3, o: 0.9 }), line(pl(h.glottisWall), C.blue, { w: 1.2, o: 0.8 }));
  q.push(line(pl(h.epiglottis), C.blue, { w: 2.2, o: 0.85 }));
  q.push(line(pl(h.upperTeeth, true), C.blueMid, { w: 1.1 }), line(pl(h.lowerTeeth, true), C.blueMid, { w: 1.1 }));
  q.push(line(pl(h.foldL, true), C.blue, { w: 1 }), line(pl(h.foldR, true), C.blue, { w: 1 }));
  q.push(`<g clip-path="url(#${id}t)">`);
  for (const fb of h.fibres) q.push(line(pl(fb), col.tongue, { w: 0.8, o: 0.32 }));
  q.push(`</g>`);
  q.push(line(pl(openTongue(h.tongue)), col.tongue, { w: 1.6, hot: true }));
  q.push(line(pl(h.jaw.lower), col.jaw, { w: 1.5, hot: true }), line(pl(h.jaw.upper), col.jaw, { w: 1, o: 0.45 }));
  q.push(line(pl(chaikin(h.upperLip), true), col.lips, { w: 1.5, hot: true }), line(pl(chaikin(h.lowerLip), true), col.lips, { w: 1.5, hot: true }));
  q.push(line(pl(chaikin(h.velum, 3), true), col.velum, { w: 1.25, o: 0.92 }));
  // turbulence: beam-dwell dots where the jet breaks up
  for (const s of F.sparks) q.push(`<circle cx="${n1(X(s.p[0]))}" cy="${n1(Y(s.p[1]))}" r="${n1(s.r)}" fill="#f4fbff" stroke="none" opacity="${n1(s.a)}"/>`);
  // handles
  const hd = F.handles;
  const ringFor = { tongue: hd.tongue, jaw: hd.jaw, lips: hd.lips, velum: hd.velum };
  q.push(handle(...P(hd.tongue), col.tongue, 8), handle(...P(hd.jaw), col.jaw, 7, 0.95), handle(...P(hd.lips), col.lips, 7, 0.95), handle(...P(hd.velum), col.velum, 6, 0.9));
  q.push(handle(...P(hd.tip), C.blueMid, 5, 0.8), handle(...P(hd.larynx), C.blueMid, 5, 0.8));
  const act = opts.active === undefined ? "tongue" : opts.active;
  if (act) {
    const [ax, ay] = P(ringFor[act]);
    q.push(circle(ax, ay, 30, col[act], 1, 0.75));
  }
  p.push(`<g clip-path="url(#${id})">${glow(q.join(""))}</g>`);
  // readouts
  const r: string[] = [];
  const fx = S.x + S.w - 18;
  const fm = opts.formants === undefined ? F.formants : opts.formants;
  let yInfo = S.y + 64;
  if (fm) {
    ["F1", "F2", "F3"].forEach((lab, i) => {
      r.push(text(fx - 78, S.y + 64 + i * 22, lab, { size: 12, color: C.amber, ls: 0.12, o: 0.75 }));
      r.push(text(fx, S.y + 64 + i * 22, String(Math.round(fm[i])), { size: 15, color: C.amber, anchor: "end", ls: 0.08 }));
    });
    r.push(text(fx, S.y + 64 + 3 * 22 - 2, "HZ", { size: 10, color: C.amber, anchor: "end", o: 0.6 }));
    yInfo = S.y + 150;
  }
  for (const s of opts.info ?? [`TRACT ${F.tractLength.toFixed(1)} CM`, `MOUTH ${F.lipArea.toFixed(1)} CM2`]) {
    r.push(text(fx, yInfo, s, { size: 10, color: C.blueMid, anchor: "end", ls: 0.14 }));
    yInfo += 16;
  }
  if (opts.slots) {
    const gx0 = X(3.35), gy0 = Y(2.05);
    const cellW = (S.x + S.w - 14 - gx0) / 2, cellH = 112;
    opts.slots.forEach((s, i) => {
      const cx = gx0 + (i % 2) * cellW, cy = gy0 + Math.floor(i / 2) * cellH;
      r.push(circle(cx + 6, cy + 4, 4, s.color, 1.2, 1, s.color));
      r.push(text(cx + 16, cy + 8, s.label, { size: 11, color: s.color, ls: 0.2, o: 0.85 }));
      if (s.ipa) r.push(ipaGlyph(cx + 26, cy + 66, s.value, 64, s.color));
      else r.push(text(cx, cy + 66, s.value, { size: s.value.length > 3 ? 36 : 44, color: s.color, weight: "m", ls: 0.04 }));
      if (s.unit) r.push(text(s.ipa ? cx + 64 : cx + (s.value.length > 3 ? 0 : 72), cy + 66, s.unit, { size: 11, color: s.color, ls: 0.16, o: 0.7 }));
    });
  }
  if (opts.extra) r.push(opts.extra(X, Y));
  p.push(glow(r.join("")));
  return p.join("");
}

// ---------------------------------------------------------------- instruments
export interface TubeData { ph: number[]; or: number[]; phLen: number; orLen: number; rms: number[]; nasalLen: number; velumPort?: number }
export function tubeScreen(S: Rect, tb: TubeData): string {
  const p: string[] = [screen(S, "TUBE", "GLOTTIS TO LIPS")];
  const q: string[] = [];
  const total = tb.phLen + tb.orLen;
  const x0 = S.x + 26, x1 = S.x + S.w - 22;
  const sx = (x1 - x0) / total;
  const axisY = S.y + S.h * 0.64;
  const dS = 13.5;
  const a2d = (A: number) => 2 * Math.sqrt(Math.max(0, A) / Math.PI);
  const secs: { s: number; d: number; lv: number }[] = [];
  const nPh = tb.ph.length, nOr = tb.or.length;
  for (let i = 0; i < nPh; i++) secs.push({ s: ((i + 0.5) / nPh) * tb.phLen, d: a2d(tb.ph[i]), lv: tb.rms[i] });
  for (let i = 0; i < nOr; i++) secs.push({ s: tb.phLen + ((i + 0.5) / nOr) * tb.orLen, d: a2d(tb.or[i]), lv: tb.rms[nPh + i] });
  const open = (tb.velumPort ?? 0) > 0.02;
  const jx = x0 + tb.phLen * sx;
  const nTop = axisY - 66;
  const nLen = tb.nasalLen * sx * 0.62;
  const nasalCol = open ? C.white : C.blueDim;
  const dash = open ? undefined : "3 4";
  q.push(line(`M${n1(jx - 5)} ${n1(axisY - 30)}L${n1(jx - 5)} ${n1(nTop + 6)}Q${n1(jx - 5)} ${n1(nTop - 6)} ${n1(jx + 8)} ${n1(nTop - 6)}L${n1(jx + nLen)} ${n1(nTop - 6)}`, nasalCol, { w: 1, dash }));
  q.push(line(`M${n1(jx + 5)} ${n1(axisY - 30)}L${n1(jx + 5)} ${n1(nTop + 6)}Q${n1(jx + 5)} ${n1(nTop + 6)} ${n1(jx + 8)} ${n1(nTop + 6)}L${n1(jx + nLen)} ${n1(nTop + 6)}`, nasalCol, { w: 1, dash }));
  q.push(text(jx + nLen + 6, nTop + 4, "NOSE", { size: 9, color: open ? C.white : C.blueDim, ls: 0.16 }));
  q.push(text(jx - 2, nTop - 14, open ? "VELUM OPEN" : "VELUM SHUT", { size: 9, color: open ? C.white : C.blueDim, ls: 0.16, anchor: "middle" }));
  const top = secs.map((qq) => [x0 + qq.s * sx, axisY - (qq.d * dS) / 2] as Pt);
  const bot = secs.map((qq) => [x0 + qq.s * sx, axisY + (qq.d * dS) / 2] as Pt);
  top.unshift([x0, axisY - 3]); bot.unshift([x0, axisY + 3]);
  top.push([x1, top[top.length - 1][1]]); bot.push([x1, bot[bot.length - 1][1]]);
  q.push(line(dPath(top), C.blue, { w: 1.3, hot: true }), line(dPath(bot), C.blue, { w: 1.3, hot: true }));
  for (const qq of secs) {
    const cx = x0 + qq.s * sx, ry = Math.max(0.6, (qq.d * dS) / 2), rx = Math.max(1.2, ry * 0.28);
    const lv = Math.pow(qq.lv, 0.85);
    q.push(`<ellipse cx="${n1(cx)}" cy="${n1(axisY)}" rx="${n1(rx)}" ry="${n1(ry)}" stroke="${heat(lv)}" stroke-width="1.1" opacity="${n1(0.35 + 0.65 * lv)}"/>`);
  }
  q.push(line(`M${x0} ${n1(axisY)}L${x1} ${n1(axisY)}`, C.blueDim, { w: 0.8, dash: "2 5", o: 0.8 }));
  q.push(text(x0 - 2, S.y + S.h - 14, "GLOTTIS", { size: 9, color: C.blueDim, ls: 0.16 }));
  q.push(text(x1 + 2, S.y + S.h - 14, "LIPS", { size: 9, color: C.blueDim, ls: 0.16, anchor: "end" }));
  for (const s of [0, 5, 10, 15]) if (s < total) q.push(line(`M${n1(x0 + s * sx)} ${S.y + S.h - 30}L${n1(x0 + s * sx)} ${S.y + S.h - 26}`, C.blueDim, { w: 1 }), text(x0 + s * sx, S.y + S.h - 34, String(s), { size: 8, color: C.blueDim, anchor: "middle" }));
  p.push(glow(q.join("")));
  return p.join("");
}

export function foldsScreen(S: Rect, ga: number[], gf: number[], sub = "SLOWED 100X"): string {
  const p: string[] = [screen(S, "VOCAL FOLDS", sub)];
  const q: string[] = [];
  const n = ga.length;
  const maxA = Math.max(0.02, ...ga);
  let idx = 0;
  for (let i = 0; i < n / 2; i++) if (ga[i] > ga[idx]) idx = i;
  idx = Math.max(0, idx - Math.round(n * 0.06));
  const a = ga[idx] ?? 0;
  const R = 64, cx = S.x + 90, cy = S.y + 118;
  q.push(circle(cx, cy, R, C.blue, 1.3), circle(cx, cy, R + 5, C.blueDim, 0.8, 0.8));
  const len = R * 1.22;
  const open = Math.min(1, a / Math.max(0.12, maxA)) * R * 0.42;
  const chink = R * 0.03;
  q.push(line(`M${n1(cx)} ${n1(cy - len / 2)}Q${n1(cx + open + chink)} ${n1(cy + len * 0.05)} ${n1(cx + chink * 0.8)} ${n1(cy + len / 2)}`, C.white, { w: 1.6, hot: true }));
  q.push(line(`M${n1(cx)} ${n1(cy - len / 2)}Q${n1(cx - open - chink)} ${n1(cy + len * 0.05)} ${n1(cx - chink * 0.8)} ${n1(cy + len / 2)}`, C.white, { w: 1.6, hot: true }));
  for (const sg of [-1, 1]) {
    q.push(line(`M${n1(cx + sg * 3)} ${n1(cy - len / 2 - 2)}Q${n1(cx + sg * R * 0.62)} ${n1(cy)} ${n1(cx + sg * R * 0.22)} ${n1(cy + len / 2 + 4)}`, C.blueMid, { w: 1.1 }));
    q.push(line(`M${n1(cx + sg * 8)} ${n1(cy - len / 2 - 8)}Q${n1(cx + sg * R * 0.92)} ${n1(cy - 2)} ${n1(cx + sg * R * 0.42)} ${n1(cy + len / 2 + 6)}`, C.blueDim, { w: 1 }));
    q.push(`<ellipse cx="${n1(cx + sg * R * 0.2)}" cy="${n1(cy + len / 2 + 9)}" rx="${n1(R * 0.13)}" ry="${n1(R * 0.09)}" stroke="${C.blueMid}" stroke-width="1.1"/>`);
  }
  q.push(line(`M${n1(cx - R * 0.42)} ${n1(cy - R * 0.62)}Q${n1(cx)} ${n1(cy - R * 0.98)} ${n1(cx + R * 0.42)} ${n1(cy - R * 0.62)}`, C.blueMid, { w: 1.2 }));
  q.push(text(cx, cy - R - 10, "FRONT", { size: 8, color: C.blueDim, anchor: "middle", ls: 0.18 }));
  const tx0 = S.x + 176, tx1 = S.x + S.w - 16;
  const band = (vals: number[], y0: number, y1: number) => {
    const mx = Math.max(...vals), mn = Math.min(...vals);
    return vals.map((v, i) => [tx0 + (i / (vals.length - 1)) * (tx1 - tx0), y1 - ((v - mn) / Math.max(1e-9, mx - mn)) * (y1 - y0)] as Pt);
  };
  q.push(text(tx1, S.y + 58, "AREA", { size: 9, color: C.blueMid, ls: 0.18, anchor: "end" }));
  q.push(line(`M${tx0} ${S.y + 110}L${tx1} ${S.y + 110}`, C.frame, { w: 1 }));
  q.push(line(dPath(band(ga, S.y + 66, S.y + 108)), C.white, { w: 1.3, hot: true }));
  q.push(text(tx1, S.y + 132, "FLOW", { size: 9, color: C.blueMid, ls: 0.18, anchor: "end" }));
  q.push(line(`M${tx0} ${S.y + 184}L${tx1} ${S.y + 184}`, C.frame, { w: 1 }));
  q.push(line(dPath(band(gf, S.y + 140, S.y + 182)), C.amber, { w: 1.3, hot: true }));
  const px = tx0 + (idx / (n - 1)) * (tx1 - tx0);
  q.push(line(`M${n1(px)} ${S.y + 62}L${n1(px)} ${S.y + 186}`, C.white, { w: 1, o: 0.55, dash: "2 3" }));
  p.push(glow(q.join("")));
  return p.join("");
}

export function spectrumScreen(S: Rect, harmonics: { f: number; db: number }[], envelope: Pt[], formants: number[], f0: number, { title = "SPECTRUM", sub = "RESONANCES", f0Color = C.green }: { title?: string; sub?: string; f0Color?: string } = {}): string {
  const p: string[] = [screen(S, title, sub)];
  const q: string[] = [];
  const x0 = S.x + 40, x1 = S.x + S.w - 20, y0 = S.y + 44, y1 = S.y + S.h - 34;
  const fx = (f: number) => x0 + (Math.log(f / 60) / Math.log(8000 / 60)) * (x1 - x0);
  const dbY = (db: number) => y0 + (Math.min(0, Math.max(-62, db)) / -62) * (y1 - y0);
  for (const f of [100, 200, 500, 1000, 2000, 5000]) {
    q.push(line(`M${n1(fx(f))} ${y0}L${n1(fx(f))} ${y1}`, C.frame, { w: 1 }));
    if (f !== 100) q.push(text(fx(f), y1 + 14, f >= 1000 ? `${f / 1000}K` : String(f), { size: 9, color: C.blueMid, anchor: "middle", ls: 0.1 }));
  }
  for (const db of [-20, -40]) {
    q.push(line(`M${x0} ${n1(dbY(db))}L${x1} ${n1(dbY(db))}`, C.frame, { w: 1, dash: "2 4" }));
    q.push(text(x0 - 6, dbY(db) + 3, String(db), { size: 8, color: C.blueDim, anchor: "end", ls: 0.05 }));
  }
  q.push(line(`M${x0} ${y1}L${x1} ${y1}`, C.blueDim, { w: 1 }));
  for (const hm of harmonics) {
    if (hm.f < 60 || hm.f > 8000) continue;
    q.push(line(`M${n1(fx(hm.f))} ${y1}L${n1(fx(hm.f))} ${n1(dbY(hm.db))}`, C.amber, { w: 1.1, o: 0.55 }));
  }
  const env = envelope.filter(([f]) => f >= 60 && f <= 8000).map(([f, db]) => [fx(f), dbY(db)] as Pt);
  q.push(line(dPath(env), C.amber, { w: 1.5, hot: true }));
  formants.slice(0, 3).forEach((F, i) => {
    const near = envelope.reduce((b, pp) => (Math.abs(pp[0] - F) < Math.abs(b[0] - F) ? pp : b));
    const x = fx(F), y = dbY(near[1]) - 10;
    q.push(line(`M${n1(x - 5)} ${n1(y - 8)}L${n1(x + 5)} ${n1(y - 8)}L${n1(x)} ${n1(y)}Z`, C.amber, { w: 1.2 }));
    q.push(text(x, y - 13, `F${i + 1}`, { size: 10, color: C.amber, anchor: "middle", ls: 0.1 }));
  });
  if (f0 > 0) {
    const xf = fx(f0);
    q.push(line(`M${n1(xf - 5)} ${y1 + 9}L${n1(xf + 5)} ${y1 + 9}L${n1(xf)} ${y1 + 1}Z`, f0Color, { w: 1.2 }));
    q.push(text(xf - 9, y1 + 13, `F0 ${Math.round(f0)}`, { size: 9, color: f0Color, ls: 0.12, anchor: "end" }));
  }
  p.push(glow(q.join("")));
  return p.join("");
}

export function vowelsScreen(S: Rect, vowels: { ipa: string; F1: number; F2: number }[], trail: Pt[], live: Pt, liveIpa: string, color = C.green): string {
  const p: string[] = [screen(S, "VOWELS", "F1 / F2")];
  const q: string[] = [];
  const r = { f1: [230, 950], f2: [700, 2700] };
  const px0 = S.x + 22, px1 = S.x + S.w - 22, py0 = S.y + 46, py1 = S.y + S.h - 18;
  const toXY = (F1: number, F2: number): Pt => [px0 + ((Math.log(r.f2[1]) - Math.log(F2)) / (Math.log(r.f2[1]) - Math.log(r.f2[0]))) * (px1 - px0), py0 + ((Math.log(F1) - Math.log(r.f1[0])) / (Math.log(r.f1[1]) - Math.log(r.f1[0]))) * (py1 - py0)];
  const corners = [toXY(300, 2500), toXY(320, 850), toXY(720, 1050), toXY(780, 1650)];
  q.push(line(dPath([...corners, corners[0]]), C.blueDim, { w: 1, o: 0.9 }));
  for (const v of vowels) {
    if (v.ipa === liveIpa) continue;
    const [x, y] = toXY(v.F1, v.F2);
    q.push(ipaGlyph(x, y + 4, v.ipa, 15, C.blueMid));
  }
  const tr = trail.map(([F1, F2]) => toXY(F1, F2));
  for (let i = 1; i < tr.length; i++) q.push(line(dPath([tr[i - 1], tr[i]]), color, { w: 1.2, o: 0.12 + (0.8 * i) / tr.length }));
  const [lx, ly] = toXY(live[0], live[1]);
  q.push(handle(lx, ly, color, 8));
  if (liveIpa) q.push(ipaGlyph(lx + 22, ly - 8, liveIpa, 17, color));
  p.push(glow(q.join("")));
  return p.join("");
}

// ---------------------------------------------------------------- page
const b64 = async (path: string) => Buffer.from(await Bun.file(path).arrayBuffer()).toString("base64");
export async function writePage(name: string, title: string, parts: string[], comment = ""): Promise<string> {
  const fontLight = await b64(FONTS + "AVHersheySimplexLight.otf");
  const fontMed = await b64(FONTS + "AVHersheySimplexMedium.otf");
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="${esc(title)}">
<defs>
  <filter id="glow" x="-3%" y="-3%" width="106%" height="106%" color-interpolation-filters="sRGB">
    <feGaussianBlur in="SourceGraphic" stdDeviation="1.15" result="b1"/>
    <feGaussianBlur in="SourceGraphic" stdDeviation="3.8" result="b2"/>
    <feComponentTransfer in="b2" result="b2a"><feFuncA type="linear" slope="0.7"/></feComponentTransfer>
    <feMerge><feMergeNode in="b2a"/><feMergeNode in="b1"/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>
</defs>
<rect width="${W}" height="${H}" fill="#000"/>
${parts.join("\n")}
</svg>`;
  const html = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${esc(title)}</title>
<!--
  CHIMERA UI mockup: WarGames-style vector display with OP-1-style colour-coded controls.
  ${comment}
  Drawn from real engine output (the TS engine at c9020c3); outlines follow src/ui/render/anatomy.ts.
  Font: AVHershey Simplex by Stewart C. Russell (CC0 / WTFPL), from A. V. Hershey's public-domain vector fonts (1967).
-->
<style>
  @font-face { font-family: "Hershey"; src: url(data:font/otf;base64,${fontLight}) format("opentype"); font-weight: 300; }
  @font-face { font-family: "Hershey"; src: url(data:font/otf;base64,${fontMed}) format("opentype"); font-weight: 500; }
  html, body { margin: 0; background: #000; }
  body { display: flex; justify-content: center; align-items: flex-start; min-height: 100vh; }
  svg { display: block; width: 100%; max-width: 1440px; height: auto; }
  text { font-family: "Hershey", monospace; }
  text.l { font-weight: 300; }
  text.m { font-weight: 500; }
  text.ipa { font-family: "DejaVu Sans", "Noto Sans", "Segoe UI", sans-serif; font-weight: 400; }
</style>
</head>
<body>
${svg}
</body>
</html>`;
  const path = OUT + name + ".html";
  await Bun.write(path, html);
  return path;
}
