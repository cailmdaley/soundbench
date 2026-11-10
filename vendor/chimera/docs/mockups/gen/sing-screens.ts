/** Sing-only screens: piano roll, lyrics chips, the singer's-ring marker on the spectrum. */
import { C, screen, glow, line, text, rect, n1, dPath, textWidth, type Rect, type Pt } from "./lib.ts";
import type { Syllable } from "../../../src/engine/phonetics/syllables.ts";

const NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const BLACK = [1, 3, 6, 8, 10];

interface RollNote { midi: number; t0: number; t1: number; text: string }

/**
 * Piano roll: the notes asked for (outlined bars, green = pitch) with their syllables, the pitch
 * actually sung (amber trace: onsets, glides, vibrato), the now line at 70 % of the width.
 */
export function rollScreen(S: Rect, notes: RollNote[], trace: { t: number; m: number; v: number }[], now: number, cur: number, t0: number, beat: number): string {
  const p: string[] = [screen(S, "PIANO ROLL", "NOTES AND THE PITCH SUNG")];
  const x0 = S.x + 46, x1 = S.x + S.w - 14, y0 = S.y + 44, y1 = S.y + S.h - 26;
  const span = 6.6;
  const tLeft = now - 0.7 * span, tRight = tLeft + span;
  const X = (t: number) => x0 + ((t - tLeft) / span) * (x1 - x0);
  const vis = notes.filter((n) => n.t1 > tLeft && n.t0 < tRight);
  const lo = Math.min(...vis.map((n) => n.midi)) - 2, hi = Math.max(...vis.map((n) => n.midi)) + 2;
  const rows = hi - lo + 1;
  const rowH = (y1 - y0) / rows;
  const yc = (m: number) => y1 - (m - lo + 0.5) * rowH;
  const q: string[] = [];
  // rows: boundaries dim, C brighter; black-key rows get a faint dotted centre line; natural notes named
  for (let m = lo; m <= hi + 1; m++) {
    const y = y1 - (m - lo) * rowH;
    q.push(line(`M${x0} ${n1(y)}L${x1} ${n1(y)}`, m % 12 === 0 ? C.blueDim : C.frame, { w: 1, o: m % 12 === 0 ? 1 : 0.55 }));
  }
  for (let m = lo; m <= hi; m++) {
    const pc = ((m % 12) + 12) % 12;
    if (BLACK.includes(pc)) q.push(line(`M${x0} ${n1(yc(m))}L${x1} ${n1(yc(m))}`, C.frame, { w: 1, dash: "1 6", o: 0.8 }));
    else q.push(text(x0 - 8, yc(m) + 3, `${NAMES[pc]}${Math.floor(m / 12) - 1}`, { size: pc === 0 ? 10 : 9, color: pc === 0 ? C.blue : C.blueMid, anchor: "end", ls: 0.08 }));
  }
  // beats and bars
  for (let k = Math.ceil((tLeft - t0) / beat); t0 + k * beat <= tRight; k++) {
    const x = X(t0 + k * beat);
    if (x < x0 || x > x1) continue;
    if (k % 4 === 0) {
      q.push(line(`M${n1(x)} ${y0}L${n1(x)} ${y1}`, C.blueDim, { w: 1, o: 0.7 }));
      q.push(text(x + 4, y1 + 14, `BAR ${k / 4 + 1}`, { size: 8, color: C.blueDim, ls: 0.14 }));
    } else q.push(line(`M${n1(x)} ${y1}L${n1(x)} ${y1 + 5}`, C.blueDim, { w: 1 }));
  }
  // notes and syllables (clipped to the plot)
  const id = `roll${Math.round(S.x)}`;
  const nq: string[] = [];
  const barH = rowH * 0.62;
  const labels: string[] = [];
  notes.forEach((n, i) => {
    if (!(n.t1 > tLeft && n.t0 < tRight)) return;
    const xa = X(n.t0) + 1, xb = X(n.t1) - 2, y = yc(n.midi);
    const r = Math.min(3, barH / 2);
    const d = `M${n1(xa + r)} ${n1(y - barH / 2)}L${n1(xb - r)} ${n1(y - barH / 2)}Q${n1(xb)} ${n1(y - barH / 2)} ${n1(xb)} ${n1(y - barH / 2 + r)}L${n1(xb)} ${n1(y + barH / 2 - r)}Q${n1(xb)} ${n1(y + barH / 2)} ${n1(xb - r)} ${n1(y + barH / 2)}L${n1(xa + r)} ${n1(y + barH / 2)}Q${n1(xa)} ${n1(y + barH / 2)} ${n1(xa)} ${n1(y + barH / 2 - r)}L${n1(xa)} ${n1(y - barH / 2 + r)}Q${n1(xa)} ${n1(y - barH / 2)} ${n1(xa + r)} ${n1(y - barH / 2)}Z`;
    const live = i === cur, past = n.t1 <= now;
    if (live) nq.push(`<path d="${d}" fill="${C.green}" fill-opacity="0.12" stroke="none"/>`, line(d, C.green, { w: 1.6, hot: true }));
    else nq.push(line(d, C.green, { w: 1.15, o: past ? 0.7 : 0.38 }));
    const prev = notes[i - 1];
    const below = !!prev && prev.midi > n.midi && prev.t1 > tLeft;
    const ly = below ? y + barH / 2 + (live ? 15 : 13) : y - barH / 2 - 5;
    labels.push(text(xa + 8, ly, n.text, { size: live ? 12 : 10, color: live ? C.white : past ? C.blueMid : C.blueDim, ls: 0.14 }));
  });
  // the pitch actually sung
  let seg: Pt[] = [];
  const flush = () => { if (seg.length > 1) nq.push(line(dPath(seg), C.amber, { w: 1.5, hot: true })); seg = []; };
  for (const s of trace) {
    if (!(s.v > 0.3) || !Number.isFinite(s.m) || s.t < tLeft) { flush(); continue; }
    seg.push([X(s.t), yc(s.m)]);
  }
  flush();
  nq.push(...labels);
  // now
  const xn = X(now);
  nq.push(line(`M${n1(xn)} ${y0 - 4}L${n1(xn)} ${y1 + 3}`, C.white, { w: 1, o: 0.7, dash: "3 3" }));
  q.push(`<clipPath id="${id}"><rect x="${x0}" y="${y0 - 16}" width="${x1 - x0}" height="${y1 - y0 + 20}"/></clipPath><g clip-path="url(#${id})">${nq.join("")}</g>`);
  p.push(glow(q.join("")));
  return p.join("");
}

/** Lyrics as syllable chips: sung (dim), now (lit), next (dashed), breath marks after phrases. */
export function lyricsScreen(S: Rect, syl: Syllable[], cur: number, sub: string): string {
  const p: string[] = [screen(S, "LYRICS", sub)];
  const q: string[] = [];
  const size = 11, h = 22, padX = 8;
  const xMin = S.x + 14, xMax = S.x + S.w - 14;
  let x = xMin, y = S.y + 36;
  // lay out by phrase so a line never breaks mid-phrase
  const phrases: number[][] = [[]];
  syl.forEach((s, i) => { phrases[phrases.length - 1].push(i); if (s.phraseEnd && i < syl.length - 1) phrases.push([]); });
  const wOf = (i: number) => textWidth(syl[i].text.toUpperCase(), size, 0.16) + padX * 2;
  const gapAfter = (i: number) => (syl[i].phraseEnd ? 22 : syl[i].wordEnd ? 10 : 3);
  for (const ph of phrases) {
    const width = ph.reduce((s, i) => s + wOf(i) + gapAfter(i), 0);
    if (x > xMin && x + width > xMax + 22) { x = xMin; y += h + 10; }
    for (const i of ph) {
      const label = syl[i].text.toUpperCase();
      const w = wOf(i);
      const state = i < cur ? "past" : i === cur ? "now" : i === cur + 1 ? "next" : "later";
      if (state === "now") q.push(rect(x, y, w, h, C.green, 1.3), rect(x + 2.5, y + 2.5, w - 5, h - 5, C.green, 0.8, 0.45));
      else if (state === "next") q.push(rect(x, y, w, h, C.green, 1, 0.7).replace("/>", ' stroke-dasharray="3 3"/>'));
      else q.push(rect(x, y, w, h, C.frame, 1, state === "past" ? 1 : 0.8));
      q.push(text(x + w / 2, y + h / 2 + size * 0.36, label, { size, color: state === "now" ? C.green : state === "past" ? C.blueMid : state === "next" ? C.green : C.blueDim, anchor: "middle", ls: 0.16, o: state === "next" ? 0.8 : 1 }));
      x += w;
      if (syl[i].phraseEnd) {
        // breath mark
        q.push(line(`M${n1(x + 9)} ${n1(y + 2)}L${n1(x + 7)} ${n1(y + 8)}`, C.blueMid, { w: 1.6 }));
      }
      x += gapAfter(i);
    }
  }
  p.push(glow(q.join("")));
  return p.join("");
}

/** White bracket over the singer's ring band (2.4 to 3.6 kHz), in the RING encoder's colour. */
export function ringOverlay(S: Rect, envelope: Pt[]): string {
  const x0 = S.x + 40, x1 = S.x + S.w - 20, y0 = S.y + 44, y1 = S.y + S.h - 34;
  const fx = (f: number) => x0 + (Math.log(f / 60) / Math.log(8000 / 60)) * (x1 - x0);
  const dbY = (db: number) => y0 + (Math.min(0, Math.max(-62, db)) / -62) * (y1 - y0);
  void dbY; void envelope;
  const a = fx(2400), b = fx(3600);
  return glow(
    line(`M${n1(a)} ${n1(y1)}L${n1(a)} ${n1(y0 - 2)}L${n1(b)} ${n1(y0 - 2)}L${n1(b)} ${n1(y1)}`, C.white, { w: 1, o: 0.55, dash: "3 3" }) +
    text(b + 5, y0 + 6, "RING", { size: 9, color: C.white, ls: 0.16, o: 0.9 }),
  );
}

/** Dashed outline on the white key that sings the next syllable (same geometry as keysScreen). */
export function nextKeyOverlay(S: Rect, midi: number, oct0 = 2, nWhite = 15): string {
  const WHITE = [0, 2, 4, 5, 7, 9, 11];
  const kx0 = S.x + 18, kx1 = S.x + S.w - 18, ky0 = S.y + 38, ky1 = S.y + S.h - 16;
  const kw = (kx1 - kx0) / nWhite;
  const i = (Math.floor(midi / 12) - (oct0 + 1)) * 7 + WHITE.indexOf(midi % 12);
  if (WHITE.indexOf(midi % 12) < 0 || i < 0 || i >= nWhite) return "";
  const BLACK = [1, 3, 6, 8, 10];
  const xa = kx0 + i * kw + 4, xb = kx0 + (i + 1) * kw - 4;
  const bh = (ky1 - ky0) * 0.58 + 4;
  const nl = BLACK.includes((midi + 11) % 12) ? kw * 0.3 : 0, nr = BLACK.includes((midi + 1) % 12) ? kw * 0.3 : 0;
  const ya = ky0 + 3, yb = ky1 - 3;
  const d = `M${n1(xa + nl)} ${n1(ya)}L${n1(xb - nr)} ${n1(ya)}L${n1(xb - nr)} ${n1(ya + bh)}L${n1(xb)} ${n1(ya + bh)}L${n1(xb)} ${n1(yb)}L${n1(xa)} ${n1(yb)}L${n1(xa)} ${n1(ya + bh)}L${n1(xa + nl)} ${n1(ya + bh)}Z`;
  return glow(line(d, C.green, { w: 1, o: 0.75, dash: "3 3" }));
}
