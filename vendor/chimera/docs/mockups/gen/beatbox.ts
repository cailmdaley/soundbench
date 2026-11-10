/**
 * BEATBOX mode mockup: the "Hum bass" preset played by the sequencer on an offline HumanVoice.
 * The face is caught on the kick that opens bar 2, the instant the lips burst on the air the
 * rising larynx has compressed. → out/beatbox.html
 */
import { HumanVoice } from "../../../src/engine/voice/humanVoice.ts";
import { BODY_MALE, type HumanArtic } from "../../../src/engine/anatomy/human.ts";
import { BEAT_SOUNDS, BEAT_PRESETS, BASS_NOTE, beatReady, parseBeat, type Airstream } from "../../../src/engine/phonetics/beatbox.ts";
import { BeatPlayer } from "../../../src/engine/control/beatPlayer.ts";
import { FFT, hann } from "../../../src/engine/analysis/fft.ts";
import { VoiceModel, buildFaceData, type Pt } from "./face.ts";
import { C, topBar, faceScreen, encoderScreen, screen, glow, text, rect, line, chip, dPath, n1, heat, textWidth, writePage, type Rect } from "./lib.ts";

const FS = 48000;
const PRESET = "Hum bass";
const preset = BEAT_PRESETS.find((p) => p.name === PRESET)!;
const AIR: Record<Airstream, string> = { pulmonic: "BREATH OUT", ejective: "LARYNX PISTON", click: "TONGUE SUCTION", ingressive: "BREATH IN", voiced: "VOICE" };
const NOTE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const noteName = (m: number) => `${NOTE[((m % 12) + 12) % 12]}${Math.floor(m / 12) - 1}`;

// ---------------------------------------------------------------- engine: the sequencer on a voice, as the worklet runs it
const model = new VoiceModel("male", FS);
const v = new HumanVoice({ fsOut: FS, body: BODY_MALE });
v.setArticulation(beatReady("male"));
v.setSource({ f0: 100, voiced: true, pressure: 8 });
v.setGate(0);
const seq = new BeatPlayer();
seq.body = "male";
seq.steps = parseBeat(preset.text);
seq.setTempo(preset.bpm, preset.swing);
seq.start(v.clock);
const L = seq.steps.length;
const hitT = (n: number) => seq.horizon + n * seq.stepDur + (n % 2 ? seq.swing * 0.5 * seq.stepDur : 0);
const STEP = L; // the kick on the first step of bar 2
const tNow = hitT(STEP);
const events = [
  { t: tNow - 0.09, kind: "ghost" }, { t: tNow - 0.06, kind: "ghost" }, { t: tNow - 0.03, kind: "ghost" },
  { t: tNow - 0.016, kind: "reset" }, { t: tNow, kind: "now" },
];
const halfStep = seq.stepDur / 2;
const total = Math.round(FS * (tNow + halfStep + 0.03));
const out = new Float32Array(total);
const ghosts: HumanArtic[] = [];
let tele = v.telemetry();
let ei = 0;
for (let i = 0; i < total; i += 128) {
  seq.tick(v.clock, (gs) => v.schedule(gs));
  v.render(out, i, Math.min(128, total - i));
  while (ei < events.length && v.clock >= events[ei].t) {
    const e = events[ei++];
    const tl = v.telemetry();
    if (e.kind === "ghost") ghosts.push(VoiceModel.articFromArray(tl.artic));
    if (e.kind === "now") tele = tl;
  }
}
const nowSample = Math.round((tNow + halfStep) * FS);
const artic = VoiceModel.articFromArray(tele.artic);
const F = buildFaceData(model, artic, tele, { ghosts, seed: 11 });
const cur = seq.steps[STEP % L]!;
const curSound = BEAT_SOUNDS.find((s) => s.id === cur.id)!;
console.log(`moment: ${cur.id} at ${tNow.toFixed(3)} s  lipAp ${artic.lipAp.toFixed(2)} jaw ${artic.jaw.toFixed(2)} larynx ${artic.larynx.toFixed(2)} noise ${tele.noiseLevel.toFixed(0)} sparks ${F.sparks.length} maxLv ${Math.max(...F.grid.map((g) => g.level)).toFixed(2)}`);

// ---------------------------------------------------------------- waterfall data: the bar that just played, newest slice in front
const barDur = L * seq.stepDur;
const NS = 2 * L, NF = 110, FMIN = 60, FMAX = 12000;
const WIN = 2048, HOP = 256;
const fft = new FFT(WIN);
const win = hann(WIN);
const re = new Float64Array(WIN), im = new Float64Array(WIN);
const freqs = Array.from({ length: NF }, (_, i) => FMIN * Math.pow(FMAX / FMIN, i / (NF - 1)));
const slices: Float64Array[] = Array.from({ length: NS }, () => new Float64Array(NF).fill(-200));
const t0 = tNow + halfStep - barDur;
const pw = new Float64Array(WIN / 2);
for (let c = Math.round(t0 * FS) + WIN / 2; c + WIN / 2 <= nowSample; c += HOP) {
  const k = Math.min(NS - 1, Math.floor(((c / FS - t0) / barDur) * NS));
  if (k < 0) continue;
  for (let j = 0; j < WIN; j++) { re[j] = (out[c - WIN / 2 + j] ?? 0) * win[j]; im[j] = 0; }
  fft.transform(re, im);
  for (let q = 0; q < WIN / 2; q++) pw[q] = 10 * Math.log10(re[q] * re[q] + im[q] * im[q] + 1e-20);
  for (let b = 0; b < NF; b++) {
    // narrow bands (low end): interpolate between bins; wide bands: the loudest bin
    const fb = (freqs[b] / FS) * WIN;
    const lo = Math.floor(((b ? Math.sqrt(freqs[b - 1] * freqs[b]) : freqs[0]) / FS) * WIN);
    const hi = Math.ceil(((b < NF - 1 ? Math.sqrt(freqs[b] * freqs[b + 1]) : freqs[b]) / FS) * WIN);
    let v = pw[Math.floor(fb)] + (pw[Math.floor(fb) + 1] - pw[Math.floor(fb)]) * (fb - Math.floor(fb));
    for (let q = lo + 1; q < hi; q++) v = Math.max(v, pw[q]);
    slices[k][b] = Math.max(slices[k][b], v);
  }
}
let gMax = -Infinity;
for (const sl of slices) for (const x of sl) gMax = Math.max(gMax, x);
for (const sl of slices) {
  const sm = Float64Array.from(sl, (_, b) => 0.25 * sl[Math.max(0, b - 1)] + 0.5 * sl[b] + 0.25 * sl[Math.min(NF - 1, b + 1)]);
  for (let b = 0; b < NF; b++) sl[b] = Math.max(-54, Math.min(0, sm[b] - gMax));
}
// which slice each hit of the bar landed in (for labels)
const hitSlices = new Map<number, string>();
for (let n = STEP - L + 1; n <= STEP + 1; n++) {
  const c = seq.steps[((n % L) + L) % L];
  if (!c) continue;
  const k = Math.floor(((hitT(n) + 0.004 - t0) / barDur) * NS);
  if (k >= 0 && k < NS) hitSlices.set(k, c.id);
}

// ---------------------------------------------------------------- composition
const parts: string[] = [topBar("BEATBOX")];
const FACE: Rect = { x: 16, y: 52, w: 700, h: 832 };
const liftCm = artic.larynx;
parts.push(faceScreen(FACE, F, {
  sub: "ADULT MALE / MIDSAGITTAL",
  formants: null,
  info: [],
  active: "lips",
  slots: [
    { label: "TONGUE", value: "", unit: "REST", color: C.green },
    { label: "JAW", value: artic.jaw.toFixed(1), unit: "CM", color: C.red },
    { label: "LIPS", value: artic.lipAp > 0.005 ? "BURST" : "SEALED", color: C.pink },
    { label: "VELUM", value: artic.velum > 0 ? "OPEN" : "SHUT", color: C.white },
  ],
  extra: (X, Y) => {
    const p: string[] = [];
    // the sound being made, big, like the film's sector numbers
    const bx = FACE.x + FACE.w - 22;
    p.push(`<text x="${n1(bx)}" y="${n1(FACE.y + 214)}" font-size="176" fill="none" stroke="${C.amber}" stroke-width="1.3" text-anchor="end" class="m">${cur.id}</text>`);
    p.push(text(bx, FACE.y + 246, curSound.name.toUpperCase(), { size: 17, color: C.amber, anchor: "end", ls: 0.22 }));
    p.push(text(bx, FACE.y + 266, AIR[curSound.airstream], { size: 10, color: C.amber, anchor: "end", ls: 0.18, o: 0.75 }));
    p.push(text(bx, FACE.y + 284, `PISTON UP ${liftCm.toFixed(1)} CM`, { size: 10, color: C.blue, anchor: "end", ls: 0.18 }));
    p.push(text(bx, FACE.y + 302, `STEP ${String((STEP % L) + 1).padStart(2, "0")} / ${L}`, { size: 10, color: C.blueMid, anchor: "end", ls: 0.18 }));
    // schwa for the tongue slot, stroked like the Play screen's vowel
    const sx = X(3.35) + 26, sy = Y(2.05) + 66, xh = 32, r = xh * 0.48, cy = sy - xh / 2;
    const arc: string[] = [];
    for (let d = 180; d >= -140; d -= 10) arc.push(`L${n1(sx + r * Math.cos((d * Math.PI) / 180))} ${n1(cy + (xh / 2) * Math.sin((d * Math.PI) / 180))}`);
    p.push(`<path d="M${n1(sx + r)} ${n1(cy)}L${n1(sx - r)} ${n1(cy)}${arc.join("")}" stroke="${C.green}" stroke-width="4.8" fill="none" stroke-linecap="round" stroke-linejoin="round"/>`);
    // the piston: the larynx pushed up, compressing the air behind the lips
    const [lx, ly] = [X(F.handles.larynx[0]), Y(F.handles.larynx[1])];
    const ax = lx + 22;
    p.push(line(`M${n1(ax)} ${n1(ly + 26)}L${n1(ax)} ${n1(ly - 22)}M${n1(ax - 5)} ${n1(ly - 15)}L${n1(ax)} ${n1(ly - 22)}L${n1(ax + 5)} ${n1(ly - 15)}`, C.blue, { w: 1.3, hot: true }));
    return p.join("");
  },
}));

// step sequencer: the hero. Glowing grid like the film's tic-tac-toe boards.
{
  const S: Rect = { x: 728, y: 52, w: 696, h: 288 };
  parts.push(screen(S, "STEP SEQUENCER", `${preset.name.toUpperCase()} / ${preset.bpm} BPM`));
  const q: string[] = [];
  const gx0 = S.x + 150, gx1 = S.x + S.w - 16, gy0 = S.y + 44, gy1 = S.y + S.h - 30;
  const cw = (gx1 - gx0) / L, rh = (gy1 - gy0) / BEAT_SOUNDS.length;
  const now = STEP % L;
  // playhead column
  const px = gx0 + now * cw;
  q.push(rect(px + 1, gy0 - 6, cw - 2, gy1 - gy0 + 12, C.white, 1.2, 0.9));
  q.push(line(`M${n1(px + cw / 2 - 5)} ${n1(gy0 - 14)}L${n1(px + cw / 2 + 5)} ${n1(gy0 - 14)}L${n1(px + cw / 2)} ${n1(gy0 - 8)}Z`, C.white, { w: 1.2 }));
  // grid lines: rows dim, steps dim, downbeats brighter
  for (let r = 0; r <= BEAT_SOUNDS.length; r++) q.push(line(`M${n1(gx0)} ${n1(gy0 + r * rh)}L${n1(gx1)} ${n1(gy0 + r * rh)}`, C.frame, { w: 1 }));
  for (let c = 0; c <= L; c++) {
    const down = c % 4 === 0;
    q.push(line(`M${n1(gx0 + c * cw)} ${n1(gy0)}L${n1(gx0 + c * cw)} ${n1(gy1)}`, down ? C.blueMid : C.frame, { w: down ? 1.2 : 1, o: down ? 0.9 : 1 }));
  }
  // row labels
  BEAT_SOUNDS.forEach((s, r) => {
    const y = gy0 + r * rh + rh / 2 + 4;
    const used = seq.steps.some((st) => st?.id === s.id);
    q.push(text(S.x + 16, y, s.id, { size: 12, color: used ? C.blue : C.blueMid, weight: "m", ls: 0.06 }));
    q.push(text(S.x + 56, y - 0.5, s.name.toUpperCase(), { size: 9, color: used ? C.blueMid : C.blueDim, ls: 0.14 }));
  });
  // hits
  seq.steps.forEach((st, c) => {
    if (!st) return;
    const r = BEAT_SOUNDS.findIndex((s) => s.id === st.id);
    const x = gx0 + c * cw, y = gy0 + r * rh;
    const live = c === now;
    const col = live ? C.white : C.amber;
    if (st.id === "m") {
      q.push(rect(x + 4, y + 4, cw - 8, rh - 8, col, live ? 1.6 : 1.3));
      q.push(text(x + cw / 2, y + rh / 2 + 3, noteName(st.note ?? BASS_NOTE), { size: 8, color: col, anchor: "middle", ls: 0.04 }));
    } else {
      const m = Math.min(cw, rh) / 2 - 5;
      const cx = x + cw / 2, cy = y + rh / 2;
      q.push(rect(cx - m, cy - m, 2 * m, 2 * m, col, live ? 1.8 : 1.3));
      q.push(line(`M${n1(cx - m + 3)} ${n1(cy - m + 3)}L${n1(cx + m - 3)} ${n1(cy + m - 3)}M${n1(cx + m - 3)} ${n1(cy - m + 3)}L${n1(cx - m + 3)} ${n1(cy + m - 3)}`, col, { w: live ? 1.4 : 1.1, hot: live }));
    }
  });
  // step numbers
  for (let c = 0; c < L; c++) q.push(text(gx0 + c * cw + cw / 2, gy1 + 16, String(c + 1), { size: 8, color: c === now ? C.white : c % 4 === 0 ? C.blueMid : C.blueDim, anchor: "middle", ls: 0 }));
  parts.push(glow(q.join("")));
}

// spectrogram as a Fairlight-style waterfall: slices back to front, hidden lines removed
{
  const S: Rect = { x: 728, y: 352, w: 696, h: 220 };
  parts.push(screen(S, "SPECTROGRAM", "LAST BAR / NEWEST IN FRONT"));
  const q: string[] = [];
  const labels: string[] = [];
  const xf0 = S.x + 56, xf1 = S.x + S.w - 160, yf = S.y + S.h - 34;
  const DX = 92 / (NS - 1), DY = -110 / (NS - 1), HGT = 50;
  const fx = (i: number) => xf0 + (i / (NF - 1)) * (xf1 - xf0);
  // hit labels along the left edge: newest first, at least 10 px apart
  const keep = new Set<number>();
  let lastY = Infinity;
  for (let k = NS - 1; k >= 0; k--) {
    if (!hitSlices.has(k)) continue;
    const y = yf + (NS - 1 - k) * DY;
    if (lastY - y >= 10) { keep.add(k); lastY = y; }
  }
  for (let k = 0; k < NS; k++) {
    const depth = NS - 1 - k; // 0 = front (newest)
    const ox = depth * DX, oy = depth * DY;
    const sl = slices[k];
    const pts: Pt[] = [];
    let energy = 0;
    for (let b = 0; b < NF; b++) {
      const u = Math.pow((sl[b] + 54) / 54, 1.25);
      pts.push([fx(b) + ox, yf + oy - u * HGT]);
      energy += u;
    }
    const lv = Math.min(1, energy / NF / 0.55);
    const base: Pt[] = [[fx(NF - 1) + ox, yf + oy], [fx(0) + ox, yf + oy]];
    const front = depth === 0;
    const age = 0.38 + 0.62 * Math.pow(1 - depth / (NS - 1), 1.3);
    const col = front ? "#fff3d6" : heat(0.3 + 0.42 * lv);
    q.push(`<path d="${dPath([...pts, ...base], true)}" fill="#000" stroke="${col}" stroke-width="${front ? 1.6 : 1.05}" opacity="${n1(front ? 1 : age)}"/>`);
    if (front) q.push(`<path d="${dPath(pts)}" fill="none" stroke="#ffffff" stroke-width="0.6"/>`);
    const lab = keep.has(k) ? hitSlices.get(k) : undefined;
    if (lab) labels.push(text(fx(0) + ox - 7, yf + oy - 1, lab, { size: 9, color: front ? C.white : C.amber, anchor: "end", ls: 0.04, o: front ? 1 : Math.max(0.55, age) }));
  }
  q.push(...labels);
  for (const f of [100, 1000, 10000]) {
    const i = (Math.log(f / FMIN) / Math.log(FMAX / FMIN)) * (NF - 1);
    q.push(line(`M${n1(fx(i))} ${n1(yf + 3)}L${n1(fx(i))} ${n1(yf + 7)}`, C.blueMid, { w: 1 }));
    q.push(text(fx(i), yf + 19, f >= 1000 ? `${f / 1000}K` : String(f), { size: 9, color: C.blueMid, anchor: "middle", ls: 0.1 }));
  }
  q.push(text(xf1 + 10, yf + 3, "NOW", { size: 9, color: C.white, ls: 0.16 }));
  q.push(text(xf1 + (NS - 1) * DX + 10, yf + (NS - 1) * DY + 3, "1 BAR AGO", { size: 9, color: C.blueMid, ls: 0.12 }));
  parts.push(glow(q.join("")));
}

// pads: the ten sounds, keys 1-0; the one being made lit
{
  const S: Rect = { x: 728, y: 584, w: 284, h: 300 };
  parts.push(screen(S, "PADS", "KEYS 1 TO 0"));
  const q: string[] = [];
  const pw = (S.w - 24 - 8) / 2, ph = (S.h - 40 - 12 - 4 * 6) / 5;
  BEAT_SOUNDS.forEach((s, i) => {
    const x = S.x + 12 + (i % 2) * (pw + 8), y = S.y + 40 + Math.floor(i / 2) * (ph + 6);
    const lit = s.id === cur.id;
    const col = lit ? C.amber : C.blueMid;
    q.push(rect(x, y, pw, ph, lit ? C.amber : C.frame, lit ? 1.5 : 1));
    q.push(text(x + 10, y + ph / 2 + 7, s.id, { size: 19, color: lit ? C.amber : C.blue, weight: "m", ls: 0.02 }));
    q.push(text(x + 54, y + ph / 2 - 2, s.name.toUpperCase(), { size: 8.5, color: col, ls: 0.1 }));
    q.push(text(x + 54, y + ph / 2 + 11, AIR[s.airstream], { size: 7, color: lit ? C.amber : C.blueDim, ls: 0.08, o: lit ? 0.8 : 1 }));
    q.push(text(x + pw - 6, y + 12, String((i + 1) % 10), { size: 8, color: lit ? C.amber : C.blueDim, anchor: "end", ls: 0 }));
  });
  parts.push(glow(q.join("")));
}

// transport: four encoders (OP-1 slots) + presets
{
  const S: Rect = { x: 1024, y: 584, w: 400, h: 300 };
  parts.push(encoderScreen(S, "TRANSPORT", [
    { label: "TEMPO", value: String(preset.bpm), unit: "BPM", frac: (preset.bpm - 60) / 120, color: C.green },
    { label: "SWING", value: String(Math.round(preset.swing * 100)), unit: "%", frac: preset.swing / 0.6, color: C.red },
    { label: "MIC", value: "CLOSE", unit: "PROXIMITY", frac: 0.85, color: C.pink },
    { label: "BASS", value: noteName(BASS_NOTE), unit: "HUM NOTE", frac: (BASS_NOTE - 28) / 24, color: C.white },
  ], 1, 1));
  const q: string[] = [];
  const tx = S.x + 14 + "TRANSPORT".length * 12 * 0.86 + 14;
  q.push(line(`M${n1(tx)} ${S.y + 13}L${n1(tx + 7)} ${S.y + 18}L${n1(tx)} ${S.y + 23}Z`, C.green, { w: 1.2, hot: true }));
  q.push(text(tx + 14, S.y + 22, "PLAYING", { size: 10, color: C.green, ls: 0.16 }));
  const names = BEAT_PRESETS.map((p) => p.name.toUpperCase());
  let x = S.x + 14, y = S.y + 214;
  for (const nm of names) {
    const w = textWidth(nm, 9, 0.16) + 16;
    if (x + w > S.x + S.w - 14) { x = S.x + 14; y += 26; }
    q.push(chip(x, y, nm, { lit: nm === PRESET.toUpperCase(), color: C.green, size: 9, padX: 8, h: 20 }).svg);
    x += w + 8;
  }
  parts.push(glow(q.join("")));
}

const path = await writePage("beatbox", "CHIMERA Beatbox mockup", parts, `Beatbox: the "${PRESET}" preset at ${preset.bpm} bpm; the face on the kick that opens bar 2, lips bursting, larynx up.`);
console.log("wrote", path);
