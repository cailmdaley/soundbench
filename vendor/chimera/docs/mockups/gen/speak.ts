/**
 * SPEAK mode mockup: "Shall we play a game?" spoken by the real engine (G2P → planner → voice),
 * frozen at one moment (default: mid /ʃ/ of "shall"). → out/speak.html
 * Usage: bun gen/speak.ts [moment-seconds] [out-name]
 */
import { activation, type Gesture } from "../../../src/engine/control/score.ts";
import { VoiceModel, buildFaceData, tubeData } from "./face.ts";
import { C, topBar, faceScreen, tubeScreen, foldsScreen, screen, glow, text, line, rect, circle, chip, dPath, n1, textWidth, writePage, type Rect, type Pt } from "./lib.ts";
import { runSpeak, PHRASE, FS } from "./speak-data.ts";
import { ipa, ipaRun } from "./speak-ipa.ts";

const NOW = Number(process.argv[2] ?? 0.24);
const NAME = process.argv[3] ?? "speak";
const run = await runSpeak(NOW);
const model = new VoiceModel("male", FS);
const artic = VoiceModel.articFromArray(run.tele.artic);
const F = buildFaceData(model, artic, run.tele, { ghosts: run.earlier.map((t) => VoiceModel.articFromArray(t.artic)) });
const now = run.now;
const segNow = run.plan.segments.find((s) => s.t0 <= now && now < s.t1) ?? run.plan.segments[0];
const t = run.tele;
console.log(`moment ${now.toFixed(3)} s: segment ${segNow.ipa} (${segNow.kind}), voicing ${t.voicing.toFixed(2)}, noise ${t.noiseLevel.toFixed(0)} @${t.noiseNode}, lungs ${t.pLung.toFixed(1)}, velum ${t.velumPort.toFixed(2)}, sparks ${F.sparks.length}, caliper ${F.caliper?.area.toFixed(3)}`);

const parts: string[] = [topBar("SPEAK")];

// ---------------------------------------------------------------- face
const noiseS = t.noiseLevel > 1 && t.noiseNode >= 0 ? model.branchS(t.noiseNode, t.phLen, t.orLen) : -1;
const info = [
  t.voicing > 0.5 ? `VOICED ${Math.round(t.voicing * 100)}%` : "VOICELESS",
  noiseS >= 0 ? `NOISE AT ${noiseS.toFixed(1)} CM` : "NO NOISE",
  `LUNGS ${t.pLung.toFixed(1)} CMH2O`,
];
const lipValue = artic.lipProt > 0.35 ? { value: "ROUND", unit: "" } : { value: artic.lipAp.toFixed(1), unit: "CM" };
parts.push(faceScreen({ x: 16, y: 52, w: 700, h: 832 }, F, {
  sub: "ADULT MALE / SPEAKING",
  formants: null,
  info,
  active: null,
  slots: [
    { label: "TONGUE", value: "", color: C.green },
    { label: "JAW", value: artic.jaw.toFixed(1), unit: "CM", color: C.red },
    { label: "LIPS", value: lipValue.value, unit: lipValue.unit, color: C.pink },
    { label: "VELUM", value: t.velumPort > 0.02 ? "OPEN" : "SHUT", color: C.white },
  ],
  extra: (X, Y) => {
    // the TONGUE slot's IPA, sized so tall glyphs (ʃ) clear the label
    const slotGlyph = ipa(X(3.35) + 26, Y(2.05) + 78, segNow.ipa, 52, C.green);
    if (!F.caliper) return slotGlyph;
    // the narrowest gap, measured like the app's caliper
    const [ax, ay] = [X(F.caliper.inner[0]), Y(F.caliper.inner[1])];
    const [bx, by] = [X(F.caliper.outer[0]), Y(F.caliper.outer[1])];
    const dx = bx - ax, dy = by - ay, L = Math.hypot(dx, dy) || 1, nx = -dy / L * 4, ny = dx / L * 4;
    const tick = (x: number, y: number) => `M${n1(x - nx)} ${n1(y - ny)}L${n1(x + nx)} ${n1(y + ny)}`;
    const lx = X(8.05), ly = Y(9.75);
    return slotGlyph + line(`M${n1(ax)} ${n1(ay)}L${n1(bx)} ${n1(by)}${tick(ax, ay)}${tick(bx, by)}`, C.white, { w: 1.2, hot: true }) +
      line(`M${n1((ax + bx) / 2 + 3)} ${n1((ay + by) / 2 - 2)}L${n1(lx - 6)} ${n1(ly - 4)}`, C.white, { w: 0.9, o: 0.55 }) +
      text(lx, ly, F.caliper.area < 0.0005 ? "CLOSED" : `${F.caliper.area.toFixed(2)} CM2`, { size: 11, color: C.white, ls: 0.12 });
  },
}));

// ---------------------------------------------------------------- SAY
{
  const S: Rect = { x: 728, y: 52, w: 696, h: 132 };
  parts.push(screen(S, "SAY", "TEXT TO SPEECH"));
  const q: string[] = [];
  const fx = S.x + 14, fy = S.y + 36, fw = S.w - 28 - 176, fh = 32;
  q.push(rect(fx, fy, fw, fh, C.blueMid, 1));
  q.push(text(fx + 12, fy + 22, PHRASE, { size: 16, color: C.white, ls: 0.06 }));
  const mixedWidth = (str: string, size: number, ls: number) => Array.from(str).reduce((w, ch) => w + size * (ch === " " ? 0.4 : "iljtfr'.,;:!".includes(ch) ? 0.26 : /[A-Z]/.test(ch) ? 0.64 : 0.5) + size * ls, 0);
  const cx = fx + 12 + mixedWidth(PHRASE, 16, 0.06) + 3;
  q.push(line(`M${n1(cx)} ${fy + 7}L${n1(cx)} ${fy + fh - 7}`, C.blue, { w: 1.2 }));
  let x = fx + fw + 12;
  const sp = chip(x, fy, "SPEAK", { lit: true, color: C.blue, size: 12, padX: 14, h: fh });
  q.push(sp.svg);
  x += sp.w + 8;
  q.push(chip(x, fy, "STOP", { size: 12, padX: 12, h: fh }).svg);
  // transcription, current segment lit
  const ty = S.y + 106;
  q.push(text(fx, ty, "/", { size: 16, color: C.amber, o: 0.7 }));
  const tr = run.transcription;
  // position of the current segment in the transcription: count segments up to now
  const segIdx = run.plan.segments.indexOf(segNow);
  let shown = 0, curChar = -1;
  const chars = Array.from(tr);
  for (let i = 0; i < chars.length; i++) {
    if (" ˈˌ?".includes(chars[i])) continue;
    // diphthongs are one character pair in the transcription and two planned segments: count chars
    if (shown === segIdx) { curChar = i; break; }
    shown++;
  }
  let tx = fx + 12;
  chars.forEach((ch, i) => {
    const r = ipaRun(tx, ty, ch, 17, i === curChar ? C.white : C.amber);
    q.push(r.svg);
    if (i === curChar) q.push(line(`M${n1(tx + 1)} ${ty + 6}L${n1(tx + r.w - 1)} ${ty + 6}`, C.white, { w: 1.3, hot: true }));
    tx += r.w;
  });
  q.push(text(tx + 2, ty, "/", { size: 16, color: C.amber, o: 0.7 }));
  // options
  x = fx + fw + 12;
  q.push(chip(x, ty - 16, "IPA", { size: 10, padX: 10, h: 22 }).svg);
  q.push(text(S.x + S.w - 14, ty - 1, "RATE 1.0X", { size: 11, color: C.green, anchor: "end", ls: 0.14 }));
  parts.push(glow(q.join("")));
}

// ---------------------------------------------------------------- gestural score
{
  const S: Rect = { x: 728, y: 196, w: 696, h: 284 };
  parts.push(screen(S, "GESTURAL SCORE", `${now.toFixed(2)} S`));
  const ROWS = [
    { name: "LIPS", keys: ["lipAp", "lipProt", "lipWidth", "labiodental"], color: C.pink },
    { name: "TONGUE TIP", keys: ["ttX", "ttY", "ttCurl", "lateral"], color: C.green },
    { name: "TONGUE BODY", keys: ["tbX", "tbY", "root"], color: C.green },
    { name: "SOFT PALATE", keys: ["velum"], color: C.white },
    { name: "GLOTTIS", keys: ["abduction", "closure"], color: C.amber },
  ];
  const rowOf = (g: Gesture) => {
    const keys = Object.keys(g.targets);
    if (keys.includes("tbX") && keys.includes("lipAp") && keys.length > 4) return 2;
    if (keys.includes("ttX") || keys.includes("ttCurl")) return 1;
    for (let r = 0; r < ROWS.length; r++) if (keys.some((k) => ROWS[r].keys.includes(k))) return r;
    return -1;
  };
  const left = S.x + 118, right = S.x + S.w - 16, top = S.y + 38, bottom = S.y + S.h - 34;
  const nRows = ROWS.length + 1;
  const rowH = (bottom - top) / nRows;
  const T0 = 0.06, T1 = run.plan.end + 0.2;
  const X = (tt: number) => left + ((tt - T0) / (T1 - T0)) * (right - left);
  const q: string[] = [];
  parts.push(`<clipPath id="scoreclip"><rect x="${left - 2}" y="${top - 14}" width="${right - left + 4}" height="${S.y + S.h - 2 - (top - 14)}"/></clipPath>`);
  for (let i = 0; i <= nRows; i++) q.push(line(`M${left} ${n1(top + rowH * i)}L${right} ${n1(top + rowH * i)}`, C.frame, { w: 1 }));
  [...ROWS.map((r) => ({ name: r.name, color: r.color })), { name: "PITCH", color: C.amber }].forEach((r, i) => q.push(text(S.x + 14, top + rowH * (i + 0.5) + 3.5, r.name, { size: 9, color: r.color, ls: 0.16, o: 0.85 })));
  // time ticks
  for (let tt = 0.2; tt < T1; tt += 0.2) q.push(line(`M${n1(X(tt))} ${bottom}L${n1(X(tt))} ${bottom + 4}`, C.blueDim, { w: 1 }));
  const f0s: { t: number; f: number }[] = [];
  const word = (l: string) => (l === "aspiration" || l === "voiceless" || l === "h" ? "OPEN" : l === "velum" ? "LOWERED" : l === "exhale" ? "EXHALE" : l === "voiced frication" ? "BUZZ" : null);
  const seen = new Set<string>();
  const labelled: { r: number; label: string; a: number; b: number }[] = [];
  for (const g of run.plan.gestures) {
    if (g.targets.f0 !== undefined) { f0s.push({ t: (g.t0 + g.t1) / 2, f: g.targets.f0 }); continue; }
    const r = rowOf(g);
    if (r < 0) continue;
    const key = `${r}|${g.label}|${g.t0.toFixed(4)}|${g.t1.toFixed(4)}`;
    if (seen.has(key)) continue;
    seen.add(key);
    const isVowel = r === 2 && Object.keys(g.targets).length > 4;
    const x0 = X(g.t0 - g.rampIn), xa = X(g.t0), xb = X(g.t1), x1 = X(g.t1 + g.rampOut);
    const inset = isVowel ? 0.26 : 0.16;
    const yT = top + rowH * (r + inset), yB = top + rowH * (r + 1 - 0.14);
    const act = activation(g, now);
    const col = ROWS[r].color;
    const d = `M${n1(x0)} ${n1(yB)}L${n1(xa)} ${n1(yT)}L${n1(xb)} ${n1(yT)}L${n1(x1)} ${n1(yB)}`;
    if (act > 0.05) q.push(line(d, col, { w: 1.7, hot: true }));
    else q.push(line(d, col, { w: isVowel ? 1 : 1.2, o: isVowel ? 0.45 : 0.8 }));
    let lab = g.label ?? "";
    // one label per sound per row (a /g/ moves the tongue body twice, with overlapping timing)
    if (labelled.some((l) => l.r === r && l.label === lab && l.a < g.t1 && g.t0 < l.b)) lab = "";
    else if (lab) labelled.push({ r, label: lab, a: g.t0, b: g.t1 });
    const wd = word(lab);
    const lx = (xa + xb) / 2, ly = (yT + yB) / 2 + 4;
    if (wd) { if (xb - xa > textWidth(wd, 8, 0.12) * 0.8) q.push(text(lx, ly - 0.5, wd, { size: 8, color: col, anchor: "middle", ls: 0.12, o: act > 0.05 ? 1 : 0.8 })); }
    else if (lab && xb - xa > 8) q.push(ipa(lx, ly + 1, lab, 11, act > 0.05 ? col : col));
  }
  // pitch contour (a question: it rises at the end)
  if (f0s.length) {
    f0s.sort((a, b) => a.t - b.t);
    const fmin = Math.min(...f0s.map((p) => p.f)) * 0.95, fmax = Math.max(...f0s.map((p) => p.f)) * 1.03;
    const yb = top + rowH * ROWS.length + rowH * 0.86, ya = top + rowH * ROWS.length + rowH * 0.14;
    const pts = f0s.map((p) => [X(p.t), yb - ((p.f - fmin) / Math.max(1, fmax - fmin)) * (yb - ya)] as Pt);
    q.push(line(dPath(pts), C.amber, { w: 1.4, hot: true }));
    for (const [px, py] of pts) q.push(circle(px, py, 1.8, C.amber, 1, 1, C.amber));
    const last = pts[pts.length - 1];
    q.push(text(last[0] + 8, last[1] + 4, `${Math.round(f0s[f0s.length - 1].f)} HZ`, { size: 9, color: C.amber, ls: 0.1 }));
  }
  // segments along the bottom
  for (const s of run.plan.segments) {
    const x = X((s.t0 + s.t1) / 2);
    const cur = s === segNow;
    q.push(ipa(x, S.y + S.h - 12, s.ipa, 13, cur ? C.white : C.blueMid));
    q.push(line(`M${n1(X(s.t0))} ${bottom + 2}L${n1(X(s.t0))} ${bottom + 8}`, C.frame, { w: 1 }));
  }
  // playhead
  const px = X(now);
  q.push(line(`M${n1(px)} ${top - 6}L${n1(px)} ${bottom + 8}`, C.white, { w: 1.2, hot: true }));
  q.push(line(`M${n1(px - 5)} ${top - 12}L${n1(px + 5)} ${top - 12}L${n1(px)} ${top - 5}Z`, C.white, { w: 1.1 }));
  const labelsAt = q.findIndex((x) => x.includes("PITCH"));
  parts.push(glow(q.slice(0, labelsAt + 1).join("")), `<g clip-path="url(#scoreclip)">${glow(q.slice(labelsAt + 1).join(""))}</g>`);
}

// ---------------------------------------------------------------- consonant chart
{
  const S: Rect = { x: 728, y: 492, w: 696, h: 180 };
  parts.push(screen(S, "CONSONANTS", "BETWEEN TWO VOWELS"));
  const PLACES = ["BILABIAL", "LABDENT", "DENTAL", "ALVEOLAR", "POSTALV", "RETROFL", "PALATAL", "VELAR", "UVULAR", "PHARYNG", "GLOTTAL"];
  const CHART: [string, (string | null)[][]][] = [
    ["PLOSIVE", [["p", "b"], [null, null], [null, null], ["t", "d"], [null, null], ["ʈ", "ɖ"], ["c", "ɟ"], ["k", "g"], ["q", "ɢ"], [null, null], ["ʔ", null]]],
    ["NASAL", [[null, "m"], [null, "ɱ"], [null, null], [null, "n"], [null, null], [null, "ɳ"], [null, "ɲ"], [null, "ŋ"], [null, "ɴ"], [null, null], [null, null]]],
    ["TRILL", [[null, "ʙ"], [null, null], [null, null], [null, "r"], [null, null], [null, null], [null, null], [null, null], [null, "ʀ"], [null, null], [null, null]]],
    ["TAP", [[null, null], [null, null], [null, null], [null, "ɾ"], [null, null], [null, "ɽ"], [null, null], [null, null], [null, null], [null, null], [null, null]]],
    ["FRICATIVE", [["ɸ", "β"], ["f", "v"], ["θ", "ð"], ["s", "z"], ["ʃ", "ʒ"], ["ʂ", "ʐ"], ["ç", "ʝ"], ["x", "ɣ"], ["χ", "ʁ"], ["ħ", "ʕ"], ["h", "ɦ"]]],
    ["LAT FRIC", [[null, null], [null, null], [null, null], ["ɬ", "ɮ"], [null, null], [null, null], [null, null], [null, null], [null, null], [null, null], [null, null]]],
    ["APPROX", [[null, null], [null, "ʋ"], [null, null], [null, "ɹ"], [null, null], [null, "ɻ"], [null, "j"], [null, "ɰ"], [null, null], [null, null], [null, null]]],
    ["LATERAL", [[null, null], [null, null], [null, null], [null, "l"], [null, null], [null, "ɭ"], [null, "ʎ"], [null, null], [null, null], [null, null], [null, null]]],
  ];
  const inPhrase = new Set(run.plan.segments.map((s) => (s.ipa === "ɫ" ? "l" : s.ipa)));
  const sounding = segNow.ipa === "ɫ" ? "l" : segNow.ipa;
  const gx0 = S.x + 96, gx1 = S.x + S.w - 12, gy0 = S.y + 50, gy1 = S.y + S.h - 10;
  const cw = (gx1 - gx0) / PLACES.length, rh = (gy1 - gy0) / CHART.length;
  const q: string[] = [];
  PLACES.forEach((p, i) => q.push(text(gx0 + cw * (i + 0.5), gy0 - 8, p, { size: 7.5, color: C.blueMid, anchor: "middle", ls: 0.1 })));
  // the grid, like the film's game boards: bright frame, dim cell lines
  for (let i = 1; i < PLACES.length; i++) q.push(line(`M${n1(gx0 + cw * i)} ${gy0}L${n1(gx0 + cw * i)} ${gy1}`, C.frame, { w: 1 }));
  for (let r = 1; r < CHART.length; r++) q.push(line(`M${gx0} ${n1(gy0 + rh * r)}L${gx1} ${n1(gy0 + rh * r)}`, C.frame, { w: 1 }));
  q.push(rect(gx0, gy0, gx1 - gx0, gy1 - gy0, C.blueDim, 1.1));
  CHART.forEach(([manner, cells], r) => {
    const yb = gy0 + rh * (r + 0.5) + 4;
    q.push(text(S.x + 14, yb - 0.5, manner, { size: 8, color: C.blueMid, ls: 0.12 }));
    cells.forEach((pair, c) => {
      pair.forEach((sym, j) => {
        if (!sym) return;
        const x = gx0 + cw * (c + 0.5) + (j ? 11 : -11);
        const isNow = sym === sounding;
        const col = isNow ? C.green : inPhrase.has(sym) ? C.blue : C.blueDim;
        q.push(ipa(x, yb, sym, 11.5, col));
        if (isNow) q.push(circle(x, yb - 4, 8.5, C.green, 1.3), circle(x, yb - 4, 13, C.green, 0.8, 0.5));
      });
    });
  });
  parts.push(glow(q.join("")));
}

// ---------------------------------------------------------------- instruments
parts.push(tubeScreen({ x: 728, y: 684, w: 342, h: 200 }, tubeData(model, run.tele)));
parts.push(foldsScreen({ x: 1082, y: 684, w: 342, h: 200 }, Array.from(run.tele.glottalArea), Array.from(run.tele.glottalFlow), run.tele.voicing > 0.3 ? "SLOWED 100X" : "OPEN, NOT VOICING"));

const path = await writePage(NAME, "CHIMERA Speak mockup", parts, `Speak: "${PHRASE}" through the real G2P, planner and voice, at ${now.toFixed(2)} s (/${segNow.ipa}/).`);
console.log("wrote", path);
