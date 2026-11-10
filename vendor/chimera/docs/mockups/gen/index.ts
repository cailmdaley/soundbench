/** The mockups' index page: one framed screen per mode (linked) and a colour key. → docs/mockups/index.html */
import { C, SLOTS, glow, line, n1, screen, text, writePage, type Rect } from "./lib.ts";

const modes: { mode: string; lines: string[] }[] = [
  { mode: "PLAY", lines: ["HOLD A VOWEL AND DRAG THE TONGUE,", "JAW AND LIPS.", "SHOWN: AH AT 110 HZ, THE TONGUE", "ARRIVING FROM EE."] },
  { mode: "SPEAK", lines: ["TEXT TO SPEECH, WITH THE GESTURAL", "SCORE OF EVERY ARTICULATOR.", "SHOWN: SHALL WE PLAY A GAME?,", "IN THE MIDDLE OF THE SH."] },
  { mode: "SING", lines: ["LYRICS ON A PIANO ROLL, WITH THE", "PITCH THE FOLDS ACTUALLY SANG.", "SHOWN: MADE OF AIR, HOLDING", "AIR ON C3 WITH VIBRATO."] },
  { mode: "BEATBOX", lines: ["A STEP SEQUENCER OF MOUTH SOUNDS,", "EACH ONE A REAL GESTURE.", "SHOWN: HUM BASS, THE KICK THAT", "OPENS BAR 2."] },
  { mode: "BESTIARY", lines: ["ANIMAL VOICES FROM THEIR OWN", "SOUND SOURCES AND AIRWAYS.", "SHOWN: THE SONGBIRD'S TWO VOICES", "IN ITS FLUTE PHRASE."] },
];

const parts: string[] = [];
parts.push(glow(text(18, 33, "CHIMERA", { size: 22, color: C.blue, weight: "m", ls: 0.34 }) + text(196, 32, "UI MOCKUPS / VECTOR DISPLAY", { size: 10, color: C.blueMid, ls: 0.18 })));
const cols = [16, 496, 976], rows = [64, 476], w = 448, h = 400;
modes.forEach((m, i) => {
  const S: Rect = { x: cols[i % 3], y: rows[Math.floor(i / 3)], w, h };
  const q: string[] = [];
  q.push(text(S.x + 28, S.y + 110, m.mode, { size: 48, color: C.blue, weight: "m", ls: 0.12 }));
  m.lines.forEach((l, j) => q.push(text(S.x + 30, S.y + 170 + j * 24 + (j >= 2 ? 14 : 0), l, { size: 12, color: j >= 2 ? C.amber : C.blueMid, ls: 0.14 })));
  q.push(line(`M${S.x + 30} ${S.y + h - 46}L${S.x + w - 30} ${S.y + h - 46}`, C.frame, { w: 1 }));
  q.push(text(S.x + w - 30, S.y + h - 22, "OPEN  >", { size: 12, color: C.blue, anchor: "end", ls: 0.2 }));
  parts.push(`<a href="${m.mode.toLowerCase()}.html">${screen(S, `0${i + 1}`, m.mode === "PLAY" ? "THE INSTRUMENT" : "MODE")}${glow(q.join(""))}<rect x="${S.x}" y="${S.y}" width="${w}" height="${h}" fill="transparent" stroke="none"/></a>`);
});
// colour key
{
  const S: Rect = { x: cols[2], y: rows[1], w, h };
  const q: string[] = [];
  const row = (y: number, colors: string[], a: string, b: string) => {
    colors.forEach((c, k) => q.push(line(`M${S.x + 30 + k * 22} ${y - 4}L${S.x + 44 + k * 22} ${y - 4}`, c, { w: 2.2, hot: true })));
    const x = S.x + 30 + colors.length * 22 + 10;
    q.push(text(x, y, a, { size: 12, color: C.blue, ls: 0.16 }), text(x, y + 20, b, { size: 11, color: C.blueMid, ls: 0.14 }));
  };
  row(S.y + 90, [C.blue], "THE BODY", "OUTLINES, GRIDS, TITLES");
  row(S.y + 160, [C.amber], "SOUND", "PRESSURE, FLOW, SPECTRA");
  row(S.y + 230, [...SLOTS], "THE FOUR CONTROLS", "EACH SCREEN MAPS ITS OWN FOUR");
  q.push(text(S.x + 30, S.y + 300, "ALL DATA COMES FROM THE ENGINE:", { size: 11, color: C.blueMid, ls: 0.14 }));
  q.push(text(S.x + 30, S.y + 320, "GEOMETRY, PRESSURE, PITCH, SPECTRA.", { size: 11, color: C.blueMid, ls: 0.14 }));
  q.push(text(S.x + 30, S.y + 352, "TYPE: HERSHEY SIMPLEX, 1967.", { size: 11, color: C.blueDim, ls: 0.14 }));
  parts.push(screen(S, "KEY", "COLOURS") + glow(q.join("")));
}
void n1;
const path = await writePage("index", "CHIMERA UI mockups", parts, "Index of the five mode mockups.");
console.log("wrote", path);
