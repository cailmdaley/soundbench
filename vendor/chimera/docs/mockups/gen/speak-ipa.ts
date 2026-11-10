/**
 * Single-stroke IPA consonants for the SPEAK mockup, on top of lib.ts ipaGlyph().
 * Latin letters come from the Hershey font; the rest are Hershey glyphs transformed
 * (rotated, mirrored, small caps) or short pen strokes, so the chart reads as one vector font.
 */
import { ipaGlyph, n1, esc } from "./lib.ts";

export function ipa(x: number, y: number, sym: string, size: number, color: string): string {
  const s = size;
  const xh = s * 0.5;
  // pen width close to Hershey Light, so drawn glyphs sit with font glyphs
  const sw = Math.max(0.62, s * 0.058);
  const st = (d: string) => `<path d="${d}" stroke="${color}" stroke-width="${n1(sw)}" fill="none"/>`;
  const font = (ch: string, { dx = 0, fs = s, tr = "" }: { dx?: number; fs?: number; tr?: string } = {}) =>
    `<text x="${n1(x + dx)}" y="${n1(y)}" font-size="${n1(fs)}" fill="${color}" text-anchor="middle" class="l"${tr ? ` transform="${tr}"` : ""}>${esc(ch)}</text>`;
  const cy = y - xh / 2;
  const rotC = (cx = x) => `rotate(180 ${n1(cx)} ${n1(cy)})`;
  const mirror = `translate(${n1(2 * x)} 0) scale(-1 1)`;
  const smallCap = (ch: string, tr = "") => font(ch, { fs: s * 0.72, tr });
  // retroflex tail from a stem bottom at xr: down below the baseline, curling right
  const tail = (xr: number) => st(`M${n1(xr)} ${n1(y - xh * 0.1)}L${n1(xr)} ${n1(y + s * 0.1)}Q${n1(xr)} ${n1(y + s * 0.24)} ${n1(xr + s * 0.14)} ${n1(y + s * 0.22)}`);
  // left/right descender hooks (ɲ, ɱ)
  const hookL = (xl: number) => st(`M${n1(xl)} ${n1(y - xh * 0.1)}L${n1(xl)} ${n1(y + s * 0.12)}Q${n1(xl)} ${n1(y + s * 0.26)} ${n1(xl - s * 0.13)} ${n1(y + s * 0.24)}`);
  const qmark = (sg: number) => st(`M${n1(x - sg * s * 0.2)} ${n1(y - s * 0.6)}Q${n1(x - sg * s * 0.2)} ${n1(y - s * 0.8)} ${n1(x)} ${n1(y - s * 0.8)}Q${n1(x + sg * s * 0.22)} ${n1(y - s * 0.8)} ${n1(x + sg * s * 0.22)} ${n1(y - s * 0.6)}Q${n1(x + sg * s * 0.22)} ${n1(y - s * 0.42)} ${n1(x)} ${n1(y - s * 0.36)}L${n1(x)} ${n1(y)}`);
  switch (sym) {
    case "θ": return st(`M${n1(x + s * 0.22)} ${n1(y - xh)}A${n1(s * 0.22)} ${n1(xh)} 0 1 0 ${n1(x - s * 0.22)} ${n1(y - xh)}A${n1(s * 0.22)} ${n1(xh)} 0 1 0 ${n1(x + s * 0.22)} ${n1(y - xh)}M${n1(x - s * 0.22)} ${n1(y - xh)}L${n1(x + s * 0.22)} ${n1(y - xh)}`);
    case "ʃ": return st(`M${n1(x + s * 0.22)} ${n1(y - xh * 1.9)}Q${n1(x)} ${n1(y - xh * 2.2)} ${n1(x)} ${n1(y - xh)}L${n1(x)} ${n1(y + xh * 0.3)}Q${n1(x)} ${n1(y + xh * 0.95)} ${n1(x - s * 0.22)} ${n1(y + xh * 0.7)}`);
    case "ŋ": return font("n") + hookL(x + s * 0.2).replace("Q", "Q").replace(/L([\d.]+) /, "L$1 ");
    case "ɪ": return st(`M${n1(x)} ${n1(y - xh)}L${n1(x)} ${n1(y)}M${n1(x - xh * 0.3)} ${n1(y - xh)}L${n1(x + xh * 0.3)} ${n1(y - xh)}M${n1(x - xh * 0.3)} ${n1(y)}L${n1(x + xh * 0.3)} ${n1(y)}`);
    case "ɫ": return font("l") + st(`M${n1(x - s * 0.24)} ${n1(y - s * 0.36)}C${n1(x - s * 0.16)} ${n1(y - s * 0.5)} ${n1(x - s * 0.06)} ${n1(y - s * 0.46)} ${n1(x)} ${n1(y - s * 0.4)}C${n1(x + s * 0.06)} ${n1(y - s * 0.34)} ${n1(x + s * 0.16)} ${n1(y - s * 0.28)} ${n1(x + s * 0.24)} ${n1(y - s * 0.44)}`);
    case "ɢ": return smallCap("G");
    case "ʙ": return smallCap("B");
    case "ʀ": return smallCap("R");
    case "ɴ": return smallCap("N");
    case "ʁ": return smallCap("R", `translate(0 ${n1(2 * cy)}) scale(1 -1)`);
    case "ʈ": return font("t") + tail(x - s * 0.02);
    case "ɖ": return font("d") + tail(x + s * 0.2);
    case "ɳ": return font("n") + tail(x + s * 0.2);
    case "ʂ": return font("s") + tail(x - s * 0.2);
    case "ʐ": return font("z") + tail(x + s * 0.18);
    case "ɭ": return font("l") + tail(x);
    case "ɻ": return font("r", { tr: rotC() }) + tail(x + s * 0.08);
    case "ɾ": return st(`M${n1(x - s * 0.08)} ${n1(y)}L${n1(x - s * 0.08)} ${n1(y - xh * 0.7)}Q${n1(x - s * 0.08)} ${n1(y - xh * 1.05)} ${n1(x + s * 0.16)} ${n1(y - xh * 0.98)}`);
    case "ɽ": return st(`M${n1(x - s * 0.08)} ${n1(y + s * 0.1)}L${n1(x - s * 0.08)} ${n1(y - xh * 0.7)}Q${n1(x - s * 0.08)} ${n1(y - xh * 1.05)} ${n1(x + s * 0.16)} ${n1(y - xh * 0.98)}M${n1(x - s * 0.08)} ${n1(y + s * 0.1)}Q${n1(x - s * 0.08)} ${n1(y + s * 0.24)} ${n1(x + s * 0.06)} ${n1(y + s * 0.22)}`);
    case "ɟ": return font("f", { tr: rotC() });
    case "ʔ": return qmark(1);
    case "ʕ": return qmark(-1);
    case "ɱ": return font("m") + hookL(x + s * 0.3);
    case "ɲ": return font("n") + hookL(x - s * 0.2);
    case "ɸ": return font("o") + st(`M${n1(x)} ${n1(y - s * 0.72)}L${n1(x)} ${n1(y + s * 0.22)}`);
    case "β": return st(`M${n1(x - s * 0.17)} ${n1(y + s * 0.24)}L${n1(x - s * 0.17)} ${n1(y - s * 0.56)}Q${n1(x - s * 0.17)} ${n1(y - s * 0.78)} ${n1(x + s * 0.02)} ${n1(y - s * 0.78)}Q${n1(x + s * 0.19)} ${n1(y - s * 0.78)} ${n1(x + s * 0.19)} ${n1(y - s * 0.6)}Q${n1(x + s * 0.19)} ${n1(y - s * 0.44)} ${n1(x - s * 0.02)} ${n1(y - s * 0.42)}Q${n1(x + s * 0.24)} ${n1(y - s * 0.4)} ${n1(x + s * 0.24)} ${n1(y - s * 0.19)}Q${n1(x + s * 0.24)} ${n1(y)} ${n1(x + s * 0.02)} ${n1(y)}Q${n1(x - s * 0.1)} ${n1(y)} ${n1(x - s * 0.17)} ${n1(y - s * 0.07)}`);
    case "ð": return st(`M${n1(x + s * 0.2)} ${n1(y - xh * 0.5)}A${n1(s * 0.2)} ${n1(xh * 0.5)} 0 1 0 ${n1(x - s * 0.2)} ${n1(y - xh * 0.5)}A${n1(s * 0.2)} ${n1(xh * 0.5)} 0 1 0 ${n1(x + s * 0.2)} ${n1(y - xh * 0.5)}Q${n1(x + s * 0.18)} ${n1(y - s * 0.62)} ${n1(x - s * 0.06)} ${n1(y - s * 0.8)}M${n1(x - s * 0.14)} ${n1(y - s * 0.6)}L${n1(x + s * 0.14)} ${n1(y - s * 0.72)}`);
    case "ʒ": return st(`M${n1(x - s * 0.18)} ${n1(y - xh)}L${n1(x + s * 0.18)} ${n1(y - xh)}L${n1(x - s * 0.04)} ${n1(y - xh * 0.48)}Q${n1(x + s * 0.24)} ${n1(y - xh * 0.48)} ${n1(x + s * 0.22)} ${n1(y + s * 0.02)}Q${n1(x + s * 0.18)} ${n1(y + s * 0.25)} ${n1(x - s * 0.04)} ${n1(y + s * 0.25)}Q${n1(x - s * 0.19)} ${n1(y + s * 0.24)} ${n1(x - s * 0.22)} ${n1(y + s * 0.13)}`);
    case "ç": return font("c") + st(`M${n1(x)} ${n1(y)}L${n1(x)} ${n1(y + s * 0.09)}Q${n1(x + s * 0.13)} ${n1(y + s * 0.11)} ${n1(x + s * 0.03)} ${n1(y + s * 0.22)}`);
    case "ʝ": return font("j") + st(`M${n1(x - s * 0.14)} ${n1(y + s * 0.08)}L${n1(x + s * 0.1)} ${n1(y + s * 0.08)}`);
    case "ɣ": return st(`M${n1(x - s * 0.2)} ${n1(y - xh)}L${n1(x)} ${n1(y + s * 0.04)}L${n1(x + s * 0.2)} ${n1(y - xh)}M${n1(x)} ${n1(y + s * 0.04)}Q${n1(x - s * 0.12)} ${n1(y + s * 0.24)} ${n1(x)} ${n1(y + s * 0.26)}Q${n1(x + s * 0.12)} ${n1(y + s * 0.24)} ${n1(x)} ${n1(y + s * 0.04)}`);
    case "χ": return st(`M${n1(x - s * 0.22)} ${n1(y - xh)}L${n1(x + s * 0.22)} ${n1(y + s * 0.24)}M${n1(x + s * 0.22)} ${n1(y - xh)}L${n1(x - s * 0.22)} ${n1(y + s * 0.24)}`);
    case "ħ": return font("h") + st(`M${n1(x - s * 0.3)} ${n1(y - s * 0.58)}L${n1(x + s * 0.04)} ${n1(y - s * 0.58)}`);
    case "ɦ": return st(`M${n1(x - s * 0.16)} ${n1(y)}L${n1(x - s * 0.16)} ${n1(y - s * 0.64)}Q${n1(x - s * 0.16)} ${n1(y - s * 0.8)} ${n1(x)} ${n1(y - s * 0.78)}M${n1(x - s * 0.16)} ${n1(y - xh * 0.62)}Q${n1(x - s * 0.04)} ${n1(y - xh * 1.02)} ${n1(x + s * 0.06)} ${n1(y - xh)}Q${n1(x + s * 0.17)} ${n1(y - xh * 0.96)} ${n1(x + s * 0.17)} ${n1(y - xh * 0.62)}L${n1(x + s * 0.17)} ${n1(y)}`);
    case "ɬ": return font("l") + st(`M${n1(x + s * 0.16)} ${n1(y - s * 0.38)}Q${n1(x + s * 0.16)} ${n1(y - s * 0.5)} ${n1(x)} ${n1(y - s * 0.5)}Q${n1(x - s * 0.16)} ${n1(y - s * 0.5)} ${n1(x - s * 0.16)} ${n1(y - s * 0.38)}L${n1(x + s * 0.12)} ${n1(y - s * 0.38)}`);
    case "ɮ": return font("l", { dx: -s * 0.16 }) + ipa(x + s * 0.12, y, "ʒ", s * 0.85, color);
    case "ʋ": return st(`M${n1(x - s * 0.19)} ${n1(y - xh)}Q${n1(x - s * 0.24)} ${n1(y)} ${n1(x)} ${n1(y)}Q${n1(x + s * 0.22)} ${n1(y)} ${n1(x + s * 0.2)} ${n1(y - xh * 0.7)}Q${n1(x + s * 0.2)} ${n1(y - xh)} ${n1(x + s * 0.06)} ${n1(y - xh)}`);
    case "ɰ": return font("m", { tr: rotC() }) + st(`M${n1(x + s * 0.28)} ${n1(y - xh * 0.1)}L${n1(x + s * 0.28)} ${n1(y + s * 0.24)}`);
    default: return ipaGlyph(x, y, sym, size, color);
  }
}

/** advance width of one symbol (fixed-ish, like a typewriter IPA font) */
export function advance(sym: string, size: number): number {
  if (sym === " ") return size * 0.42;
  if (sym === "ˈ" || sym === "ˌ") return size * 0.22;
  if ("ilɫɪjtfrɹɾʃ'".includes(sym)) return size * 0.42;
  if ("mwæœɯɱɰ".includes(sym)) return size * 0.86;
  return size * 0.62;
}

/** lay out an IPA string (stress marks as ticks); returns svg and width */
export function ipaRun(x: number, y: number, str: string, size: number, color: string): { svg: string; w: number } {
  const parts: string[] = [];
  let cx = x;
  for (const ch of Array.from(str)) {
    const a = advance(ch, size);
    if (ch === "ˈ") parts.push(`<path d="M${n1(cx + a / 2)} ${n1(y - size * 0.78)}L${n1(cx + a / 2)} ${n1(y - size * 0.56)}" stroke="${color}" stroke-width="${n1(Math.max(0.9, size * 0.075))}"/>`);
    else if (ch === "ˌ") parts.push(`<path d="M${n1(cx + a / 2)} ${n1(y + size * 0.02)}L${n1(cx + a / 2)} ${n1(y + size * 0.22)}" stroke="${color}" stroke-width="${n1(Math.max(0.9, size * 0.075))}"/>`);
    else if (ch !== " ") parts.push(ipa(cx + a / 2, y, ch, size, color));
    cx += a;
  }
  return { svg: parts.join(""), w: cx - x };
}
