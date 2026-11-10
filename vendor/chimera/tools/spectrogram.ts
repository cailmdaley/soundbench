/**
 * Spectrogram PNG of a WAV (the renders of `chimera render …`; the first channel).
 * Usage: bun tools/spectrogram.ts in.wav out.png [title] [maxFreq Hz, default 8000] [windowMs, default 5]
 */
import { readWav } from "./lib/wav.ts";
import { renderSpectrogram } from "./lib/spectrogram.ts";

const [input, output, title, maxFreq, windowMs] = process.argv.slice(2);
if (!input || !output) {
  console.error("usage: bun tools/spectrogram.ts in.wav out.png [title] [maxFreq] [windowMs]");
  process.exit(1);
}
const { sampleRate, channels } = readWav(input);
const name = input.split("/").pop() ?? input;
renderSpectrogram(output, channels[0], sampleRate, {
  title: (title ?? name).replace(/[^\x20-\x7e]/g, "?"),
  maxFreq: maxFreq ? Number(maxFreq) : 8000,
  windowMs: windowMs ? Number(windowMs) : 5,
});
console.log(`wrote ${output}  ${(channels[0].length / sampleRate).toFixed(2)} s`);
