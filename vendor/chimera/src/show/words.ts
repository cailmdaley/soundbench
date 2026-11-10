/**
 * The words at a duet head's mouth: each sung note's syllable over the note's time, from the voice's
 * melody file (the one `chimera render notes` sang), as Sing mode's demo shows them.
 */
import type { Syllable } from "../wasm/types.ts";
import type { MouthWord } from "../ui/mouthWords.ts";

/** a melody file: `{"bpm", "t0"?, "lyrics", "notes": [[midi, beats], ...]}` (midi <= 0 is a rest) */
export interface Melody {
  bpm: number;
  t0?: number;
  lyrics: string;
  notes: [number, number][];
}

export interface SungWords {
  /** sounding notes' start and end (s, on the render's clock) */
  times: [number, number][];
  /** what the mouth shows per note: its syllable, or a melisma's parent's */
  shown: string[];
  /** the note whose syllable each note sings (a melisma's parent) */
  parent: number[];
}

/**
 * Note times and syllables for a melody. `syls` is lyricsToSyllables(melody.lyrics); `spell` writes
 * a syllable another way at the mouth (lyrics spelled for the dictionary: "owr" → "our").
 * Throws when the syllables don't match the sounding notes, as the CLI does.
 */
export function sungWords(m: Melody, syls: Syllable[], spell: Record<string, string> = {}): SungWords {
  const beat = 60 / m.bpm;
  const times: [number, number][] = [];
  let t = m.t0 ?? 0.3; // the CLI's default
  for (const [midi, beats] of m.notes) {
    if (midi > 0) times.push([t, t + beats * beat]);
    t += beats * beat;
  }
  if (syls.length !== times.length) throw new Error(`${syls.length} syllables for ${times.length} notes`);
  const respell = new Map(Object.entries(spell).map(([k, v]) => [k.toLowerCase(), v]));
  const parent: number[] = [];
  syls.forEach((s, i) => parent.push(s.melisma && i > 0 ? parent[i - 1] : i));
  const shown = parent.map((p) => {
    const s = syls[p];
    return (respell.get(s.text.toLowerCase()) ?? s.text) + (s.wordEnd ? "" : "-");
  });
  return { times, shown, parent };
}

/** the word sounding at time t (s), keyed by its note so a repeated word is still a new one */
export function wordAt(w: SungWords, t: number): MouthWord | null {
  // binary search: the last note starting at or before t
  let lo = 0, hi = w.times.length - 1, k = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (w.times[mid][0] <= t) {
      k = mid;
      lo = mid + 1;
    } else hi = mid - 1;
  }
  if (k < 0 || t >= w.times[k][1]) return null;
  return { key: w.parent[k], text: w.shown[k] };
}
