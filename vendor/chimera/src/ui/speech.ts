/** Main-thread speech planning: text/IPA → phrases → gestural plan → worklet. */
import type { BodyMix } from "../wasm/types.ts";
import { EnglishG2P, type Phrase } from "../wasm/engine.ts";
import { tokenizeIpa } from "../wasm/engine.ts";
import { planPhrases, type Plan, type Segment } from "../wasm/engine.ts";
import { engine } from "../audio/AudioEngine.ts";
import type { Gesture } from "../wasm/types.ts";
import { wordSpans, type WordSpan } from "./words.ts";

export const g2p = new EnglishG2P();

export interface SpokenPlan {
  plan: Plan;
  /** worklet-clock offset of the plan's time 0 (set when the worklet confirms) */
  offset: number | null;
  id: number;
  transcription: string;
  /** each word's span in the plan */
  words: WordSpan[];
}

let current: SpokenPlan | null = null;
/** the plan a stop cut short: its words are no longer being said */
let cutId = -1;
const listeners = new Set<(p: SpokenPlan | null) => void>();
engine.onPlanned((m) => {
  if (current && m.id === current.id) {
    current = { ...current, offset: m.t0 };
    for (const l of listeners) l(current);
  }
});

export function onPlan(l: (p: SpokenPlan | null) => void): () => void {
  listeners.add(l);
  return () => listeners.delete(l);
}

export function currentPlan(): SpokenPlan | null {
  return current;
}

export function ipaToPhrases(text: string): Phrase[] {
  return text
    .split(/[|.?!]+/)
    .map((p) => p.trim())
    .filter(Boolean)
    .map((p) => ({ segments: tokenizeIpa(p), intonation: "statement" as const, pause: 0.3, words: [] }));
}

export function transcribe(phrases: { segments: Segment[]; intonation?: string }[]): string {
  return phrases.map((p) => p.segments.map((s) => (s.stress === 1 ? "ˈ" : s.stress === 2 ? "ˌ" : "") + s.ipa + (s.wordEnd ? " " : "")).join("").trim() + (p.intonation === "question" ? "?" : "")).join(" | ");
}

export function speakPhrases(phrases: { segments: Segment[]; intonation?: "statement" | "question" | "flat"; pause?: number; words?: string[] }[], o: { body: BodyMix; rate: number; f0: number }): SpokenPlan {
  const plan = planPhrases(phrases, { body: o.body, t0: 0, rate: o.rate, f0: o.f0 });
  const id = engine.schedule({ gestures: plan.gestures.map((g: Gesture) => ({ ...g, targets: { ...g.targets } })), segments: plan.segments, end: plan.end });
  current = { plan, offset: null, id, transcription: transcribe(phrases), words: wordSpans(phrases, plan) };
  for (const l of listeners) l(current);
  return current;
}

export async function speakText(text: string, o: { ipa: boolean; body: BodyMix; rate: number; f0: number }): Promise<SpokenPlan | null> {
  if (!text.trim()) return null;
  let phrases: Phrase[];
  if (o.ipa) phrases = ipaToPhrases(text);
  else {
    try {
      await g2p.load();
    } catch {
      // dictionary unavailable: the letter-to-sound rules still work
    }
    phrases = g2p.toPhrases(text);
  }
  if (!phrases.length) return null;
  return speakPhrases(phrases, o);
}

export function stopSpeaking(): void {
  engine.stopSpeech();
  if (current) cutId = current.id;
}

/** The word being spoken now (null between words and once the speech is over or stopped). */
export function spokenWordNow(): { key: string; text: string } | null {
  const p = current;
  const t = engine.voiceClock();
  if (!p || p.offset === null || t === null || p.id === cutId) return null;
  const pt = t - p.offset;
  const i = p.words.findIndex((w) => pt >= w.t0 && pt < w.t1 + 0.05);
  return i < 0 ? null : { key: `${p.id}:${i}`, text: p.words[i].text };
}
