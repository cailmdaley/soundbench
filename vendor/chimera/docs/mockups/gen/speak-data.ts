/**
 * Engine run for the SPEAK mockup: "Shall we play a game?" through the real G2P, planner and voice.
 * runSpeak(moment) renders offline and snapshots telemetry at the chosen moment and 30/60/90 ms before.
 */
import { HumanVoice, type VoiceTelemetry } from "../../../src/engine/voice/humanVoice.ts";
import { BODY_MALE } from "../../../src/engine/anatomy/human.ts";
import { EnglishG2P } from "../../../src/engine/phonetics/english.ts";
import { planPhrases, type Plan } from "../../../src/engine/phonetics/planner.ts";
import { REPO } from "./face.ts";

export const PHRASE = "Shall we play a game?";
export const FS = 48000;

/** same as src/ui/speech.ts transcribe() */
export function transcribe(phrases: { segments: { ipa: string; stress?: number; wordEnd?: boolean }[]; intonation?: string }[]): string {
  return phrases.map((p) => p.segments.map((s) => (s.stress === 1 ? "ˈ" : s.stress === 2 ? "ˌ" : "") + s.ipa + (s.wordEnd ? " " : "")).join("").trim() + (p.intonation === "question" ? "?" : "")).join(" | ");
}

export interface SpeakRun {
  plan: Plan;
  transcription: string;
  /** telemetry at the moment, and 90/60/30 ms before (oldest first) */
  tele: VoiceTelemetry;
  earlier: VoiceTelemetry[];
  /** voice-clock time of the moment (s) */
  now: number;
  out: Float32Array;
  /** sample index of the moment */
  at: number;
}

export async function runSpeak(now: number): Promise<SpeakRun> {
  const g2p = new EnglishG2P();
  const gz = new Uint8Array(await Bun.file(REPO + "/public/data/cmudict.txt.gz").arrayBuffer());
  g2p.setDictionaryText(new TextDecoder().decode(Bun.gunzipSync(gz)));
  const phrases = g2p.toPhrases(PHRASE);
  const plan = planPhrases(phrases, { body: "male", t0: 0.1, rate: 1, f0: 110 });
  const voice = new HumanVoice({ fsOut: FS, body: BODY_MALE });
  voice.setSource({ f0: 110 });
  voice.schedule(plan.gestures);
  const total = Math.round(FS * (plan.end + 0.4));
  const out = new Float32Array(total);
  const B = 128;
  const block = (t: number) => Math.round((t * FS) / B) * B;
  const snaps = [now - 0.09, now - 0.06, now - 0.03, now].map(block);
  const resets = new Set(snaps.map((s) => s - 4 * B)); // ~10 ms of accumulation per snapshot
  const got = new Map<number, VoiceTelemetry>();
  for (let i = 0; i < total; i += B) {
    voice.render(out, i, Math.min(B, total - i));
    const end = i + B;
    if (resets.has(end)) voice.telemetry();
    if (snaps.includes(end)) got.set(end, voice.telemetry());
  }
  const tele = got.get(snaps[3])!;
  return { plan, transcription: transcribe(phrases), tele, earlier: snaps.slice(0, 3).map((s) => got.get(s)!), now: snaps[3] / FS, out, at: snaps[3] };
}
