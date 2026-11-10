import { useEffect, useRef } from "react";
import { VoiceModel } from "./analysis.ts";
import { useStore } from "./store.ts";
import { engine } from "../audio/AudioEngine.ts";
import { bodyKey, type BodyMix } from "./body.ts";

/**
 * Shared main-thread acoustic models, most recent last (rebuilt on body / sample-rate change).
 * A body is continuous, so only the last few are kept; an evicted one is freed in the engine too
 * (components read the current body's model on every render, so none holds an evicted one).
 */
const cache = new Map<string, VoiceModel>();
const KEEP = 8;

export function getModel(body: BodyMix, sampleRate: number): VoiceModel {
  const key = `${bodyKey(body)}@${sampleRate}`;
  let m = cache.get(key);
  if (m) cache.delete(key);
  else m = new VoiceModel(body, sampleRate);
  cache.set(key, m);
  if (cache.size > KEEP) {
    const [oldest, old] = cache.entries().next().value!;
    cache.delete(oldest);
    old.dispose();
  }
  return m;
}

export function useModel(): React.MutableRefObject<VoiceModel | null> {
  const body = useStore((s) => s.body);
  const started = useStore((s) => s.started);
  const ref = useRef<VoiceModel | null>(null);
  ref.current = getModel(body, started ? engine.sampleRate : 48000);
  useEffect(() => {
    ref.current = getModel(body, started ? engine.sampleRate : 48000);
  }, [body, started]);
  return ref;
}
