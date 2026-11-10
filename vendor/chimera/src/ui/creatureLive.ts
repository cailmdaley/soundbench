import { useEffect, useState } from "react";
import { engine } from "../audio/AudioEngine.ts";
import { creaturePreview, type CreatureTelemetry } from "../wasm/engine.ts";
import type { CreatureDef } from "../wasm/types.ts";

const previews = new Map<string, CreatureTelemetry>();

/**
 * A still frame of a creature's airway at the given settings, for drawing before audio starts:
 * the engine builds the simulation on the main thread (well under a millisecond) and reads its geometry.
 */
export function previewTelemetry(def: CreatureDef, params?: Record<string, number>): CreatureTelemetry {
  const key = def.id + JSON.stringify(params ?? {});
  let t = previews.get(key);
  if (!t) {
    t = creaturePreview(def.id, params);
    previews.set(key, t);
    if (previews.size > 48) previews.delete(previews.keys().next().value!);
  }
  return t;
}

/** Live telemetry for this creature, or null when it is not the voice that is playing. */
export function liveTelemetry(id: string): CreatureTelemetry | null {
  const m = engine.latestCreature;
  return m && m.t.id === id ? m.t : null;
}

/** Re-render at about 20 Hz with the latest telemetry of this creature. */
export function useLiveCreature(id: string): CreatureTelemetry | null {
  const [t, setT] = useState<CreatureTelemetry | null>(null);
  useEffect(() => {
    let raf = 0, last = 0;
    const loop = (now: number) => {
      if (now - last > 50) {
        last = now;
        const next = liveTelemetry(id);
        setT((prev) => (prev === next ? prev : next));
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [id]);
  return t;
}
