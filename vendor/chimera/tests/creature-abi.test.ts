/**
 * Creatures through the worklet: after a switch, the telemetry the processor decodes has the
 * creature's real layout (every tube's nodes, every source, every param), matching what the
 * engine reports for the same creature on the main thread.
 */
import { describe, expect, test } from "bun:test";
import { initEngine, wasmModule } from "./harness.ts";

await initEngine();
const { CREATURES, creaturePreview } = await import("../src/wasm/engine.ts");

describe("creature telemetry through the worklet", () => {
  test("every creature's tubes, sources and params arrive with their real sizes", async () => {
    type Proc = { port: { onmessage: ((e: { data: unknown }) => void) | null }; process: (i: unknown, o: Float32Array[][]) => boolean };
    let Ctor: (new (o: unknown) => Proc) | null = null;
    const posted: { type: string; t?: unknown; message?: string }[] = [];
    const g = globalThis as Record<string, unknown>;
    g.sampleRate = 48000;
    g.AudioWorkletProcessor = class {
      port = { onmessage: null as unknown, postMessage: (m: { type: string }) => posted.push(m) };
    };
    g.registerProcessor = (_name: string, ctor: typeof Ctor) => (Ctor = ctor);
    // @ts-expect-error the query makes bun evaluate the module again, so this file registers its own processor
    await import("../src/worklet/processor.ts?creature-abi");
    const p = new Ctor!({ processorOptions: { module: wasmModule } });
    const send = (data: unknown) => p.port.onmessage!({ data });
    expect(CREATURES.length).toBeGreaterThan(0);
    for (const def of CREATURES) {
      send({ type: "creature", id: def.id });
      send({ type: "gate", level: 1 });
      send({ type: "call", id: def.calls[0].id });
      posted.length = 0;
      for (let i = 0; i < 24; i++) p.process([], [[new Float32Array(128)]]);
      const tele = posted.filter((m) => m.type === "ctelemetry").map((m) => m.t as import("../src/wasm/types.ts").CreatureTelemetry);
      expect(tele.length).toBeGreaterThan(0);
      const t = tele.at(-1)!;
      const ref = creaturePreview(def.id);
      expect(t.id).toBe(def.id);
      expect(Object.keys(t.params)).toEqual(def.params.map((q) => q.key));
      expect(Object.keys(t.source)).toEqual(Object.keys(ref.source));
      expect(Object.keys(t.tubes)).toEqual(Object.keys(ref.tubes));
      for (const [name, tube] of Object.entries(ref.tubes)) {
        expect(t.tubes[name].areas.length).toBe(tube.areas.length);
        expect(t.tubes[name].rms.length).toBe(tube.areas.length);
        expect(tube.areas.length).toBeGreaterThan(0);
        for (const a of t.tubes[name].areas) expect(Number.isFinite(a)).toBe(true);
      }
      expect(t.call).toBe(def.calls[0].id);
      expect(t.outRms).toBeGreaterThan(0);
    }
    send({ type: "creature", id: "human" });
    posted.length = 0;
    for (let i = 0; i < 6; i++) p.process([], [[new Float32Array(128)]]);
    expect(posted.some((m) => m.type === "telemetry")).toBe(true);
    expect(posted.filter((m) => m.type === "error")).toEqual([]);
  });
});
