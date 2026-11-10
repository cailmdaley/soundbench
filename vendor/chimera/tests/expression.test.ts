/**
 * The face's expression from the real engine: the cheeks puff on p and b (lips shut, the air held
 * behind them) but not on m (out through the nose), t or k (shut further back), and the brows
 * rise on a high note.
 */
import { describe, expect, test } from "bun:test";
import { initEngine, WasmVoice } from "./harness.ts";
import type { TelemetryMessage } from "../src/worklet/protocol.ts";
import { Expression } from "../src/ui/render/expression.ts";

await initEngine();
const { planUtterance } = await import("../src/wasm/engine.ts");
const MALE = { female: 0, child: 0 };
const FS = 48000;

/** the telemetry block (as src/worklet/processor.ts reads it), with what Expression uses */
function telemetry(v: WasmVoice): TelemetryMessage {
  const { x, h } = v as unknown as { x: { memory: WebAssembly.Memory; engine_telemetry: (h: number) => number }; h: number };
  const p = x.engine_telemetry(h);
  const [phLen, orLen, velumPort, f0, voicing, pLung, outRms, noiseLevel, noiseNode, gate, clock, nArtic, nPh, nOr] = new Float64Array(x.memory.buffer, p, 16);
  const all = new Float64Array(x.memory.buffer, p, 16 + nArtic + 3 * (nPh + nOr));
  const o = 16 + nArtic + nPh + nOr;
  const pressureRms = Float32Array.from(all.subarray(o, o + nPh + nOr));
  const none = new Float32Array(0);
  return { type: "telemetry", t: { artic: none, phAreas: none, orAreas: none, phLen, orLen, pressureRms, flow: none, velumPort, glottalArea: none, glottalFlow: none, f0, voicing, pLung, outRms, noiseLevel, noiseNode, gate, clock, active: [] }, load: 0, body: MALE, nPh, nOr, beatStep: -1 };
}

/** say a run of segments, stepping an Expression at 100 frames a second; the largest of each value by segment */
function run(ipa: string[], f0 = 110, dur = (s: string) => ("aiu".includes(s) ? 0.3 : 0.12)) {
  const v = new WasmVoice(FS);
  const plan = planUtterance(ipa.map((s) => ({ ipa: s, dur: dur(s) })), { body: MALE, t0: 0.05, rate: 1, f0, intonation: "flat" });
  v.schedule(plan.gestures, 0, true);
  const e = new Expression();
  const peak = new Map<string, { puff: number; brow: number; lid: number }>();
  for (let i = 0; i < FS * (plan.end + 0.3); i += 480) {
    v.render(480);
    const tele = telemetry(v);
    e.update(tele, 0.01);
    // the cheeks lag the pressure a little: credit each moment to the segment of 40 ms before
    const seg = plan.segments.find((s) => s.t0 <= tele.t.clock - 0.04 && tele.t.clock - 0.04 < s.t1)?.ipa;
    if (!seg) continue;
    const pk = peak.get(seg) ?? { puff: 0, brow: -1, lid: 0 };
    peak.set(seg, { puff: Math.max(pk.puff, e.puff), brow: Math.max(pk.brow, e.brow), lid: Math.max(pk.lid, e.lid) });
  }
  return peak;
}

describe("expression", () => {
  test("cheeks puff behind shut lips, not for m, t or k", () => {
    const pk = run(["a", "p", "a", "t", "a", "m", "a", "b", "a", "k", "a"]);
    expect(pk.get("p")!.puff).toBeGreaterThan(0.6);
    expect(pk.get("b")!.puff).toBeGreaterThan(0.6);
    for (const s of ["m", "t", "k"]) expect(pk.get(s)!.puff).toBeLessThan(0.15);
  });

  test("brows rise on a note well above the recent pitch, and a long note closes the eyes", () => {
    // a voiced frame at a pitch (only what Expression reads differs from silence)
    const v = new WasmVoice(FS);
    v.render(480);
    const base = telemetry(v);
    const at = (f0: number): TelemetryMessage => ({ ...base, t: { ...base.t, f0, voicing: 1, gate: 1, outRms: 0.1 } });
    const e = new Expression();
    const brows: number[] = [], lids: number[] = [];
    for (let i = 0; i < 400; i++) {
      // two seconds around 110 Hz, then two an octave up
      e.update(at(i < 200 ? 110 * 2 ** (0.08 * Math.sin(i / 3)) : 220), 0.01);
      brows.push(e.brow);
      lids.push(e.lid);
    }
    expect(Math.max(...brows.slice(50, 200))).toBeLessThan(0.3);
    expect(Math.max(...brows.slice(220, 260))).toBeGreaterThan(0.6);
    // the octave note held for two seconds: eyes nearly shut by its end
    expect(lids[399]).toBeGreaterThan(0.6);
  });

  test("a slow frame rate keeps time, and a skipped beat step still nods", () => {
    const v = new WasmVoice(FS);
    v.render(480);
    const base = telemetry(v);
    const at = (f0: number, beatStep = -1): TelemetryMessage => ({ ...base, beatStep, t: { ...base.t, f0, voicing: 1, gate: 1, outRms: 0.1 } });
    // three frames a second: two seconds low, then two an octave up
    const e = new Expression();
    for (let i = 0; i < 6; i++) e.update(at(110), 1 / 3);
    for (let i = 0; i < 6; i++) e.update(at(220), 1 / 3);
    expect(e.lid).toBeGreaterThan(0.6);
    // the sequencer moves from step 3 to 5 between frames: the beat at 4 nods the head
    const b = new Expression();
    b.update(at(110, 3), 0.01);
    b.update(at(110, 5), 0.01);
    let nod = 0;
    for (let i = 0; i < 30; i++) {
      b.update(at(110, 5), 0.01);
      nod = Math.min(nod, b.nod);
    }
    expect(nod).toBeLessThan(-0.02);
  });
});
