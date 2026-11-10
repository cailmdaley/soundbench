/**
 * The duet page's recordings (`chimera render notes … --tele`): a file built by hand in the CLI's
 * layout decodes into the worklet's telemetry messages at the right times, and damaged files are
 * refused rather than replayed as garbage.
 */
import { describe, expect, test } from "bun:test";
import { TeleTrack } from "../src/show/tele.ts";

const [nArtic, nPh, nOr, nScope] = [17, 3, 2, 32];
const LEN = 11 + nArtic + 3 * (nPh + nOr) + 2 * nScope;

/** a file of `frames` frames: frame i has f0 = 100 + i, clock = i·768/48000 and artic[k] = i + k/100 */
function file(frames: number, o: { version?: number; cut?: number } = {}): ArrayBuffer {
  const buf = new ArrayBuffer(48 + 4 * frames * LEN);
  const d = new DataView(buf);
  new Uint8Array(buf).set(new TextEncoder().encode("CHTL"));
  [o.version ?? 1, 48000, 768, frames, LEN, nArtic, nPh, nOr, nScope].forEach((x, i) => d.setUint32(4 + 4 * i, x, true));
  d.setFloat32(40, 1, true); // female
  d.setFloat32(44, 0, true);
  const f = new Float32Array(buf, 48);
  for (let i = 0; i < frames; i++) {
    f[i * LEN + 3] = 100 + i;
    f[i * LEN + 10] = (i * 768) / 48000;
    for (let k = 0; k < nArtic; k++) f[i * LEN + 11 + k] = i + k / 100;
    f[i * LEN + 11 + nArtic + 3 * (nPh + nOr)] = 0.5; // glottalArea[0]
  }
  return o.cut ? buf.slice(0, buf.byteLength - o.cut) : buf;
}

describe("TeleTrack", () => {
  test("reads the header and each frame as a telemetry message", () => {
    const t = new TeleTrack(file(10));
    expect([t.sampleRate, t.hop, t.frames, t.nPh, t.nOr]).toEqual([48000, 768, 10, nPh, nOr]);
    expect(t.body).toEqual({ female: 1, child: 0 });
    expect(t.rate).toBe(62.5);
    const m = t.message(4, 7);
    expect(m.t.f0).toBe(104);
    expect(m.t.clock).toBeCloseTo(0.064, 6);
    expect(m.t.artic.length).toBe(nArtic);
    expect(m.t.artic[2]).toBeCloseTo(4.02, 5);
    expect([m.t.phAreas.length, m.t.orAreas.length, m.t.pressureRms.length, m.t.flow.length]).toEqual([nPh, nOr, nPh + nOr, nPh + nOr]);
    expect(m.t.glottalArea.length).toBe(nScope);
    expect(m.t.glottalArea[0]).toBe(0.5);
    expect([m.nPh, m.nOr, m.beatStep]).toEqual([nPh, nOr, 7]);
  });

  test("the frame at a time is the last one recorded by then, clamped to the recording", () => {
    const t = new TeleTrack(file(10));
    expect(t.indexAt(0)).toBe(0);
    expect(t.indexAt(0.016)).toBe(1); // exactly frame 1's time
    expect(t.indexAt(0.0159)).toBe(0);
    expect(t.indexAt(0.1)).toBe(6);
    expect(t.indexAt(-1)).toBe(0);
    expect(t.indexAt(99)).toBe(9);
    expect(t.indexAt(NaN)).toBe(0);
    expect(t.duration).toBeCloseTo(0.144, 6);
  });

  test("refuses a truncated file, another version or another format", () => {
    expect(() => new TeleTrack(file(10, { cut: 4 }))).toThrow(/bytes/);
    expect(() => new TeleTrack(file(10).slice(0, 20))).toThrow(/too short/);
    expect(() => new TeleTrack(file(10, { version: 2 }))).toThrow(/version 2/);
    const other = file(2);
    new Uint8Array(other).set(new TextEncoder().encode("RIFF"));
    expect(() => new TeleTrack(other)).toThrow(/not a CHIMERA/);
  });
});
