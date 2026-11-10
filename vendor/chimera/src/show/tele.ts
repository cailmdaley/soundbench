/**
 * A voice's recorded telemetry (`chimera render notes … --tele voice.tele`), replayed as the live
 * worklet's telemetry messages, so a 3D head moves exactly as the engine did for the sound.
 *
 * File layout (little-endian; crates/chimera-cli/src/render.rs `TeleRecorder`): "CHTL", u32
 * version, sampleRate, hop, frames, frameLen, nArtic, nPh, nOr, nScope, f32 bodyFemale,
 * bodyChild; then `frames` × `frameLen` f32. A frame: phLen, orLen, velumPort, f0, voicing, pLung,
 * outRms, noiseLevel, noiseNode, gate, clock, then artic, phAreas, orAreas, pressureRms, flow,
 * glottalArea, glottalFlow. Frame i is the state at sample i·hop.
 */
import type { TelemetryMessage } from "../worklet/protocol.ts";
import type { BodyMix } from "../wasm/types.ts";

export const TELE_VERSION = 1;
const HEADER_BYTES = 48;
const SCALARS = 11;

export class TeleTrack {
  readonly sampleRate: number;
  readonly hop: number;
  readonly frames: number;
  readonly body: BodyMix;
  readonly nPh: number;
  readonly nOr: number;
  private readonly nArtic: number;
  private readonly nScope: number;
  private readonly len: number;
  private readonly data: Float32Array;

  constructor(buf: ArrayBuffer) {
    if (buf.byteLength < HEADER_BYTES) throw new Error("telemetry file is too short for its header");
    const magic = new TextDecoder().decode(new Uint8Array(buf, 0, 4));
    if (magic !== "CHTL") throw new Error("not a CHIMERA telemetry file");
    const u = new DataView(buf);
    const at = (i: number) => u.getUint32(4 + 4 * i, true);
    const version = at(0);
    if (version !== TELE_VERSION) throw new Error(`telemetry file version ${version}; this app reads ${TELE_VERSION}`);
    this.sampleRate = at(1);
    this.hop = at(2);
    this.frames = at(3);
    this.len = at(4);
    this.nArtic = at(5);
    this.nPh = at(6);
    this.nOr = at(7);
    this.nScope = at(8);
    this.body = { female: u.getFloat32(40, true), child: u.getFloat32(44, true) };
    const want = SCALARS + this.nArtic + 3 * (this.nPh + this.nOr) + 2 * this.nScope;
    if (this.len !== want || this.frames < 1 || this.hop < 1 || this.sampleRate < 1) throw new Error("telemetry file header is inconsistent");
    if (buf.byteLength !== HEADER_BYTES + 4 * this.frames * this.len)
      throw new Error(`telemetry file is ${buf.byteLength} bytes; its header says ${HEADER_BYTES + 4 * this.frames * this.len}`);
    this.data = new Float32Array(buf, HEADER_BYTES, this.frames * this.len);
  }

  /** frames per second */
  get rate(): number {
    return this.sampleRate / this.hop;
  }

  /** seconds the recording covers */
  get duration(): number {
    return ((this.frames - 1) * this.hop) / this.sampleRate;
  }

  /** the frame at time t (s) on the recording's clock, clamped to the recording */
  indexAt(t: number): number {
    const i = Math.floor((t * this.sampleRate) / this.hop + 1e-6);
    return Math.min(this.frames - 1, Math.max(0, Number.isFinite(i) ? i : 0));
  }

  /** frame i as the worklet's telemetry message (arrays are views into the recording) */
  message(i: number, beatStep = -1): TelemetryMessage {
    const o0 = i * this.len;
    const d = this.data;
    let o = o0 + SCALARS;
    const take = (n: number) => d.subarray(o, (o += n));
    const nT = this.nPh + this.nOr;
    const artic = take(this.nArtic), phAreas = take(this.nPh), orAreas = take(this.nOr), pressureRms = take(nT), flow = take(nT), glottalArea = take(this.nScope), glottalFlow = take(this.nScope);
    const s = (k: number) => d[o0 + k];
    return {
      type: "telemetry",
      t: { phLen: s(0), orLen: s(1), velumPort: s(2), f0: s(3), voicing: s(4), pLung: s(5), outRms: s(6), noiseLevel: s(7), noiseNode: s(8), gate: s(9), clock: s(10), artic, phAreas, orAreas, pressureRms, flow, glottalArea, glottalFlow, active: [] },
      load: 0,
      body: this.body,
      nPh: this.nPh,
      nOr: this.nOr,
      beatStep,
    };
  }
}
