import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname } from "node:path";

/** Write mono or stereo 16-bit PCM WAV. Channels are Float32/64 arrays in [-1, 1]. */
export function writeWav(path: string, channels: ArrayLike<number>[], sampleRate: number): void {
  const nCh = channels.length;
  const n = channels[0].length;
  const bytesPerSample = 2;
  const dataBytes = n * nCh * bytesPerSample;
  const buf = Buffer.alloc(44 + dataBytes);
  buf.write("RIFF", 0);
  buf.writeUInt32LE(36 + dataBytes, 4);
  buf.write("WAVE", 8);
  buf.write("fmt ", 12);
  buf.writeUInt32LE(16, 16);
  buf.writeUInt16LE(1, 20);
  buf.writeUInt16LE(nCh, 22);
  buf.writeUInt32LE(sampleRate, 24);
  buf.writeUInt32LE(sampleRate * nCh * bytesPerSample, 28);
  buf.writeUInt16LE(nCh * bytesPerSample, 32);
  buf.writeUInt16LE(16, 34);
  buf.write("data", 36);
  buf.writeUInt32LE(dataBytes, 40);
  let o = 44;
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < nCh; c++) {
      let v = channels[c][i];
      if (!Number.isFinite(v)) v = 0;
      v = Math.max(-1, Math.min(1, v));
      buf.writeInt16LE(Math.round(v * 32767), o);
      o += 2;
    }
  }
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, buf);
}

export function readWav(path: string): { sampleRate: number; channels: Float32Array[] } {
  const buf = readFileSync(path);
  if (buf.toString("ascii", 0, 4) !== "RIFF" || buf.toString("ascii", 8, 12) !== "WAVE") throw new Error("not a WAV");
  let off = 12;
  let fmt: { ch: number; sr: number; bits: number; format: number } | null = null;
  while (off + 8 <= buf.length) {
    const id = buf.toString("ascii", off, off + 4);
    const size = buf.readUInt32LE(off + 4);
    const body = off + 8;
    if (id === "fmt ") {
      fmt = { format: buf.readUInt16LE(body), ch: buf.readUInt16LE(body + 2), sr: buf.readUInt32LE(body + 4), bits: buf.readUInt16LE(body + 14) };
    } else if (id === "data" && fmt) {
      const bps = fmt.bits / 8;
      const frames = Math.floor(size / (bps * fmt.ch));
      const channels = Array.from({ length: fmt.ch }, () => new Float32Array(frames));
      for (let i = 0; i < frames; i++) {
        for (let c = 0; c < fmt.ch; c++) {
          const p = body + (i * fmt.ch + c) * bps;
          let v = 0;
          if (fmt.format === 3 && fmt.bits === 32) v = buf.readFloatLE(p);
          else if (fmt.bits === 16) v = buf.readInt16LE(p) / 32768;
          else if (fmt.bits === 24) v = (buf.readIntLE(p, 3)) / 8388608;
          else if (fmt.bits === 32) v = buf.readInt32LE(p) / 2147483648;
          channels[c][i] = v;
        }
      }
      return { sampleRate: fmt.sr, channels };
    }
    off = body + size + (size & 1);
  }
  throw new Error("no data chunk");
}
