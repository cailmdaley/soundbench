/**
 * The duet page's transport on a fake AudioContext whose clock and output latency the test sets:
 * the playhead is what the listener hears, Play never moves it back, the end comes when it's heard,
 * and audio that can't resume leaves the transport paused.
 */
import { describe, expect, test } from "bun:test";
import { Transport } from "../src/show/transport.ts";

class FakeSource {
  onended: (() => void) | null = null;
  started: [number, number] | null = null;
  stopped = false;
  buffer: unknown = null;
  connect() {}
  disconnect() {}
  start(when: number, offset: number) {
    this.started = [when, offset];
  }
  stop() {
    this.stopped = true;
  }
}

class FakeContext {
  currentTime = 0;
  outputLatency = 0.2;
  baseLatency = 0;
  state = "suspended";
  onstatechange: (() => void) | null = null;
  destination = {};
  sources: FakeSource[] = [];
  resumeResult: "ok" | "fail" = "ok";
  createBufferSource() {
    const s = new FakeSource();
    this.sources.push(s);
    return s;
  }
  async decodeAudioData() {
    return { duration: 100 };
  }
  async resume() {
    if (this.resumeResult === "fail") throw new Error("interrupted");
    this.setState("running");
  }
  async close() {
    this.setState("closed");
  }
  setState(s: string) {
    this.state = s;
    this.onstatechange?.();
  }
}

async function make() {
  const ctx = new FakeContext();
  const t = new Transport(ctx as unknown as AudioContext);
  await t.load(new ArrayBuffer(0));
  return { ctx, t };
}

describe("Transport", () => {
  test("the playhead is the audible time: still where Play was pressed until the output catches up", async () => {
    const { ctx, t } = await make();
    t.seek(60);
    ctx.currentTime = 5;
    t.play();
    expect(ctx.sources.at(-1)!.started).toEqual([5, 60]);
    expect(t.time()).toBe(60); // not 59.8: the first 0.2 s are still in the output buffer
    ctx.currentTime = 5.7;
    expect(t.time()).toBeCloseTo(60.5, 6);
  });

  test("tapping Play and Pause quickly never moves the playhead back", async () => {
    const { ctx, t } = await make();
    t.seek(60);
    for (let i = 0; i < 5; i++) {
      t.play();
      ctx.currentTime += 0.05;
      t.pause();
    }
    expect(t.time()).toBe(60);
  });

  test("the end comes when it is heard, not when the source finishes rendering", async () => {
    const { ctx, t } = await make();
    t.seek(99);
    t.play();
    ctx.currentTime = 1.0;
    ctx.sources.at(-1)!.onended!(); // rendered; 0.2 s still to come out of the speakers
    expect(t.isPlaying).toBe(true);
    expect(t.time()).toBeCloseTo(99.8, 6);
    ctx.currentTime = 1.25;
    expect(t.time()).toBe(100);
    expect(t.isPlaying).toBe(false);
    // Play at the end starts again from the top
    t.play();
    expect(ctx.sources.at(-1)!.started![1]).toBe(0);
  });

  test("a seek while playing restarts the source there and bumps the epoch", async () => {
    const { ctx, t } = await make();
    t.play();
    const first = ctx.sources.at(-1)!;
    const e = t.epoch;
    ctx.currentTime = 2;
    t.seek(30);
    expect(first.stopped).toBe(true);
    expect(ctx.sources.at(-1)!.started).toEqual([2, 30]);
    expect(t.epoch).toBe(e + 1);
    // the old source's end means nothing now
    first.onended?.();
    ctx.currentTime = 100;
    expect(t.isPlaying).toBe(true);
  });

  test("audio that can't resume leaves the transport paused where it was", async () => {
    const { ctx, t } = await make();
    ctx.resumeResult = "fail";
    t.seek(10);
    t.play();
    await Promise.resolve();
    await Promise.resolve();
    expect(t.isPlaying).toBe(false);
    expect(t.time()).toBe(10);
  });

  test("an interruption while playing pauses it", async () => {
    const { ctx, t } = await make();
    t.play();
    await Promise.resolve();
    expect(ctx.state).toBe("running");
    ctx.currentTime = 3;
    ctx.setState("interrupted");
    expect(t.isPlaying).toBe(false);
    expect(t.time()).toBeCloseTo(2.8, 6);
  });
});
