/**
 * Plays one decoded track and keeps the one playhead every head reads: what the listener hears
 * now, in seconds into the track. Its own AudioContext (the voice engine doesn't run here).
 *
 * Paused, the playhead is the saved position. Playing, it is that position plus the audible time
 * since the source started (never less than none: the first moments after Play are still in the
 * output buffer): the context's output timestamp where the browser has one (it already allows for
 * output latency), else currentTime less the reported latency. A seek bumps `epoch` (so animation
 * history can be reset) and restarts a fresh source. The track ends when its end is heard, not when
 * the source finishes rendering. The context leaving "running" (an iOS interruption, the page
 * going to the background), or failing to resume after Play, pauses the transport where it was.
 */
export type TransportState = { playing: boolean; time: number; duration: number; epoch: number };

export class Transport {
  readonly ctx: AudioContext;
  private buffer: AudioBuffer | null = null;
  private src: AudioBufferSourceNode | null = null;
  private offset = 0;
  private startedAt = 0;
  private playing = false;
  /** the source has rendered its last sample; the track ends once that is heard */
  private rendered = false;
  /** bumped by every play, pause, seek and dispose: a late resume result for an old one is ignored */
  private generation = 0;
  private wasRunning = false;
  epoch = 0;
  private readonly listeners = new Set<() => void>();

  constructor(ctx: AudioContext = new AudioContext({ latencyHint: "playback" })) {
    this.ctx = ctx;
    this.ctx.onstatechange = () => {
      const running = this.ctx.state === "running";
      if (this.wasRunning && !running && this.playing) this.pause();
      this.wasRunning = running;
    };
  }

  async load(data: ArrayBuffer): Promise<void> {
    this.buffer = await this.ctx.decodeAudioData(data);
    this.emit();
  }

  get duration(): number {
    return this.buffer?.duration ?? 0;
  }

  get isPlaying(): boolean {
    return this.playing;
  }

  get state(): TransportState {
    return { playing: this.playing, time: this.time(), duration: this.duration, epoch: this.epoch };
  }

  subscribe(f: () => void): () => void {
    this.listeners.add(f);
    return () => this.listeners.delete(f);
  }

  private emit(): void {
    for (const f of this.listeners) f();
  }

  /** Call from the tap itself: iOS only lets a user gesture start audio. */
  play(): void {
    if (!this.buffer || this.playing) return;
    const gen = ++this.generation;
    this.ctx.resume().catch(() => {
      // no audio (an interruption, a browser's refusal): back to paused, where it was
      if (gen === this.generation && this.playing) this.pause();
    });
    if (this.offset >= this.duration - 0.05) this.offset = 0;
    this.startSource();
    this.playing = true;
    this.emit();
  }

  pause(): void {
    if (!this.playing) return;
    this.generation++;
    this.offset = this.time();
    this.stopSource();
    this.playing = false;
    this.emit();
  }

  seek(t: number): void {
    this.generation++;
    this.offset = Math.min(this.duration, Math.max(0, t));
    this.epoch++;
    if (this.playing) {
      this.stopSource();
      this.startSource();
    }
    this.emit();
  }

  /** the audible playhead (s) */
  time(): number {
    if (!this.playing) return this.offset;
    const t = Math.min(this.duration, this.offset + Math.max(0, this.audibleNow() - this.startedAt));
    if (this.rendered && t >= this.duration) {
      // the end has been heard: stop here (listeners hear of it after the caller's frame)
      this.playing = false;
      this.offset = this.duration;
      queueMicrotask(() => this.emit());
    }
    return t;
  }

  dispose(): void {
    this.generation++;
    this.stopSource();
    this.listeners.clear();
    void this.ctx.close().catch(() => {});
  }

  private startSource(): void {
    const src = this.ctx.createBufferSource();
    src.buffer = this.buffer;
    src.connect(this.ctx.destination);
    src.onended = () => {
      if (this.src === src) this.rendered = true;
    };
    this.src = src;
    this.rendered = false;
    this.startedAt = this.ctx.currentTime;
    src.start(this.startedAt, this.offset);
  }

  private stopSource(): void {
    const src = this.src;
    this.src = null;
    this.rendered = false;
    if (!src) return;
    src.onended = null;
    try {
      src.stop();
    } catch {
      // never started (the context was closed)
    }
    src.disconnect();
  }

  /** the context time being heard now */
  private audibleNow(): number {
    const c = this.ctx;
    const ts = typeof c.getOutputTimestamp === "function" ? c.getOutputTimestamp() : null;
    if (ts && ts.contextTime !== undefined && ts.performanceTime !== undefined && ts.contextTime > 0) {
      // the stamp is a moment ago; carry it to now, but never ahead of what has been rendered
      const t = ts.contextTime + Math.max(0, (performance.now() - ts.performanceTime) / 1000);
      return Math.min(t, c.currentTime);
    }
    return c.currentTime - (c.outputLatency || c.baseLatency || 0);
  }
}
