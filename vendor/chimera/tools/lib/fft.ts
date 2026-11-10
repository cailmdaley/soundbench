/**
 * FFT and windows for the plotting tools (the engine has its own, in Rust: chimera::analysis::fft).
 * In-place iterative radix-2 complex FFT on separate real/imag Float64Arrays.
 * Size must be a power of two. Pure TS, allocation-free after plan creation.
 */
export class FFT {
  readonly n: number;
  private readonly cos: Float64Array;
  private readonly sin: Float64Array;
  private readonly rev: Uint32Array;

  constructor(n: number) {
    if ((n & (n - 1)) !== 0 || n < 2) throw new Error(`FFT size must be a power of two, got ${n}`);
    this.n = n;
    this.cos = new Float64Array(n / 2);
    this.sin = new Float64Array(n / 2);
    for (let i = 0; i < n / 2; i++) {
      this.cos[i] = Math.cos((2 * Math.PI * i) / n);
      this.sin[i] = -Math.sin((2 * Math.PI * i) / n);
    }
    this.rev = new Uint32Array(n);
    const bits = Math.log2(n);
    for (let i = 0; i < n; i++) {
      let r = 0;
      for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
      this.rev[i] = r;
    }
  }

  /** Forward transform (sign -1). */
  transform(re: Float64Array, im: Float64Array): void {
    const n = this.n;
    const rev = this.rev;
    for (let i = 0; i < n; i++) {
      const j = rev[i];
      if (j > i) {
        let t = re[i]; re[i] = re[j]; re[j] = t;
        t = im[i]; im[i] = im[j]; im[j] = t;
      }
    }
    for (let size = 2; size <= n; size <<= 1) {
      const half = size >> 1;
      const step = n / size;
      for (let start = 0; start < n; start += size) {
        for (let k = 0; k < half; k++) {
          const wr = this.cos[k * step];
          const wi = this.sin[k * step];
          const a = start + k;
          const b = a + half;
          const xr = re[b] * wr - im[b] * wi;
          const xi = re[b] * wi + im[b] * wr;
          re[b] = re[a] - xr;
          im[b] = im[a] - xi;
          re[a] += xr;
          im[a] += xi;
        }
      }
    }
  }
}

export function hann(n: number): Float64Array {
  const w = new Float64Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1));
  return w;
}

export function hamming(n: number): Float64Array {
  const w = new Float64Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (n - 1));
  return w;
}

/** Magnitude spectrum in dB (n/2+1 bins) of a real frame, windowed. */
export function powerSpectrumDb(frame: ArrayLike<number>, fft: FFT, window: Float64Array): Float64Array {
  const n = fft.n;
  const re = new Float64Array(n);
  const im = new Float64Array(n);
  const m = Math.min(frame.length, window.length, n);
  for (let i = 0; i < m; i++) re[i] = frame[i] * window[i];
  fft.transform(re, im);
  const out = new Float64Array(n / 2 + 1);
  for (let k = 0; k <= n / 2; k++) out[k] = 10 * Math.log10(re[k] * re[k] + im[k] * im[k] + 1e-30);
  return out;
}

export function nextPow2(x: number): number {
  let n = 1;
  while (n < x) n <<= 1;
  return n;
}
