//! LPC formants and YIN pitch on synthetic vowels (port of tests/analysis.test.ts).
use chimera::analysis::lpc::{FormantOpts, estimate_formants};
use chimera::analysis::pitch::{YinOpts, yin_pitch};
use chimera::math::{PI, cos, exp, js_round};

/// Impulse train at f0 through cascaded 2-pole resonators at known formants.
fn synth_vowel(sr: f64, f0: f64, formants: &[(f64, f64)], dur: f64) -> Vec<f64> {
    let n = js_round(sr * dur) as usize;
    let mut x = vec![0.0; n];
    let period = sr / f0;
    let mut k = 0usize;
    while (k as f64) * period < n as f64 {
        let i = js_round(k as f64 * period) as usize;
        // (an index rounding up to n was a no-op write past a typed array's end)
        if i < n {
            x[i] = 1.0;
        }
        k += 1;
    }
    for &(f, bw) in formants {
        let r = exp((-PI * bw) / sr);
        let a1 = 2.0 * r * cos((2.0 * PI * f) / sr);
        let a2 = -r * r;
        let mut y = vec![0.0; n];
        for i in 0..n {
            y[i] = x[i] + a1 * (if i > 0 { y[i - 1] } else { 0.0 }) + a2 * (if i > 1 { y[i - 2] } else { 0.0 });
        }
        x = y;
    }
    x
}

#[test]
fn lpc_recovers_known_formants() {
    let sr = 16000.0;
    let target = [(730.0, 60.0), (1090.0, 80.0), (2440.0, 120.0), (3400.0, 150.0)];
    let x = synth_vowel(sr, 110.0, &target, 0.3);
    let frame = &x[1600..1600 + 512];
    // flat-spectrum source: no pre-emphasis
    let fm = estimate_formants(frame, sr, FormantOpts { order: Some(14), pre_emph: Some(0.0), ..Default::default() });
    for (f, _) in target {
        let nearest = fm.iter().copied().reduce(|b, c| if (c.freq - f).abs() < (b.freq - f).abs() { c } else { b }).expect("no formants");
        let e = (nearest.freq - f).abs() / f;
        println!("{f} Hz: {:.1} Hz ({:.2} %)", nearest.freq, 100.0 * e);
        assert!(e < 0.05);
    }
}

#[test]
fn yin_recovers_f0() {
    let sr = 48000.0;
    for f0 in [82.0, 110.0, 220.0, 440.0, 880.0] {
        let x = synth_vowel(sr, f0, &[(500.0, 80.0), (1500.0, 100.0)], 0.2);
        let est = yin_pitch(&x[2000..2000 + 2400], sr, YinOpts { min_f0: Some(60.0), ..Default::default() }).f0;
        let e = (est - f0).abs() / f0;
        println!("{f0} Hz: {est:.3} Hz ({:.3} %)", 100.0 * e);
        assert!(e < 0.01);
    }
}
