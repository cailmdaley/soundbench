//! Uniform-tube resonances against analytic values, and the FDTD against the chain matrix
//! (port of tests/airway.test.ts, same thresholds).
use chimera::airway::frequency::{Peak, find_peaks, linspace, transfer_function};
use chimera::airway::network::{AirwayNetwork, Area, Tube, TubeSpec};
use chimera::analysis::fft::Fft;
use chimera::math::{PI, hypot, js_max, js_round, log2, pow, sin, sqrt};
use chimera::physics::{C_SOUND, WallParams};

fn uniform_tube(fs: f64, walls: bool) -> (AirwayNetwork, Tube) {
    let mut net = AirwayNetwork::new(fs);
    let wall = if walls { Some(WallParams { mass: 1.5, resistance: 1600.0, stiffness: 3e5 }) } else { None };
    let tube = net.add_tube(TubeSpec::new("tract", 40, 17.5, Area::Uniform(3.0)).wall(wall));
    net.add_radiation(tube.last_node());
    net.finalize();
    (net, tube)
}

/// The peak nearest to `f` (the first of equals, as `reduce` kept).
fn nearest(peaks: &[Peak], f: f64) -> Peak {
    peaks.iter().copied().reduce(|b, c| if (c.freq - f).abs() < (b.freq - f).abs() { c } else { b }).expect("no peaks")
}

/// FDTD impulse response → |spectrum| peaks.
fn fdtd_peaks(net: &mut AirwayNetwork, inject: usize, seconds: f64) -> Vec<Peak> {
    let n = js_round(net.fs * seconds) as usize;
    let nfft = 1usize << (log2(n as f64).ceil() as u32);
    let mut re = vec![0.0; nfft];
    let mut im = vec![0.0; nfft];
    net.begin_block(1);
    for i in 0..n {
        if i == 0 {
            net.ext_flow[inject] = net.fs; // unit volume impulse
        }
        re[i] = net.step();
        assert!(re[i].is_finite(), "non-finite at {i}");
    }
    Fft::new(nfft).transform(&mut re, &mut im);
    let mut freqs = vec![0.0; nfft / 2];
    let mut mag = vec![0.0; nfft / 2];
    for k in 0..nfft / 2 {
        freqs[k] = (k as f64 * net.fs) / nfft as f64;
        // divide by ω to compare with |U_rad| (output is dU/dt)
        mag[k] = hypot(re[k], im[k]) / (2.0 * PI * js_max(freqs[k], 1.0));
    }
    find_peaks(&freqs, &mag, 100.0).into_iter().filter(|p| p.freq < 5000.0).collect()
}

#[test]
fn uniform_tube_resonances_match_quarter_wave_theory() {
    let (net, tube) = uniform_tube(96000.0, false);
    let el = net.linear_elements();
    let fr = linspace(50.0, 5000.0, 4000);
    let h = transfer_function(&el, tube.first_node(), &fr, &[]);
    let peaks = find_peaks(&fr, &h.mag, 80.0);
    let a = sqrt(3.0 / PI);
    // node-centred discretisation: radiation load sits at the last node centre (Δ/2 inside)
    let l_eff = 17.5 - 17.5 / 40.0 / 2.0 + (8.0 * a) / (3.0 * PI) + 17.5 / 40.0 / 2.0;
    for n in 1..=4 {
        let f_theory = ((2.0 * n as f64 - 1.0) * C_SOUND) / (4.0 * l_eff);
        let p = nearest(&peaks, f_theory);
        let e = (p.freq - f_theory).abs() / f_theory;
        println!("R{n}: {:.2} Hz vs theory {f_theory:.2} Hz ({:.3} %)", p.freq, 100.0 * e);
        assert!(e < 0.03);
    }
}

#[test]
fn fdtd_impulse_response_agrees_with_frequency_domain_model() {
    for walls in [false, true] {
        let (mut net, tube) = uniform_tube(96000.0, walls);
        let el = net.linear_elements();
        let fr = linspace(50.0, 5000.0, 6000);
        let reference: Vec<Peak> = find_peaks(&fr, &transfer_function(&el, tube.first_node(), &fr, &[]).mag, 80.0).into_iter().filter(|p| p.freq < 4800.0).collect();
        let got = fdtd_peaks(&mut net, tube.first_node(), 0.5);
        assert!(got.len() >= reference.len());
        for r in &reference {
            let g = nearest(&got, r.freq);
            let e = (g.freq - r.freq).abs() / r.freq;
            println!("walls={walls}: {:.2} vs {:.2} Hz ({:.3} %)", g.freq, r.freq, 100.0 * e);
            assert!(e < 0.012);
        }
    }
}

#[test]
fn fdtd_stays_stable_with_a_non_uniform_tube_and_tiny_constriction() {
    let mut net = AirwayNetwork::new(96000.0);
    let areas: Vec<f64> = (0..40).map(|i| if i == 25 { 0.0005 } else { 0.5 + 4.0 * pow(sin((i as f64 / 39.0) * PI), 2.0) }).collect();
    let t = net.add_tube(TubeSpec::new("t", 40, 17.0, Area::Profile(areas)).wall(Some(WallParams { mass: 1.5, resistance: 1600.0, stiffness: 3e5 })));
    net.add_radiation(t.last_node());
    net.finalize();
    net.begin_block(1);
    let mut peak: f64 = 0.0;
    for i in 0..96000 {
        if i < 2000 {
            net.ext_flow[t.first_node()] = 300.0 * sin(i as f64 * 0.05);
        }
        let y = net.step();
        peak = js_max(peak, y.abs());
        assert!(y.is_finite());
    }
    // energy must decay after excitation stops
    let mut tail: f64 = 0.0;
    for _ in 0..9600 {
        tail = js_max(tail, net.step().abs());
    }
    assert!(tail < peak * 0.01, "tail {tail:e} vs peak {peak:e}");
}
