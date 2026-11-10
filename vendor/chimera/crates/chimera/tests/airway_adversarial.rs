//! Adversarial airway cases: nearly closed junctions, sealed cavities, Bernoulli flow, FDTD
//! bandwidths, sliding and fast closures (port of tests/airway-adversarial.test.ts).
use chimera::airway::frequency::{Peak, find_peaks, linspace, transfer_function};
use chimera::airway::network::{AirwayNetwork, Area, PortSpec, TubeSpec};
use chimera::analysis::fft::Fft;
use chimera::math::{PI, exp, hypot, js_max, js_min, ln, pow, sin, sqrt};
use chimera::physics::{CM_H2O, RHO, RHO_C2, WALL_SOFT};

const FS: f64 = 96000.0;

fn nearest(peaks: &[Peak], f: f64) -> Peak {
    peaks.iter().copied().reduce(|b, c| if (c.freq - f).abs() < (b.freq - f).abs() { c } else { b }).expect("no peaks")
}

#[test]
fn nearly_closed_three_way_junction_stays_stable_at_the_working_cfl() {
    let mut net = AirwayNetwork::new(FS);
    let (n, len) = (20, 8.0); // Δ = 0.4 cm → CFL ≈ 0.91
    let a = net.add_tube(TubeSpec::new("a", n, len, Area::Uniform(3.0)));
    let b = net.add_tube(TubeSpec::new("b", n, len, Area::Uniform(3.0)));
    let c = net.add_tube(TubeSpec::new("c", n, len, Area::Uniform(3.0)));
    // junction = last node of a, which we squeeze to a tiny area
    net.connect(a.last_node(), b.first_node());
    net.connect(a.last_node(), c.first_node());
    net.add_radiation(b.last_node());
    net.add_radiation(c.last_node());
    net.finalize();
    let mut areas = vec![3.0; n];
    areas[n - 1] = 0.002;
    net.set_tube_targets(a, &areas, None, None);
    net.begin_block(64);
    let mut peak: f64 = 0.0;
    for i in 0..FS as usize {
        if i % 64 == 0 {
            net.begin_block(64);
        }
        if i < 400 {
            net.ext_flow[a.first_node()] = 200.0 * sin(i as f64 * 0.3);
        }
        let y = net.step();
        assert!(y.is_finite());
        peak = js_max(peak, net.p[a.last_node()].abs());
    }
    assert!(peak < 1e7, "peak {peak:e}");
    // decays once excitation stops
    let mut tail: f64 = 0.0;
    for _ in 0..(FS / 10.0) as usize {
        // (the pressure is read before the step, as `p[…] + net.step() * 0` evaluated)
        let p = net.p[a.last_node()];
        tail = js_max(tail, (p + net.step() * 0.0).abs());
    }
    assert!(tail < peak, "tail {tail:e} vs peak {peak:e}");
}

#[test]
fn sealed_cavity_compression_follows_the_linearised_adiabatic_law() {
    let mut net = AirwayNetwork::new(FS);
    let t = net.add_tube(TubeSpec::new("cav", 10, 4.0, Area::Uniform(2.0)));
    net.finalize();
    // shrink volume by 5 % over 20 ms
    let steps = FS * 0.02;
    let block = 64;
    let mut i = 0usize;
    while (i as f64) < steps {
        if i % block == 0 {
            let f = js_min(1.0, (i + block) as f64 / steps);
            net.set_tube_targets(t, &[2.0 * (1.0 - 0.05 * f); 10], None, None);
            net.begin_block(block);
        }
        net.step();
        i += 1;
    }
    let mut i = 0usize;
    while (i as f64) < FS * 0.02 {
        if i % block == 0 {
            net.begin_block(block);
        }
        net.step();
        i += 1;
    }
    let expected = RHO_C2 * ln(1.0 / 0.95); // ∫ρc² dV/V
    let mut mean = 0.0;
    for k in 0..10 {
        mean += net.p[t.node(k)] / 10.0;
    }
    let e = (mean - expected).abs() / expected;
    println!("cavity: {mean:.2} vs {expected:.2} dyn/cm² ({:.3} %)", 100.0 * e);
    assert!(e < 0.03);
}

#[test]
fn steady_flow_through_a_short_constriction_obeys_bernoulli_with_jet_loss() {
    for length_frac in [1.0, 0.25] {
        let mut net = AirwayNetwork::new(FS);
        let n = 30;
        let t = net.add_tube(TubeSpec::new("t", n, 12.0, Area::Uniform(3.0)));
        let lung = net.add_pressure_node("lung");
        net.add_port(lung, t.first_node(), PortSpec::new(3.0, 1.0).kinetic(0.0, 0.0));
        net.add_radiation(t.last_node());
        net.finalize();
        let ac = 0.08;
        let mut areas = vec![3.0; n];
        let mut mins = vec![3.0; n];
        // constriction occupying a whole section, or only a quarter of it (sub-grid)
        let harmonic = 1.0 / (length_frac / ac + (1.0 - length_frac) / 3.0);
        areas[20] = harmonic;
        mins[20] = ac;
        net.set_tube_targets(t, &areas, None, Some(&mins));
        let p = 6.0 * CM_H2O;
        let mut i = 0usize;
        while (i as f64) < FS * 0.25 {
            if i % 64 == 0 {
                net.begin_block(64);
            }
            net.set_fixed_pressure(lung, p * js_min(1.0, i as f64 / (FS * 0.02)));
            net.step();
            i += 1;
        }
        let u = net.u[t.branch(20)].abs();
        let bernoulli = ac * sqrt((2.0 * p) / RHO);
        let e = (u - bernoulli).abs() / bernoulli;
        println!("lengthFrac {length_frac}: U {u:.2} vs Bernoulli {bernoulli:.2} cm³/s ({:.2} %)", 100.0 * e);
        assert!(e < 0.15);
    }
}

#[test]
fn formant_bandwidths_agree_between_fdtd_and_the_frequency_domain_model() {
    let mut net = AirwayNetwork::new(FS);
    let areas: Vec<f64> = (0..40).map(|i| 1.5 + 2.5 * pow(sin((i as f64 / 39.0) * 2.5), 2.0)).collect();
    let t = net.add_tube(TubeSpec::new("t", 40, 17.5, Area::Profile(areas)).wall(Some(WALL_SOFT)));
    net.add_radiation(t.last_node());
    net.finalize();
    let fr = linspace(80.0, 4500.0, 6000);
    let reference: Vec<Peak> = find_peaks(&fr, &transfer_function(&net.linear_elements(), t.first_node(), &fr, &[]).mag, 80.0).into_iter().filter(|p| p.freq < 4000.0).collect();
    let n = FS as usize;
    let nfft = 1 << 17;
    let (mut re, mut im) = (vec![0.0; nfft], vec![0.0; nfft]);
    net.begin_block(1);
    for i in 0..n {
        if i == 0 {
            net.ext_flow[t.first_node()] = FS;
        }
        re[i] = net.step();
    }
    Fft::new(nfft).transform(&mut re, &mut im);
    let (mut freqs, mut mag) = (vec![0.0; nfft / 2], vec![0.0; nfft / 2]);
    for k in 0..nfft / 2 {
        freqs[k] = (k as f64 * FS) / nfft as f64;
        mag[k] = hypot(re[k], im[k]) / (2.0 * PI * js_max(freqs[k], 1.0));
    }
    let got: Vec<Peak> = find_peaks(&freqs, &mag, 80.0).into_iter().filter(|p| p.freq < 4200.0).collect();
    for r in &reference {
        let g = nearest(&got, r.freq);
        let e = (g.freq - r.freq).abs() / r.freq;
        println!("{:.1} vs {:.1} Hz ({:.3} %), bw {:.1} vs {:.1} Hz", g.freq, r.freq, 100.0 * e, g.bw, r.bw);
        assert!(e < 0.012);
        if r.bw.is_finite() && g.bw.is_finite() {
            assert!((g.bw - r.bw).abs() / r.bw < 0.2);
        }
    }
}

#[test]
fn a_realistic_sliding_closure_with_trapped_pressure_releases_without_spikes() {
    // closure forms over 25 ms (as articulator dynamics produce), slides 6 sections back toward the
    // lungs while held (trapped pressure behind it), then releases over 12 ms
    let mut net = AirwayNetwork::new(FS);
    let n = 40;
    let t = net.add_tube(TubeSpec::new("t", n, 17.0, Area::Uniform(3.0)).wall(Some(WALL_SOFT)));
    let lung = net.add_pressure_node("lung");
    net.add_port(lung, t.first_node(), PortSpec::new(3.0, 1.0));
    net.add_radiation(t.last_node());
    net.finalize();
    let block = 64;
    let (mut peak_p, mut peak_out, mut peak_open_out, mut peak_slide, mut peak_release): (f64, f64, f64, f64, f64) = (0.0, 0.0, 0.0, 0.0, 0.0);
    let total = FS * 0.6;
    let p = 8.0 * CM_H2O;
    let mut areas = vec![0.0; n];
    let mut i = 0usize;
    while (i as f64) < total {
        let time = i as f64 / FS;
        if i % block == 0 {
            let tt = (i + block) as f64 / FS;
            let depth = if tt < 0.05 {
                0.0
            } else if tt < 0.075 {
                (1.3 * (tt - 0.05)) / 0.025
            } else if tt < 0.4 {
                1.3
            } else {
                js_max(0.0, 1.3 - (1.3 * (tt - 0.4)) / 0.012)
            };
            let pos = 30.0 - 6.0 * js_min(1.0, js_max(0.0, (tt - 0.1) / 0.25));
            for (k, a) in areas.iter_mut().enumerate() {
                *a = js_max(0.0, 3.0 * (1.0 - depth * exp(-pow((k as f64 - pos) / 1.6, 2.0))));
            }
            net.set_tube_targets(t, &areas, None, None);
            net.begin_block(block);
        }
        net.set_fixed_pressure(lung, p * js_min(1.0, time / 0.03));
        let y = net.step();
        assert!(y.is_finite());
        if time > 0.03 {
            peak_out = js_max(peak_out, y.abs());
        }
        if time > 0.03 && time < 0.05 {
            peak_open_out = js_max(peak_open_out, y.abs());
        }
        if time > 0.09 && time < 0.39 {
            peak_slide = js_max(peak_slide, y.abs());
        }
        if time > 0.395 && time < 0.45 {
            peak_release = js_max(peak_release, y.abs());
        }
        // acoustically open cells only: squeeze-film pressure inside a closing gap is physical
        for k in 0..n {
            if net.area[t.node(k)] > 0.1 {
                peak_p = js_max(peak_p, net.p[t.node(k)].abs());
            }
        }
        i += 1;
    }
    println!("peakP {peak_p:.1} (P {p:.1}), slide {peak_slide:e}, release {peak_release:e}, open {peak_open_out:e}, out {peak_out:e}");
    assert!(peak_p < 2.5 * p);
    // a held, sliding closure must stay quiet (no spurious pops); the release burst is the
    // largest event and is of physical size (comparable to the breathing onset, not orders above)
    assert!(peak_slide < 0.25 * peak_release);
    assert!(peak_release < 10.0 * peak_open_out);
}

#[test]
fn fast_but_articulator_limited_closure_gives_bounded_piston_pressure() {
    // closing 3 cm² → 0 over 8 ms is near the fastest lip/tongue motion; pressure must stay sane
    let mut net = AirwayNetwork::new(FS);
    let n = 40;
    let t = net.add_tube(TubeSpec::new("t", n, 17.0, Area::Uniform(3.0)).wall(Some(WALL_SOFT)));
    net.add_radiation(t.last_node());
    net.finalize();
    let mut peak_p: f64 = 0.0;
    let mut areas = vec![0.0; n];
    let mut i = 0usize;
    while (i as f64) < FS * 0.1 {
        if i % 64 == 0 {
            let tt = (i + 64) as f64 / FS;
            let depth = js_min(1.2, (1.2 * tt) / 0.008);
            for (k, a) in areas.iter_mut().enumerate() {
                *a = js_max(0.0, 3.0 * (1.0 - depth * exp(-pow((k as f64 - 30.0) / 1.6, 2.0))));
            }
            net.set_tube_targets(t, &areas, None, None);
            net.begin_block(64);
        }
        assert!(net.step().is_finite());
        for k in 0..n {
            if net.area[t.node(k)] > 0.1 {
                peak_p = js_max(peak_p, net.p[t.node(k)].abs());
            }
        }
        i += 1;
    }
    println!("peakP {peak_p:.1} dyn/cm² (limit {:.1})", 20.0 * CM_H2O);
    assert!(peak_p < 20.0 * CM_H2O);
}
