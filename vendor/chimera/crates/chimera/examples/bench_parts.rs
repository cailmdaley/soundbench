//! Where a human voice's time goes: geometry (per control step) vs the FDTD step (per sample).
use chimera::anatomy::human::{BODY_MALE, NEUTRAL};
use chimera::voice::human_airway::HumanAirway;
use std::time::Instant;
fn main() {
    let fs = 96000.0;
    let mut air = HumanAirway::new(fs, BODY_MALE);
    let mut a = NEUTRAL;
    let n_ctrl = 750 * 4; // 4 s of control steps at 96 kHz / 128
    let t0 = Instant::now();
    for i in 0..n_ctrl {
        a.tb_x = 2.9 + 0.5 * ((i as f64) * 0.01).sin();
        air.apply(&a, 128.0 / fs);
        air.net.begin_block(128);
    }
    let geo = t0.elapsed().as_secs_f64();
    let net = &mut air.net;
    let n = (fs * 4.0) as usize;
    let t0 = Instant::now();
    let mut acc = 0.0;
    for _ in 0..n {
        net.set_fixed_pressure(0, 8000.0);
        acc += net.step();
    }
    let step = t0.elapsed().as_secs_f64();
    println!("per 4 s of audio: geometry {:.1} ms ({:.1}%), FDTD steps {:.1} ms ({:.1}%) [{acc:e}] nodes {} branches {}", geo * 1e3, geo / 4.0 * 100.0, step * 1e3, step / 4.0 * 100.0, net.n_nodes, net.n_branches);
}
