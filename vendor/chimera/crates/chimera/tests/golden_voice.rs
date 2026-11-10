//! The human voice against the TypeScript engine, sample by sample, rendered in several block
//! sizes (the output must not depend on them).
mod common;
use chimera::anatomy::human::{ArticPatch, BodyName};
use chimera::control::score::Gesture;
use chimera::voice::human_voice::{HumanVoice, SourcePatch};
use common::*;
use serde_json::Value;

fn run(s: &Value, block: usize) -> Vec<f32> {
    let body = BodyName::parse(s["body"].as_str().unwrap()).body();
    let fs = num(&s["fsOut"]);
    let mut v = HumanVoice::new(fs, body, s["oversample"].as_u64().unwrap() as usize);
    let n = s["samples"].as_u64().unwrap() as usize;
    let mut ops: Vec<&Value> = s["ops"].as_array().unwrap().iter().collect();
    ops.sort_by_key(|o| o["at"].as_u64().unwrap());
    let mut out = vec![0.0f32; n];
    let (mut i, mut oi) = (0usize, 0usize);
    while i < n {
        while oi < ops.len() && ops[oi]["at"].as_u64().unwrap() as usize <= i {
            let o = ops[oi];
            oi += 1;
            match o["op"].as_str().unwrap() {
                "gate" => v.set_gate(num(&o["level"])),
                "artic" => {
                    let p: ArticPatch = serde_json::from_value(o["values"].clone()).unwrap();
                    v.set_articulation(&p, o["tipFree"].as_bool());
                }
                "source" => v.set_source(&serde_json::from_value::<SourcePatch>(o["values"].clone()).unwrap()),
                "schedule" => {
                    let mut gs: Vec<Gesture> = serde_json::from_value(o["gestures"].clone()).unwrap();
                    let off = v.clock + num(&o["offset"]);
                    for g in gs.iter_mut() {
                        g.t0 += off;
                        g.t1 += off;
                    }
                    v.schedule(&gs);
                }
                "release" => v.release(num(&o["tag"])),
                "cancel" => v.cancel_score(),
                op => panic!("unknown op {op}"),
            }
        }
        let next_op = if oi < ops.len() { (ops[oi]["at"].as_u64().unwrap() as usize).max(i + 1) } else { usize::MAX };
        let end = n.min(i + block).min(next_op);
        v.render(&mut out[i..end]);
        i = end;
    }
    out
}

#[test]
fn voice_matches_typescript() {
    let scenarios = load("voice");
    for s in scenarios.as_array().unwrap() {
        let name = s["name"].as_str().unwrap();
        let reference = load_f32(&format!("voice-{name}"));
        let got = run(s, 128);
        if blessing() {
            save_f32(&format!("voice-{name}"), &got);
            continue;
        }
        let d = compare_audio(&reference, &got, 1e-6, 480);
        println!("{name:>20}: peak {:.3} max err {:.2e} first >1e-6 at {:?} env {:.4} dB", d.peak, d.max_err, d.first_diverge, d.env_db);
        // the human voice is linear enough (and the port literal enough) to agree throughout
        assert!(d.first_diverge.is_none(), "{name}: renders diverge from sample {:?} (max error {:.2e} of peak)", d.first_diverge, d.max_err);
        for block in [1usize, 77, 4096] {
            assert_eq!(run(s, block), got, "{name}: block size {block} changed the output");
        }
    }
}
