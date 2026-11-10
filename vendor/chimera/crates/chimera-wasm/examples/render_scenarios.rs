//! Render the voice scenarios of crates/chimera/tests/golden/voice.json natively, through the
//! same C ABI the browser uses, into target/native-render/<name>.f32. The wasm test
//! (tests/wasm-engine.test.ts) renders them through the wasm build and requires identical bytes.
use chimera::control::score::Gesture;
use chimera_wasm::*;
use serde_json::Value;

fn main() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
    let scenarios: Value = serde_json::from_str(&std::fs::read_to_string(format!("{root}/crates/chimera/tests/golden/voice.json")).unwrap()).unwrap();
    let out_dir = format!("{root}/target/native-render");
    std::fs::create_dir_all(&out_dir).unwrap();
    for s in scenarios.as_array().unwrap() {
        let name = s["name"].as_str().unwrap();
        let out = render(s);
        let bytes: Vec<u8> = out.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(format!("{out_dir}/{name}.f32"), bytes).unwrap();
        println!("{name}: {} samples", out.len());
    }
}

/// The scenario's operations through the engine ABI, one 128-frame quantum at a time.
fn render(s: &Value) -> Vec<f32> {
    let fs = s["fsOut"].as_f64().unwrap();
    let n = s["samples"].as_u64().unwrap() as usize;
    let h = engine_new(fs);
    unsafe {
        let (female, child) = match s["body"].as_str().unwrap() {
            "female" => (1.0, 0.0),
            "child" => (0.0, 1.0),
            _ => (0.0, 0.0),
        };
        if (female, child) != (0.0, 0.0) {
            engine_set_body(h, female, child);
        }
        let mut ops: Vec<&Value> = s["ops"].as_array().unwrap().iter().collect();
        ops.sort_by_key(|o| o["at"].as_u64().unwrap());
        let mut out = Vec::with_capacity(n);
        let mut oi = 0;
        while out.len() < n {
            while oi < ops.len() && ops[oi]["at"].as_u64().unwrap() as usize <= out.len() {
                apply(h, ops[oi]);
                oi += 1;
            }
            let next = if oi < ops.len() { ops[oi]["at"].as_u64().unwrap() as usize } else { usize::MAX };
            let m = 128.min(n - out.len()).min(next.max(out.len() + 1) - out.len());
            let p = engine_render(h, m);
            out.extend_from_slice(std::slice::from_raw_parts(p, m));
        }
        engine_free(h);
        out
    }
}

unsafe fn apply(h: *mut Handle, o: &Value) {
    unsafe {
        let vals = std::slice::from_raw_parts_mut(engine_values(h), 32);
        match o["op"].as_str().unwrap() {
            "gate" => engine_set_gate(h, o["level"].as_f64().unwrap()),
            "artic" => {
                vals.fill(f64::NAN);
                for (k, v) in o["values"].as_object().unwrap() {
                    let i = chimera::anatomy::human::ArticKey::from_name(k).unwrap().index();
                    vals[i] = v.as_f64().unwrap();
                }
                engine_set_artic(h, match o["tipFree"].as_bool() {
                    Some(false) => 0,
                    Some(true) => 1,
                    None => 2,
                });
            }
            "source" => {
                vals.fill(f64::NAN);
                let names = ["f0", "glide", "pressure", "breathiness", "tension", "fry", "roughness", "vibratoDepth", "vibratoRate", "drift", "voiced"];
                for (k, v) in o["values"].as_object().unwrap() {
                    let i = names.iter().position(|n| n == k).unwrap();
                    vals[i] = v.as_f64().unwrap_or_else(|| if v.as_bool().unwrap() { 1.0 } else { 0.0 });
                }
                engine_set_source(h);
            }
            "schedule" => {
                let gs: Vec<Gesture> = serde_json::from_value(o["gestures"].clone()).unwrap();
                let rec = std::slice::from_raw_parts_mut(engine_records(h), gs.len() * GESTURE_RECORD);
                for (i, g) in gs.iter().enumerate() {
                    write_gesture(g, &mut rec[i * GESTURE_RECORD..(i + 1) * GESTURE_RECORD]);
                }
                engine_schedule(h, gs.len(), o["offset"].as_f64().unwrap(), 0);
            }
            "release" => engine_release(h, o["tag"].as_f64().unwrap()),
            "cancel" => engine_stop_speech(h),
            op => panic!("{op}"),
        }
    }
}
