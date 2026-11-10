//! The creatures against the TypeScript engine: every call of every creature, pitched calls,
//! parameter changes on a held gate, stops, initial parameters and a 96 kHz case, rendered in
//! several block sizes (the output must not depend on them), plus telemetry snapshots and the
//! valve sources sample by sample. Fixtures from `bun tools/golden/creatures.ts`.
mod common;
use chimera::creatures::{CREATURES, CreatureTelemetry, CreatureVoice, creature_by_id};
use chimera::sources::syrinx::SyrinxLabia;
use chimera::sources::valves::{Bubble, ClickTrain, LipReed, Whistle};
use common::*;
use serde_json::Value;

fn values(def: &chimera::creatures::CreatureDef, v: &Value) -> Vec<(usize, f64)> {
    v.as_object().unwrap().iter().map(|(k, x)| (def.param_index(k).unwrap_or_else(|| panic!("param {k}")), num(x))).collect()
}

fn run(s: &Value, block: usize) -> (Vec<f32>, Vec<CreatureTelemetry>) {
    let def = creature_by_id(s["creature"].as_str().unwrap()).unwrap();
    let fs = num(&s["fsOut"]);
    let initial = if s["initial"].is_object() { values(def, &s["initial"]) } else { vec![] };
    let mut v = CreatureVoice::new(def, fs, &initial);
    let n = s["samples"].as_u64().unwrap() as usize;
    let mut ops: Vec<&Value> = s["ops"].as_array().unwrap().iter().collect();
    ops.sort_by_key(|o| o["at"].as_u64().unwrap());
    let mut out = vec![0.0f32; n];
    let mut snaps = Vec::new();
    let mut apply = |v: &mut CreatureVoice, o: &Value| match o["op"].as_str().unwrap() {
        "params" => v.set_params(&values(def, &o["values"])),
        "gate" => v.set_gate(num(&o["level"])),
        "call" => {
            let pitch = o.get("pitch").map(num);
            let amp = o.get("amp").map_or(1.0, num);
            v.play_call_id(o["id"].as_str().unwrap(), pitch, amp);
        }
        "stop" => v.stop_call(),
        "telemetry" => {
            let mut t = v.new_telemetry();
            v.telemetry(&mut t);
            snaps.push(t);
        }
        op => panic!("unknown op {op}"),
    };
    let (mut i, mut oi) = (0usize, 0usize);
    while i < n {
        while oi < ops.len() && ops[oi]["at"].as_u64().unwrap() as usize <= i {
            apply(&mut v, ops[oi]);
            oi += 1;
        }
        let next_op = if oi < ops.len() { (ops[oi]["at"].as_u64().unwrap() as usize).max(i + 1) } else { usize::MAX };
        let end = n.min(i + block).min(next_op);
        v.render(&mut out[i..end]);
        i = end;
    }
    for o in &ops[oi..] {
        apply(&mut v, o);
    }
    (out, snaps)
}

/// Compare a telemetry snapshot with the TS one (1e-9 relative; f32 fields exactly as stored).
fn telemetry_diff(name: &str, t: &CreatureTelemetry, r: &Value) -> f64 {
    let got = serde_json::to_value(t).unwrap();
    for k in ["kind", "id", "call"] {
        assert_eq!(got[k], r[k], "{name}: telemetry {k}");
    }
    let mut w = 0.0f64;
    for k in ["gate", "outRms", "clock"] {
        w = w.max(rel(num(&got[k]), num(&r[k]), 1e-12));
    }
    for k in ["params", "source"] {
        let (a, b) = (got[k].as_object().unwrap(), r[k].as_object().unwrap());
        assert_eq!(a.keys().collect::<Vec<_>>(), b.keys().collect::<Vec<_>>(), "{name}: telemetry {k} keys");
        for (key, x) in a {
            w = w.max(rel(num(x), num(&b[key]), 1e-12));
        }
    }
    let (a, b) = (got["tubes"].as_object().unwrap(), r["tubes"].as_object().unwrap());
    assert_eq!(a.keys().collect::<Vec<_>>(), b.keys().collect::<Vec<_>>(), "{name}: tube names");
    for (key, x) in a {
        for f in ["areas", "rms"] {
            w = w.max(worst(&format!("{name} {key}.{f}"), &nums(&x[f]), &nums(&b[key][f]), 1e-9));
        }
    }
    w
}

struct Outcome {
    name: String,
    line: String,
    diff: AudioDiff,
    env_db: f64,
    tel: f64,
}

fn check(s: &Value, audio: &[f32]) -> Outcome {
    let name = s["name"].as_str().unwrap();
    let (got, snaps) = run(s, 128);
    let win = s["win"].as_u64().unwrap() as usize;
    let windows: Vec<usize> = s["windows"].as_array().unwrap().iter().map(|w| w.as_u64().unwrap() as usize).collect();
    let off = s["offset"].as_u64().unwrap() as usize;
    let len = s["length"].as_u64().unwrap() as usize;
    let reference = &audio[off..off + len];
    // the kept windows of the Rust render, in the fixture's layout
    let mut kept = Vec::with_capacity(len);
    let mut starts = Vec::new();
    for &w in &windows {
        starts.push((kept.len(), w * win));
        kept.extend_from_slice(&got[w * win..((w + 1) * win).min(got.len())]);
    }
    let d = compare_audio(reference, &kept, 1e-6, win);
    let first = d.first_diverge.map(|i| {
        let (k, abs) = starts.iter().rev().find(|(k, _)| *k <= i).unwrap();
        (abs + i - k) as f64 / num(&s["fsOut"])
    });
    // the whole render's envelope against the TS window RMS
    let env = nums(&s["env"]);
    let mut env_db = 0.0f64;
    for (wi, e) in env.iter().enumerate() {
        let c = &got[wi * win..((wi + 1) * win).min(got.len())];
        let r = (c.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / c.len() as f64).sqrt();
        if r.max(*e) / d.peak < 1e-3 {
            continue;
        }
        env_db = env_db.max((20.0 * ((r + 1e-30) / (e + 1e-30)).log10()).abs());
    }
    let refs = s["snaps"].as_array().unwrap();
    assert_eq!(snaps.len(), refs.len(), "{name}: snapshot count");
    let tel = snaps.iter().zip(refs).map(|(t, r)| telemetry_diff(name, t, r)).fold(0.0, f64::max);
    let line = format!("{name:>24}: peak {:.3} max err {:.2e} first >1e-6 at {first:?} s, kept-window env {:.4} dB, full env {env_db:.4} dB, telemetry {tel:.1e}", d.peak, d.max_err, d.env_db);
    for block in [1usize, 77, 4096] {
        assert!(run(s, block).0 == got, "{name}: block size {block} changed the output");
    }
    Outcome { name: name.to_string(), line, diff: d, env_db, tel }
}

#[test]
fn creatures_match_typescript() {
    let scenarios = load("creatures");
    let audio = load_f32("creatures");
    let list = scenarios.as_array().unwrap();
    // scenarios are independent: render them on all cores
    let outcomes: Vec<Outcome> = std::thread::scope(|sc| {
        let hs: Vec<_> = list.iter().map(|s| sc.spawn(|| check(s, &audio))).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for o in &outcomes {
        println!("{}", o.line);
    }
    for o in &outcomes {
        assert_close(&o.name, &o.diff);
        assert!(o.env_db < 0.05, "{}: full-render envelope differs by {:.3} dB", o.name, o.env_db);
        assert!(o.tel < 1e-9, "{}: telemetry differs by {:.2e}", o.name, o.tel);
    }
}

/// JSON equality with numbers compared as f64 (Rust writes `0.0` where JS writes `0`).
fn same_json(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => x.as_f64() == y.as_f64(),
        (Value::Array(x), Value::Array(y)) => x.len() == y.len() && x.iter().zip(y).all(|(p, q)| same_json(p, q)),
        (Value::Object(x), Value::Object(y)) => x.len() == y.len() && x.iter().all(|(k, v)| y.get(k).is_some_and(|w| same_json(v, w))),
        _ => a == b,
    }
}

/// Positions of `"key":` in a JSON string, which must increase (serde_json's `Value` sorts keys,
/// so key order is checked on the text).
fn assert_key_order(label: &str, json: &str, keys: &[&str]) {
    let pos: Vec<usize> = keys.iter().map(|k| json.find(&format!("\"{k}\":")).unwrap_or_else(|| panic!("{label}: no key {k}"))).collect();
    assert!(pos.windows(2).all(|w| w[0] < w[1]), "{label}: key order {keys:?}");
}

#[test]
fn preview_telemetry_serialises_like_typescript() {
    // the first call scenario of each creature captures the telemetry of a fresh voice
    let scenarios = load("creatures");
    let mut seen = 0;
    for s in scenarios.as_array().unwrap() {
        let name = s["name"].as_str().unwrap();
        if !name.ends_with("-initial") {
            continue;
        }
        let def = creature_by_id(s["creature"].as_str().unwrap()).unwrap();
        let mut v = CreatureVoice::new(def, num(&s["fsOut"]), &values(def, &s["initial"]));
        let mut t = v.new_telemetry();
        v.telemetry(&mut t);
        let reference = &s["snaps"][0];
        let text = serde_json::to_string(&t).unwrap();
        let got: Value = serde_json::from_str(&text).unwrap();
        assert!(same_json(&got, reference), "{name}: preview telemetry JSON\n{got}\n{reference}");
        assert_key_order(name, &text, &["kind", "id", "params", "gate", "outRms", "clock", "tubes", "source", "call"]);
        let params: Vec<&str> = def.params.iter().map(|p| p.key).collect();
        assert_key_order(name, &text, &params);
        seen += 1;
    }
    assert_eq!(seen, CREATURES.len());
}

#[test]
fn defs_serialise_like_typescript() {
    let u = load("creature-units");
    let text = serde_json::to_string(&*CREATURES).unwrap();
    let got: Value = serde_json::from_str(&text).unwrap();
    assert!(same_json(&got, &u["defs"]), "defs differ\n{got}\n{}", u["defs"]);
    assert_key_order("def", &text, &["id", "name", "latin", "group", "blurb", "role", "oversample"]);
}

fn seq_check(label: &str, got: &[f64], reference: &Value) -> f64 {
    let r = nums(reference);
    let w = worst(label, got, &r, 1e-12);
    let first = got.iter().zip(&r).position(|(a, b)| rel(*a, *b, 1e-12) > 1e-9);
    println!("{label:>14}: worst {w:.2e}, first >1e-9 at {first:?} of {}", r.len());
    w
}

#[test]
fn valves_match_typescript() {
    use chimera::math::{PI, sin};
    let u = load("creature-units");
    let n = nums(&u["lipDefault"]).len();
    let mut checks = Vec::new();
    {
        let fs = 48000.0;
        let mut a = LipReed::new(fs);
        let g: Vec<f64> = (0..n).map(|i| a.next(2500.0 + 3000.0 * sin((2.0 * PI * 180.0 * i as f64) / fs))).collect();
        checks.push(("lipDefault", seq_check("lipDefault", &g, &u["lipDefault"])));
        let mut b = LipReed::new(fs);
        b.f = 470.0;
        b.zeta = 0.08;
        b.h0 = 0.003;
        b.width = 2.4;
        b.surface_per_mass = 40.0;
        let mut y = 0.0;
        let nb = nums(&u["lipBuzz"]).len();
        let g: Vec<f64> = (0..nb)
            .map(|i| {
                let dp = (if i < nb * 2 / 3 { 400.0 } else { -300.0 }) - 2e3 * y;
                y = b.next(dp);
                y
            })
            .collect();
        checks.push(("lipBuzz", seq_check("lipBuzz", &g, &u["lipBuzz"])));
    }
    {
        let fs = 96000.0;
        let mut w = Whistle::new(fs);
        w.f = 1800.0;
        let nw = nums(&u["whistle"]).len();
        let g: Vec<f64> = (0..nw).map(|i| w.next(if i < nw * 11 / 15 { 0.8 } else { 0.01 }, 900.0 * sin((2.0 * PI * 1750.0 * i as f64) / fs), 1.5)).collect();
        checks.push(("whistle", seq_check("whistle", &g, &u["whistle"])));
    }
    {
        let mut c = ClickTrain::new(48000.0);
        c.click(0.001, 1.0);
        c.click(0.0012, 0.5);
        c.click(0.004, -0.8);
        let g: Vec<f64> = (0..nums(&u["clicksPlain"]).len()).map(|_| c.next()).collect();
        checks.push(("clicksPlain", seq_check("clicksPlain", &g, &u["clicksPlain"])));
        c.ring = 3200.0;
        c.ring_decay = 0.0015;
        c.click(0.0005, 1.0);
        c.click(0.003, 0.6);
        c.click(0.0031, 0.3);
        let total = nums(&u["clicksRing"]).len();
        let mut g: Vec<f64> = (0..total - 200).map(|_| c.next()).collect();
        c.click(0.002, 1.0);
        c.clear();
        g.extend((0..200).map(|_| c.next()));
        checks.push(("clicksRing", seq_check("clicksRing", &g, &u["clicksRing"])));
    }
    {
        let mut b = Bubble::new(96000.0);
        let nb = nums(&u["bubble"]).len();
        let (mut ys, mut rs) = (Vec::new(), Vec::new());
        for i in 0..nb {
            if i == 100 {
                b.snap(0.8, Bubble::DEFAULT_SNAP);
            }
            if i == nb * 7 / 12 {
                b.snap(0.3, 0.002);
            }
            ys.push(b.next());
            rs.push(b.radius());
        }
        checks.push(("bubble", seq_check("bubble", &ys, &u["bubble"])));
        checks.push(("bubbleRadius", seq_check("bubbleRadius", &rs, &u["bubbleRadius"])));
    }
    {
        let mut l = SyrinxLabia::new(192000.0, 1.0);
        let ns = nums(&u["syrinx"]).len();
        let g: Vec<f64> = (0..ns)
            .map(|i| {
                l.drive = if i < ns / 2 { 0.3 } else { 0.9 };
                l.tension = if i < ns / 3 { 0.4 } else { 0.05 };
                l.f0 = 2500.0 + i as f64;
                l.gate = if i < ns * 5 / 6 { 1.0 } else { 0.2 };
                l.next()
            })
            .collect();
        checks.push(("syrinx", seq_check("syrinx", &g, &u["syrinx"])));
    }
    for (label, w) in checks {
        assert!(w < 1e-9, "{label}: differs by {w:.2e}");
    }
}
