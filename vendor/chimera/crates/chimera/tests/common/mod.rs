//! Shared helpers for the golden tests: fixtures written by `bun tools/golden.ts` from the
//! TypeScript engine the Rust crate was ported from.
#![allow(dead_code)]

use serde_json::Value;

pub fn load(name: &str) -> Value {
    let path = format!("{}/tests/golden/{name}.json", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    serde_json::from_str(&text).unwrap()
}

pub fn num(v: &Value) -> f64 {
    match v {
        Value::Number(n) => n.as_f64().unwrap(),
        Value::String(s) => match s.as_str() {
            "NaN" => f64::NAN,
            "Infinity" => f64::INFINITY,
            "-Infinity" => f64::NEG_INFINITY,
            _ => panic!("not a number: {s}"),
        },
        Value::Null => f64::NAN,
        _ => panic!("not a number: {v}"),
    }
}

pub fn nums(v: &Value) -> Vec<f64> {
    v.as_array().unwrap_or_else(|| panic!("not an array: {v}")).iter().map(num).collect()
}

/// Relative difference with an absolute floor (values near zero compare absolutely).
pub fn rel(a: f64, b: f64, floor: f64) -> f64 {
    if a == b || (a.is_nan() && b.is_nan()) {
        return 0.0;
    }
    (a - b).abs() / a.abs().max(b.abs()).max(floor)
}

/// Worst relative difference between two equal-length sequences (panics on a length mismatch).
pub fn worst(label: &str, a: &[f64], b: &[f64], floor: f64) -> f64 {
    assert_eq!(a.len(), b.len(), "{label}: length {} vs {}", a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| rel(*x, *y, floor)).fold(0.0, f64::max)
}

/// `CHIMERA_BLESS=1 cargo test`: write the Rust engine's output as the new fixtures (after an
/// intended change to the sound; the TypeScript engine that made the first ones is gone).
pub fn blessing() -> bool {
    std::env::var("CHIMERA_BLESS").is_ok_and(|v| v == "1")
}

/// Replace a fixture in one step (write a temporary file, then rename it over the old one), so a
/// test reading it while another blesses it sees the old file or the new, never half of one.
fn write_fixture(path: &str, bytes: &[u8]) {
    let tmp = format!("{path}.{}.tmp", std::process::id());
    std::fs::write(&tmp, bytes).unwrap();
    std::fs::rename(&tmp, path).unwrap();
    println!("blessed {path}");
}

pub fn save_f32(name: &str, data: &[f32]) {
    let path = format!("{}/tests/golden/{name}.f32", env!("CARGO_MANIFEST_DIR"));
    let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
    write_fixture(&path, &bytes);
}

pub fn save_json(name: &str, v: &Value) {
    let path = format!("{}/tests/golden/{name}.json", env!("CARGO_MANIFEST_DIR"));
    write_fixture(&path, serde_json::to_string(v).unwrap().as_bytes());
}

/// For renders that may legitimately part after a discrete decision goes the other way on a
/// last-ulp difference (V8's libm and Rust's differ in the last bit of some functions; chaotic or
/// self-oscillating sources can amplify that): the loudness envelope must still agree to 0.05 dB
/// in every 10 ms window. Renders expected to agree throughout assert `first_diverge.is_none()`.
pub fn assert_close(name: &str, d: &AudioDiff) {
    assert!(d.env_db < 0.05, "{name}: RMS envelope differs by {:.3} dB (max sample error {:.2e} of peak, first at {:?})", d.env_db, d.max_err, d.first_diverge);
}

pub fn load_f32(name: &str) -> Vec<f32> {
    let path = format!("{}/tests/golden/{name}.f32", env!("CARGO_MANIFEST_DIR"));
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// How two renders compare: worst sample error relative to the reference peak, the first sample
/// where they differ by more than `tol`·peak, and the RMS-envelope difference (dB, 10 ms windows
/// louder than −60 dB re peak).
#[derive(Debug)]
pub struct AudioDiff {
    pub peak: f64,
    pub max_err: f64,
    pub first_diverge: Option<usize>,
    pub env_db: f64,
}

pub fn compare_audio(reference: &[f32], got: &[f32], tol: f64, win: usize) -> AudioDiff {
    assert_eq!(reference.len(), got.len(), "length");
    let peak = reference.iter().fold(0.0f64, |m, x| m.max(x.abs() as f64)).max(1e-30);
    let mut max_err = 0.0f64;
    let mut first = None;
    for (i, (a, b)) in reference.iter().zip(got).enumerate() {
        let e = (*a as f64 - *b as f64).abs() / peak;
        if e > max_err {
            max_err = e;
        }
        if first.is_none() && e > tol {
            first = Some(i);
        }
    }
    let mut env_db = 0.0f64;
    for (ca, cb) in reference.chunks(win).zip(got.chunks(win)) {
        let ra = (ca.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / ca.len() as f64).sqrt();
        let rb = (cb.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / cb.len() as f64).sqrt();
        if ra.max(rb) / peak < 1e-3 {
            continue;
        }
        env_db = env_db.max((20.0 * ((rb + 1e-30) / (ra + 1e-30)).log10()).abs());
    }
    AudioDiff { peak, max_err, first_diverge: first, env_db }
}
