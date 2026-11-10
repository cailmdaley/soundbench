//! The text side of phonetics against the TypeScript engine: IPA tokenizing, G2P (with and
//! without the dictionary), phrases, lyrics → syllables, spelling splits, songs and vowel data.
//! Structures must agree exactly (numbers compare as doubles, so `1` equals `1.0`).
mod common;
use chimera::phonetics::english::{EnglishG2P, arpabet_to_segments, letter_to_sound};
use chimera::phonetics::ipa::tokenize_ipa;
use chimera::phonetics::syllables::{lyrics_to_syllables, split_spelling, syllabify_segments};
use chimera::phonetics::{songs, vowel_data};
use common::load;
use serde::Serialize;
use serde_json::Value;
use std::io::Read;

fn dict_text() -> String {
    let path = format!("{}/../../public/data/cmudict.txt.gz", env!("CARGO_MANIFEST_DIR"));
    let gz = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut text = String::new();
    flate2::read::GzDecoder::new(&gz[..]).read_to_string(&mut text).unwrap();
    text
}
fn g2ps() -> (EnglishG2P, EnglishG2P) {
    let mut d = EnglishG2P::new();
    d.load_dict(&dict_text());
    (d, EnglishG2P::new())
}

/// Deep equality; numbers compare as f64 (JSON.stringify writes 1.0 as 1).
fn same(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => x.as_f64() == y.as_f64(),
        (Value::Array(x), Value::Array(y)) => x.len() == y.len() && x.iter().zip(y).all(|(p, q)| same(p, q)),
        (Value::Object(x), Value::Object(y)) => x.len() == y.len() && x.iter().all(|(k, v)| y.get(k).is_some_and(|w| same(v, w))),
        _ => a == b,
    }
}
fn check(label: &str, expected: &Value, got: impl Serialize, failures: &mut Vec<String>) {
    let got = serde_json::to_value(got).unwrap();
    if !same(expected, &got) {
        failures.push(format!("{label}\n  ts:   {expected}\n  rust: {got}"));
    }
}
fn report(area: &str, n: usize, failures: Vec<String>) {
    println!("{area}: {n} cases, {} mismatches", failures.len());
    assert!(failures.is_empty(), "{area}: {} of {n} differ:\n{}", failures.len(), failures.iter().take(20).cloned().collect::<Vec<_>>().join("\n"));
}

#[test]
fn ipa_matches_typescript() {
    let cases = load("phonetics-ipa");
    let cases = cases.as_array().unwrap();
    let mut f = Vec::new();
    for c in cases {
        let input = c["input"].as_str().unwrap();
        check(&format!("{input:?}"), &c["segments"], tokenize_ipa(input), &mut f);
    }
    report("tokenize_ipa", cases.len(), f);
}

#[test]
fn g2p_matches_typescript() {
    let (d, nd) = g2ps();
    let fx = load("phonetics-g2p");
    let words = fx["words"].as_array().unwrap();
    let mut f = Vec::new();
    for c in words {
        let w = c["w"].as_str().unwrap();
        check(&format!("{w:?} dict reduce"), &c["dict"]["reduce"], d.word(w, true), &mut f);
        check(&format!("{w:?} dict full"), &c["dict"]["full"], d.word(w, false), &mut f);
        check(&format!("{w:?} no dict reduce"), &c["noDict"]["reduce"], nd.word(w, true), &mut f);
        check(&format!("{w:?} no dict full"), &c["noDict"]["full"], nd.word(w, false), &mut f);
        check(&format!("{w:?} letter_to_sound"), &c["lts"], letter_to_sound(w), &mut f);
    }
    let arpa = fx["arpabet"].as_array().unwrap();
    for c in arpa {
        let p = c["p"].as_str().unwrap();
        check(&format!("arpabet {p:?}"), &c["segments"], arpabet_to_segments(p), &mut f);
    }
    report("g2p", words.len() * 5 + arpa.len(), f);
}

#[test]
fn phrases_match_typescript() {
    let (d, nd) = g2ps();
    let cases = load("phonetics-phrases");
    let cases = cases.as_array().unwrap();
    let mut f = Vec::new();
    for c in cases {
        let text = c["text"].as_str().unwrap();
        check(&format!("{text:?} dict"), &c["dict"], d.to_phrases(text), &mut f);
        check(&format!("{text:?} no dict"), &c["noDict"], nd.to_phrases(text), &mut f);
    }
    report("to_phrases", cases.len() * 2, f);
}

#[test]
fn syllables_match_typescript() {
    let (d, nd) = g2ps();
    let fx = load("phonetics-syllables");
    let mut f = Vec::new();
    let mut n = 0;
    for c in fx["lyrics"].as_array().unwrap() {
        let text = c["text"].as_str().unwrap();
        check(&format!("{text:?} dict"), &c["dict"], lyrics_to_syllables(text, &d), &mut f);
        check(&format!("{text:?} no dict"), &c["noDict"], lyrics_to_syllables(text, &nd), &mut f);
        n += 2;
    }
    for c in fx["spelling"].as_array().unwrap() {
        let (w, k) = (c["word"].as_str().unwrap(), c["n"].as_u64().unwrap() as usize);
        check(&format!("split_spelling({w:?}, {k})"), &c["parts"], split_spelling(w, k), &mut f);
        n += 1;
    }
    for c in fx["syllabify"].as_array().unwrap() {
        let w = c["w"].as_str().unwrap();
        check(&format!("syllabify {w:?}"), &c["parts"], syllabify_segments(&d.word(w, false)), &mut f);
        n += 1;
    }
    report("syllables", n, f);
}

#[test]
fn songs_and_vowel_data_match_typescript() {
    let fx = load("phonetics-data");
    let mut f = Vec::new();
    check("demo songs", &fx["songs"], songs::DEMO_SONGS, &mut f);
    check("vowel refs", &fx["vowels"], vowel_data::VOWELS, &mut f);
    report("data", 2, f);
}
