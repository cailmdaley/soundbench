//! IPA string → segments (stress ˈ ˌ, length ː, tie bars, affricates, diphthongs, word gaps).

use super::Segment;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

const VOWEL_CHARS: &str = "iyɨʉɯuɪʏʊeøɘɵɤoəɛœɜɞʌɔæɐaɶɑɒɝɚ";
const MULTI: [&str; 10] = ["tʃ", "dʒ", "ts", "dz", "pf", "aɪ", "aʊ", "ɔɪ", "eɪ", "oʊ"];
const ALIASES: [(&str, &str); 9] = [("t͡ʃ", "tʃ"), ("d͡ʒ", "dʒ"), ("t͡s", "ts"), ("d͡z", "dz"), ("ɡ", "g"), ("ʧ", "tʃ"), ("ʤ", "dʒ"), ("r̥", "r"), ("ɐ̯", "ɐ")];


fn known() -> &'static HashSet<&'static str> {
    static K: OnceLock<HashSet<&'static str>> = OnceLock::new();
    K.get_or_init(|| {
        let mut k: HashSet<&'static str> = super::inventory::consonant_symbols().iter().copied().collect();
        for (i, c) in VOWEL_CHARS.char_indices() {
            k.insert(&VOWEL_CHARS[i..i + c.len_utf8()]);
        }
        k.extend(MULTI);
        k
    })
}

/// Force the tables (the known-symbol set and the NFC data), so nothing initialises lazily later.
pub fn init() {
    known();
    nfc_tables();
}

pub fn tokenize_ipa(input: &str) -> Vec<Segment> {
    let mut s = nfc(input);
    for (a, b) in ALIASES {
        s = s.replace(a, b);
    }
    s = s.replace(['\u{35C}', '\u{361}'], ""); // tie bars
    let s: Vec<char> = s.chars().collect();
    let mut out: Vec<Segment> = Vec::new();
    let mut stress = 0u8;
    let mut i = 0;
    let known = known();
    while i < s.len() {
        let ch = s[i];
        if ch == 'ˈ' || ch == '\'' {
            stress = 1;
            i += 1;
            continue;
        }
        if ch == 'ˌ' {
            stress = 2;
            i += 1;
            continue;
        }
        if ch == ' ' || ch == '.' || ch == '|' || ch == ',' {
            if let Some(last) = out.last_mut() {
                last.word_end = Some(true);
            }
            if ch == ',' || ch == '|' {
                out.push(Segment { ipa: "_".into(), dur: Some(0.16), ..Default::default() });
            }
            i += 1;
            continue;
        }
        if ch == 'ː' || ch == ':' {
            if let Some(last) = out.last_mut() {
                let first = last.ipa.chars().next();
                last.dur = Some(last.dur.unwrap_or(if first.is_some_and(|c| VOWEL_CHARS.contains(c)) { 0.15 } else { 0.09 }) * 1.7);
            }
            i += 1;
            continue;
        }
        let mut tok = String::new();
        for m in MULTI {
            if starts_with_at(&s, i, m) {
                tok = m.to_string();
            }
        }
        if tok.is_empty() {
            tok = ch.to_string();
        }
        i += tok.chars().count();
        // skip combining diacritics we do not model
        while i < s.len() && ('\u{300}'..='\u{36F}').contains(&s[i]) {
            i += 1;
        }
        if !known.contains(tok.as_str()) {
            continue;
        }
        let is_v = tok.chars().next().is_some_and(|c| VOWEL_CHARS.contains(c)) && !super::inventory::consonant_symbols().contains(&tok.as_str());
        out.push(Segment { ipa: tok, stress: Some(if is_v { stress } else { 0 }), ..Default::default() });
        if is_v {
            stress = 0;
        }
    }
    out
}

fn starts_with_at(s: &[char], i: usize, m: &str) -> bool {
    m.chars().enumerate().all(|(k, c)| s.get(i + k) == Some(&c))
}

// ---------------------------------------------------------------- NFC
// `String.prototype.normalize("NFC")` without a Unicode crate: canonical decomposition of the code
// points NFC rewrites on their own, canonical ordering, canonical composition. The tables
// (`tools/gen-ipa-nfc.ts`, from the JS engine's ICU) hold every combining class but only the
// compositions whose first character the tokenizer can emit; any other starter is dropped by the
// tokenizer whether it composes or not, so the tokens come out as the TypeScript engine's did.

#[derive(Deserialize)]
struct NfcData {
    ccc: Vec<(u32, u32, u8)>,
    decomp: Vec<(u32, Vec<u32>)>,
    comp: Vec<(u32, u32, u32)>,
}

struct NfcTables {
    /// (first, last, rank) ranges, sorted
    ccc: Vec<(u32, u32, u8)>,
    decomp: HashMap<char, Vec<char>>,
    comp: HashMap<(char, char), char>,
}

fn nfc_tables() -> &'static NfcTables {
    static T: OnceLock<NfcTables> = OnceLock::new();
    T.get_or_init(|| {
        let d: NfcData = serde_json::from_str(include_str!("ipaNfc.generated.json")).expect("ipaNfc.generated.json");
        let ch = |u: u32| char::from_u32(u).expect("code point");
        NfcTables {
            ccc: d.ccc,
            decomp: d.decomp.into_iter().map(|(c, v)| (ch(c), v.into_iter().map(ch).collect())).collect(),
            comp: d.comp.into_iter().map(|(a, b, c)| ((ch(a), ch(b)), ch(c))).collect(),
        }
    })
}

impl NfcTables {
    /// canonical-ordering rank (0 for a starter)
    fn rank(&self, c: char) -> u8 {
        let u = c as u32;
        match self.ccc.binary_search_by(|&(a, b, _)| if b < u { std::cmp::Ordering::Less } else if a > u { std::cmp::Ordering::Greater } else { std::cmp::Ordering::Equal }) {
            Ok(k) => self.ccc[k].2,
            Err(_) => 0,
        }
    }
}

fn nfc(input: &str) -> String {
    let t = nfc_tables();
    let mut cps: Vec<char> = Vec::with_capacity(input.len());
    for c in input.chars() {
        match t.decomp.get(&c) {
            Some(d) => cps.extend(d),
            None => cps.push(c),
        }
    }
    // canonical ordering: each run of non-starters, stably by class
    let mut i = 0;
    while i < cps.len() {
        if t.rank(cps[i]) == 0 {
            i += 1;
            continue;
        }
        let start = i;
        while i < cps.len() && t.rank(cps[i]) != 0 {
            i += 1;
        }
        cps[start..i].sort_by_key(|&c| t.rank(c));
    }
    // canonical composition
    let mut out: Vec<char> = Vec::with_capacity(cps.len());
    let mut starter: Option<usize> = None;
    let mut last: Option<u8> = None;
    for c in cps {
        let r = t.rank(c);
        if let Some(si) = starter
            && last.is_none_or(|l| l < r)
            && let Some(&composed) = t.comp.get(&(out[si], c))
        {
            out[si] = composed;
            continue;
        }
        if r == 0 {
            starter = Some(out.len());
            last = None;
        } else {
            last = Some(r);
        }
        out.push(c);
    }
    out.into_iter().collect()
}
