//! JavaScript formatting, so the printed summaries and the regenerated tables read exactly as the
//! TypeScript tools wrote them: `Number.prototype.toFixed`, `String(number)`, `padStart`/`padEnd`
//! and `JSON.stringify(value, null, 1)` over an insertion-ordered value.

use serde::de::{Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};

/// `x.toFixed(d)`: the exact decimal value rounded half away from zero (JS picks the larger n on
/// a tie), the sign printed whenever x < 0 (so not for −0).
pub fn fixed(x: f64, d: usize) -> String {
    if !x.is_finite() || x.abs() >= 1e21 {
        return num(x);
    }
    let neg = x < 0.0;
    // the exact binary value in decimal (a double has at most 1074 fractional digits)
    let s = format!("{:.1100}", x.abs());
    let (int, frac) = s.split_once('.').unwrap();
    let mut digits: Vec<u8> = int.bytes().chain(frac.bytes().take(d)).collect();
    if frac.as_bytes()[d] >= b'5' {
        let mut i = digits.len();
        loop {
            if i == 0 {
                digits.insert(0, b'1');
                break;
            }
            i -= 1;
            if digits[i] == b'9' {
                digits[i] = b'0';
            } else {
                digits[i] += 1;
                break;
            }
        }
    }
    let n_int = digits.len() - d;
    let mut out = String::with_capacity(digits.len() + 2);
    if neg {
        out.push('-');
    }
    out.push_str(core::str::from_utf8(&digits[..n_int]).unwrap());
    if d > 0 {
        out.push('.');
        out.push_str(core::str::from_utf8(&digits[n_int..]).unwrap());
    }
    out
}

/// `String(x)`: the shortest round-trip digits, in exponent form outside [1e-6, 1e21).
pub fn num(x: f64) -> String {
    if x.is_nan() {
        return "NaN".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "Infinity" } else { "-Infinity" }.into();
    }
    if x == 0.0 {
        return "0".into();
    }
    let a = x.abs();
    if (1e-6..1e21).contains(&a) {
        return format!("{x}");
    }
    let s = format!("{x:e}");
    match s.split_once('e') {
        Some((m, e)) if !e.starts_with('-') => format!("{m}e+{e}"),
        _ => s,
    }
}

/// `s.padStart(n)` (JS counts UTF-16 units; every symbol printed here is one unit).
pub fn pad_start(s: &str, n: usize) -> String {
    format!("{s:>n$}")
}

/// `s.padEnd(n)`.
pub fn pad_end(s: &str, n: usize) -> String {
    format!("{s:<n$}")
}

/// A JSON value that keeps object keys in insertion order, as JS objects do (for the string
/// keys used here).
#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    /// `obj[key] = v`: replace in place (the key keeps its position) or append.
    pub fn set(&mut self, key: &str, v: Json) {
        let Json::Obj(m) = self else {
            panic!("not an object")
        };
        match m.iter_mut().find(|(k, _)| k == key) {
            Some((_, slot)) => *slot = v,
            None => m.push((key.to_string(), v)),
        }
    }

    /// `JSON.stringify(self, null, indent)`.
    pub fn stringify(&self, indent: usize) -> String {
        let mut out = String::new();
        self.write(&mut out, indent, 0);
        out
    }

    fn write(&self, out: &mut String, indent: usize, depth: usize) {
        let nl = |out: &mut String, d: usize| {
            out.push('\n');
            out.extend(core::iter::repeat_n(' ', indent * d));
        };
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Json::Num(x) => out.push_str(&if x.is_finite() {
                num(*x)
            } else {
                "null".into()
            }),
            Json::Str(s) => out.push_str(&serde_json::to_string(s).unwrap()),
            Json::Arr(a) if a.is_empty() => out.push_str("[]"),
            Json::Obj(m) if m.is_empty() => out.push_str("{}"),
            Json::Arr(a) => {
                out.push('[');
                for (i, v) in a.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    nl(out, depth + 1);
                    v.write(out, indent, depth + 1);
                }
                nl(out, depth);
                out.push(']');
            }
            Json::Obj(m) => {
                out.push('{');
                for (i, (k, v)) in m.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    nl(out, depth + 1);
                    out.push_str(&serde_json::to_string(k).unwrap());
                    out.push_str(": ");
                    v.write(out, indent, depth + 1);
                }
                nl(out, depth);
                out.push('}');
            }
        }
    }

    pub fn parse(text: &str) -> serde_json::Result<Json> {
        serde_json::from_str(text)
    }
}

impl<'de> Deserialize<'de> for Json {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Json;
            fn expecting(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                f.write_str("JSON")
            }
            fn visit_unit<E>(self) -> Result<Json, E> {
                Ok(Json::Null)
            }
            fn visit_bool<E>(self, b: bool) -> Result<Json, E> {
                Ok(Json::Bool(b))
            }
            fn visit_i64<E>(self, x: i64) -> Result<Json, E> {
                Ok(Json::Num(x as f64))
            }
            fn visit_u64<E>(self, x: u64) -> Result<Json, E> {
                Ok(Json::Num(x as f64))
            }
            fn visit_f64<E>(self, x: f64) -> Result<Json, E> {
                Ok(Json::Num(x))
            }
            fn visit_str<E>(self, s: &str) -> Result<Json, E> {
                Ok(Json::Str(s.to_string()))
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut a: A) -> Result<Json, A::Error> {
                let mut v = Vec::new();
                while let Some(x) = a.next_element()? {
                    v.push(x);
                }
                Ok(Json::Arr(v))
            }
            fn visit_map<A: MapAccess<'de>>(self, mut a: A) -> Result<Json, A::Error> {
                let mut m: Vec<(String, Json)> = Vec::new();
                while let Some((k, v)) = a.next_entry::<String, Json>()? {
                    // a repeated key keeps its first position and takes the last value (JSON.parse)
                    match m.iter_mut().find(|(k0, _)| *k0 == k) {
                        Some((_, slot)) => *slot = v,
                        None => m.push((k, v)),
                    }
                }
                Ok(Json::Obj(m))
            }
        }
        d.deserialize_any(V)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_fixed_matches_js() {
        assert_eq!(fixed(0.125, 2), "0.13");
        assert_eq!(fixed(1.005, 2), "1.00"); // 1.005 is just below in binary
        assert_eq!(fixed(-0.04, 1), "-0.0");
        assert_eq!(fixed(-0.0, 1), "0.0");
        assert_eq!(fixed(9.96, 1), "10.0");
        assert_eq!(fixed(2.5, 0), "3");
        assert_eq!(fixed(f64::NAN, 1), "NaN");
        assert_eq!(fixed(123.456, 0), "123");
    }

    #[test]
    fn number_strings_match_js() {
        assert_eq!(num(3.0), "3");
        assert_eq!(num(-0.0), "0");
        assert_eq!(num(0.1), "0.1");
        assert_eq!(num(1e-7), "1e-7");
        assert_eq!(num(1e21), "1e+21");
        assert_eq!(num(96000.0), "96000");
    }

    #[test]
    fn stringify_keeps_order_and_js_layout() {
        let j = Json::parse(r#"{"b":{"x":[1,2.5,-0]},"a":[],"c":{}}"#).unwrap();
        assert_eq!(
            j.stringify(1),
            "{\n \"b\": {\n  \"x\": [\n   1,\n   2.5,\n   0\n  ]\n },\n \"a\": [],\n \"c\": {}\n}"
        );
    }
}
