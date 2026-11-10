//! The CSD comparison as one number, and the verdict on a change.
//!
//! Each term is a song-level measure both voices have: her median H1−H2 in a song against the
//! synth's in the same song, her share of voiced frames against the synth's, and so on. A term's
//! gap is the mean over songs of |synth − singer|, divided by the spread of her own values over
//! the training songs (median absolute deviation, scaled to a standard deviation; taken from all
//! her notes, so nothing the synth does can move it). A gap within one spread is about as close
//! as two of her own songs are to each other and costs nothing; beyond that, the excess is the
//! term's cost. The score is the weighted mean of the costs: lower is better, and 0 means every
//! term is within her own variation.
//!
//! Every fifth song (en005, en010, …, in both keys) is held out. The spreads and the verdict use
//! the training songs only; the held-out songs are scored with the same spreads, as a check that
//! a change carries over to songs it was not tuned on.
//!
//! `--against <score.json>` compares the training score with an earlier run's. A change is
//! better when the score falls by at least `MARGIN`, no term's cost rises by more than
//! `TERM_SLACK`, and no guard breaks. The verdict compares like for like, note by note: each
//! song's value of each term is recomputed for both runs over the notes on which both runs give
//! it. A note that stops being measured (a stray voiced frame gone, a vibrato that no longer
//! reads as vibrato) leaves both runs alike, so it can neither lower nor raise the score; making
//! notes unmeasurable by not voicing them still costs, through the voiced share.

use super::{mean, median};
use serde_json::{Value, json};

pub const N_TERMS: usize = 13;

/// (name, weight). The weights give each group one share: spectrum, H1−H2, vibrato, voicing,
/// onsets and noise.
pub const TERMS: [(&str, f64); N_TERMS] = [
    ("level 0-.5k", 0.2),
    ("level .5-1k", 0.2),
    ("level 1-2k", 0.2),
    ("level 2-4k", 0.2),
    ("level 4-8k", 0.2),
    ("H1-H2", 1.0),
    ("vibrato share", 1.0 / 3.0),
    ("vibrato depth", 1.0 / 3.0),
    ("vibrato rate", 1.0 / 3.0),
    ("voiced share", 1.0),
    ("onset, vowel-initial", 0.5),
    ("onset, voiceless-initial", 0.5),
    ("aperiodicity", 1.0),
];

/// fewest training songs a spread is taken over
const MIN_SPREAD_SONGS: usize = 5;
/// fewest songs a gap is averaged over
const MIN_GAP_SONGS: usize = 3;
/// a change must lower the training score by this much to be better
pub const MARGIN: f64 = 0.01;
/// no term's cost may rise by more than this, whatever the score does
const TERM_SLACK: f64 = 0.25;
/// guards: largest rise in note MCD (dB) and synth pitch error (cents), largest fall in envelope
/// correlation
const MCD_SLACK: f64 = 0.1;
const PITCH_SLACK: f64 = 2.0;
const CORR_SLACK: f64 = 0.01;

/// One voice's measures on one note, in this order: the five band levels, H1−H2, long enough
/// to look for vibrato (1 or 0), vibrato depth and rate (NaN without vibrato), voiced share,
/// voice onset (NaN unless phrase-initial), aperiodicity, cents off the score.
pub const N_NOTE: usize = 13;
pub type NoteRow = [f64; N_NOTE];
const H1H2: usize = 5;
const ELIGIBLE: usize = 6;
const DEPTH: usize = 7;
const RATE: usize = 8;
const VOICED: usize = 9;
const ONSET: usize = 10;
const APER: usize = 11;
const CENTS: usize = 12;

/// What a note opens: 0 not a phrase, else a phrase on a vowel, a voiceless or a voiced consonant.
pub const PHRASE_VOWEL: u8 = 1;
pub const PHRASE_VOICELESS: u8 = 2;
pub const PHRASE_VOICED: u8 = 3;

/// Held out of tuning: every fifth song number, in both keys.
pub fn held_out(id: &str) -> bool {
    let digits: String = id.chars().filter(|c| c.is_ascii_digit()).collect();
    digits.parse::<u32>().is_ok_and(|n| n % 5 == 0)
}

/// Whether a voice's note gives term `k` a value.
fn gives(k: usize, row: &NoteRow, start: u8) -> bool {
    match k {
        6 => row[ELIGIBLE] == 1.0,
        7 | 8 => row[DEPTH].is_finite() && row[RATE].is_finite(),
        10 => start == PHRASE_VOWEL && row[ONSET].is_finite(),
        11 => start == PHRASE_VOICELESS && row[ONSET].is_finite(),
        _ => row[column(k)].is_finite(),
    }
}

fn column(k: usize) -> usize {
    match k {
        0..=4 => k,
        5 => H1H2,
        7 => DEPTH,
        8 => RATE,
        9 => VOICED,
        10 | 11 => ONSET,
        12 => APER,
        _ => unreachable!("term {k} has no single column"),
    }
}

/// A voice's value of term `k` in one song, over the notes `keep` admits (and that give it).
fn song_value(k: usize, rows: &[NoteRow], starts: &[u8], keep: &dyn Fn(usize) -> bool) -> f64 {
    let kept: Vec<usize> = (0..rows.len())
        .filter(|&i| keep(i) && gives(k, &rows[i], starts[i]))
        .collect();
    match k {
        // the share of the notes long enough for vibrato that have it
        6 => {
            if kept.is_empty() {
                f64::NAN
            } else {
                kept.iter().filter(|&&i| rows[i][DEPTH].is_finite()).count() as f64
                    / kept.len() as f64
            }
        }
        0..=4 | 9 => mean(&kept.iter().map(|&i| rows[i][column(k)]).collect::<Vec<_>>()),
        _ => median(&kept.iter().map(|&i| rows[i][column(k)]).collect::<Vec<_>>()),
    }
}

/// One song's inputs to the score: every annotated note, measured on both voices.
pub struct SongScore {
    pub id: String,
    /// per note: PHRASE_* or 0
    pub starts: Vec<u8>,
    /// per note: the planner placed its vowel on time, so it is compared
    pub aligned: Vec<bool>,
    pub singer: Vec<NoteRow>,
    pub synth: Vec<NoteRow>,
    pub dropped: u64,
    /// mean note MCD (dB) over the notes compared
    pub mcd: f64,
    pub env_corr: f64,
}

impl SongScore {
    fn misaligned(&self) -> usize {
        self.aligned.iter().filter(|a| !**a).count()
    }

    /// per term (singer, synth) over the notes compared
    fn pairs(&self) -> [(f64, f64); N_TERMS] {
        let keep = |i: usize| self.aligned[i];
        std::array::from_fn(|k| {
            (
                song_value(k, &self.singer, &self.starts, &keep),
                song_value(k, &self.synth, &self.starts, &keep),
            )
        })
    }

    /// per term, the singer's value over all her notes: the spreads are taken from these
    fn singer_all(&self) -> [f64; N_TERMS] {
        std::array::from_fn(|k| song_value(k, &self.singer, &self.starts, &|_| true))
    }

    /// the synth's median |cents| off the score over the notes compared that `keep` admits
    fn synth_pitch(&self, keep: &dyn Fn(usize) -> bool) -> f64 {
        median(
            &(0..self.synth.len())
                .filter(|&i| self.aligned[i] && keep(i))
                .map(|i| self.synth[i][CENTS].abs())
                .collect::<Vec<_>>(),
        )
    }
}

fn mad_sd(xs: &[f64]) -> f64 {
    let m = median(xs);
    1.4826 * median(&xs.iter().map(|x| (x - m).abs()).collect::<Vec<_>>())
}

/// The singer's spread of each term over these songs; NaN with too few songs or none at all.
fn spreads(train: &[&SongScore]) -> [f64; N_TERMS] {
    let all: Vec<[f64; N_TERMS]> = train.iter().map(|s| s.singer_all()).collect();
    std::array::from_fn(|k| {
        let h: Vec<f64> = all.iter().map(|v| v[k]).filter(|v| v.is_finite()).collect();
        let sd = mad_sd(&h);
        if h.len() >= MIN_SPREAD_SONGS && sd > 0.0 {
            sd
        } else {
            f64::NAN
        }
    })
}

fn r3(x: f64) -> Value {
    if x.is_finite() {
        json!((x * 1000.0).round() / 1000.0)
    } else {
        Value::Null
    }
}

fn r4(x: f64) -> Value {
    if x.is_finite() {
        json!((x * 10000.0).round() / 10000.0)
    } else {
        Value::Null
    }
}

/// A number, or null when it isn't finite (full precision: the verdict recomputes from these).
fn fin(x: f64) -> Value {
    if x.is_finite() { json!(x) } else { Value::Null }
}

/// A term's cost from its per-song gaps |synth − singer|: the mean gap in units, less one unit.
/// NaN with too few songs or no unit (`f64::max` would turn that into a cost of 0).
fn term_cost(gaps: &[f64], unit: f64) -> f64 {
    let z = if gaps.len() >= MIN_GAP_SONGS {
        mean(gaps) / unit
    } else {
        f64::NAN
    };
    if z.is_finite() {
        (z - 1.0).max(0.0)
    } else {
        f64::NAN
    }
}

/// Score and guards of a set of songs, with the spreads of the training songs; with `notes`,
/// every note's measures too, for a later verdict to compare against.
fn score_set(songs: &[&SongScore], spread: &[f64; N_TERMS], notes: bool) -> Value {
    let all_pairs: Vec<[(f64, f64); N_TERMS]> = songs.iter().map(|s| s.pairs()).collect();
    let mut terms = Vec::new();
    let (mut cost, mut weight) = (0.0, 0.0);
    for (k, (name, w)) in TERMS.iter().enumerate() {
        let pairs: Vec<(f64, f64)> = all_pairs
            .iter()
            .map(|p| p[k])
            .filter(|(h, s)| h.is_finite() && s.is_finite())
            .collect();
        let gaps: Vec<f64> = pairs.iter().map(|(h, s)| (s - h).abs()).collect();
        let gap = if gaps.len() >= MIN_GAP_SONGS {
            mean(&gaps)
        } else {
            f64::NAN
        };
        let excess = term_cost(&gaps, spread[k]);
        if excess.is_finite() {
            cost += w * excess;
            weight += w;
        }
        terms.push(json!({
            "name": name, "weight": r3(*w), "songs": pairs.len(),
            "singer": r4(median(&pairs.iter().map(|p| p.0).collect::<Vec<_>>())),
            "synth": r4(median(&pairs.iter().map(|p| p.1).collect::<Vec<_>>())),
            "gap": r4(gap), "spread": r4(spread[k]), "z": r3(gap / spread[k]), "cost": r3(excess),
        }));
    }
    let avg = |f: &dyn Fn(&SongScore) -> f64| mean(&songs.iter().map(|s| f(s)).collect::<Vec<_>>());
    let mut out = json!({
        "songs": songs.iter().map(|s| s.id.clone()).collect::<Vec<_>>(),
        "score": r3(if weight > 0.0 { cost / weight } else { f64::NAN }),
        "terms": terms,
        "guards": {
            "notes": songs.iter().map(|s| s.aligned.len()).sum::<usize>(),
            "notesMisaligned": songs.iter().map(|s| s.misaligned()).sum::<usize>(),
            "droppedGestures": songs.iter().map(|s| s.dropped).sum::<u64>(),
            "noteMcdDb": r3(avg(&|s| s.mcd)),
            "envelopeCorr": r3(avg(&|s| s.env_corr)),
            "synthPitchAbsCents": r3(avg(&|s| s.synth_pitch(&|_| true))),
        },
    });
    if notes {
        let rows = |r: &[NoteRow]| {
            r.iter()
                .map(|n| n.iter().map(|v| fin(*v)).collect::<Vec<_>>())
                .collect::<Vec<_>>()
        };
        out["notes"] = songs
            .iter()
            .map(|s| {
                (
                    s.id.clone(),
                    json!({
                        "starts": s.starts, "aligned": s.aligned,
                        "singer": rows(&s.singer), "synth": rows(&s.synth),
                        "dropped": s.dropped, "mcd": fin(s.mcd), "envCorr": fin(s.env_corr),
                    }),
                )
            })
            .collect::<serde_json::Map<_, _>>()
            .into();
    }
    out
}

/// A song back from `score.json`'s notes.
fn song_from(id: &str, v: &Value) -> Result<SongScore, String> {
    let bad = || format!("score.json: no notes for {id}; rescore the baseline with this build");
    let rows = |side: &str| -> Result<Vec<NoteRow>, String> {
        v[side]
            .as_array()
            .ok_or_else(bad)?
            .iter()
            .map(|r| {
                let r = r.as_array().filter(|r| r.len() == N_NOTE).ok_or_else(bad)?;
                Ok(std::array::from_fn(|k| num(&r[k])))
            })
            .collect()
    };
    let starts: Vec<u8> = v["starts"]
        .as_array()
        .ok_or_else(bad)?
        .iter()
        .map(|x| x.as_u64().unwrap_or(0) as u8)
        .collect();
    let aligned: Vec<bool> = v["aligned"]
        .as_array()
        .ok_or_else(bad)?
        .iter()
        .map(|x| x.as_bool().unwrap_or(false))
        .collect();
    let (singer, synth) = (rows("singer")?, rows("synth")?);
    if singer.len() != starts.len() || synth.len() != starts.len() || aligned.len() != starts.len()
    {
        return Err(bad());
    }
    Ok(SongScore {
        id: id.to_string(),
        starts,
        aligned,
        singer,
        synth,
        dropped: v["dropped"].as_u64().unwrap_or(0),
        mcd: num(&v["mcd"]),
        env_corr: num(&v["envCorr"]),
    })
}

#[derive(Debug, PartialEq)]
pub enum Verdict {
    Better,
    Worse,
    Same,
}

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap_or(f64::NAN)
}

/// The verdict on a change in score (like for like) once the guards are checked: the change is
/// taken in thousandths, so 1.001 → 0.991 is a fall of exactly 0.010.
fn decide(change: f64, mut why: Vec<String>) -> (Verdict, Vec<String>) {
    let milli = |x: f64| (x * 1000.0).round() as i64;
    let (d, margin) = (milli(change), milli(MARGIN));
    if !why.is_empty() {
        (Verdict::Worse, why)
    } else if d <= -margin {
        (Verdict::Better, why)
    } else if d >= margin {
        why.push(format!("score up by {:.3}", change));
        (Verdict::Worse, why)
    } else {
        (Verdict::Same, why)
    }
}

/// The training score of `new` against `old` (both as written to `score.json`): the verdict,
/// why when worse, and the change in score, like for like.
pub fn verdict(old: &Value, new: &Value) -> Result<(Verdict, Vec<String>, f64), String> {
    let (o, n) = (&old["train"], &new["train"]);
    if o["songs"] != n["songs"] || old["body"] != new["body"] {
        return Err("the earlier score covers other songs or another body; score both runs with the same --songs and --body".into());
    }
    if old["units"] != new["units"] {
        return Err(
            "the singer's spreads differ between the runs; rescore the baseline with this build"
                .into(),
        );
    }
    let ids: Vec<&str> = o["songs"]
        .as_array()
        .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();
    let mut why = Vec::new();
    // run-level guards: a guard that had a value and lost it breaks
    let mut guard = |k: &str, broken: &dyn Fn(f64, f64) -> bool, what: &str| {
        let (a, b) = (num(&o["guards"][k]), num(&n["guards"][k]));
        if a.is_finite() && !b.is_finite() {
            why.push(format!("{what}: no value any more"));
        } else if a.is_finite() && broken(a, b) {
            why.push(format!("{what}: {a} → {b}"));
        }
    };
    guard("notesMisaligned", &|a, b| b > a, "more notes off time");
    guard("droppedGestures", &|a, b| b > a, "more dropped gestures");
    guard("noteMcdDb", &|a, b| b > a + MCD_SLACK, "note MCD up (dB)");
    guard(
        "envelopeCorr",
        &|a, b| b < a - CORR_SLACK,
        "envelope correlation down",
    );
    // the pitch error's rise is judged like for like below; here only its loss
    guard("synthPitchAbsCents", &|_, _| false, "pitch error");
    let empty = vec![];
    let (to, tn) = (
        o["terms"].as_array().unwrap_or(&empty),
        n["terms"].as_array().unwrap_or(&empty),
    );
    if to.len() != N_TERMS
        || tn.len() != N_TERMS
        || to.iter().zip(tn).any(|(a, b)| a["name"] != b["name"])
    {
        return Err(
            "the earlier score has other terms; rescore the baseline with this build".into(),
        );
    }
    // like for like, note by note
    let mut gaps: Vec<(Vec<f64>, Vec<f64>)> = vec![(Vec::new(), Vec::new()); N_TERMS];
    let (mut pitch_o, mut pitch_n) = (Vec::new(), Vec::new());
    for id in &ids {
        let (so, sn) = (
            song_from(id, &o["notes"][id])?,
            song_from(id, &n["notes"][id])?,
        );
        if so.starts != sn.starts {
            return Err(format!("{id}: the two runs have different notes"));
        }
        let both = |i: usize| so.aligned[i] && sn.aligned[i];
        for (k, (go, gn)) in gaps.iter_mut().enumerate() {
            let synth = |i: usize| {
                both(i)
                    && gives(k, &so.synth[i], so.starts[i])
                    && gives(k, &sn.synth[i], sn.starts[i])
            };
            let h = song_value(k, &sn.singer, &sn.starts, &both);
            let (vo, vn) = (
                song_value(k, &so.synth, &so.starts, &synth),
                song_value(k, &sn.synth, &sn.starts, &synth),
            );
            if [h, vo, vn].iter().all(|v| v.is_finite()) {
                go.push((vo - h).abs());
                gn.push((vn - h).abs());
            }
        }
        let pitched =
            |i: usize| both(i) && so.synth[i][CENTS].is_finite() && sn.synth[i][CENTS].is_finite();
        pitch_o.push(so.synth_pitch(&pitched));
        pitch_n.push(sn.synth_pitch(&pitched));
    }
    let (po, pn) = (mean(&pitch_o), mean(&pitch_n));
    if pn > po + PITCH_SLACK {
        why.push(format!("pitch error up (cents): {po:.2} → {pn:.2}"));
    }
    // every term the earlier score counted keeps its weight: a term with no notes in common
    // changes by nothing, rather than leaving the denominator and magnifying the rest; a term
    // that has no value at all any more breaks
    let (mut change, mut weight) = (0.0, 0.0);
    for (k, (name, w)) in TERMS.iter().enumerate() {
        if !num(&to[k]["cost"]).is_finite() {
            continue;
        }
        weight += w;
        if !num(&tn[k]["cost"]).is_finite() {
            why.push(format!("{name}: no value any more"));
            continue;
        }
        let unit = num(&old["units"][k]);
        let (co, cn) = (term_cost(&gaps[k].0, unit), term_cost(&gaps[k].1, unit));
        if co.is_finite() && cn.is_finite() {
            if cn > co + TERM_SLACK {
                why.push(format!("{name} cost up: {co:.3} → {cn:.3}"));
            }
            change += w * (cn - co);
        }
    }
    if weight == 0.0 {
        return Err("no term is measured in both runs".into());
    }
    let (v, why) = decide(change / weight, why);
    Ok((v, why, change / weight))
}

/// `score.json` and the report's score section for these songs, with the verdict against an
/// earlier `score.json` if one is given.
pub fn report(
    body: &str,
    songs: &[&SongScore],
    against: Option<&str>,
) -> Result<(Value, String, Option<String>), String> {
    let (held, train): (Vec<&SongScore>, Vec<&SongScore>) =
        songs.iter().copied().partition(|s| held_out(&s.id));
    if train.is_empty() {
        if against.is_some() {
            return Err("--against: this run has no training song to score".into());
        }
        let md = "## Score\n\nNo score: the run has no training song to take the singer's spreads from.\n\n";
        return Ok((json!({ "body": body, "train": null }), md.to_string(), None));
    }
    let spread = spreads(&train);
    let mut out = json!({
        "body": body,
        "units": spread.iter().map(|u| fin(*u)).collect::<Vec<_>>(),
        "train": score_set(&train, &spread, true),
    });
    if !held.is_empty() {
        out["heldout"] = score_set(&held, &spread, false);
    }
    let t = &out["train"];
    let fmt = |v: &Value, d: usize| v.as_f64().map_or("—".to_string(), |x| format!("{x:.d$}"));
    let mut md = format!(
        "## Score\n\nTraining songs ({}): **{}**. Held out ({}): {}. Lower is better; 0 means \
         every term is within the singer's own song-to-song spread (see docs/csd-comparison.md).\n\n\
         | term | weight | singer | synth | mean gap | her spread | gap/spread | cost | held-out cost |\n\
         |---|---|---|---|---|---|---|---|---|\n",
        train.len(),
        fmt(&t["score"], 3),
        held.len(),
        out.get("heldout")
            .map_or("—".to_string(), |h| fmt(&h["score"], 3)),
    );
    for (k, term) in t["terms"].as_array().unwrap().iter().enumerate() {
        let held_cost = out
            .get("heldout")
            .map_or("—".to_string(), |h| fmt(&h["terms"][k]["cost"], 2));
        md += &format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {held_cost} |\n",
            term["name"].as_str().unwrap_or("?"),
            fmt(&term["weight"], 2),
            fmt(&term["singer"], 3),
            fmt(&term["synth"], 3),
            fmt(&term["gap"], 3),
            fmt(&term["spread"], 3),
            fmt(&term["z"], 2),
            fmt(&term["cost"], 2),
        );
    }
    let g = &t["guards"];
    md += &format!(
        "\nGuards (training songs): {} of {} notes off time, {} dropped gestures, note MCD {} dB, \
         envelope correlation {}, synth pitch error {} cents.\n\n",
        g["notesMisaligned"],
        g["notes"],
        g["droppedGestures"],
        fmt(&g["noteMcdDb"], 3),
        fmt(&g["envelopeCorr"], 3),
        fmt(&g["synthPitchAbsCents"], 2),
    );
    let mut line = None;
    if let Some(path) = against {
        let old: Value = serde_json::from_str(
            &std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?,
        )
        .map_err(|e| format!("{path}: {e}"))?;
        let (v, why, change) = verdict(&old, &out)?;
        let word = match v {
            Verdict::Better => "BETTER",
            Verdict::Worse => "WORSE",
            Verdict::Same => "NO CLEAR CHANGE",
        };
        let text = format!(
            "Verdict against {path}: {word} (training score {} → {}, {change:+.3} like for like){}",
            fmt(&old["train"]["score"], 3),
            fmt(&t["score"], 3),
            if why.is_empty() {
                String::new()
            } else {
                format!(": {}", why.join("; "))
            }
        );
        md += &format!("{text}\n\n");
        out["verdict"] =
            json!({ "against": path, "result": word, "change": r3(change), "reasons": why });
        line = Some(text);
    }
    Ok((out, md, line))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A song of ten notes, all compared and long enough for vibrato; the first opens a phrase on
    /// a vowel and the sixth one on a voiceless consonant. Every measure of a voice is its value.
    fn song(id: &str, singer: f64, synth: f64) -> SongScore {
        let row = |v: f64| -> NoteRow {
            let mut r = [v; N_NOTE];
            r[ELIGIBLE] = 1.0;
            r
        };
        let mut starts = vec![0; 10];
        starts[0] = PHRASE_VOWEL;
        starts[5] = PHRASE_VOICELESS;
        SongScore {
            id: id.into(),
            starts,
            aligned: vec![true; 10],
            singer: vec![row(singer); 10],
            synth: vec![row(synth); 10],
            dropped: 0,
            mcd: 9.0,
            env_corr: 0.8,
        }
    }

    fn report_of(songs: &[SongScore]) -> Value {
        let refs: Vec<&SongScore> = songs.iter().collect();
        report("female", &refs, None).unwrap().0
    }

    /// Nine songs, the synth `gap` above her on every measure, then `f` applied to each song.
    fn scored(gap: f64, f: impl Fn(&mut SongScore)) -> Value {
        let mut songs: Vec<SongScore> = (1..=9)
            .map(|i| song(&format!("en{i:03}a"), i as f64, i as f64 + gap))
            .collect();
        songs.iter_mut().for_each(&f);
        report_of(&songs)
    }

    #[test]
    fn every_fifth_song_is_held_out_in_both_keys() {
        assert!(held_out("en005a") && held_out("en005b") && held_out("en050a"));
        assert!(!held_out("en001a") && !held_out("en049b") && !held_out("x"));
    }

    #[test]
    fn a_gap_costs_its_excess_over_one_singer_spread() {
        // eight training songs (en005 and en010 are held out); her values 0 1 2 3 5 6 7 8 have
        // median 4 and MAD 2.5, a spread of 3.7065, so a gap of two spreads costs 1
        let songs: Vec<SongScore> = (1..=10)
            .map(|i| {
                song(
                    &format!("en{i:03}a"),
                    (i - 1) as f64,
                    (i - 1) as f64 + 7.413,
                )
            })
            .collect();
        let out = report_of(&songs);
        let t = &out["train"]["terms"][0];
        assert_eq!(out["train"]["songs"].as_array().unwrap().len(), 8);
        assert!((num(&t["spread"]) - 3.7065).abs() < 1e-3, "{t}");
        assert!((num(&t["z"]) - 2.0).abs() < 1e-3, "{t}");
        assert!((num(&out["train"]["score"]) - 1.0).abs() < 1e-3);
        // two held-out songs are too few to average a gap over
        assert!(out["heldout"]["score"].is_null());
        // within one spread costs nothing
        let close: Vec<SongScore> = (1..=9)
            .map(|i| song(&format!("en{i:03}a"), i as f64, i as f64 + 1.0))
            .collect();
        assert_eq!(num(&report_of(&close)["train"]["score"]), 0.0);
    }

    #[test]
    fn the_margin_is_exact_in_thousandths() {
        // 0.991 − 1.001 is −0.0099999… in floating point, and still a fall of 0.010
        assert_eq!(decide(0.991 - 1.001, vec![]).0, Verdict::Better);
        assert_eq!(decide(1.001 - 0.991, vec![]).0, Verdict::Worse);
        assert_eq!(decide(0.992 - 1.001, vec![]).0, Verdict::Same);
        assert_eq!(decide(-0.5, vec!["a guard".into()]).0, Verdict::Worse);
    }

    #[test]
    fn the_verdict_wants_a_lower_score_and_every_guard_kept() {
        let base = scored(8.0, |_| {});
        let (v, _, change) = verdict(&base, &scored(6.0, |_| {})).unwrap();
        assert_eq!(v, Verdict::Better);
        assert!(change < -0.01, "{change}");
        assert_eq!(
            verdict(&base, &scored(8.0, |_| {})).unwrap().0,
            Verdict::Same
        );
        assert_eq!(
            verdict(&base, &scored(9.0, |_| {})).unwrap().0,
            Verdict::Worse
        );
        // better on the score, but a note leaves its time: worse
        let (v, why, _) = verdict(&base, &scored(6.0, |s| s.aligned[3] = false)).unwrap();
        assert_eq!(v, Verdict::Worse);
        assert!(why[0].contains("off time"), "{why:?}");
        let (v, _, _) = verdict(&base, &scored(6.0, |s| s.mcd = 9.5)).unwrap();
        assert_eq!(v, Verdict::Worse);
        let (v, why, _) = verdict(&base, &scored(6.0, |s| s.mcd = f64::NAN)).unwrap();
        assert_eq!(v, Verdict::Worse);
        assert!(why[0].contains("note MCD up (dB): no value"), "{why:?}");
        // one term much worse while the rest improve: worse
        let one_worse = scored(6.0, |s| s.synth.iter_mut().for_each(|r| r[H1H2] += 30.0));
        let (v, why, _) = verdict(&base, &one_worse).unwrap();
        assert_eq!(v, Verdict::Worse);
        assert!(why.iter().any(|w| w.contains("H1-H2")), "{why:?}");
        // the synth with no pitch at all: worse, though like for like there is nothing to compare
        let unpitched = scored(6.0, |s| {
            s.synth.iter_mut().for_each(|r| r[CENTS] = f64::NAN)
        });
        let (v, why, _) = verdict(&base, &unpitched).unwrap();
        assert_eq!(v, Verdict::Worse);
        assert!(
            why.iter().any(|w| w.contains("pitch error: no value")),
            "{why:?}"
        );
        // the synth further off the score: worse
        let off_key = scored(6.0, |s| s.synth.iter_mut().for_each(|r| r[CENTS] += 10.0));
        let (v, why, _) = verdict(&base, &off_key).unwrap();
        assert_eq!(v, Verdict::Worse);
        assert!(why.iter().any(|w| w.contains("pitch error up")), "{why:?}");
        // different songs are not comparable
        let mut other = scored(6.0, |_| {});
        other["train"]["songs"] = json!(["en001a"]);
        assert!(verdict(&base, &other).is_err());
    }

    #[test]
    fn notes_that_stop_being_measured_take_nothing_with_them() {
        // in en001a, six of ten notes have the synth's H1−H2 60 off: they carry its median
        let mk = |gone: bool| {
            let mut songs: Vec<SongScore> = (1..=9)
                .map(|i| song(&format!("en{i:03}a"), i as f64, i as f64 + 3.7))
                .collect();
            for r in songs[0].synth.iter_mut().take(6) {
                r[H1H2] = if gone { f64::NAN } else { 61.0 };
            }
            report_of(&songs)
        };
        let (base, gone) = (mk(false), mk(true));
        // counted plainly, losing them looks like a gain
        assert!(num(&gone["train"]["score"]) < num(&base["train"]["score"]) - 0.01);
        // like for like, both runs are judged on the four notes both measure
        let (v, _, change) = verdict(&base, &gone).unwrap();
        assert_eq!(v, Verdict::Same);
        assert_eq!(change, 0.0);
    }

    #[test]
    fn a_vibrato_that_stops_reading_as_vibrato_takes_no_gap_with_it() {
        // en001a's vibrato rate is far off; then its modulation leaves the vibrato range on
        // every note, so the synth has no rate there at all
        let mk = |gone: bool| {
            let mut songs: Vec<SongScore> = (1..=9)
                .map(|i| song(&format!("en{i:03}a"), i as f64, i as f64 + 3.7))
                .collect();
            for r in songs[0].synth.iter_mut() {
                (r[DEPTH], r[RATE]) = if gone {
                    (f64::NAN, f64::NAN)
                } else {
                    (4.7, 41.0)
                };
            }
            report_of(&songs)
        };
        let (base, gone) = (mk(false), mk(true));
        assert!(num(&gone["train"]["score"]) < num(&base["train"]["score"]) - 0.01);
        let (v, _, change) = verdict(&base, &gone).unwrap();
        assert_eq!(v, Verdict::Same);
        assert_eq!(change, 0.0);
    }

    #[test]
    fn a_term_with_nothing_in_common_keeps_its_weight() {
        // the synth's vibrato moves from notes 0–4 to notes 5–9, so depth and rate have no note
        // in common; the bands improve. The change is the bands' gain over the earlier score's
        // whole weight (vibrato share counts nowhere: her share is 1 in every song, no spread).
        let mk = |moved: bool, bands: f64| {
            scored(8.0, move |s| {
                for (i, r) in s.synth.iter_mut().enumerate() {
                    if (i < 5) == moved {
                        (r[DEPTH], r[RATE]) = (f64::NAN, f64::NAN);
                    }
                    for b in &mut r[0..5] {
                        *b += bands - 8.0;
                    }
                }
            })
        };
        let (base, moved) = (mk(false, 8.0), mk(true, 6.0));
        let unit = 1.4826 * 2.5;
        let gain = 0.2 * 5.0 * ((6.0 / unit - 1.0) - (8.0 / unit - 1.0));
        let (_, _, change) = verdict(&base, &moved).unwrap();
        assert!((change - gain / (6.0 - 1.0 / 3.0)).abs() < 1e-9, "{change}");
        // and a term that loses every value breaks
        let gone = scored(8.0, |s| {
            s.synth
                .iter_mut()
                .for_each(|r| (r[DEPTH], r[RATE]) = (f64::NAN, f64::NAN))
        });
        let (v, why, _) = verdict(&base, &gone).unwrap();
        assert_eq!(v, Verdict::Worse);
        assert!(
            why.iter().any(|w| w.contains("vibrato depth: no value")),
            "{why:?}"
        );
    }

    #[test]
    fn a_song_reads_back_from_score_json_as_it_was() {
        let mut s = song("en001a", 2.0, 3.0);
        s.synth[4][ONSET] = f64::NAN;
        s.aligned[7] = false;
        let out = report_of(&[s, song("en002a", 1.0, 2.0)]);
        let back = song_from("en001a", &out["train"]["notes"]["en001a"]).unwrap();
        assert!(back.synth[4][ONSET].is_nan());
        assert!(!back.aligned[7]);
        assert_eq!(back.starts[5], PHRASE_VOICELESS);
        assert_eq!(back.singer[0][0], 2.0);
    }
}
