//! Port of tests/speech-quality.test.ts: running-speech guards (nasality, Klatt durations,
//! phrases ending in silence) and contact-model continuity.
use chimera::anatomy::human::{BODY_MALE, HumanArtic, HumanVocalTract, NEUTRAL};
use chimera::phonetics::Plan;
use chimera::phonetics::ipa::tokenize_ipa;
use chimera::phonetics::planner::{PlanOptions, plan_utterance};
use chimera::voice::human_voice::HumanVoice;

const FS: f64 = 48000.0;

struct Render {
    plan: Plan,
    out: Vec<f32>,
    port: Vec<f64>,
    voicing: Vec<f64>,
}

impl Render {
    fn frame(&self, t: f64) -> usize {
        ((self.port.len() - 1) as f64).min(((t * FS) / 128.0).round()) as usize
    }
    fn at(&self, t: f64) -> f64 {
        self.port[self.frame(t)]
    }
}

fn render(ipa: &str, tail: f64) -> Render {
    let mut v = HumanVoice::with_defaults(FS);
    let plan = plan_utterance(&tokenize_ipa(ipa), &PlanOptions { body: "male".into(), t0: 0.05, f0: Some(110.0), ..Default::default() });
    v.schedule(&plan.gestures);
    let n = (FS * (plan.end + tail)).round() as usize;
    let mut out = vec![0.0f32; n];
    let (mut port, mut voicing) = (vec![], vec![]);
    let mut i = 0;
    while i < n {
        let m = 128.min(n - i);
        v.render(&mut out[i..i + m]);
        port.push(v.airway.frame.as_ref().map_or(0.0, |f| f.velum_port));
        voicing.push(v.glottis.voicing);
        i += m;
    }
    Render { plan, out, port, voicing }
}

/// RMS level (dB re the utterance's peak) over [t0, t1).
fn level_db(out: &[f32], t0: f64, t1: f64) -> f64 {
    let peak = out.iter().fold(0.0f64, |m, x| m.max(x.abs() as f64));
    let (a, b) = (((t0 * FS).round() as usize).min(out.len()), ((t1 * FS).round() as usize).min(out.len()));
    let e: f64 = out[a..b].iter().map(|x| (*x as f64).powi(2)).sum();
    20.0 * ((e / (b - a).max(1) as f64).sqrt() / peak + 1e-12).log10()
}

#[test]
fn nasality_does_not_spread_into_oral_vowels() {
    // [ɑ n ə b ɑ b ɑ]: the last two vowels have no nasal neighbour (Bunton & Story: audible from ~0.01 cm²)
    let r = render("ˈɑnəbɑbɑ", 0.15);
    let vowels: Vec<_> = r.plan.segments.iter().filter(|s| s.kind == "vowel").collect();
    assert_eq!(vowels.len(), 4);
    for v in &vowels[2..] {
        let mut t = v.t0;
        while t <= v.t1 {
            assert!(r.at(t) < 0.01, "velum port {} at {t}", r.at(t));
            t += 0.005;
        }
    }
    // and the nasal itself is nasal
    let n = r.plan.segments.iter().find(|s| s.ipa == "n").unwrap();
    assert!(r.at((n.t0 + n.t1) / 2.0) > 0.2);
}

#[test]
fn phrase_final_lengthening_and_stress_contrast() {
    let r = render("ˈbɑbəˈbɑb", 0.15);
    let v: Vec<_> = r.plan.segments.iter().filter(|s| s.kind == "vowel").collect();
    let d = |s: &chimera::phonetics::PlannedSegment| s.t1 - s.t0;
    assert!(d(v[2]) / d(v[0]) > 1.25);
    assert!(d(v[0]) / d(v[1]) > 1.4);
}

#[test]
fn a_phrase_ends_in_silence() {
    // no vowel added after a final consonant, no lingering breath
    for ipa in ["ˈbæd", "ˈkæt", "ˈsi", "ˈðɛm"] {
        let r = render(ipa, 0.45);
        let fin = r.plan.segments.last().unwrap();
        let sonorant = fin.kind == "vowel" || fin.kind == "nasal";
        // the folds stop at a final obstruent's release; a final vowel or nasal fades out
        let voiced_for = if sonorant { 0.05 } else { 0.015 };
        for f in r.frame(r.plan.end + voiced_for)..r.voicing.len() {
            assert!(r.voicing[f] < 0.3, "{ipa}: voicing {} at frame {f}", r.voicing[f]);
        }
        assert!(level_db(&r.out, r.plan.end + 0.15, r.plan.end + 0.45) < -50.0, "{ipa}");
    }
}

#[test]
fn the_contact_model_engages_smoothly() {
    // the tongue on its way from a final /d/ back to rest: the body's gap to the roof crosses
    // the contact regime's edge (0.6 cm) at tbY ≈ 6.90
    let base = HumanArtic { jaw: 0.604, tb_x: 2.935, tt_x: 6.272, tt_y: 6.89, lip_ap: 0.703, lip_prot: 0.2, lip_width: 0.5, ..NEUTRAL };
    let mut tract = HumanVocalTract::new(BODY_MALE);
    let (mut ph, mut or) = (vec![0.0; 20], vec![0.0; 24]);
    let mut prev: Option<Vec<f64>> = None;
    let mut worst = 0.0f64;
    let mut y = 6.85;
    while y <= 6.95 {
        let a = HumanArtic { tb_y: y, ..base };
        let f = tract.compute_new(&a);
        tract.resample(&a, &f, &mut ph, &mut or, None, None);
        let cur: Vec<f64> = ph.iter().chain(&or).copied().collect();
        if let Some(p) = &prev {
            for j in 0..cur.len() {
                worst = worst.max((cur[j] - p[j]).abs());
            }
        }
        prev = Some(cur);
        y += 0.001;
    }
    assert!(worst < 0.05, "{worst} cm² per 0.01 mm of tongue travel (was 0.87 at the edge)");
}
