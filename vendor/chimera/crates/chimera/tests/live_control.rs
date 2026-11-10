//! Port of the tagged-release part of tests/live-control.test.ts: a note released early drops
//! what has not begun and fades what has, never rising again.
use chimera::control::score::{GestureScore, activation};
use chimera::phonetics::sing::{NoteOnOptions, plan_note_on};
use chimera::phonetics::{Segment, Syllable};

#[test]
fn a_released_note_never_rises_again() {
    // /strɑ/ held, released 30 ms in: the vowel and its pitch have not begun yet
    let syl = Syllable { text: "stra".into(), onset: vec![Segment::new("s"), Segment::new("t"), Segment::new("ɹ")], nucleus: Segment { stress: Some(1), ..Segment::new("ɑ") }, coda: vec![], melisma: None, word_end: true, phrase_end: false };
    let gs = plan_note_on(&syl, 57.0, &NoteOnOptions { body: "male".into(), vibrato: 30.0, tag: 4.0, legato_from: None });
    let mut score = GestureScore::new();
    for g in &gs {
        let mut g = *g;
        g.t0 += 0.05;
        g.t1 += 0.05;
        score.add(g);
    }
    let t = 0.08;
    // activation of each gesture before the release. A release only drops gestures (those of the
    // released tags not yet begun) and keeps the order, so the survivors line up with `was`.
    let dropped = |g: &chimera::control::score::Gesture| (g.tag == Some(4.0) || g.tag == Some(5.0)) && !(t > g.t0 - g.ramp_in);
    let was: Vec<f64> = score.gestures().iter().filter(|g| !dropped(g)).map(|g| activation(g, t)).collect();
    score.release(4.0, t);
    score.release(5.0, t);
    assert_eq!(score.size(), was.len());
    for (i, g) in score.gestures().iter().enumerate() {
        assert!((activation(g, t) - was[i]).abs() < 1e-6, "continuous at the release");
    }
    // after the release, no released gesture's activation exceeds what it had at the release
    let mut tt = t;
    while tt < t + 0.5 {
        for (i, g) in score.gestures().iter().enumerate() {
            if (g.tag == Some(4.0) || g.tag == Some(5.0)) && activation(g, tt) > 0.0 {
                assert!(activation(g, tt) <= was[i] + 1e-9, "{:?} rose again at {tt}", g.label);
            }
        }
        tt += 0.002;
    }
    // the vowel (not begun at 80 ms) is gone, and everything is over soon after
    assert_eq!(score.active_at(0.3).count(), 0);
}
