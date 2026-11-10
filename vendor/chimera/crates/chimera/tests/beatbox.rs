//! Port of tests/beatbox.test.ts: beat notation, the sounds, and the beat player.
use chimera::control::beat_player::BeatPlayer;
use chimera::control::score::Gesture;
use chimera::phonetics::beatbox::{BEAT_PRESETS, BEAT_SOUNDS, BeatBuild, BeatStep, format_beat, parse_beat};

struct Hit {
    t: f64,
    #[allow(dead_code)]
    label: String,
}

/// Hit time of each scheduled sound: the planned hit is where its gestures were shifted to.
fn run(p: &mut BeatPlayer, from: f64, to: f64) -> Vec<Hit> {
    let dt = 128.0 / 48000.0;
    let mut batches: Vec<Vec<Gesture>> = Vec::new();
    let mut clock = from;
    while clock < to {
        p.tick(clock, &mut |gs: &[Gesture]| batches.push(gs.to_vec()));
        clock += dt;
    }
    batches
        .iter()
        .map(|gs| {
            let label = gs[0].label_str().to_string();
            let mut r = Vec::new();
            p.gestures(&BeatStep::new(if label == "m" { "m" } else { &label }), 0.0, p.step_dur(), &mut r);
            Hit { t: gs[0].t0 - r[0].t0, label }
        })
        .collect()
}

fn ids(s: &[Option<BeatStep>]) -> Vec<Option<String>> {
    s.iter().map(|x| x.map(|x| x.id.as_str().to_string())).collect()
}

#[test]
fn parses_symbols_rests_notes_accents_and_bar_lines() {
    let s = parse_beat("B . t | Pf m:43 ^K> -");
    let want: Vec<Option<String>> = [Some("B"), None, Some("t"), Some("Pf"), Some("m"), Some("^K"), None].iter().map(|x| x.map(String::from)).collect();
    assert_eq!(ids(&s), want);
    assert_eq!(s[4].unwrap().note, Some(43.0));
    assert!(s[5].unwrap().accent.unwrap() > 1.0);
}

#[test]
fn format_and_parse_round_trip_every_preset() {
    for p in &BEAT_PRESETS {
        let steps = parse_beat(p.text);
        assert_eq!(steps.len(), 16);
        assert_eq!(parse_beat(&format_beat(&steps)), steps);
    }
}

#[test]
fn every_sound_builds_gestures_that_start_before_and_end_after_its_hit() {
    for s in &BEAT_SOUNDS {
        for body in ["male", "female", "child"] {
            let gs = s.build_vec(body.into(), &BeatBuild { f0: 82.0, len: 0.15, accent: 1.0 });
            assert!(!gs.is_empty());
            assert!(gs.iter().map(|g| g.t0 - g.ramp_in).fold(f64::INFINITY, f64::min) < 0.0);
            assert!(gs.iter().map(|g| g.t1 + g.ramp_out).fold(f64::NEG_INFINITY, f64::max) > 0.0);
            // everything a sound needs fits inside the sequencer's look-ahead
            assert!(BeatPlayer::lead_in(s.id, body.into()) < BeatPlayer::HORIZON, "{} {body}", s.id);
        }
    }
}

#[test]
fn schedules_each_step_once_on_the_grid_with_swing_on_the_off_beats() {
    let mut p = BeatPlayer::new();
    p.set_steps(&parse_beat("B t B t"));
    p.set_tempo(120.0, 0.5);
    p.start(0.0);
    let hits = run(&mut p, 0.0, 2.2);
    let step = 60.0 / 120.0 / 4.0;
    // 2 s of sixteenths at 120 bpm = 16 steps, starting one horizon in
    assert!(hits.len() >= 15);
    for (n, h) in hits.iter().take(8).enumerate() {
        let want = BeatPlayer::HORIZON + n as f64 * step + if n % 2 == 1 { 0.25 * step } else { 0.0 };
        assert!((h.t - want).abs() < 1e-9);
    }
}

/// `toBeCloseTo(x, 9)`: |a − b| < 5e-10
fn close(a: f64, b: f64, digits: i32) -> bool {
    (a - b).abs() < 10f64.powi(-digits) / 2.0
}

#[test]
fn tempo_changes_keep_the_next_steps_time() {
    let mut p = BeatPlayer::new();
    p.set_steps(&parse_beat("t t t t"));
    p.set_tempo(100.0, 0.0);
    p.start(0.0);
    let before = run(&mut p, 0.0, 0.6);
    let last = before[before.len() - 1].t;
    p.set_tempo(160.0, 0.0);
    let after = run(&mut p, 0.6, 1.2);
    let gap = after[0].t - last;
    // the first step after the change is spaced by at most one old step, then the new tempo
    assert!(gap > 0.0);
    assert!(gap <= 60.0 / 100.0 / 4.0 + 1e-9);
    assert!(close(after[2].t - after[1].t, 60.0 / 160.0 / 4.0, 9));
}

#[test]
fn the_playhead_follows_the_scheduled_hits_swing_included_and_nothing_when_stopped() {
    let mut p = BeatPlayer::new();
    p.set_steps(&parse_beat("B . t ."));
    p.set_tempo(120.0, 0.6);
    assert_eq!(p.step_at(1.0), -1);
    p.start(0.0);
    run(&mut p, 0.0, 1.2);
    let (step, late) = (0.125, 0.3 * 0.125);
    let h = BeatPlayer::HORIZON;
    assert_eq!(p.step_at(h - 0.01), -1);
    assert_eq!(p.step_at(h + 0.01), 0);
    // step 1 is swung: until its hit the playhead stays on step 0
    assert_eq!(p.step_at(h + step + late - 0.005), 0);
    assert_eq!(p.step_at(h + step + late + 0.005), 1);
    assert_eq!(p.step_at(h + 4.0 * step + 0.01), 0);
    p.stop();
    assert_eq!(p.step_at(1.0), -1);
}

#[test]
fn after_the_voice_is_replaced_steps_not_yet_heard_are_scheduled_again_at_their_times() {
    let mut p = BeatPlayer::new();
    p.set_steps(&parse_beat("B t Pf t"));
    p.set_tempo(120.0, 0.3);
    p.start(0.0);
    let first = run(&mut p, 0.0, 0.5);
    // a new voice whose clock restarts at 0 while the old one was at 0.5
    p.rebase(-0.5);
    p.rewind(0.0);
    let again = run(&mut p, 0.0, 0.3);
    let pending: Vec<f64> = first.iter().filter(|h| h.t > 0.5).map(|h| h.t - 0.5).collect();
    assert!(again.len() >= pending.len());
    for (i, t) in pending.iter().enumerate() {
        assert!(close(again[i].t, *t, 9));
    }
}
