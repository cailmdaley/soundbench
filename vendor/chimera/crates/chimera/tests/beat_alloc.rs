//! The beat sequencer runs on the audio thread: once warm, ticking (every sound, pattern edits,
//! tempo changes, rewinds), pad-hit builds and lead-in queries must not allocate.
use chimera::control::beat_player::BeatPlayer;
use chimera::control::score::Gesture;
use chimera::phonetics::beatbox::{BEAT_SOUNDS, BeatStep, MAX_SOUND_GESTURES, parse_beat};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct Counting;

thread_local! {
    static ALLOCS: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.with(|n| n.set(n.get() + 1));
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        ALLOCS.with(|c| c.set(c.get() + 1));
        unsafe { System.realloc(p, l, n) }
    }
}

#[global_allocator]
static A: Counting = Counting;

fn allocs() -> usize {
    ALLOCS.with(|n| n.get())
}

#[test]
fn the_beat_player_does_not_allocate() {
    chimera::phonetics::beatbox::init();
    // prepared off the audio thread
    let all = parse_beat("B t tss Pf | K ^K ! P! | kch m:43> . m");
    let other = parse_beat("m . . kch | t t");
    let mut p = BeatPlayer::new();
    let mut scratch: Vec<Gesture> = Vec::with_capacity(MAX_SOUND_GESTURES);
    let mut scheduled = 0usize;

    let before = allocs();
    p.set_steps(&all);
    p.set_tempo(140.0, 0.3);
    p.start(0.0);
    let mut clock = 0.0;
    for b in 0..2000 {
        if b == 500 {
            p.set_tempo(90.0, 0.0);
        }
        if b == 900 {
            p.set_steps(&other);
        }
        if b == 1300 {
            p.body = chimera::anatomy::human::BodyName::Child.into();
            p.rebase(-clock);
            clock = 0.0;
            p.rewind(clock);
        }
        p.tick(clock, &mut |gs: &[Gesture]| scheduled += gs.len());
        let _ = p.step_at(clock);
        clock += 128.0 / 48000.0;
    }
    for s in &BEAT_SOUNDS {
        let lead = BeatPlayer::lead_in_with(s.id, "female".into(), &mut scratch);
        scratch.clear();
        p.gestures(&BeatStep { note: Some(45.0), accent: Some(1.2), ..BeatStep::new(s.id) }, clock + lead, 0.18, &mut scratch);
        assert!(!scratch.is_empty() && scratch.len() <= MAX_SOUND_GESTURES);
    }
    let n = allocs() - before;
    assert!(scheduled > 150, "{scheduled}");
    assert_eq!(n, 0, "the beat player allocated {n} times");
}
