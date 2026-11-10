//! The creature voice's real-time contract: once warm, rendering, control steps, parameter and
//! gate changes, call playback and telemetry allocate nothing.
use chimera::creatures::{CREATURES, CreatureVoice};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct Counting;

thread_local! {
    static ALLOCS: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
        unsafe { System.realloc(p, l, n) }
    }
}

#[global_allocator]
static A: Counting = Counting;

fn allocs() -> usize {
    ALLOCS.with(|c| c.get())
}

#[test]
fn creature_voice_is_send() {
    fn send<T: Send>() {}
    send::<CreatureVoice>();
}

#[test]
fn warm_creature_voice_does_not_allocate() {
    chimera::creatures::init();
    for def in CREATURES.iter() {
        for fs in [48000.0, 96000.0] {
            let mut v = CreatureVoice::new(def, fs, &[(0, def.params[0].default)]);
            let mut tel = v.new_telemetry();
            let mut buf = vec![0.0f32; 4096];
            // warm: one render, a call, a telemetry read
            v.render(&mut buf[..256]);
            v.play_call(0, None, 1.0);
            v.render(&mut buf[..256]);
            v.telemetry(&mut tel);
            let probe = allocs();
            drop(std::hint::black_box(vec![1u8; 16]));
            assert!(allocs() > probe, "the counting allocator counts");
            let before = allocs();
            v.set_gate(1.0);
            for (ci, _) in def.calls.iter().enumerate() {
                v.play_call(ci, Some(def.params[0].default * 1.3), 0.8);
                v.render(&mut buf[..1000]);
                v.set_params(&[(0, def.params[0].max), (def.params.len() - 1, def.params[def.params.len() - 1].min)]);
                v.render(&mut buf[..77]);
                v.telemetry(&mut tel);
                v.stop_call();
                v.render(&mut buf);
            }
            v.play_call_id(def.calls[0].id, None, 1.0);
            v.render(&mut buf);
            v.set_gate(0.0);
            v.render(&mut buf);
            v.telemetry(&mut tel);
            assert_eq!(allocs() - before, 0, "{} at {fs} Hz allocated on the audio path", def.id);
        }
    }
}

#[test]
fn every_call_key_resolves() {
    use chimera::creatures::types::CallTarget;
    for def in CREATURES.iter() {
        for c in &def.calls {
            for k in &c.plan.keys {
                for (t, _) in &k.v {
                    assert!(!matches!(t, CallTarget::Unknown | CallTarget::Event(None)), "{}/{}: unresolved key", def.id, c.id);
                }
            }
        }
    }
}
