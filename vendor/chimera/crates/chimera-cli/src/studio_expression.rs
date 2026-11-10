//! Native-only performance intent; the browser protocol and SungNote stay unchanged.
use chimera::anatomy::human::BodyMix;
use chimera::control::score::{ControlKey, Gesture, GestureScore, Targets};
use chimera::phonetics::Plan;
use chimera::voice::human_voice::SourcePatch;
use serde::Deserialize;

fn one() -> f64 {
    1.0
}
fn diagnostic() -> bool {
    true
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Phrase {
    pub t0: f64,
    pub notes: Vec<chimera::phonetics::sing::SungNote>,
    pub vibrato: f64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Control {
    pub time: f64,
    pub pressure: f64,
    pub tension: f64,
    pub breathiness: f64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MidiControl {
    pub time: f64,
    pub control: u8,
    pub value: f64,
}
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Intent {
    pub effort_center: f64,
    pub effort_span: f64,
    pub vibrato_rate: f64,
    pub vibrato_extent: f64,
    pub vibrato_delay: f64,
    pub vibrato: String,
    pub onset: String,
    pub attack_ms: f64,
    pub release: String,
    pub breath_tail_ms: f64,
    pub breath_gain: f64,
    pub roughness: f64,
    pub sag_cents: f64,
    pub vowel_opening_hz: f64,
}
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Expression {
    pub time: f64,
    pub off: f64,
    pub velocity: f64,
    pub intent: Intent,
    pub emerging_progress: f64,
    pub marked_onset: bool,
    pub marked_release: bool,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub seconds: f64,
    pub body: BodyMix,
    pub phrases: Vec<Phrase>,
    pub controls: Vec<Control>,
    #[serde(default = "diagnostic")]
    pub baseline: bool,
    #[serde(default)]
    pub singer: String,
    #[serde(default)]
    pub seed: u32,
    #[serde(default = "one")]
    pub consonant_scale: f64,
    #[serde(default)]
    pub level_db: f64,
    #[serde(default)]
    pub tension: f64,
    #[serde(default = "one")]
    pub breathiness: f64,
    #[serde(default = "one")]
    pub vibrato_scale: f64,
    #[serde(default)]
    pub expression: Vec<Expression>,
    #[serde(default)]
    pub midi_controls: Vec<MidiControl>,
}

impl Input {
    pub fn body(&self) -> Result<BodyMix, &'static str> {
        if self.baseline {
            return Ok(self.body);
        }
        // Provisional continuous adult anatomy, not fitted singer identities.
        let female = match self.singer.as_str() {
            "bass" => 0.0,
            "baritone" => 0.15,
            "tenor" => 0.30,
            "alto" => 0.85,
            "soprano" => 1.0,
            _ => return Err("native CHIMERA requires bass/baritone/tenor/alto/soprano"),
        };
        Ok(BodyMix::new(female, 0.0))
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if !self.seconds.is_finite() || self.seconds <= 0.0 || self.controls.is_empty() {
            return Err("positive duration and source controls required");
        }
        self.body()?;
        if !self.baseline {
            if self.expression.is_empty() {
                return Err("native expression requires performed notes");
            }
            for n in &self.expression {
                if n.intent.vowel_opening_hz != 0.0 {
                    return Err(
                        "nonzero vowel_opening_hz is unsupported: no calibrated Hz-to-jaw mapping",
                    );
                }
                if !["straight", "emerging", "natural"].contains(&n.intent.vibrato.as_str())
                    || !["soft", "breathy", "hard", "scooped"].contains(&n.intent.onset.as_str())
                    || !["clean", "held", "breath", "fall"].contains(&n.intent.release.as_str())
                {
                    return Err("unsupported native vibrato/onset/release mode");
                }
            }
            if self
                .midi_controls
                .iter()
                .any(|c| c.control != 2 && c.control != 11)
            {
                return Err("native source MIDI lanes support CC2 and CC11 only");
            }
        }
        Ok(())
    }

    /// Linearly interpolate MIDI lanes; a lane holds its first value before it, as in voice3, and is the default when empty.
    fn cc(&self, cc: u8, t: f64, default: f64) -> f64 {
        let mut lane = self.midi_controls.iter().filter(|c| c.control == cc);
        let Some(first) = lane.next() else {
            return default;
        };
        let (mut at, mut value) = (first.time, first.value / 127.0);
        for c in lane {
            if c.time > t {
                return mix(value, c.value / 127.0, (t - at) / (c.time - at));
            }
            at = c.time;
            value = c.value / 127.0;
        }
        value
    }

    fn note_source(&self, n: &Expression, t: f64) -> SourcePatch {
        let ch = &n.intent;
        let cc11 = self.cc(11, t, 1.0);
        let cc2 = self.cc(2, t, 0.0);
        let effort =
            (ch.effort_center + ch.effort_span * (n.velocity / 127.0 - 0.5)).clamp(0.0, 1.0);
        let pressure = ((4.0 + 8.0 * effort) * cc11 * (1.0 - 0.5 * cc2)).clamp(0.0, 24.0);
        let age = t - n.time;
        let duration = n.off - n.time;
        let extent = match ch.vibrato.as_str() {
            "straight" => 0.0,
            "emerging" => ch.vibrato_extent * n.emerging_progress,
            _ => ch.vibrato_extent,
        };
        let depth = if duration >= 0.5 {
            extent
                * self.vibrato_scale
                * ramp((age - ch.vibrato_delay) / 0.25)
                * (1.0 - ramp((t - n.off + 0.08) / 0.08))
        } else {
            0.0
        };
        // A plain note at mid effort is the engine's default source: pressure 8,
        // tension .5, breathiness .15, roughness .1.
        SourcePatch {
            pressure: Some(pressure),
            tension: Some((0.35 + 0.30 * effort * cc11 + 0.25 * self.tension).clamp(0.0, 1.0)),
            breathiness: Some(
                (0.15 * ch.breath_gain * self.breathiness + 0.75 * cc2).clamp(0.0, 1.0),
            ),
            brightness: Some(0.20 + 0.15 * effort * cc11),
            // The engine's healthy trained voice (roughness .1, ~0.45% jitter) is the
            // plain floor; without it straight tones whine as pure partials.
            roughness: Some(0.10 * (1.0 + ch.roughness)),
            drift: Some(0.0),
            vibrato_depth: Some(depth),
            vibrato_rate: Some(ch.vibrato_rate),
            ..Default::default()
        }
    }

    /// Source controls are interpolated across 40 ms around each vowel beat.
    pub fn source_at(&self, t: f64) -> SourcePatch {
        let k = self
            .expression
            .partition_point(|n| n.time - 0.02 <= t)
            .saturating_sub(1);
        let n = &self.expression[k];
        let mut patch = self.note_source(n, t);
        if k > 0 {
            let a = self.note_source(&self.expression[k - 1], t);
            let f = ramp((t - n.time + 0.02) / 0.04);
            macro_rules! blend {
                ($($field:ident),*) => { $(patch.$field = Some(mix(a.$field.unwrap(), patch.$field.unwrap(), f));)* };
            }
            blend!(
                pressure,
                tension,
                breathiness,
                brightness,
                roughness,
                vibrato_depth,
                vibrato_rate
            );
        }
        patch
    }
}

/// Sung level grows about as lung pressure^2.1 (measured on held vowels).
const LEVEL_EXPONENT: f64 = 2.1;

impl Input {
    /// Microphone placement for the note's character, in dB: a close character
    /// stands nearer, so every character is heard at one level for one velocity
    /// while keeping its sound. Velocity and CC dynamics are untouched. Blended
    /// over the same 40 ms as the source.
    pub fn mic_db(&self, t: f64) -> f64 {
        let at = |n: &Expression| {
            let center = 4.0 + 8.0 * n.intent.effort_center.clamp(0.0, 1.0);
            20.0 * LEVEL_EXPONENT * (8.0 / center).log10()
        };
        let k = self
            .expression
            .partition_point(|n| n.time - 0.02 <= t)
            .saturating_sub(1);
        let here = at(&self.expression[k]);
        if k == 0 {
            return here;
        }
        let f = ramp((t - self.expression[k].time + 0.02) / 0.04);
        mix(at(&self.expression[k - 1]), here, f)
    }
}

fn mix(a: f64, b: f64, f: f64) -> f64 {
    a + f.clamp(0.0, 1.0) * (b - a)
}
fn ramp(x: f64) -> f64 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

/// Change only breath/fold gestures. Segment positions and oral gestures remain
/// exactly those of plan_song, including onsets, diphthongs and deferred codas.
pub fn express_plan(plan: &mut Plan, notes: &[Expression]) {
    let breaths: Vec<_> = plan
        .gestures
        .iter()
        .filter(|g| g.label_str() == "breath")
        .copied()
        .collect();
    for breath in breaths {
        let group: Vec<_> = notes
            .iter()
            .filter(|n| n.time >= breath.t0 - 0.001 && n.time < breath.t1)
            .collect();
        let (Some(first), Some(last)) = (group.first(), group.last()) else {
            continue;
        };
        for g in &mut plan.gestures {
            if g.label_str() == "breath" && g.t0 == breath.t0 {
                g.ramp_in = first.intent.attack_ms / 1000.0
                    * if first.intent.onset == "hard" {
                        0.5
                    } else {
                        1.0
                    };
                g.ramp_out = match last.intent.release.as_str() {
                    "breath" => last.intent.breath_tail_ms / 1000.0,
                    "clean" => 0.03,
                    _ => 0.05,
                };
                // Keep the drive through final consonants; no added voiced tail.
                if last.intent.release == "held" {
                    g.t1 = last.off;
                }
            }
            if g.label_str() == "exhale" && g.t0 >= breath.t1 && g.t0 < breath.t1 + 0.1 {
                if last.intent.release == "held" {
                    g.t0 = g.t0.max(last.off + 0.02);
                    g.t1 = g.t1.max(g.t0);
                } else if last.intent.release == "breath" {
                    g.t1 = g.t1.max(breath.t1 + last.intent.breath_tail_ms / 1000.0);
                }
            }
        }
        for n in &group {
            if n.time == first.time || n.marked_onset {
                let attack = n.intent.attack_ms / 1000.0;
                let (key, target) = match n.intent.onset.as_str() {
                    "hard" => (ControlKey::CLOSURE, 0.25),
                    "breathy" => (ControlKey::ABDUCTION, 0.06),
                    _ => (ControlKey::ABDUCTION, 0.015),
                };
                // This small fold gesture ends at the vowel; consonant gestures
                // retain their stronger closure/abduction targets.
                plan.gestures.push(
                    Gesture::new(
                        n.time - 0.015,
                        n.time,
                        attack,
                        if n.intent.onset == "hard" {
                            0.012
                        } else {
                            attack * 0.5
                        },
                        2.0,
                        Targets::default().with(key, target),
                    )
                    .label("studio onset"),
                );
            }
            if n.marked_release && n.time != last.time && n.intent.release != "held" {
                // The note boundary caps a marked tail; a legato boundary gets
                // a brief fold rearticulation, not a cut in the oral/consonant plan.
                let next = notes
                    .iter()
                    .find(|next| next.time > n.time)
                    .map_or(n.off + 0.2, |next| next.time);
                let at = n.off.min(next);
                let end = (at + n.intent.breath_tail_ms / 1000.0).min(next + 0.005);
                if end > at {
                    let target = if n.intent.release == "breath" {
                        0.12
                    } else {
                        0.04
                    };
                    plan.gestures.push(
                        Gesture::new(
                            at,
                            end,
                            0.005,
                            0.005,
                            3.0,
                            Targets::default().with(ControlKey::ABDUCTION, target),
                        )
                        .label("studio release"),
                    );
                }
            }
        }
    }
}

/// Intentional scoops/falls are source pitch gestures, not pitch correction.
/// They settle to the written pitch; no wandering center or transposition.
pub fn pitch_offset(
    notes: &[Expression],
    t: f64,
    phrase_starts: &[f64],
    phrase_ends: &[f64],
) -> f64 {
    let mut cents = 0.0;
    for n in notes {
        if (n.marked_onset || phrase_starts.contains(&n.time))
            && n.intent.onset == "scooped"
            && t >= n.time - 0.04
            && t < n.time + 0.09
        {
            cents -= 30.0 * (1.0 - ramp((t - n.time) / 0.09));
        }
        if (n.marked_release || phrase_ends.contains(&n.off))
            && t >= n.off - 0.10
            && t < n.off + 0.15
        {
            let fall = if n.intent.release == "fall" {
                50.0
            } else {
                0.0
            };
            cents -= (fall + n.intent.sag_cents) * ramp((t - n.off + 0.10) / 0.15);
        }
    }
    cents
}

/// Split native pitch gestures from articulation. Feeding their blended pitch
/// to the source avoids HumanVoice's fixed-base-pitch bias on wide intervals.
pub fn split_pitch(plan: &mut Plan, pitch: &mut GestureScore) {
    plan.gestures.retain(|g| {
        if g.targets.has(ControlKey::F0) {
            pitch.add(*g);
            return false;
        }
        !g.targets.has(ControlKey::VIBRATO)
    });
}

pub fn phrase_edges(plan: &Plan, notes: &[Expression]) -> (Vec<f64>, Vec<f64>) {
    let mut starts = Vec::new();
    let mut ends = Vec::new();
    for g in plan.gestures.iter().filter(|g| g.label_str() == "breath") {
        let group: Vec<_> = notes
            .iter()
            .filter(|n| n.time >= g.t0 - 0.001 && n.time < g.t1)
            .collect();
        if let (Some(first), Some(last)) = (group.first(), group.last()) {
            starts.push(first.time);
            ends.push(last.off);
        }
    }
    (starts, ends)
}
