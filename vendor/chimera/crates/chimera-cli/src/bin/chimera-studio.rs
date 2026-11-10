//! Studio timeline: explicit score syllables, native source controls, float output.
use chimera::control::score::{ControlKey, GestureScore};
use chimera::phonetics::sing::{SongOptions, plan_song, plan_song_with_consonant_scale};
use chimera::voice::human_voice::{HumanVoice, SourcePatch};

#[path = "../studio_expression.rs"]
mod expression;
#[path = "../wav.rs"]
mod wav;
use expression::{Input, express_plan, phrase_edges, pitch_offset, split_pitch};

/// Constant output trim for expressive renders; the limiter stays a safety only.
const STUDIO_GAIN_DB: f64 = -24.0;

fn voice(
    input: &Input,
) -> Result<(HumanVoice, GestureScore, Vec<f64>, Vec<f64>), Box<dyn std::error::Error>> {
    input.validate()?;
    let body = input.body()?;
    let mut v = HumanVoice::new(48000.0, body.body(), 2);
    let first = input
        .phrases
        .first()
        .and_then(|p| p.notes.first())
        .ok_or("no sung notes")?;
    v.set_source(&SourcePatch {
        f0: Some(chimera::math::midi_to_hz(first.midi)),
        voiced: Some(true),
        drift: Some(0.0),
        roughness: Some(0.0),
        ..Default::default()
    });
    if !input.baseline {
        v.glottis.seed_variation(input.seed);
        // Fixed studio headroom keeps the engine's output limiter (0.75 peak) a
        // safety, so pressure changes loudness as well as timbre. level_db is the
        // part's constant level.
        v.output_gain *= 10f64.powf((STUDIO_GAIN_DB + input.level_db) / 20.0);
        let noise = &mut v.airway.noise_gain;
        noise.glottis *= input.aspiration;
        noise.constriction *= input.frication;
        noise.teeth *= input.frication;
    }
    let mut pitch = GestureScore::new();
    let (mut starts, mut ends) = (Vec::new(), Vec::new());
    for phrase in &input.phrases {
        let options = SongOptions {
            body,
            bpm: 60.0,
            t0: phrase.t0,
            vibrato: phrase.vibrato,
        };
        let mut plan = if input.baseline {
            plan_song(&phrase.notes, &options)
        } else {
            plan_song_with_consonant_scale(&phrase.notes, &options, input.consonant_scale)
        };
        if !input.baseline {
            let (a, b) = phrase_edges(&plan, &input.expression);
            starts.extend(a);
            ends.extend(b);
            express_plan(&mut plan, &input.expression);
            split_pitch(&mut plan, &mut pitch);
        }
        v.schedule(&plan.gestures);
    }
    if v.score.dropped > 0 || pitch.dropped > 0 {
        return Err("native gestural score capacity exceeded".into());
    }
    Ok((v, pitch, starts, ends))
}

fn source(
    input: &Input,
    v: &mut HumanVoice,
    pitch: &mut GestureScore,
    starts: &[f64],
    ends: &[f64],
    t: f64,
    k: &mut usize,
) {
    if input.baseline {
        while *k + 1 < input.controls.len() && input.controls[*k + 1].time <= t {
            *k += 1;
        }
        let a = &input.controls[*k];
        let b = input.controls.get(*k + 1).unwrap_or(a);
        let f = if b.time > a.time {
            ((t - a.time) / (b.time - a.time)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let lerp = |x: f64, y: f64| x + f * (y - x);
        v.set_source(&SourcePatch {
            pressure: Some(lerp(a.pressure, b.pressure)),
            tension: Some(lerp(a.tension, b.tension)),
            breathiness: Some(lerp(a.breathiness, b.breathiness)),
            ..Default::default()
        });
    } else {
        let mut patch = input.source_at(t);
        let target = pitch.blend(ControlKey::F0, t);
        if target.weight > 0.0 {
            patch.f0 = Some(
                target.value * 2f64.powf(pitch_offset(&input.expression, t, starts, ends) / 1200.0),
            );
        }
        patch.glide = Some(0.03);
        v.set_source(&patch);
        pitch.prune(t - 0.2);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        return Err("chimera-studio input.json output.wav".into());
    }
    let input: Input = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    let (mut v, mut pitch, starts, ends) = voice(&input)?;
    let mut audio = vec![0f32; (input.seconds * 48000.0).ceil() as usize];
    let mut k = 0;
    let gain = v.output_gain;
    for (block, out) in audio.chunks_mut(128).enumerate() {
        let t = (block * 128) as f64 / 48000.0;
        source(&input, &mut v, &mut pitch, &starts, &ends, t, &mut k);
        if !input.baseline {
            v.output_gain = gain * 10f64.powf(input.mic_db(t) / 20.0);
        }
        v.render(out);
    }
    if audio.iter().any(|x| !x.is_finite()) {
        return Err("non-finite engine output".into());
    }
    wav::write(&args[2], &audio, 48000, true)?;
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chimera::anatomy::human::BodyMix;
    use chimera::phonetics::sing::SungNote;
    #[test]
    fn punctuated_melisma_has_one_coda_and_one_breath() {
        let notes: Vec<SungNote> = serde_json::from_value(serde_json::json!([
            {"midi":60,"beats":1,"syl":{"text":"here.","onset":[{"ipa":"h"}],"nucleus":{"ipa":"ɪ"},"coda":[{"ipa":"ɹ"}],"wordEnd":true,"phraseEnd":false}},
            {"midi":62,"beats":1,"syl":{"text":"here.","onset":[],"nucleus":{"ipa":"ɪ"},"coda":[],"melisma":true,"wordEnd":true,"phraseEnd":false}},
            {"midi":60,"beats":1,"syl":{"text":"here.","onset":[],"nucleus":{"ipa":"ɪ"},"coda":[],"melisma":true,"wordEnd":true,"phraseEnd":true}}
        ])).unwrap();
        let plan = plan_song(
            &notes,
            &SongOptions {
                body: BodyMix::MALE,
                bpm: 60.0,
                t0: 0.4,
                vibrato: 35.0,
            },
        );
        assert_eq!(plan.segments.iter().filter(|s| s.ipa == "ɹ").count(), 1);
        assert_eq!(
            plan.gestures
                .iter()
                .filter(|g| g.label.is_some_and(|label| label.as_str() == "breath"))
                .count(),
            1
        );
        let vowels: Vec<_> = plan.segments.iter().filter(|s| s.ipa == "ɪ").collect();
        assert_eq!(vowels.len(), 3);
        for (v, expected) in vowels.iter().zip([0.4, 1.4, 2.4]) {
            assert!((v.t0 - expected).abs() < 1e-9);
        }
    }
}

#[cfg(test)]
mod expression_tests {
    use super::*;
    use chimera::anatomy::human::{ArticKey, BodyMix};
    use chimera::voice::human_voice::VoiceTelemetry;
    use serde_json::{Value, json};

    fn spec() -> Value {
        json!({"seconds":2.0,"body":"male","baseline":false,"singer":"tenor","seed":7,
            "controls":[{"time":0.0,"pressure":9.0,"tension":0.55,"breathiness":0.15}],
            "phrases":[{"t0":0.4,"vibrato":0.0,"notes":[{"midi":60.0,"beats":1.0,
                "syl":{"text":"a","onset":[],"nucleus":{"ipa":"ɑ"},"coda":[],"wordEnd":true,"phraseEnd":true}}]}],
            "expression":[{"time":0.4,"off":1.4,"velocity":64.0,"emerging_progress":1.0,
                "marked_onset":false,"marked_release":false,"intent":{
                "effort_center":0.5,"effort_span":1.0,"vibrato_rate":6.8,"vibrato_extent":27.0,
                "vibrato_delay":0.2,"vibrato":"natural","onset":"soft","attack_ms":70.0,
                "release":"held","breath_tail_ms":100.0,"breath_gain":1.0,
                "roughness":0.0,"sag_cents":0.0,"vowel_opening_hz":0.0}}]})
    }
    fn input(value: Value) -> Input {
        serde_json::from_value(value).unwrap()
    }
    fn plan(i: &Input) -> chimera::phonetics::Plan {
        plan_song_with_consonant_scale(
            &i.phrases[0].notes,
            &SongOptions {
                body: i.body().unwrap(),
                bpm: 60.0,
                t0: 0.4,
                vibrato: 0.0,
            },
            i.consonant_scale,
        )
    }

    #[test]
    fn continuous_adult_casts_are_native_defaults() {
        let mut s = spec();
        let mut previous = -1.0;
        for singer in ["bass", "baritone", "tenor", "alto", "soprano"] {
            s["singer"] = json!(singer);
            let body = input(s.clone()).body().unwrap();
            assert!(body.female > previous);
            assert_eq!(body.child, 0.0);
            previous = body.female;
        }
        s["baseline"] = json!(true);
        assert_eq!(input(s).body().unwrap(), BodyMix::MALE);
    }

    #[test]
    fn dynamics_and_cc_change_physical_source_not_output_gain() {
        let mut s = spec();
        s["expression"][0]["velocity"] = json!(30);
        let quiet = input(s.clone());
        s["expression"][0]["velocity"] = json!(110);
        let loud = input(s.clone());
        let q = quiet.source_at(0.9);
        let l = loud.source_at(0.9);
        assert!(l.pressure.unwrap() > q.pressure.unwrap());
        assert!(l.tension.unwrap() > q.tension.unwrap());
        let (mut v, _, _, _) = voice(&loud).unwrap();
        let gain = v.output_gain;
        v.set_source(&q);
        let quiet_closure = v.glottis.params.closure_sharpness;
        v.set_source(&l);
        assert!(v.glottis.params.closure_sharpness > quiet_closure);
        v.render(&mut [0.0; 128]);
        let mut telemetry = VoiceTelemetry::default();
        v.telemetry(&mut telemetry);
        assert!(telemetry.p_lung >= 0.0);
        assert_eq!(v.output_gain, gain);
        s["midi_controls"] = json!([
            {"time":0.0,"control":11,"value":127}, {"time":1.0,"control":11,"value":0},
            {"time":0.0,"control":2,"value":0}, {"time":1.0,"control":2,"value":127}]);
        let with_cc = input(s);
        let mid = with_cc.source_at(0.5);
        assert!((mid.pressure.unwrap() / l.pressure.unwrap() - 0.375).abs() < 1e-12);
        assert!(mid.breathiness.unwrap() > l.breathiness.unwrap());
        assert_eq!(with_cc.source_at(1.0).pressure, Some(0.0));
        assert_eq!(v.output_gain, gain);
    }

    #[test]
    fn characters_stand_at_the_mic_for_one_level_per_velocity() {
        let mut s = spec();
        assert!(input(s.clone()).mic_db(0.9).abs() < 1e-12);
        s["expression"][0]["intent"]["effort_center"] = json!(0.22);
        let close = input(s.clone()).mic_db(0.9);
        s["expression"][0]["intent"]["effort_center"] = json!(0.67);
        let far = input(s).mic_db(0.9);
        assert!(close > 5.0 && close < 7.0, "{close}");
        assert!(far < -2.0 && far > -4.0, "{far}");
    }

    #[test]
    fn aspiration_and_frication_scale_the_engine_noise() {
        let mut s = spec();
        let (plain, _, _, _) = voice(&input(s.clone())).unwrap();
        s["aspiration"] = json!(2.0);
        s["frication"] = json!(5.0);
        let (v, _, _, _) = voice(&input(s)).unwrap();
        let (a, b) = (plain.airway.noise_gain, v.airway.noise_gain);
        assert!((b.glottis / a.glottis - 2.0).abs() < 1e-12);
        assert!((b.constriction / a.constriction - 5.0).abs() < 1e-12);
        assert!((b.teeth / a.teeth - 5.0).abs() < 1e-12);
    }

    #[test]
    fn a_cc_lane_holds_its_first_value_before_it() {
        let mut s = spec();
        let open = input(s.clone()).source_at(0.9).pressure.unwrap();
        s["midi_controls"] = json!([{"time":10.0,"control":11,"value":63.5}]);
        let held = input(s).source_at(0.9).pressure.unwrap();
        assert!((held / open - 0.5).abs() < 1e-12);
    }

    #[test]
    fn level_is_constant_output_gain_and_source_is_untouched() {
        let s = spec();
        let base = input(s.clone());
        let mut raised = s;
        raised["level_db"] = json!(12.0);
        let i = input(raised);
        assert_eq!(i.source_at(0.9).pressure, base.source_at(0.9).pressure);
        assert_eq!(i.source_at(0.9).tension, base.source_at(0.9).tension);
        let (a, _, _, _) = voice(&i).unwrap();
        let (b, _, _, _) = voice(&base).unwrap();
        assert!((a.output_gain / b.output_gain - 10f64.powf(0.6)).abs() < 1e-12);
    }

    #[test]
    fn straight_emerging_rate_extent_and_delay_are_source_parameters() {
        let mut s = spec();
        let ch = &mut s["expression"][0]["intent"];
        ch["vibrato"] = json!("straight");
        assert_eq!(input(s.clone()).source_at(0.9).vibrato_depth, Some(0.0));
        s["expression"][0]["intent"]["vibrato"] = json!("emerging");
        let mut last = 0.0;
        for progress in [0.0, 0.25, 0.5, 1.0] {
            s["expression"][0]["emerging_progress"] = json!(progress);
            let i = input(s.clone());
            let patch = i.source_at(0.9);
            assert_eq!(patch.vibrato_depth, Some(27.0 * progress));
            assert!(patch.vibrato_depth.unwrap() >= last);
            last = patch.vibrato_depth.unwrap();
            assert_eq!(i.source_at(0.5).vibrato_depth, Some(0.0));
            assert_eq!(patch.vibrato_rate, Some(6.8));
            let (mut v, _, _, _) = voice(&i).unwrap();
            v.set_source(&patch);
            assert_eq!(v.glottis.params.vibrato_depth, last);
        }
        s["vibrato_scale"] = json!(0.5);
        assert_eq!(input(s).source_at(0.9).vibrato_depth, Some(13.5));
    }

    #[test]
    fn phrase_onset_release_and_note_marks_change_gestures_not_segments() {
        let mut s = spec();
        // Three legato vowels; the second note has explicit attack/release marks.
        let mut n = s["phrases"][0]["notes"][0].clone();
        n["syl"]["phraseEnd"] = json!(false);
        s["phrases"][0]["notes"] = json!([n.clone(), n.clone(), n]);
        let e = s["expression"][0].clone();
        s["expression"] = json!([e.clone(), e.clone(), e]);
        for (k, e) in s["expression"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .enumerate()
        {
            e["time"] = json!(0.4 + k as f64);
            e["off"] = json!(1.4 + k as f64);
        }
        s["expression"][1]["marked_onset"] = json!(true);
        s["expression"][1]["marked_release"] = json!(true);
        s["expression"][1]["intent"]["onset"] = json!("hard");
        s["expression"][1]["intent"]["release"] = json!("breath");
        s["expression"][2]["intent"]["release"] = json!("breath");
        let i = input(s);
        let mut p = plan(&i);
        let segments = p.segments.clone();
        let oral: Vec<_> = p
            .gestures
            .iter()
            .filter(|g| g.targets.has(ArticKey::Jaw))
            .copied()
            .collect();
        express_plan(&mut p, &i.expression);
        assert_eq!(segments, p.segments);
        assert_eq!(
            serde_json::to_value(oral).unwrap(),
            serde_json::to_value(
                p.gestures
                    .iter()
                    .filter(|g| g.targets.has(ArticKey::Jaw))
                    .copied()
                    .collect::<Vec<_>>()
            )
            .unwrap()
        );
        let onset: Vec<_> = p
            .gestures
            .iter()
            .filter(|g| g.label_str() == "studio onset")
            .collect();
        assert_eq!(onset.len(), 2); // phrase start + marked second note
        assert_eq!(onset[1].targets.get(ControlKey::CLOSURE), Some(0.25));
        assert_eq!(
            p.gestures
                .iter()
                .filter(|g| g.label_str() == "studio release")
                .count(),
            1
        );
        assert_eq!(
            p.gestures
                .iter()
                .find(|g| g.label_str() == "breath")
                .unwrap()
                .ramp_out,
            0.1
        );
        for (v, expected) in p
            .segments
            .iter()
            .filter(|s| s.ipa == "ɑ")
            .zip([0.4, 1.4, 2.4])
        {
            assert!((v.t0 - expected).abs() < 1e-9);
        }
    }

    #[test]
    fn consonant_scale_retains_vowel_beats_and_melisma_coda() {
        let mut s = spec();
        s["phrases"][0]["notes"][0]["syl"]["onset"] = json!([{"ipa":"s"}]);
        s["phrases"][0]["notes"][0]["syl"]["coda"] = json!([{"ipa":"m"}]);
        let a = plan(&input(s.clone()));
        s["consonant_scale"] = json!(1.5);
        let b = plan(&input(s));
        let cons = |p: &chimera::phonetics::Plan| {
            p.segments
                .iter()
                .find(|s| s.ipa == "s")
                .map(|s| s.t1 - s.t0)
                .unwrap()
        };
        assert!((cons(&b) / cons(&a) - 1.5).abs() < 1e-9);
        for p in [&a, &b] {
            assert_eq!(p.segments.iter().filter(|s| s.ipa == "m").count(), 1);
            assert!((p.segments.iter().find(|s| s.ipa == "ɑ").unwrap().t0 - 0.4).abs() < 1e-9);
        }
    }

    #[test]
    fn settled_intonation_has_no_fixed_first_note_bias_over_octaves() {
        let mut s = spec();
        s["expression"][0]["intent"]["vibrato"] = json!("straight");
        let mut notes = Vec::new();
        let mut expressions = Vec::new();
        for (k, midi) in [48, 60, 72, 55].iter().enumerate() {
            let mut n = s["phrases"][0]["notes"][0].clone();
            n["midi"] = json!(midi);
            n["syl"]["phraseEnd"] = json!(false);
            notes.push(n);
            let mut e = s["expression"][0].clone();
            e["time"] = json!(0.4 + k as f64);
            e["off"] = json!(1.4 + k as f64);
            expressions.push(e);
        }
        s["phrases"][0]["notes"] = json!(notes);
        s["expression"] = json!(expressions);
        let i = input(s);
        let (mut v, mut pitch, starts, ends) = voice(&i).unwrap();
        let gain = v.output_gain;
        let mut cents = vec![Vec::new(); 4];
        let mut k = 0;
        for block in 0..1650 {
            let t = block as f64 * 128.0 / 48000.0;
            source(&i, &mut v, &mut pitch, &starts, &ends, t, &mut k);
            let mut out = [0.0; 128];
            v.render(&mut out);
            assert!(out.iter().all(|s| s.is_finite()));
            for (j, midi) in [48.0, 60.0, 72.0, 55.0].iter().enumerate() {
                if t > 0.8 + j as f64 && t < 1.2 + j as f64 {
                    cents[j].push(
                        1200.0 * (v.glottis.f0_now / chimera::math::midi_to_hz(*midi)).log2(),
                    );
                }
            }
        }
        for c in cents {
            assert!(!c.is_empty());
            let mean = c.iter().sum::<f64>() / c.len() as f64;
            assert!(mean.abs() < 3.0, "settled source pitch error: {mean} cents");
        }
        assert_eq!(gain, v.output_gain);
    }

    #[test]
    fn unsupported_native_intent_is_not_silent() {
        let mut s = spec();
        s["expression"][0]["intent"]["vowel_opening_hz"] = json!(10);
        assert!(input(s.clone()).validate().is_err());
        s["expression"][0]["intent"]["invented"] = json!(1);
        assert!(serde_json::from_value::<Input>(s).is_err());
    }
}
