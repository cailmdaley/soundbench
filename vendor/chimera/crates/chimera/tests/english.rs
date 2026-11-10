//! Port of tests/english.test.ts.
use chimera::phonetics::Intonation;
use chimera::phonetics::english::{EnglishG2P, letter_to_sound};
use chimera::phonetics::Segment;
use std::io::Read;
use std::sync::OnceLock;

fn g2p() -> &'static EnglishG2P {
    static G: OnceLock<EnglishG2P> = OnceLock::new();
    G.get_or_init(|| {
        let path = format!("{}/../../public/data/cmudict.txt.gz", env!("CARGO_MANIFEST_DIR"));
        let gz = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let mut text = String::new();
        flate2::read::GzDecoder::new(&gz[..]).read_to_string(&mut text).unwrap();
        let mut g = EnglishG2P::new();
        g.load_dict(&text);
        assert!(g.is_loaded());
        g
    })
}
fn ipa(s: &[Segment]) -> String {
    s.iter().map(|x| x.ipa.as_str()).collect()
}

#[test]
fn dictionary_words_with_stress_and_allophones() {
    let p = &g2p().to_phrases("Hello world")[0];
    assert_eq!(ipa(&p.segments), "həloʊwɝɫd");
    assert_eq!(p.segments.iter().find(|s| s.ipa == "oʊ").unwrap().stress, Some(1));
    let w = &g2p().to_phrases("water")[0];
    assert_eq!(ipa(&w.segments), "wɔɾɚ");
}

#[test]
fn punctuation_makes_phrases_and_question_intonation() {
    let ph = g2p().to_phrases("Is it on? Yes, it is.");
    assert_eq!(ph.len(), 3);
    assert_eq!(ph[0].intonation, Intonation::Question);
    assert_eq!(ph[2].intonation, Intonation::Statement);
}

#[test]
fn numbers_are_spelled_out_and_unknown_words_fall_back_to_rules() {
    let p = &g2p().to_phrases("42")[0];
    assert_eq!(p.words, ["forty", "two"]);
    assert!(ipa(&letter_to_sound("zorblat")).chars().count() > 4);
}

#[test]
fn a_phrase_lists_the_words_said_one_per_word_end() {
    // "--" and a lone apostrophe have no sound: they are not words of the phrase
    for text in ["hello -- world", "rock ' roll", "it's - fine"] {
        for p in g2p().to_phrases(text) {
            let ends = p.segments.iter().filter(|s| s.word_end == Some(true)).count();
            assert_eq!(p.words.len(), ends, "{text:?}: {:?}", p.words);
        }
    }
    assert_eq!(g2p().to_phrases("hello -- world")[0].words, ["hello", "world"]);
}

#[test]
fn words_that_are_also_object_property_names_do_not_break_the_front_end() {
    for w in ["constructor", "toString", "valueOf", "hasOwnProperty", "__proto__", "prototype"] {
        let p = &g2p().to_phrases(&format!("The {w} is ready."))[0];
        assert!(p.segments.len() > 5, "{w}");
    }
    assert!(!EnglishG2P::new().is_loaded());
    assert!(EnglishG2P::new().to_phrases("constructor")[0].segments.len() > 3); // no dictionary
}
