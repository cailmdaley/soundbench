# Comparing the synth to a singer: the Children's Song Dataset

`chimera compare csd` sings the English songs of the Children's Song Dataset (CSD) and
measures each render against the recording, note by note.

## The dataset

CSD (Choi, Kim, Park, Yong & Nam, ISMIR 2020; [Zenodo 10.5281/zenodo.4916302](https://zenodo.org/record/4916302), v1.1):

- one professional female singer
- 50 English and 50 Korean children's songs, each recorded in two keys (`a`, `b`)
- 44.1 kHz WAVs, with note onsets, offsets, MIDI pitch and a phonemic syllable for every note

It is licensed CC BY-NC-SA 4.0. Use it for research and tuning only. Don't ship its audio, or
anything derived from it, in the product.

```sh
tools/fetch-csd.sh            # 1.85 GB to ~/datasets/csd; unpacks CSD/english
```

## Running it

```sh
cargo run --release -p chimera-cli -- compare csd                      # the 50 `a` songs, female body, all cores (~2 min)
cargo run --release -p chimera-cli -- compare csd ~/datasets/csd/CSD out/csd --songs en001a,en002a
#   --songs a | b | all | <ids>   --body female|male|child   --jobs n   --float
cargo run --release -p chimera-cli -- compare csd ~/datasets/csd/CSD out/csd-new --against out/csd/score.json
#   score this build and give the verdict against an earlier run's score
```

It writes the following to the output directory (`out/csd` by default):

- `<id>-synth.wav`: the synth's render, on the same timeline as `english/wav/<id>.wav`. You can
  load both into a DAW, or make spectrograms of the pair with `bun tools/spectrogram.ts`.
- `report.md`: the score, medians over songs, spectral balance per vowel, and one line per song.
- `report.json`: every note's measures for the singer and the synth.
- `score.json`: the score, its terms and guards, for the training and held-out songs (below), and
  the verdict when run with `--against`.

## Method

**What the synth sings.** CSD's syllables (`k_y_uu`, `eo_n_d`) are mapped to the engine's IPA
by the table in `crates/chimera-cli/src/compare.rs`. An unmapped symbol is an error. So the synth
sings exactly the phonemes the singer sang, at her note times and her score pitches; G2P is not
under test.

- **Phrases.** A gap of more than 80 ms between notes ends a phrase. Each phrase is one
  `plan_song` call at 60 bpm, so a beat is one second.
- **Melisma.** A vowel-only syllable on the same vowel, starting within 20 ms of the previous
  note, continues that note as a melisma.
- **Settings.** The body, the source settings and the 35-cent vibrato match `render song`.

**Notes the planner can't place on time.** `plan_song` lays a phrase's segments end to end and
gives each vowel at least 50 ms. A note too short for its consonants therefore pushes the
following vowels late. Any note whose planned vowel starts more than 30 ms from its annotated
start is kept in `report.json` (`"aligned": false`) but left out of every statistic, because its
measurement window would hold the wrong sound. The report gives the count.

**What is measured.** Each note is measured over its steady part: from 30 % into the note to
20 % before its end, or from 10 % to 10 % for melisma notes.

| measure | what it tells you |
|---|---|
| pitch vs score, synth vs singer (cents) | intonation; YIN searched ±7 semitones around the score note, so it cannot jump octaves |
| vibrato share, depth (peak cents) and rate (Hz) | over the note's longest unbroken voiced run, if it lasts 0.4 s: the detrended contour counts as vibrato only if its fundamental period (the first autocorrelation peak of 0.5 or more) gives a rate of 4–9 Hz. The share is how many such notes have vibrato |
| aperiodicity | YIN's aperiodicity: noise and roughness in the voice |
| H1−H2 (dB) | first minus second harmonic, the standard voice-quality measure: high is breathy or lax, low is pressed. Measured in each voiced frame at that frame's own f0, then the median is taken |
| voiced share | share of steady frames the synth actually voices; a frame is voiced when YIN finds a period and it is within 35 dB of the signal's loud level |
| voice onset (s) | first voiced frame of a phrase-initial note, relative to the note start, by what the phrase opens on: a vowel, a voiceless consonant (the voicing is the vowel's) or a voiced one (the voicing is the consonant's) |
| band levels, synth − singer (dB) | 0–0.5, 0.5–1, 1–2, 2–4 and 4–8 kHz, each relative to the 0–8 kHz total: spectral tilt and vowel colour, per vowel |
| note MCD (dB) | mel-cepstral distance between the note's averaged log-mel spectra (40 bands, 13 coefficients); on the usual MCD scale, but note-level, not frame-aligned |
| envelope lag and correlation | the shift (within ±0.3 s) that best lines up the two level envelopes: overall timing, and how much the loudness shape matches |

**Caveats.**

- **Notes are the alignment.** There is no phoneme-level timing in CSD. Consonant timing is
  only seen through voice onsets and the envelope.
- **No formants.** At this singer's pitch (C4–C6) the harmonics are 260 Hz or more apart, so LPC
  formants snap to harmonics. That is why the comparison uses band levels and cepstra.
- **Microphone and room.** Level and spectrum compare a recording made with a real microphone
  in a real room against the dry synth. Some of the high-band difference may be the recording
  chain.
- **One singer.** Treat the numbers as directions to tune in, not as targets for every voice.

## The score

`score.json` folds the measures into one number: how far the synth is from the singer, in units of
her own variation. It is what an automated tuning loop climbs (lower is better), so it is built to
be hard to improve without sounding closer to her.

- **Terms.** Each term is a song-level value both voices have: her median H1−H2 in a song against
  the synth's in that song, the two voiced shares, and so on. There are 13 terms: the five band
  levels, H1−H2, vibrato share, depth and rate, voiced share, voice onset of vowel-initial phrases
  and of phrases that open on a voiceless consonant, and aperiodicity.
- **Gap.** A term's gap is the mean over songs of |synth − singer|. It is a mean rather than a
  median so that every song's change counts, and one song gone badly wrong shows.
- **Unit.** The gap is divided by her spread on that term over the training songs (median absolute
  deviation, scaled to a standard deviation). The spread comes from all her notes, including any
  the planner couldn't place on time, so a change to the synth can't move it.
- **Cost.** A gap within one spread is about as close as two of her songs are to each other, so it
  costs nothing. Beyond that, the excess is the term's cost. Nothing is gained by matching her
  medians more closely than her own variation, which keeps a tuning loop from overfitting one
  singer.
- **Score.** The score is the weighted mean of the costs. Each of six groups gets one share:
  spectrum (the five bands), H1−H2, vibrato (its three terms), voicing, onsets (its two terms) and
  noise (aperiodicity). 0 means every term is within her own variation.

**Held-out songs.** Every fifth song (en005, en010, …, en050, in both keys) is held out. The
spreads and the verdict use the other 40 songs only. The held-out songs are scored with the same
spreads, to check that a change carries over to songs it wasn't tuned on.

**Not scored.** Voice onsets of phrases that open on a voiced consonant are reported but not
scored. CSD's note starts mark the start of the syllable, while `plan_song` sings onset consonants
before the beat so that the vowel lands on it. Measured against CSD, those onsets mix that
convention with real timing errors. Pitch is not scored either: the synth sings the score, and her
deviations from it can't be predicted from the score.

**The verdict.** `--against <earlier score.json>` compares the training scores of two runs over
the same songs and body. It compares like for like, note by note. `score.json` keeps every note's
measures on both voices, and the verdict recomputes each song's value of each term for both runs
over the notes on which both runs give that term. A measure that stops being measurable on a
note leaves both runs alike, so it can neither help nor hurt. That covers a stray voiced frame
that comes or goes in a note the synth barely voices, and a vibrato that no longer reads as
vibrato. Not voicing a note still costs, through the voiced share, which every note has.

- **Better:** the like-for-like score falls by at least 0.01 (in thousandths), no term's cost
  rises by more than 0.25, and every guard holds.
- **Guards:**
  - no more notes off time (so a change can't drop hard notes from the statistics)
  - no more dropped gestures
  - note MCD up by no more than 0.1 dB
  - envelope correlation down by no more than 0.01
  - the synth's pitch error up by no more than 2 cents, over the notes both runs pitch
  - a guard that had a value and has none now breaks too.
- **Terms:** a term the earlier score counted keeps its weight in the comparison, so a term
  with no note in common changes by nothing rather than leaving the denominator and magnifying
  the rest. A term that has no value at all any more breaks.
- **Worse:** any guard or term breaks, or the score rises by 0.01 or more.
- **No clear change:** anything else.

## Baseline (2026-11-07, the 50 `a` songs, female body)

All 7,220 notes are compared: the planner now places every vowel within 30 ms of its annotated
start (on 2026-11-05, 188 notes were too short for their consonants and ran late).

**Score: 1.433 on the 40 training songs, 1.590 on the 10 held-out songs.** The costs, largest
first:

| term | singer | synth | cost |
|---|---|---|---|
| level 4–8 kHz (dB re total) | −24.8 | −17.3 | 5.38 |
| aperiodicity | 0.005 | 0.010 | 2.88 |
| H1−H2 (dB) | +3.2 | −0.8 | 2.08 |
| level 2–4 kHz | −18.4 | −14.0 | 1.43 |
| level 0–0.5 kHz | −4.0 | −7.4 | 1.39 |
| onset, vowel-initial (s) | +0.005 | −0.015 | 1.39 |
| onset, voiceless-initial (s) | +0.018 | −0.024 | 1.11 |
| vibrato rate (Hz) | 6.1 | 5.5 | 0.67 |
| vibrato share | 0.64 | 1.00 | 0.62 |
| level 0.5–1 kHz | −6.7 | −6.8 | 0.28 |
| voiced share | 0.93 | 0.88 | 0.26 |
| vibrato depth (peak cents) | 39 | 35 | 0.01 |
| level 1–2 kHz | −15.1 | −13.4 | 0 |

Singer and synth values are medians over the training songs. Note MCD is 9.24 dB and the envelope
correlation 0.80.

What this points at:

1. **The spectrum tilts too little.** The synth has about 7.5 dB too much at 4–8 kHz and 4.4 dB too
   much at 2–4 kHz, and 3.4 dB too little below 500 Hz, for every vowel.
2. **The fundamental doesn't track pitch the way hers does.** H1−H2 on notes ≤ C4 is −5.8 dB
   (singer +4.0), and on notes ≥ G4 it is +8.2 dB (singer +0.1). Her H1−H2 falls about 4 dB from
   her low notes to her high ones; the synth's rises by 14 dB. Points 1 and 2 together sound like
   a source that is too pressed on most notes.
3. **The synth is less periodic than she is.** Its aperiodicity is about twice hers, which points
   at jitter, shimmer or noise in the source.
4. **Voicing starts early at phrase starts.** It starts 20 ms early before a vowel. Before a
   voiceless consonant it starts about 40 ms early, where both voices should only start voicing at
   the vowel. Before /s/, /f/ and /h/ the synth voices 100–130 ms before the note, so the voicing
   runs into the fricative or comes before it.
5. **Vibrato is too uniform.** The synth puts 35-cent vibrato on every long note. She uses it on
   about two thirds of them, a little faster.
6. **Voicing gaps in the steady part of notes.** These are worst in en018a, en041a and en043a
   (0.65–0.70 voiced).
