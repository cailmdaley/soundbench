# CHIMERA in Soundbench

This is Soundbench's source fork of [CHIMERA](https://github.com/jamesondh/chimera-vocal-synth), pinned to upstream commit `d361a82ce1b18d0ac19001aab19169ff2c1eb9e8`. Jameson Hodge and CHIMERA's contributors retain their MIT copyright; see `LICENSE`. The CMU dictionary and fonts retain the licences beside their files. No Children's Song Dataset recordings or derived audio are included.

Build the native engine with `cargo build --locked --release -p chimera-cli -j 2` in this folder. The Cargo lockfile pins Rust dependencies. The browser interface is upstream source, not part of the studio build.

`chimera-studio input.json output.wav` renders explicit syllables and performed times without asking a second lyric aligner to reinterpret the score. The studio adapter writes that input. Audio is mono 48 kHz float, with no normalisation, EQ or pitch correction. Source controls act inside the physical engine.

A sidecar part chooses it with `model = "chimera"` and keeps its `[voice3.parts]` entry and `[[voice3.marks]]`: the same singer, lyrics, character and marks voice3 reads. Syllables pass from voice3's ARPAbet alignment straight to the engine's IPA inventory, so CHIMERA's own English front end is not used. One singer per part; sections, `machine`, `scatter`, `spread`, `timing_sd_s` and a nonzero `vowel_opening_hz` are rejected rather than ignored.

The expressive render maps the performance into the physical engine:

- **Body.** The singer picks a continuous adult anatomy (`BodyMix` female fraction: bass 0, baritone .15, tenor .30, alto .85, soprano 1). These are provisional casts, not fitted singer identities.
- **Dynamics.** Each note's effort is the character's `effort_center + effort_span·(velocity/127 − ½)`. Effort sets lung pressure (4–12 cmH₂O), fold tension and source brightness; CC11 scales pressure and CC2 opens the folds (breath) and lowers pressure. Controls blend over 40 ms around each vowel beat.
- **Level.** A fixed −24 dB output trim keeps the engine's peak limiter a safety only, so pressure changes loudness as well as timbre. A part's `level_db` is a constant output gain, as in voice3. Each character also stands at its own distance from the microphone: sung level grows about as lung pressure^2.1, so the character's centre effort sets a gain that hears every character at one level for one velocity (intimate about +6 dB, projected about −3 dB). Velocity and CC dynamics still move the level.
- **Vibrato.** `straight`, `emerging` (extent grows with the mark's progress) or `natural`, with the character's rate, extent and delay, on notes of at least half a second, as a glottal source parameter.
- **Onsets and releases.** Soft, breathy, hard and scooped onsets are fold gestures (and a 30-cent scoop) at the phrase's first note or a marked note; clean, held, breath and fall releases reshape the phrase's breath gesture or add a fold tail. Vowel beats and consonant gestures are those of `plan_song`; `consonant_scale` scales consonant durations.
- **Noise.** `[chimera.parts.<part>]` takes `aspiration` (glottal turbulence) and `frication` (consonant turbulence at a constriction and the teeth), each a multiple of the engine's own. A soft, breathy lead with consonants that carry at 3–8 kHz uses about 2 and 5.
- **Variation.** Seeded cycle-to-cycle source variation; roughness is the engine's healthy default (.1, about 0.45% jitter) plus the character's roughness; no pitch drift. A plain note at mid effort is the engine's default source.

`render_chimera.py --baseline` renders the frozen steady-source diagnostic instead: pressure 9 cmH₂O, tension .55, breathiness .15, held-note vibrato 35 cents, drift and roughness off, no character or dynamics.

Run adapter tests with `uv run python -m unittest tests.test_chimera` from the Soundbench root.
Run the native planner and expression tests with `cargo test --locked -p chimera-cli --bin chimera-studio` from this folder.

