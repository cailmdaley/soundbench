# Pipe organ: what a score should do

`model = "pipe-organ"` is a sample-free church organ (`soundbench/scripts/models/pipe_organ.py`, presets in `soundbench/instruments/models/pipe-organ.toml`). The Hammond-style `organ` is a different instrument.

## Sidecar

```toml
[instruments]
"Great" = { model = "pipe-organ" }
"Swell" = { model = "pipe-organ" }
"Pedal" = { model = "pipe-organ" }

[pipe_organ_model]
registration = "principal"          # default for tracks not listed below
[pipe_organ_model.tracks]           # MIDI track name (case-insensitive) -> registration
Great = "plenum"
Swell = "flutes"
Pedal = "pedal"
```

One registration per track: for a registration change, move the music to another track. `gain_db` in the `[instruments]` table balances divisions (a four-note chord renders near -22 dBFS RMS at 0 dB).

## Registrations

| name | stops | use |
| --- | --- | --- |
| `flutes` | Gedackt 8' + Flute 4' | accompaniment under voices: stopped, soft, round (odd harmonics) |
| `principal` | Principal 8' + Octave 4' | the open-diapason church sound; hymns, chorale lines |
| `plenum` | Principal 8' 4' 2' + Mixture III (1 1/3', 1', 2/3') | full climax; dark-voiced, the mixture breaks back an octave above about 2.6 kHz |
| `pedal` | Subbass 16' + Octave 8' | the floor; MIDI 36-67 (C2-G4) sounds an octave lower on the 16' |

Add stops or registrations by editing the TOML (`[registration.<name>]` with `[[...stop]]` entries: family, footage in feet, dB).

## Compass

Manuals MIDI 36-96 (C2-C7, 61 keys). Pedal MIDI 36-67 (C-g' in organists' naming). Out-of-range notes raise an error, they are not folded. Equal temperament at A4 = 440; every pipe carries a fixed tuning offset of a few cents (same pipe, same offset, every time), so unison ranks beat gently.

## Dynamics

An organ has no touch dynamics: velocity is ignored. Louder means more stops (move music from `flutes` to `principal` to `plenum`) or an opening swell box.

- **CC11** (0-127, default 127 = open) is the swell box: gain falls by up to 26 dB and the top end closes together. Use it on the Swell track for crescendi and decrescendi; leave it out on the Great and Pedal.
- Notes need a note-off. The pipe stops speaking in 70-150 ms; in a church acoustic the room carries a tail of about 4 s, so leave room for it at the end of a piece.

## Writing for it

- Hold chords; repeated notes re-speak with a small chiff (stronger on flutes). Attacks are legato-soft: harmonics build in over 25-80 ms, so very short staccato notes sound like breaths.
- Give the pedal long notes an octave below the bass voice; the Subbass is nearly pure and carries little on small speakers.
- Under voices keep to `flutes`, and use `principal` for interludes; save `plenum` for the arrival.
- The preset is deliberately dark; doubling at 2' changes that balance.
