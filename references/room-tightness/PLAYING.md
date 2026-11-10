# Composing ensemble tightness

Use the performance sidecar, not MIDI CCs or note velocities.
No new instrument gestures, keyswitches or track-name conventions are needed.

```toml
[room]
tightness = 0.8       # default: rehearsed folk band

[room.players]
piano = ["upper", "lower"]
guitar = ["guitar", "guitarboom"]
bass = "bass"

[[room.passage]]
bars = [21, 28]
tightness = 0.2       # deliberate loose noodling

[[room.passage]]
bars = [29, 44]
tightness = 0.95      # tune head
```

- Range: finite 0–1; 0 means first rehearsal, 1 means long-established band.
  These are musical endpoints, not a measured mapping from experience to precision.
- Piece default: **0.8**.
  **0.5** preserves the base room performance byte-for-byte for a fixed seed/night.
- Passages use inclusive, one-based **unfolded MIDI bars**, like `room.lead`.
  The last matching `room.passage` wins; outside passages the piece setting applies.
- A passage changes noise and timing correction, not the accumulated clock state.
  Start the tight passage a bar before the head if it must arrive settled.
- Player-feel span is bounded to **8 ms at 0.8+**, preserving the ensemble's common feel and the relative push/layback.
  This leaves 7 ms for timing weather within the 15-ms sustained-mean target; it is not a maximum-note clamp.
  Night draws still vary the profile, and accent/velocity patterns remain intact.
- The clock hears groove, strum centroids, melody anticipation and calibrated instrument attacks.
  Followers align to the lead's performed mean while retaining the permitted feel difference.
  Strum widths and within-instrument melodic gestures are not shortened or removed.
- Mean coordination and feel compression fade continuously from zero at 0.5 to full strength at 0.8.
  An intentionally wider piece or passage can use `feel_span_ms = 25`; a passage may override this without changing tightness.
- Velocity, accents, yield/answer and loudness contagion do not change.
  Use score dynamics and the existing feel controls for those.

Values below 0.5 are deliberate excursions, not a humanization preset to sprinkle everywhere.
At 0, the period-walk bound is four times the base bound (12% with the base parameters), so exposed entrances can sound conspicuously unrehearsed.
At 1, motor noise retains a floor and feel remains; the band does not become a quantized machine.
Tightness cannot repair an 85-ms composed strum or an intentionally late note, and it cannot guarantee a maximum onset separation at a silent entrance.

Measured instrument attack references are technical calibration, not another musical looseness control:

```toml
[tracks]
upper = { velocity_scale = 0.88, attack_latency_ms = 10 }
lower = { velocity_scale = 0.82, attack_latency_ms = 10 }
```

The room uses this reference in its heard pulse; it does not add 10 ms to those notes.
It has no effect on a roomless performance or on the byte-exact 0.5 compatibility mode.
The grand's 10-ms reference comes from an isolated attack measurement; vocal attack is not independently calibrated.
Microphone distances remain physical recording geometry.
