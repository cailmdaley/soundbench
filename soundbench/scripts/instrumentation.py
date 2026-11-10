"""Shared MIDI track selection, instrument registry and ensemble routing."""
from __future__ import annotations
from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Any
import mido

# Sampled instruments a project may supply, relative to its working directory.
# The library bundles none of them; the physical models below need no files.
SFZ_INSTRUMENTS = {
    "grand": Path("soundfonts/SalamanderGrandPiano-SFZ+FLAC-V3+20200602/SalamanderGrandPiano-V3+20200602.sfz"),
    "upright": Path("instruments/upright.sfz"),
    "bass": Path("instruments/bass.sfz"),
    "guitar": Path("instruments/guitar.sfz"),
    "banjo": Path("instruments/banjo.sfz"),
    "banjo-thumb": Path("instruments/banjo-thumb.sfz"),
}
# Physical models; a model without a sampled counterpart is its own instrument.
MODELS = frozenset(("banjo", "voice", "fiddle", "voice2", "voice3", "chimera", "machine-voice", "guitar", "mandolin", "dobro", "flabiol", "xeremia", "tambori", "castanyoles", "ximbomba", "piano", "drums", "organ", "pipe-organ", "electric-piano", "electric-bass", "bass", "skinpipe", "tr808", "tr808-bass", "tr909", "linn"))
MODEL_ONLY_INSTRUMENTS = frozenset(("voice", "voice2", "voice3", "chimera", "machine-voice", "flabiol", "xeremia", "tambori", "castanyoles", "ximbomba", "fiddle", "mandolin", "dobro", "drums", "organ", "pipe-organ", "electric-piano", "electric-bass", "skinpipe", "tr808", "tr808-bass", "tr909", "linn"))
ALIASES = {"piano": "grand", "upright-bass": "bass", "meatbass": "bass"}
PIANOS = frozenset(("grand", "upright"))
STEREO_INSTRUMENTS = frozenset(("grand", "upright", "drums", "organ", "pipe-organ", "electric-piano", "voice3", "tr808", "tr808-bass", "tr909", "linn"))
PLAYABLE_KEYS = {
    "bass": frozenset(range(12, 66)) | frozenset(range(72, 80)),
    "guitar": frozenset(range(39, 86)),
    "banjo": frozenset((36, 38, 96, 98, 100, 101, 103, 105)) | frozenset(range(41, 46)) | frozenset(range(48, 94)),
    "banjo-thumb": frozenset(range(65, 70)),
}
# Relative, pre-master gains; center uses -3 dB per channel for mono sources.
MIX_DEFAULTS = {
    "grand": (0.0, 0.15), "upright": (0.0, 0.15),
    "bass": (6.0, 0.0), "guitar": (-6.0, -0.35),
    "fiddle": (0.0, 0.0), "mandolin": (0.0, 0.0), "dobro": (0.0, 0.0),
    "banjo": (2.0, 0.35), "banjo-thumb": (2.0, 0.35),
    "voice": (0.0, 0.0), "voice2": (0.0, 0.0), "voice3": (0.0, 0.0), "chimera": (0.0, 0.0), "machine-voice": (0.0, 0.0), "flabiol": (0.0, -0.2), "xeremia": (0.0, 0.2), "skinpipe": (0.0, -0.1),
    "organ": (0.0, 0.0), "pipe-organ": (0.0, 0.0), "electric-piano": (0.0, 0.0),
    "electric-bass": (0.0, 0.0),
    "tambori": (0.0, 0.0), "castanyoles": (0.0, 0.0), "ximbomba": (0.0, 0.0),
    "drums": (0.0, 0.0),
    "tr808": (0.0, 0.0), "tr808-bass": (0.0, 0.0), "tr909": (0.0, 0.0), "linn": (0.0, 0.0),
}
# Instruments whose sustain mechanism is verified by tests/probes. Others use note-off extension.
CC64_SUSTAIN = frozenset(("grand", "upright", "bass", "guitar", "banjo", "banjo-thumb", "electric-piano"))


def aliases(name: str) -> set[str]:
    name = name.strip().casefold().rstrip(":")
    return {name, *[part.strip() for part in name.split(":") if part.strip()]}


def matching_tracks(selector: Any, note_tracks: list[int], names: dict[int, str]) -> list[int]:
    if isinstance(selector, int) or str(selector).isdigit():
        ordinal = int(selector)
        if not 1 <= ordinal <= len(note_tracks):
            raise ValueError(f"track {selector!r} does not select a note-bearing MIDI track")
        return [note_tracks[ordinal-1]]
    target = str(selector).strip().casefold().rstrip(":")
    exact = [i for i in note_tracks if names.get(i, "").strip().casefold().rstrip(":") == target]
    return exact or [i for i in note_tracks if target in aliases(names.get(i, ""))]


@dataclass(frozen=True)
class Binding:
    instrument: str
    gain_db: float
    pan: float
    model: str | None = None
    eq: tuple[tuple[str, float, float], ...] = ()
    room_send: float = 1.0   # share of the stem sent to the convolution room (its dry share is unchanged)


EQ_TYPES = ("highshelf", "lowshelf", "peak", "lowpass", "highpass")


def parse_eq(selector: str, bands) -> tuple[tuple[str, float, float], ...]:
    """`eq = [{type, hz, db}]`: the instrument's tone in this piece, applied to its raw render.

    Shelves and peaks take `db` (peaks are one octave wide); lowpass/highpass are 12 dB/octave
    and take no `db`.
    """
    if not isinstance(bands, list):
        raise ValueError(f"eq for {selector!r} must be a list of {{type, hz, db}} tables")
    out = []
    for band in bands:
        if not isinstance(band, dict) or band.get("type") not in EQ_TYPES or set(band) - {"type", "hz", "db"}:
            raise ValueError(f"eq band for {selector!r} needs type (one of {EQ_TYPES}), hz and db: {band!r}")
        hz = float(band.get("hz", 0))
        db = float(band.get("db", 0))
        cut = band["type"] in ("lowpass", "highpass")
        if not 20 <= hz <= 20000 or not -30 <= db <= 12 or (cut and "db" in band) or (not cut and "db" not in band):
            raise ValueError(f"eq band for {selector!r}: hz 20..20000, db -30..12 (none for lowpass/highpass): {band!r}")
        out.append((band["type"], hz, db))
    return tuple(out)


def eq_filter(eq) -> str:
    """The ffmpeg filter chain for a binding's eq, or '' when it has none."""
    parts = []
    for kind, hz, db in eq:
        if kind in ("lowpass", "highpass"):
            parts.append(f"{kind}=f={hz:g}:p=2")
        elif kind == "peak":
            parts.append(f"equalizer=f={hz:g}:t=o:w=1:g={db:g}")
        else:
            parts.append(f"{kind}=f={hz:g}:g={db:g}")
    return ",".join(parts)


def resolve_instruments(midi: mido.MidiFile, settings: dict) -> dict[int, Binding]:
    """All note tracks must be mapped; a named banjo thumb inherits its band's banjo mix."""
    note_tracks = [i for i, t in enumerate(midi.tracks)
                   if any(m.type == "note_on" and m.velocity for m in t)]
    names = {i: t.name for i, t in enumerate(midi.tracks)}
    table = settings.get("instruments")
    if table is None:
        return {i: Binding("grand", 0.0, 0.0) for i in note_tracks}
    if not isinstance(table, dict) or not table:
        raise ValueError("instruments must be a nonempty track-to-instrument table")
    bindings = {}
    for selector, value in table.items():
        options = {"instrument": value} if isinstance(value, str) else value
        if not isinstance(options, dict) or not ({"instrument", "model"} & set(options)):
            raise ValueError(f"instrument mapping {selector!r} needs instrument or model = 'name'")
        model = options.get("model")
        if "model" in options and (model not in MODELS or "instrument" in options):
            raise ValueError(f"model mappings need model = one of {sorted(MODELS)}, without instrument")
        unknown = set(options) - {"instrument", "model", "gain_db", "gain", "pan", "eq", "room_send"}
        if unknown:
            raise ValueError(f"unknown instrument options for {selector!r}: {sorted(unknown)}")
        requested = options.get("instrument", model)
        if not isinstance(requested, str):
            raise ValueError(f"instrument name for {selector!r} must be a string")
        instrument = ALIASES.get(requested, requested)
        if instrument in MODEL_ONLY_INSTRUMENTS and model != instrument:
            raise ValueError(f"{instrument!r} exists only as a physical model: use model = {instrument!r}")
        if instrument not in SFZ_INSTRUMENTS and instrument not in MODEL_ONLY_INSTRUMENTS:
            raise ValueError(f"unknown instrument {instrument!r}; choices: {', '.join(SFZ_INSTRUMENTS)}")
        gain, pan = MIX_DEFAULTS[instrument]
        gain = float(options.get("gain_db", options.get("gain", gain)))
        pan = float(options.get("pan", pan))
        if not math.isfinite(gain) or not -60 <= gain <= 24 or not math.isfinite(pan) or not -1 <= pan <= 1:
            raise ValueError(f"invalid gain/pan for {selector!r}: gain -60..24 dB, pan -1..1")
        room_send = float(options.get("room_send", 1.0))
        if not math.isfinite(room_send) or not 0 <= room_send <= 1:
            raise ValueError(f"room_send for {selector!r} must be 0..1")
        matches = matching_tracks(selector, note_tracks, names)
        if not matches:
            raise ValueError(f"instrument track {selector!r} matched no note tracks; names: {names}")
        for index in matches:
            if index in bindings:
                raise ValueError(f"MIDI track {names[index]!r} mapped more than once")
            bindings[index] = Binding(instrument, gain, pan, model, parse_eq(selector, options.get("eq", [])), room_send)
    banjos = set(b for b in bindings.values() if b.instrument == "banjo")
    skinpipes = [b for i, b in bindings.items() if b.model == "skinpipe" and "skinthumb" not in aliases(names[i])]
    for index in note_tracks:
        if index not in bindings and "skinthumb" in aliases(names[index]) and len(skinpipes) == 1:
            bindings[index] = skinpipes[0]
        thumb = "thumb" in aliases(names[index])
        if index not in bindings and thumb and len(banjos) == 1:
            gm_names = [m.name.casefold() for m in midi.tracks[index] if m.type == "instrument_name"]
            if "banjo" in gm_names or next(iter(banjos)).model == "banjo":
                bindings[index] = next(iter(banjos))
        if index not in bindings:
            raise ValueError(f"unmapped note track {names[index]!r} (ordinal {note_tracks.index(index)+1})")
        if thumb and bindings[index].instrument == "banjo":
            bindings[index] = replace(bindings[index], instrument="banjo-thumb")
    return bindings
