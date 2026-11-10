"""Play a physical instrument, inspect a recording, or fit a banjo model."""
import argparse
import importlib
from pathlib import Path
import sys

import soundfile as sf
from .api import Note, RENDERERS, render


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    play = commands.add_parser("play", help="Render a probe or MIDI file to WAV")
    play.add_argument("instrument", choices=sorted(RENDERERS))
    play.add_argument("--midi", type=Path)
    play.add_argument("--notes", type=Path, help="JSON array of Note fields (pitch, start, duration, velocity, lyric)")
    play.add_argument("--settings", type=Path, help="Instrument/articulation TOML settings")
    play.add_argument("--out", type=Path, required=True)
    play.add_argument("--pitch", type=int, default=60, help="Probe pitch when no input is supplied")
    for name, description in [("spectrum", "A-weighted spectrogram"), ("listen", "Audio diagnostics"),
                              ("tempo", "MIDI tempo curve"), ("compare", "Compare audio and a reference"),
                              ("fit-banjo", "Fit a physical banjo to recording notes")]:
        commands.add_parser(name, help=description, add_help=False)
    args, remaining = parser.parse_known_args(argv)
    if args.command != "play":
        module_name = {"spectrum": "spectrogram", "listen": "listen", "tempo": "tempo_curve", "compare": "compare_instrument",
                       "fit-banjo": "models.banjo_fit.fit"}[args.command]
        sys.argv = ["soundbench " + args.command, *remaining]
        return importlib.import_module("soundbench.scripts." + module_name).main()
    if remaining:
        parser.error("unrecognized arguments: " + " ".join(remaining))
    if args.notes and args.midi:
        parser.error("use --notes or --midi, not both")
    import json
    import tomllib
    settings = tomllib.loads(args.settings.read_text()) if args.settings else None
    notes = [Note(**row) for row in json.loads(args.notes.read_text())] if args.notes else None
    if notes is None and args.midi is None:
        lyric = "ah" if args.instrument in {"voice", "voice2", "voice3", "machine-voice"} else None
        notes = [Note(args.pitch, 0, .4, 72, lyric), Note(args.pitch, .6, .4, 80, lyric), Note(args.pitch, 1.2, .7, 76, lyric)]
    audio, sr = render(args.instrument, notes, midi=args.midi, settings=settings)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out, audio, sr, subtype="FLOAT")
    print(f"{args.out}: {len(audio)/sr:.2f} s, {sr} Hz, {audio.shape[1]} channel(s)")
    return 0
