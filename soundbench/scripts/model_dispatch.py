"""Isolated build dispatch; the physical-model dependencies live in a uv process.

Each model has its own stem renderer, run once per build on all of that
model's tracks, with a JSON manifest beside the stems.
"""
from __future__ import annotations
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
# model name -> (renderer, manifest suffix)
RENDERERS = {
    "banjo": (ROOT / "scripts/render_model.py", "models"),
    "fiddle": (ROOT / "scripts/render_fiddle.py", "fiddle"),
    "chimera": (ROOT / "scripts/render_chimera.py", "chimera"),
    "flabiol": (ROOT / "scripts/render_flabiol.py", "flabiol"),
    "xeremia": (ROOT / "scripts/render_xeremia.py", "xeremia"),
    "skinpipe": (ROOT / "scripts/render_skinpipe.py", "skinpipe"),
    "guitar": (ROOT / "scripts/render_guitar.py", "guitar"),
    "mandolin": (ROOT / "scripts/render_folk_strings.py", "mandolin"),
    "dobro": (ROOT / "scripts/render_folk_strings.py", "dobro"),
    "electric-bass": (ROOT / "scripts/render_electric_bass.py", "electric-bass"),
    "tambori": (ROOT / "scripts/render_mallorca_percussion.py", "tambori"),
    "castanyoles": (ROOT / "scripts/render_mallorca_percussion.py", "castanyoles"),
    "ximbomba": (ROOT / "scripts/render_mallorca_percussion.py", "ximbomba"),
    "piano": (ROOT / "scripts/render_piano.py", "piano"),
    "drums": (ROOT / "scripts/render_drums.py", "drums"),
    "organ": (ROOT / "scripts/render_keys.py", "organ"),
    "pipe-organ": (ROOT / "scripts/render_keys.py", "pipe-organ"),
    "electric-piano": (ROOT / "scripts/render_keys.py", "electric-piano"),
    "bass": (ROOT / "scripts/render_bass.py", "bass"),
    "tr808": (ROOT / "scripts/render_drum_machine.py", "tr808"),
    "tr808-bass": (ROOT / "scripts/render_drum_machine.py", "tr808-bass"),
    "tr909": (ROOT / "scripts/render_drum_machine.py", "tr909"),
    "linn": (ROOT / "scripts/render_drum_machine.py", "linn"),
}
# Models that sing a [singing] block; its marks are routed to each by part.
SINGERS = {"chimera"}


def renderer_command(renderer):
    """The command that runs one renderer script on a manifest path.

    A host project that runs renderers in their own environments rebinds this.
    """
    return [sys.executable, "-m", "soundbench.scripts." + renderer.stem]


def vocal_marks(marks, bindings, names, model):
    """Route shared score marks to their selected singer model, without mutation.

    A mark's part is matched among the sung parts only (every SINGERS model, in
    track order), so a numeric part counts singers.
    """
    from soundbench.scripts.instrumentation import matching_tracks
    sung = [i for i in sorted(bindings) if bindings[i].model in SINGERS]
    selected = []
    for mark in marks:
        matches = matching_tracks(mark["part"], sung, names)
        if len(matches) != 1:
            raise ValueError("a singing mark's part must select one sung part")
        if bindings[matches[0]].model == model:
            model_tracks = [i for i in sorted(bindings) if bindings[i].model == model]
            selected.append({**mark, "part": str(model_tracks.index(matches[0]) + 1)})
    return selected


def render_model_stems(performed_midi, score_midi, settings, bindings, output_dir, piece):
    import mido
    tracks = mido.MidiFile(performed_midi).tracks
    jobs = {}
    for ordinal, (index, binding) in enumerate(sorted(bindings.items()), 1):
        if binding.model is None:
            continue
        slug = re.sub(r"[^a-z0-9_-]+", "-", tracks[index].name.casefold()).strip("-") or "track"
        jobs.setdefault(binding.model, []).append(
            {"index": index, "output": str(output_dir / f"{piece}.stem.{ordinal:02d}-{slug}.raw.wav")})
    stems = {}
    for model, model_jobs in jobs.items():
        renderer, suffix = RENDERERS[model]
        manifest = output_dir / f"{piece}.{suffix}.json"
        model_settings = settings
        if model in SINGERS and settings.get("singing", {}).get("marks"):
            names = {i: tracks[i].name for i in bindings}
            model_settings = {**model_settings, "singing": {**settings["singing"],
                "marks": vocal_marks(settings["singing"]["marks"], bindings, names, model)}}
        manifest.write_text(json.dumps({"performed_midi": str(performed_midi),
                                        "score_midi": str(score_midi), "settings": model_settings,
                                        "jobs": model_jobs}, indent=2) + "\n")
        process = subprocess.run([*renderer_command(renderer), str(manifest)],
                                 text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if process.returncode:
            raise ValueError(f"{model} model render failed:\n{process.stdout}{process.stderr}")
        print(process.stdout, end="")
        stems.update(json.loads(manifest.with_suffix(".metrics.json").read_text())["stems"])
    return stems
