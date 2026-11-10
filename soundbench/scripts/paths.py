"""Where an instrument's presets live, how one is chosen, read and named.

Each instrument has one physical model and a folder of presets,
``presets/<instrument>/``: ``default.toml`` and named settings of the same
model. A preset may begin with ``inherits = "<preset>"`` and then holds only
what differs from that preset. A model's internal data sits in the folder's
``data/``. Recording chains (tape, microphones, rooms) live in ``rooms/``.
"""
import json
from pathlib import Path
import tomllib

PACKAGE = Path(__file__).resolve().parents[1]
PRESETS = PACKAGE / "presets"
ROOMS = PACKAGE / "rooms"


def preset_names(instrument):
    """The bundled presets of an instrument, default first."""
    names = sorted(p.stem for p in (PRESETS / instrument).glob("*.toml"))
    return sorted(names, key=lambda name: name != "default")


def preset_path(instrument, preset=None):
    """A preset's file: a bundled name (``None`` is the default) or a path to one's own.

    A value containing ``/`` or ending in ``.toml`` is a path; a relative path
    is taken from the working directory.
    """
    preset = "default" if preset is None else str(preset)
    if "/" in preset or preset.endswith(".toml"):
        path = Path(preset)
        if not path.exists():
            raise ValueError(f"no preset file {preset}")
        return path.resolve()
    path = PRESETS / instrument / f"{preset}.toml"
    if not path.exists():
        raise ValueError(f"{instrument} has no preset {preset!r}; choose from "
                         f"{', '.join(preset_names(instrument))} or give a .toml path")
    return path


def data_path(instrument, name):
    """One of an instrument's internal data files."""
    return PRESETS / instrument / "data" / name


def merged(base, top):
    out = dict(base)
    for key, value in top.items():
        out[key] = merged(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def read_preset(path):
    """A preset's table, with any inherited preset's values beneath its own.

    ``inherits`` names a preset beside this file (``"default"``) or a bundled
    one by instrument (``"banjo/steel"``).
    """
    path = Path(path)
    table = tomllib.loads(path.read_text())
    base = table.pop("inherits", None)
    if base is None:
        return table
    parent = path.parent / f"{base}.toml"
    if not parent.exists():
        parent = PRESETS / f"{base}.toml"
    return merged(read_preset(parent), table)


def preset_bytes(path):
    """A preset's resolved values as bytes, for a digest that follows its inherited preset."""
    return json.dumps(read_preset(path), sort_keys=True).encode()


def preset_label(path):
    """A preset path relative to the package, or else the working directory."""
    path = Path(path)
    for base in (PACKAGE, Path.cwd()):
        if path.is_relative_to(base):
            return str(path.relative_to(base))
    return str(path)
