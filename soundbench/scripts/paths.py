"""Where a renderer finds a parameter file, and how a report names it."""
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]


def parameters_path(value):
    """Resolve a parameter file named in settings.

    An absolute path is used as given. A relative path is looked up in the
    working directory first, so a project can keep its own presets, then
    among the bundled files (``instruments/models/...``) in this package.
    """
    path = Path(value)
    if path.is_absolute():
        return path
    local = Path.cwd() / path
    return local if local.exists() else PACKAGE / path


def parameters_label(path):
    """A parameter path relative to the package, or else the working directory."""
    path = Path(path)
    for base in (PACKAGE, Path.cwd()):
        if path.is_relative_to(base):
            return str(path.relative_to(base))
    return str(path)
