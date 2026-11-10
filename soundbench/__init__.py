"""Playable physical instruments, audio measurement and recording fits."""
__all__ = ["Note", "render", "presets", "notes_to_midi"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    from . import api
    return getattr(api, name)
