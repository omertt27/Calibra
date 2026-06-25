"""
Reader registry — auto-detection and dispatch.

Adapters register themselves at import time via @register. The public entry
point is `load(path)`, which probes registered readers in priority order.
"""

from __future__ import annotations

from calibra.ingestion.base import DatasetReader
from calibra.schema.episode import EpisodeBatch

_READERS: list[type[DatasetReader]] = []


def register(reader_cls: type[DatasetReader]) -> type[DatasetReader]:
    """Class decorator. Call order determines probe priority."""
    _READERS.append(reader_cls)
    return reader_cls


def detect_reader(path: str) -> type[DatasetReader]:
    """Return the first registered reader that claims it can handle `path`."""
    for cls in _READERS:
        if cls.can_read(path):
            return cls
    registered = [cls.__name__ for cls in _READERS]
    raise ValueError(
        f"No reader found for '{path}'.\n"
        f"Registered readers: {registered}\n"
        "Install optional format dependencies: pip install 'calibra[hdf5]', "
        "'calibra[lerobot]', 'calibra[rlds]', or 'calibra[mcap]'."
    )


def load(path: str, reader: DatasetReader | None = None) -> EpisodeBatch:
    """
    Load a dataset into an EpisodeBatch.

    Pass `reader` to bypass auto-detection (useful in tests or when the
    path heuristic is ambiguous).
    """
    if reader is not None:
        return reader.read(path)
    return detect_reader(path)().read(path)


def registered_formats() -> list[str]:
    return [cls().format_name for cls in _READERS]
