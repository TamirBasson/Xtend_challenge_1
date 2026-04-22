"""Frame loading utilities for Phase 1.

Responsibilities:
- Enumerate image files from an input folder
- Provide a lightweight `Frame` abstraction (path metadata)
- Allow on-demand decoding via an iterator

Scope is intentionally minimal: no preprocessing, no feature extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, Tuple

import cv2
import numpy as np


DEFAULT_EXTENSIONS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


@dataclass(frozen=True)
class Frame:
    """Metadata for a single input frame.

    The image pixels are NOT stored here; they are loaded on demand via
    `load_image()` to keep memory usage low for large datasets.
    """

    index: int
    name: str
    path: Path

    def load_image(self) -> np.ndarray:
        """Decode and return the frame as a BGR numpy array (OpenCV convention)."""
        image = cv2.imread(str(self.path), cv2.IMREAD_COLOR)
        if image is None:
            raise IOError(f"Failed to read image: {self.path}")
        return image


def load_frames(
    folder: str | Path,
    extensions: Sequence[str] = DEFAULT_EXTENSIONS,
) -> List[Frame]:
    """Return a sorted list of `Frame` objects discovered in `folder`.

    Parameters
    ----------
    folder : str | Path
        Directory containing image files (non-recursive).
    extensions : sequence of str
        Allowed file extensions (case-insensitive), including the leading dot.

    Returns
    -------
    List[Frame]
        Sorted by filename for deterministic ordering.
    """
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise FileNotFoundError(f"Input folder does not exist: {folder_path}")

    allowed = {ext.lower() for ext in extensions}
    files = [
        p for p in folder_path.iterdir()
        if p.is_file() and p.suffix.lower() in allowed
    ]
    files.sort(key=lambda p: p.name)

    return [
        Frame(index=i, name=p.name, path=p)
        for i, p in enumerate(files)
    ]


def iterate_frames(frames: Iterable[Frame]) -> Iterator[Tuple[Frame, np.ndarray]]:
    """Yield `(frame, image)` pairs, decoding each image lazily on demand."""
    for frame in frames:
        yield frame, frame.load_image()
