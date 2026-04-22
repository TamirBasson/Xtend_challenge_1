"""Simple visualization helpers for Phase 1.

Provides minimal preview utilities used to manually inspect the dataset
before moving on to preprocessing and geometric stages.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt

from .frame_loader import Frame


def _bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert an OpenCV BGR image to RGB for matplotlib display."""
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def show_image(
    image: np.ndarray,
    title: Optional[str] = None,
    figsize: tuple = (8, 6),
) -> None:
    """Display a single BGR image using matplotlib."""
    plt.figure(figsize=figsize)
    plt.imshow(_bgr_to_rgb(image))
    if title:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def show_grid(
    frames: Iterable[Frame],
    max_images: int = 6,
    cols: int = 3,
    figsize_per_cell: tuple = (4, 3),
) -> None:
    """Display up to `max_images` frames arranged in a grid.

    Images are loaded on demand, so passing a large list is safe.
    """
    frames_list = list(frames)[:max_images]
    if not frames_list:
        print("No frames to display.")
        return

    n = len(frames_list)
    cols = max(1, min(cols, n))
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(figsize_per_cell[0] * cols, figsize_per_cell[1] * rows),
    )
    axes = np.atleast_1d(axes).flatten()

    for ax, frame in zip(axes, frames_list):
        image = frame.load_image()
        ax.imshow(_bgr_to_rgb(image))
        ax.set_title(f"[{frame.index}] {frame.name}", fontsize=9)
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()
