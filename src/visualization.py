"""Helpers for drawing results and saving the comparison figures used in the README."""
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_comparison(images, titles, out_path):
    """Save the images (BGR or single-channel) side by side, each with its title."""
    fig, axes = plt.subplots(1, len(images), figsize=(6 * len(images), 3.6))
    for ax, img, title in zip(axes, images, titles):
        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=14)
        ax.axis("off")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
