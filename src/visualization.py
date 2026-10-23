"""Helpers for drawing results and saving the comparison figures used in the README."""
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_comparison(images, titles, out_path):
    """Save the images (BGR or single-channel) side by side, each with its title."""
    fig, axes = plt.subplots(1, len(images), figsize=(6 * len(images), 3.6), squeeze=False)
    for ax, img, title in zip(axes[0], images, titles):
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


def draw_lane(undist, left_fit, right_fit, Minv, pixels=None):
    """Draw the lane area between the fits (green) back onto the undistorted camera image.

    The drawing is made in the birds-eye view, where the fits are defined, and warped back with
    Minv. pixels = (leftx, lefty, rightx, righty) additionally colors the detected line pixels,
    left red and right blue.
    """
    h, w = undist.shape[:2]
    ploty = np.arange(h)
    left = np.column_stack([np.polyval(left_fit, ploty), ploty])
    right = np.column_stack([np.polyval(right_fit, ploty), ploty])[::-1]
    area = np.zeros_like(undist)
    cv2.fillPoly(area, [np.vstack([left, right]).astype(np.int32)], (0, 255, 0))
    out = cv2.addWeighted(undist, 1, cv2.warpPerspective(area, Minv, (w, h)), 0.3, 0)

    if pixels is not None:
        leftx, lefty, rightx, righty = pixels
        lines = np.zeros_like(undist)
        lines[lefty, leftx] = (0, 0, 255)
        lines[righty, rightx] = (255, 0, 0)
        lines = cv2.warpPerspective(lines, Minv, (w, h))
        mask = lines.any(axis=2)
        out[mask] = lines[mask]
    return out


def save_histogram(hist, title, out_path):
    """Save a line plot of per-column pixel counts."""
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(hist)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("pixels")
    ax.set_xlim(0, len(hist))
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    from calibration import ROOT, OUT_DIR, load_calibration, correct_distortion
    from lane_detection import find_lane_pixels, fit_lane
    from perspective import get_perspective_matrices, warp
    from thresholding import threshold_binary

    mtx, dist, _ = load_calibration()
    M, Minv = get_perspective_matrices()
    for name in ["straight_lines1.jpg", "test2.jpg", "test5.jpg"]:
        undist = correct_distortion(cv2.imread(str(ROOT / "test_images" / name)), mtx, dist)
        binary_warped = warp(threshold_binary(undist)[0], M)
        leftx, lefty, rightx, righty, _ = find_lane_pixels(binary_warped)
        left_fit, right_fit = fit_lane(leftx, lefty, rightx, righty, binary_warped.shape[0])
        lane = draw_lane(undist, left_fit, right_fit, Minv, (leftx, lefty, rightx, righty))
        cv2.imwrite(str(OUT_DIR / f"lane_{name.split('.')[0]}.jpg"), lane)
