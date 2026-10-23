"""Color and gradient thresholds that turn a road image into a binary mask of lane-line pixels."""
from pathlib import Path

import cv2
import numpy as np

from calibration import ROOT, OUT_DIR, load_calibration, correct_distortion
from visualization import save_comparison


def sobel_x_mask(channel, thresh=(20, 100), ksize=3):
    """Pixels whose horizontal gradient, scaled to 0..255, lies within thresh.

    Lane lines run roughly vertically in the image, so their edges have a strong x gradient.
    """
    sobel = np.absolute(cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=ksize))
    scaled = np.uint8(255 * sobel / max(sobel.max(), 1))
    return (scaled >= thresh[0]) & (scaled <= thresh[1])


def color_mask(hls, s_thresh=(170, 255), min_lightness=60):
    """Pixels that look like yellow or white paint.

    Saturation picks out yellow and white lines under changing light; the lightness floor
    drops dark, saturated shadow pixels. The explicit yellow and white ranges catch paint on
    pale concrete, where saturation alone is weak.
    """
    h, l, s = hls[..., 0], hls[..., 1], hls[..., 2]
    saturated = (s >= s_thresh[0]) & (s <= s_thresh[1]) & (l >= min_lightness)
    yellow = (h >= 15) & (h <= 35) & (s >= 100) & (l >= 80)
    white = l >= 205
    return saturated | yellow | white


def threshold_binary(img):
    """Binary image (uint8, 0 or 1) of likely lane-line pixels in a BGR road image.

    Also returns the two masks it combines, (gradient, color), for visualization.
    """
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    gradient = sobel_x_mask(hls[..., 1])
    color = color_mask(hls)
    return (gradient | color).astype(np.uint8), gradient, color


def contributions(gradient, color):
    """Color image showing which mask found each pixel: green = gradient, blue = color."""
    return np.dstack([color, gradient, np.zeros_like(color)]).astype(np.uint8) * 255


if __name__ == "__main__":
    mtx, dist, _ = load_calibration()
    # test1/test4/test5 have shadows and pale concrete; straight_lines1 is a clean reference
    for name in ["test1.jpg", "test4.jpg", "test5.jpg", "straight_lines1.jpg"]:
        img = correct_distortion(cv2.imread(str(ROOT / "test_images" / name)), mtx, dist)
        binary, gradient, color = threshold_binary(img)
        save_comparison([img, contributions(gradient, color), binary * 255],
                        ["Undistorted", "Gradient (green) / color (blue)", "Binary"],
                        OUT_DIR / f"binary_{Path(name).stem}.jpg")
