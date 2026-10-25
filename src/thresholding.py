"""Turn a road image into a binary mask of lane-line pixels by finding narrow bright stripes.

A painted lane line is a stripe that is lighter (white paint) or more saturated (yellow paint)
than the road on both of its sides. Edges such as asphalt seams, shadow borders or the road edge
have a different brightness on each side, so they are not stripes and are not selected.
"""
from pathlib import Path

import cv2
import numpy as np

from calibration import ROOT, DOCS_DIR, load_calibration, correct_distortion
from perspective import SRC
from visualization import save_comparison

# Horizontal width (px) of the structuring element: wider than a lane line near the car, so
# lines of any width up to that survive the top-hat, while wider bright areas don't.
STRIPE_WIDTH = 31
# Rows of the road area the thresholds are computed on: from the far edge of the birds-eye
# trapezoid to its near edge above the hood. Sky, trees and the hood would distort them.
ROAD_ROWS = slice(int(SRC[:, 1].min()), int(SRC[:, 1].max()))
# The road area is split into this many horizontal bands with a threshold each: lines far
# ahead are thinner and blurrier, so they stand out less than lines near the car.
ROAD_BANDS = 4
# Lowest contrast (grey levels) that counts as a stripe, for frames without any lane line
# where the automatic threshold would split pure road texture.
MIN_WHITE_CONTRAST = 10
MIN_YELLOW_CONTRAST = 6
# Hue range of yellow paint (OpenCV hue runs 0-179). Saturated stripes of other hues, such as
# dark shadow pixels, foliage or sky, are not yellow paint.
YELLOW_HUE = (15, 35)


def stripe_response(channel, width=STRIPE_WIDTH):
    """How much each pixel stands out from the road on both sides.

    The morphological opening removes bright structures narrower than width; subtracting it
    from the channel leaves exactly those structures. Flat areas and edges give 0.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width, 1))
    return cv2.morphologyEx(channel, cv2.MORPH_TOPHAT, kernel)


def stripe_mask(channel, min_contrast, road_rows=ROAD_ROWS, bands=ROAD_BANDS):
    """Pixels on the road whose stripe response is above a threshold chosen for this image.

    In every horizontal band of the road area, Otsu's method splits the stripe responses into
    two groups, road texture and stripes, so the threshold follows the light and contrast of
    every frame and every distance. min_contrast applies when a band has no stripes and Otsu
    would split pure texture. Rows outside the road area stay empty.
    """
    response = stripe_response(channel)
    mask = np.zeros(response.shape, bool)
    edges = np.linspace(road_rows.start, road_rows.stop, bands + 1).astype(int)
    for top, bottom in zip(edges[:-1], edges[1:]):
        band = response[top:bottom]
        values = band[band > 0].reshape(1, -1)
        otsu = cv2.threshold(values, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0] if values.size else 255
        mask[top:bottom] = band > max(min_contrast, otsu)
    return mask


def threshold_binary(img, road_rows=ROAD_ROWS):
    """Binary image (uint8, 0 or 1) of likely lane-line pixels in a BGR road image.

    White paint stands out in the HLS lightness channel. Yellow paint stands out in the
    saturation channel, where it is visible even on pale concrete where it isn't lighter than
    the road; of the saturated stripes only those with a yellow hue are kept.
    Also returns the two masks it combines, (white, yellow), for visualization.
    """
    hue, lightness, saturation = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HLS))
    white = stripe_mask(lightness, MIN_WHITE_CONTRAST, road_rows)
    yellow_hue = (hue >= YELLOW_HUE[0]) & (hue <= YELLOW_HUE[1])
    yellow = stripe_mask(saturation, MIN_YELLOW_CONTRAST, road_rows) & yellow_hue
    return (white | yellow).astype(np.uint8), white, yellow


def contributions(white, yellow):
    """Color image showing which mask found each pixel: green = white stripes, blue = yellow."""
    return np.dstack([yellow, white, np.zeros_like(white)]).astype(np.uint8) * 255


if __name__ == "__main__":
    mtx, dist, _ = load_calibration()
    # test1/test4/test5 have shadows and pale concrete; straight_lines1 is a clean reference
    for name in ["test1.jpg", "test4.jpg", "test5.jpg", "straight_lines1.jpg"]:
        img = correct_distortion(cv2.imread(str(ROOT / "test_images" / name)), mtx, dist)
        binary, white, yellow = threshold_binary(img)
        save_comparison([img, contributions(white, yellow), binary * 255],
                        ["Undistorted", "White stripes (green) / yellow stripes (blue)", "Binary"],
                        DOCS_DIR / f"binary_{Path(name).stem}.jpg")
