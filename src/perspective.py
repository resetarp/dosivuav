"""Perspective transform between the camera view and a birds-eye view of the road."""
import cv2
import numpy as np

from calibration import ROOT, DOCS_DIR, IMAGE_SIZE, load_calibration, correct_distortion
from visualization import save_comparison

# Lane-line positions measured on the undistorted straight_lines1/2 images (averaged): where the
# lines cross y=680 (just above the hood), and their vanishing point, where they meet on the
# horizon. They define a trapezoid on the flat road that becomes a rectangle in the birds-eye
# view: straight lane lines map to vertical lines.

# NOTE: These points describe how the camera sees a flat road ahead, a fixed setup
# parameter like the calibration. They do not indicate where the lanes are in any image.
VANISHING_POINT = (639.4, 419.7)
BOTTOM_Y = 680
BOTTOM_X = (269, 1044)
# The far edge of the trapezoid lies this many rows below the horizon. Keeping the offset when
# the horizon moves keeps the birds-eye view looking the same distance ahead.
TOP_OFFSET = 40
DST = np.float32([[320, 0], [320, 720], [960, 720], [960, 0]])


def source_points(horizon_y=VANISHING_POINT[1]):
    """Road trapezoid (top-left, bottom-left, bottom-right, top-right) for a horizon row.

    The bottom corners stay where the lane lines cross the bottom row; the sides point at the
    vanishing point on the given horizon. When the car pitches or the road slope changes, the
    horizon moves up or down in the image while the camera's sideways direction stays the same.
    """
    vx = VANISHING_POINT[0]
    top_y = horizon_y + TOP_OFFSET
    t = (BOTTOM_Y - top_y) / (BOTTOM_Y - horizon_y)  # fraction of the way to the vanishing point
    return np.float32([[BOTTOM_X[0] + (vx - BOTTOM_X[0]) * t, top_y],
                       [BOTTOM_X[0], BOTTOM_Y],
                       [BOTTOM_X[1], BOTTOM_Y],
                       [BOTTOM_X[1] + (vx - BOTTOM_X[1]) * t, top_y]])


SRC = source_points()


def get_perspective_matrices(src=SRC, dst=DST):
    """Return (M, Minv): camera view -> birds-eye view and back."""
    return cv2.getPerspectiveTransform(src, dst), cv2.getPerspectiveTransform(dst, src)


def warp(img, M, size=IMAGE_SIZE):
    """Apply a perspective matrix, keeping the image size."""
    return cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR)


def draw_polygon(img, pts, color=(0, 0, 255)):
    out = img.copy()
    cv2.polylines(out, [pts.astype(np.int32)], isClosed=True, color=color, thickness=3)
    return out


if __name__ == "__main__":
    from thresholding import threshold_binary

    mtx, dist, _ = load_calibration()
    M, Minv = get_perspective_matrices()

    # Straight road: both lines must come out vertical and parallel
    for name in ["straight_lines1.jpg", "straight_lines2.jpg"]:
        img = correct_distortion(cv2.imread(str(ROOT / "test_images" / name)), mtx, dist)
        save_comparison([draw_polygon(img, SRC), draw_polygon(warp(img, M), DST)],
                        ["Undistorted with source points", "Warped with destination points"],
                        DOCS_DIR / f"warped_{name.split('.')[0]}.jpg")

    # Curved road: binary image before and after warping
    img = correct_distortion(cv2.imread(str(ROOT / "test_images" / "test5.jpg")), mtx, dist)
    binary, _, _ = threshold_binary(img)
    save_comparison([draw_polygon(img, SRC), warp(binary * 255, M)],
                    ["Undistorted with source points", "Warped binary"],
                    DOCS_DIR / "warped_binary_test5.jpg")
