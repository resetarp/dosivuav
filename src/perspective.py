"""Perspective transform between the camera view and a birds-eye view of the road."""
import cv2
import numpy as np

from calibration import ROOT, OUT_DIR, IMAGE_SIZE, load_calibration, correct_distortion
from thresholding import threshold_binary
from visualization import save_comparison

# Lane-line positions measured on the undistorted straight_lines1/2 images (averaged), at
# y=460 (far) and y=680 (just above the hood). They form a trapezoid on the flat road that
# becomes a rectangle in the birds-eye view: straight lane lines map to vertical lines.

# NOTE: This is here only to help with visualization and is not an attempt at hard-coded 
# lane detection.
SRC = np.float32([[582, 460], [269, 680], [1044, 680], [702, 460]])
DST = np.float32([[320, 0], [320, 720], [960, 720], [960, 0]])


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
    mtx, dist, _ = load_calibration()
    M, Minv = get_perspective_matrices()

    # Straight road: both lines must come out vertical and parallel
    for name in ["straight_lines1.jpg", "straight_lines2.jpg"]:
        img = correct_distortion(cv2.imread(str(ROOT / "test_images" / name)), mtx, dist)
        save_comparison([draw_polygon(img, SRC), draw_polygon(warp(img, M), DST)],
                        ["Undistorted with source points", "Warped with destination points"],
                        OUT_DIR / f"warped_{name.split('.')[0]}.jpg")

    # Curved road: binary image before and after warping
    img = correct_distortion(cv2.imread(str(ROOT / "test_images" / "test5.jpg")), mtx, dist)
    binary, _, _ = threshold_binary(img)
    save_comparison([draw_polygon(img, SRC), warp(binary * 255, M)],
                    ["Undistorted with source points", "Warped binary"],
                    OUT_DIR / "warped_binary_test5.jpg")
