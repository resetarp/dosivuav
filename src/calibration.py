"""Camera calibration from chessboard images and distortion correction."""
import argparse
from pathlib import Path

import cv2
import numpy as np

from visualization import save_comparison

ROOT = Path(__file__).resolve().parent.parent
CAL_DIR = ROOT / "camera_cal"
CAL_FILE = CAL_DIR / "calibration.npz"
DOCS_DIR = ROOT / "docs"  # figures linked in README.md

NX, NY = 9, 6  # inner chessboard corners per row and per column
IMAGE_SIZE = (1280, 720)  # (width, height) of the camera that took the chessboard images


def calibrate_camera(image_dir=CAL_DIR, nx=NX, ny=NY, image_size=IMAGE_SIZE):
    """Find chessboard corners in every image in image_dir and compute the camera matrix.

    Returns (mtx, dist, rms, used, skipped) where used/skipped are image file names.
    """
    # Object points of the chessboard corners in its own plane: (0,0,0), (1,0,0), ..., (nx-1,ny-1,0)
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objpoints, imgpoints, used, skipped = [], [], [], []
    for path in sorted(Path(image_dir).iterdir()):
        if not cv2.haveImageReader(str(path)):
            continue
        gray = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2GRAY)
        # A couple of images are 1281x721; cropping keeps pixel coordinates unchanged
        gray = gray[: image_size[1], : image_size[0]]
        found, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if not found:
            skipped.append(path.name)
            continue
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners)
        used.append(path.name)

    rms, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)
    return mtx, dist, rms, used, skipped


def save_calibration(mtx, dist, path=CAL_FILE, image_size=IMAGE_SIZE):
    np.savez(path, mtx=mtx, dist=dist, image_size=np.array(image_size))


def load_calibration(path=CAL_FILE):
    """Return (mtx, dist, image_size) saved by save_calibration()."""
    data = np.load(path)
    return data["mtx"], data["dist"], tuple(int(v) for v in data["image_size"])


def undistort(img, mtx, dist):
    return cv2.undistort(img, mtx, dist, None, mtx)


def matches_calibration(img, image_size=IMAGE_SIZE):
    """True if img has the resolution of the calibrated camera."""
    return (img.shape[1], img.shape[0]) == tuple(image_size)


def correct_distortion(img, mtx, dist, image_size=IMAGE_SIZE):
    """Undistort a road image taken at the calibrated resolution.

    The calibration describes the camera's full 1280x720 frame. Images at other resolutions
    may be scaled or cropped versions of it, which the pixels alone can't tell apart, so they
    are returned unchanged.
    """
    if not matches_calibration(img, image_size):
        return img
    return undistort(img, mtx, dist)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recalibrate", action="store_true",
                        help=f"recompute the calibration even if {CAL_FILE.name} exists")
    args = parser.parse_args()

    if CAL_FILE.exists() and not args.recalibrate:
        mtx, dist, _ = load_calibration()
        print(f"Loaded {CAL_FILE.relative_to(ROOT)} (pass --recalibrate to recompute)")
    else:
        mtx, dist, rms, used, skipped = calibrate_camera()
        save_calibration(mtx, dist)
        print(f"Calibrated from {len(used)} images, RMS reprojection error {rms:.3f} px")
        print(f"Skipped (not all {NX}x{NY} corners visible): {', '.join(skipped) or 'none'}")
        print(f"Saved to {CAL_FILE.relative_to(ROOT)}")
    print(f"Camera matrix:\n{mtx}\nDistortion coefficients: {dist.ravel()}")

    # Detected corners on one of the boards that was used
    img = cv2.imread(str(CAL_DIR / "calibration2.jpg"))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, (NX, NY), None)
    corners_img = cv2.drawChessboardCorners(img.copy(), (NX, NY), corners, found)
    save_comparison([img, corners_img], ["Original", "Detected corners"],
                    DOCS_DIR / "corners_calibration2.jpg")

    # calibration1.jpg shows the strongest barrel distortion
    img = cv2.imread(str(CAL_DIR / "calibration1.jpg"))
    save_comparison([img, undistort(img, mtx, dist)], ["Original", "Undistorted"],
                    DOCS_DIR / "undistort_calibration1.jpg")

    # Road image: the change is subtle, so also show the amplified per-pixel difference
    img = cv2.imread(str(ROOT / "test_images" / "test1.jpg"))
    undist = correct_distortion(img, mtx, dist)
    diff = cv2.cvtColor(cv2.absdiff(img, undist), cv2.COLOR_BGR2GRAY)
    diff = cv2.convertScaleAbs(diff, alpha=3)
    save_comparison([img, undist, diff], ["Original", "Undistorted", "Difference (x3)"],
                    DOCS_DIR / "undistort_test1.jpg")

    skipped = [p.name for p in sorted((ROOT / "test_images").iterdir())
               if cv2.haveImageReader(str(p)) and not matches_calibration(cv2.imread(str(p)))]
    print(f"Test images not at the calibrated resolution, left distorted: {', '.join(skipped)}")
