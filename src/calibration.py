"""Camera calibration from chessboard images and distortion correction."""
import re
from pathlib import Path

import cv2
import numpy as np

from visualization import save_comparison

ROOT = Path(__file__).resolve().parent.parent
CAL_DIR = ROOT / "camera_cal"
CAL_FILE = CAL_DIR / "calibration.npz"
OUT_DIR = ROOT / "output_images"

NX, NY = 9, 6  # inner chessboard corners per row and per column
IMAGE_SIZE = (1280, 720)  # (width, height) of the camera that took the chessboard images


def _natural_key(path):
    return int(re.search(r"\d+", path.stem).group())


def calibrate_camera(image_dir=CAL_DIR, nx=NX, ny=NY, image_size=IMAGE_SIZE):
    """Find chessboard corners in every calibration image and compute the camera matrix.

    Returns (mtx, dist, rms, used, skipped) where used/skipped are image file names.
    """
    # Object points of the chessboard corners in its own plane: (0,0,0), (1,0,0), ..., (nx-1,ny-1,0)
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objpoints, imgpoints, used, skipped = [], [], [], []
    for path in sorted(Path(image_dir).glob("calibration*.jpg"), key=_natural_key):
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


if __name__ == "__main__":
    mtx, dist, rms, used, skipped = calibrate_camera()
    save_calibration(mtx, dist)
    print(f"Calibrated from {len(used)} images, RMS reprojection error {rms:.3f} px")
    print(f"Skipped (not all {NX}x{NY} corners visible): {', '.join(skipped) or 'none'}")
    print(f"Camera matrix:\n{mtx}\nDistortion coefficients: {dist.ravel()}")
    print(f"Saved to {CAL_FILE.relative_to(ROOT)}")

    # Detected corners on one of the boards that was used
    img = cv2.imread(str(CAL_DIR / "calibration2.jpg"))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, (NX, NY), None)
    corners_img = cv2.drawChessboardCorners(img.copy(), (NX, NY), corners, found)
    save_comparison([img, corners_img], ["Original", "Detected corners"],
                    OUT_DIR / "corners_calibration2.jpg")

    # calibration1.jpg shows the strongest barrel distortion
    img = cv2.imread(str(CAL_DIR / "calibration1.jpg"))
    save_comparison([img, undistort(img, mtx, dist)], ["Original", "Undistorted"],
                    OUT_DIR / "undistort_calibration1.jpg")
