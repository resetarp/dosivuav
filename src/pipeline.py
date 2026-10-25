"""Full lane-finding pipeline for images and videos, with lane tracking across video frames.

Usage: python src/pipeline.py INPUT [INPUT ...]
Each INPUT is an image, a video or a directory of them. Results are written to output_images/
(images) and output_videos/ (videos). Inputs that don't have the calibrated camera's resolution
are skipped, since the calibration and the perspective transform only apply to that camera.
"""
import argparse
from collections import deque
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

from calibration import ROOT, IMAGE_SIZE, load_calibration, correct_distortion, matches_calibration
from curvature import car_position_px, lane_measurements
from lane_detection import find_lane_pixels, fit_lane, search_around_poly
from horizon import HorizonEstimator
from perspective import DST, VANISHING_POINT, get_perspective_matrices, source_points, warp
from thresholding import road_rows_of, threshold_binary
from visualization import annotate, draw_lane

IMAGE_DIR = ROOT / "output_images"
VIDEO_DIR = ROOT / "output_videos"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
LANE_WIDTH_PX = DST[2, 0] - DST[1, 0]  # lane width in the birds-eye view
# The birds-eye view is rebuilt when the estimated horizon moved by at least this many rows
HORIZON_UPDATE_ROWS = 2
# The horizon is only estimated while the tracked lane is at least this straight (radius in m):
# in a bend the line segments near the car don't point at the vanishing point
MIN_RADIUS_FOR_HORIZON = 500


class Line:
    """One lane line across video frames: the last few accepted fits, averaged for display."""

    def __init__(self, n_smooth=5):
        self.recent_fits = deque(maxlen=n_smooth)

    @property
    def best_fit(self):
        return np.mean(self.recent_fits, axis=0) if self.recent_fits else None

    def add(self, fit):
        self.recent_fits.append(fit)

    def reset(self):
        self.recent_fits.clear()

    def reproject(self, transform, height):
        """Express the remembered fits in another birds-eye view (transform: old view -> new)."""
        ys = np.linspace(0, height - 1, 20)
        for i, fit in enumerate(self.recent_fits):
            points = np.float32(np.column_stack([np.polyval(fit, ys), ys]))[None]
            moved = cv2.perspectiveTransform(points, transform)[0]
            self.recent_fits[i] = np.polyfit(moved[:, 1], moved[:, 0], 2)


def is_plausible(left_fit, right_fit, height, width_tolerance=0.3, parallel_tolerance=0.25):
    """Sanity check that two fits describe a lane, using only what holds on any road.

    Both lines must be found, the lane must be about as wide as the birds-eye view's lane
    (within width_tolerance), and the lines must be roughly parallel: the lane width may vary
    by at most parallel_tolerance of the lane width between the top and the bottom of the image.
    The radius of curvature is not checked, because tight curves are legitimate.
    """
    if left_fit is None or right_fit is None:
        return False
    ys = np.array([0, height / 2, height - 1])
    widths = np.polyval(right_fit, ys) - np.polyval(left_fit, ys)
    if abs(widths[-1] - LANE_WIDTH_PX) > width_tolerance * LANE_WIDTH_PX:
        return False
    return np.ptp(widths) <= parallel_tolerance * LANE_WIDTH_PX


class LanePipeline:
    """Runs all steps on a frame. With track=True it keeps state between consecutive frames.

    While tracking, the horizon of the video is also estimated from its frames, and the
    birds-eye view follows it.
    """

    def __init__(self, track=False, max_misses=5):
        self.mtx, self.dist, _ = load_calibration()
        self.track = track
        self.max_misses = max_misses
        self.left, self.right = Line(), Line()
        self.misses = 0
        self.horizon = HorizonEstimator() if track else None
        self.radius = np.inf  # radius of the lane in the previous frame
        self.set_horizon(VANISHING_POINT[1])

    def set_horizon(self, horizon_y):
        """Build the birds-eye view for a horizon row and move the tracked lane into it."""
        src = source_points(horizon_y)
        M, Minv = get_perspective_matrices(src)
        if hasattr(self, "Minv"):
            for line in (self.left, self.right):
                line.reproject(M @ self.Minv, IMAGE_SIZE[1])
        self.horizon_y, self.M, self.Minv = horizon_y, M, Minv
        self.road_rows = road_rows_of(src)
        self.car_x = car_position_px(M)

    def detect(self, binary_warped):
        """Fits for this frame, or (None, None) if no plausible lane was found.

        While tracking, the search starts around the previous lane; if that result is not
        plausible, the sliding-window search runs on the same frame.
        """
        height = binary_warped.shape[0]
        if self.track and self.left.best_fit is not None:
            pixels = search_around_poly(binary_warped, self.left.best_fit, self.right.best_fit)
            fits = fit_lane(*pixels, height)
            if is_plausible(*fits, height):
                return fits
        fits = fit_lane(*find_lane_pixels(binary_warped)[:4], height)
        # A single image has no history to fall back on, so its best guess is always shown
        return fits if not self.track or is_plausible(*fits, height) else (None, None)

    def process(self, img):
        undist = correct_distortion(img, self.mtx, self.dist)
        binary = threshold_binary(undist, self.road_rows)[0]
        if self.horizon is not None and self.radius >= MIN_RADIUS_FOR_HORIZON:
            estimate = self.horizon.update(binary, self.horizon_y)
            if estimate is not None and abs(estimate - self.horizon_y) >= HORIZON_UPDATE_ROWS:
                self.set_horizon(estimate)
        binary_warped = warp(binary, self.M)
        left_fit, right_fit = self.detect(binary_warped)

        if not self.track:
            if left_fit is None or right_fit is None:
                return undist
        elif left_fit is not None:
            self.left.add(left_fit)
            self.right.add(right_fit)
            self.misses = 0
            left_fit, right_fit = self.left.best_fit, self.right.best_fit
        else:
            # Keep showing the last good lane for a few frames, then start over from scratch
            self.misses += 1
            if self.misses >= self.max_misses:
                self.left.reset()
                self.right.reset()
            if self.left.best_fit is None:
                return undist
            left_fit, right_fit = self.left.best_fit, self.right.best_fit

        radius, offset = lane_measurements(left_fit, right_fit, self.car_x, binary_warped.shape[0] - 1)
        self.radius = radius
        return annotate(draw_lane(undist, left_fit, right_fit, self.Minv), radius, offset)


def process_image(path):
    img = cv2.imread(str(path))
    if not matches_calibration(img):
        print(f"skipped {path.name}: {img.shape[1]}x{img.shape[0]} is not the calibrated resolution")
        return
    IMAGE_DIR.mkdir(exist_ok=True)
    out = IMAGE_DIR / f"final_{path.stem}.jpg"
    cv2.imwrite(str(out), LanePipeline().process(img))
    print(f"{path.name} -> {out.relative_to(ROOT)}")


def process_video(path):
    cap = cv2.VideoCapture(str(path))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if size != IMAGE_SIZE:
        print(f"skipped {path.name}: {size[0]}x{size[1]} is not the calibrated resolution")
        return
    VIDEO_DIR.mkdir(exist_ok=True)
    out = VIDEO_DIR / f"{path.stem}_output.mp4"
    # H.264 through the ffmpeg binary bundled with imageio-ffmpeg: OpenCV's pip build can only
    # write mp4v (MPEG-4 Part 2), which many players and browsers can't play
    writer = imageio_ffmpeg.write_frames(str(out), IMAGE_SIZE, fps=cap.get(cv2.CAP_PROP_FPS),
                                         codec="libx264", quality=None,
                                         output_params=["-crf", "23", "-preset", "veryfast"])
    writer.send(None)  # start the encoder
    pipeline = LanePipeline(track=True)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.send(cv2.cvtColor(pipeline.process(frame), cv2.COLOR_BGR2RGB))
    cap.release()
    writer.close()
    print(f"{path.name} -> {out.relative_to(ROOT)}")


def expand(inputs):
    for p in map(Path, inputs):
        yield from sorted(p.iterdir()) if p.is_dir() else [p]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="images, videos or directories")
    for path in expand(parser.parse_args().inputs):
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            process_video(path)
        elif cv2.haveImageReader(str(path)):
            process_image(path)
