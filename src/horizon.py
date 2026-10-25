"""Estimating the horizon row of a video from the lane lines, while the video is processed.

On a flat road the lane lines meet in the vanishing point on the horizon. Where the horizon
lies in the image depends on the camera's pitch and the road's slope, so it differs between
videos and changes on hills. The birds-eye view is only correct for the right horizon: with a
wrong one, parallel lane lines converge or diverge in the warped image.
"""
from collections import deque

import cv2
import numpy as np

from perspective import VANISHING_POINT, BOTTOM_Y

# Line segments: minimum number of votes, minimum length and largest gap in pixels
HOUGH_THRESHOLD = 30
MIN_SEGMENT_LENGTH = 40
MAX_SEGMENT_GAP = 20
# A lane-line segment leans by this much (dx/dy) to the left or right. Flatter segments are
# road texture or shadows across the road, steeper ones are not on the road surface.
MIN_SLOPE, MAX_SLOPE = 0.5, 3.0
# Only segments this many rows below the current horizon are used: near the car the lines are
# nearly straight even in a bend, further ahead a bend would bias the intersection.
NEAR_ROWS_BELOW_HORIZON = 80
# A horizon further than this from the measured one is not a plausible pitch or slope change
MAX_DEVIATION = 60
# The estimate is the median of the last frames' rows (3 s at 25 fps), once enough are known
HISTORY = 75
MIN_ESTIMATES = 5


def side_line(segments):
    """One line x = k*y + c through segments (x1, y1, x2, y2), weighted by their length."""
    x = segments[:, [0, 2]].ravel()
    y = segments[:, [1, 3]].ravel()
    length = np.hypot(segments[:, 2] - segments[:, 0], segments[:, 3] - segments[:, 1])
    return np.polyfit(y, x, 1, w=np.repeat(length, 2))


def vanishing_row(binary, horizon_y):
    """Row where the left and the right lane lines of one camera-view binary image meet.

    Line segments are found with the probabilistic Hough transform. Segments leaning right
    (going up) on the left half of the image form the left line, segments leaning left on the
    right half the right line; each side is averaged into one straight line. Returns None if a
    side has no segment.
    """
    road = binary.copy()
    road[:int(horizon_y) + NEAR_ROWS_BELOW_HORIZON] = 0
    road[BOTTOM_Y:] = 0
    segments = cv2.HoughLinesP(road * 255, 1, np.pi / 180, HOUGH_THRESHOLD,
                               minLineLength=MIN_SEGMENT_LENGTH, maxLineGap=MAX_SEGMENT_GAP)
    if segments is None:
        return None
    segments = segments.reshape(-1, 4).astype(np.float64)
    segments = segments[segments[:, 1] != segments[:, 3]]
    slope = (segments[:, 2] - segments[:, 0]) / (segments[:, 3] - segments[:, 1])
    mid_x = (segments[:, 0] + segments[:, 2]) / 2
    center = binary.shape[1] / 2
    left = segments[(slope < -MIN_SLOPE) & (slope > -MAX_SLOPE) & (mid_x < center)]
    right = segments[(slope > MIN_SLOPE) & (slope < MAX_SLOPE) & (mid_x > center)]
    if len(left) == 0 or len(right) == 0:
        return None
    (kl, cl), (kr, cr) = side_line(left), side_line(right)
    return (cr - cl) / (kl - kr)


class HorizonEstimator:
    """Median of the vanishing-point rows of the last frames of a video."""

    def __init__(self):
        self.rows = deque(maxlen=HISTORY)

    def update(self, binary, horizon_y):
        """Add one frame. Returns the current estimate, or None while it isn't known yet."""
        row = vanishing_row(binary, horizon_y)
        if row is not None and abs(row - VANISHING_POINT[1]) <= MAX_DEVIATION:
            self.rows.append(row)
        return float(np.median(self.rows)) if len(self.rows) >= MIN_ESTIMATES else None
