"""Find lane-line pixels in a warped binary image and fit a second-order polynomial to each line."""
import cv2
import numpy as np

from calibration import ROOT, DOCS_DIR, load_calibration, correct_distortion
from perspective import get_perspective_matrices, warp
from thresholding import threshold_binary
from visualization import save_comparison, save_histogram


def lane_histogram(binary_warped):
    """Column sums of the bottom half, where the lines are closest to the car and most vertical."""
    return np.sum(binary_warped[binary_warped.shape[0] // 2:], axis=0)


def find_lane_pixels(binary_warped, nwindows=9, margin=100, minpix=50):
    """Sliding-window search starting from the two histogram peaks.

    A window of +-margin px around each line's current x position moves up the image in
    nwindows steps. Pixels inside it belong to that line. If a window contains more than
    minpix pixels, the next window is centred on their mean x, so the windows follow the curve.

    Returns (leftx, lefty, rightx, righty, windows), windows being a list of rectangles
    ((x0, y0), (x1, y1)) for visualization.
    """
    histogram = lane_histogram(binary_warped)
    midpoint = histogram.shape[0] // 2
    x_current = [np.argmax(histogram[:midpoint]), midpoint + np.argmax(histogram[midpoint:])]

    height = binary_warped.shape[0]
    window_height = height // nwindows
    nonzeroy, nonzerox = binary_warped.nonzero()
    lane_inds = [[], []]
    windows = []
    for window in range(nwindows):
        y_high = height - window * window_height
        y_low = y_high - window_height
        for side in (0, 1):
            x_low, x_high = x_current[side] - margin, x_current[side] + margin
            windows.append(((x_low, y_low), (x_high, y_high)))
            good = ((nonzeroy >= y_low) & (nonzeroy < y_high) &
                    (nonzerox >= x_low) & (nonzerox < x_high)).nonzero()[0]
            lane_inds[side].append(good)
            if len(good) > minpix:
                x_current[side] = int(np.mean(nonzerox[good]))

    left, right = np.concatenate(lane_inds[0]), np.concatenate(lane_inds[1])
    return nonzerox[left], nonzeroy[left], nonzerox[right], nonzeroy[right], windows


def search_around_poly(binary_warped, left_fit, right_fit, margin=100):
    """Pixels within +-margin px of the previous frame's fits.

    In a video the lines move little between frames, so this replaces the sliding-window
    search once both lines have been found. Returns (leftx, lefty, rightx, righty).
    """
    nonzeroy, nonzerox = binary_warped.nonzero()
    result = []
    for fit in (left_fit, right_fit):
        inds = np.abs(nonzerox - np.polyval(fit, nonzeroy)) < margin
        result += [nonzerox[inds], nonzeroy[inds]]
    return tuple(result)


def fit_polynomial(x, y, min_pixels=50):
    """Fit x = A*y^2 + B*y + C (x as a function of y, since the lines are nearly vertical).

    Returns the coefficients [A, B, C], or None if there are too few pixels for a reliable fit.
    """
    if len(x) < min_pixels:
        return None
    return np.polyfit(y, x, 2)


def fit_lane(leftx, lefty, rightx, righty, height, min_span=0.4):
    """Fit both lines; a line seen only over a short stretch borrows the other line's shape.

    A second-order fit through a short stretch of pixels, such as a single dash, can bend in
    any direction. If a line's pixels cover less than min_span of the image height and the
    other line's cover more, the lines' parallelism is used: the weak line gets the other
    line's A and B, and only its offset C is fitted (as the mean residual of its pixels).
    The vertical extent is measured between the 5th and 95th percentile of the pixels' y,
    so a few stray pixels far from the line don't count as coverage.

    Returns (left_fit, right_fit); either may be None if a line has too few pixels.
    """
    fits = [fit_polynomial(leftx, lefty), fit_polynomial(rightx, righty)]
    pixels = [(leftx, lefty), (rightx, righty)]
    spans = [np.subtract(*np.percentile(y, [95, 5])) / height if fit is not None else 0
             for fit, (_, y) in zip(fits, pixels)]
    for weak, strong in ((0, 1), (1, 0)):
        if fits[weak] is not None and spans[weak] < min_span <= spans[strong]:
            x, y = pixels[weak]
            A, B, _ = fits[strong]
            fits[weak] = np.array([A, B, np.mean(x - A * y ** 2 - B * y)])
    return fits[0], fits[1]


def draw_fit(binary_warped, leftx, lefty, rightx, righty, left_fit, right_fit, windows=(),
             search_margin=None):
    """Line pixels in red (left) and blue (right), windows or search band in green, fits in yellow."""
    out = np.dstack([binary_warped * 255] * 3).astype(np.uint8)
    ploty = np.arange(binary_warped.shape[0])
    if search_margin is not None:
        band = np.zeros_like(out)
        for fit in (left_fit, right_fit):
            fitx = np.polyval(fit, ploty)
            pts = np.vstack([np.column_stack([fitx - search_margin, ploty]),
                             np.column_stack([fitx + search_margin, ploty])[::-1]])
            cv2.fillPoly(band, [pts.astype(np.int32)], (0, 255, 0))
        out = cv2.addWeighted(out, 1, band, 0.3, 0)
    for (p0, p1) in windows:
        cv2.rectangle(out, p0, p1, (0, 255, 0), 2)
    out[lefty, leftx] = (0, 0, 255)
    out[righty, rightx] = (255, 0, 0)
    for fit in (left_fit, right_fit):
        pts = np.column_stack([np.polyval(fit, ploty), ploty]).astype(np.int32)
        cv2.polylines(out, [pts], isClosed=False, color=(0, 255, 255), thickness=3)
    return out


if __name__ == "__main__":
    mtx, dist, _ = load_calibration()
    M, _ = get_perspective_matrices()
    for name in ["test1.jpg", "test3.jpg", "test5.jpg", "test6.jpg"]:
        img = correct_distortion(cv2.imread(str(ROOT / "test_images" / name)), mtx, dist)
        binary_warped = warp(threshold_binary(img)[0], M)
        leftx, lefty, rightx, righty, windows = find_lane_pixels(binary_warped)
        left_fit, right_fit = fit_lane(leftx, lefty, rightx, righty, binary_warped.shape[0])
        stem = name.split(".")[0]
        save_comparison([binary_warped * 255,
                         draw_fit(binary_warped, leftx, lefty, rightx, righty,
                                  left_fit, right_fit, windows)],
                        ["Warped binary", "Sliding windows and fit"],
                        DOCS_DIR / f"fit_{stem}.jpg")
        print(f"{name}: left {np.round(left_fit, 5)}, right {np.round(right_fit, 5)}")

        if name == "test5.jpg":
            save_histogram(lane_histogram(binary_warped), "Bottom-half column sums of the warped binary",
                           DOCS_DIR / "histogram_test5.jpg")
            # Search around the fit just found, as the next video frame would
            leftx, lefty, rightx, righty = search_around_poly(binary_warped, left_fit, right_fit)
            save_comparison([draw_fit(binary_warped, leftx, lefty, rightx, righty, left_fit,
                                      right_fit, search_margin=100)],
                            ["Search around previous fit"], DOCS_DIR / "search_around_test5.jpg")
