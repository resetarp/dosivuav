"""Radius of curvature of the lane and the vehicle's position relative to the lane center, in meters."""
import cv2
import numpy as np

from calibration import ROOT, IMAGE_SIZE, load_calibration, correct_distortion
from lane_detection import find_lane_pixels, fit_lane
from perspective import SRC, get_perspective_matrices, warp
from thresholding import threshold_binary

# Scale of the birds-eye view in meters per pixel. Like the calibration and SRC/DST, this is a
# setup parameter of the camera mounting: some known distance on the road is needed to express
# results in meters, and it must be re-estimated if the camera or the transform changes. For this
# camera: about 3.7 m across the 640 px between DST's lines, about 24 m over the 720 px height.
XM_PER_PIX = 3.7 / 640
YM_PER_PIX = 24 / 720


def fit_to_meters(fit, xm=XM_PER_PIX, ym=YM_PER_PIX):
    """Convert x = A*y^2 + B*y + C from pixels to meters.

    With x_m = xm * x and y_m = ym * y, the coefficients scale as
    A_m = A * xm / ym^2, B_m = B * xm / ym, C_m = C * xm. This avoids refitting the pixels.
    """
    A, B, C = fit
    return np.array([A * xm / ym ** 2, B * xm / ym, C * xm])


def radius_of_curvature(fit, y_eval, xm=XM_PER_PIX, ym=YM_PER_PIX):
    """Radius in meters of the pixel-space fit at image row y_eval.

    R = (1 + (2*A*y + B)^2)^(3/2) / |2*A|, evaluated in meters. A straight line gives infinity.
    """
    A, B, _ = fit_to_meters(fit, xm, ym)
    y = y_eval * ym
    return np.inf if A == 0 else (1 + (2 * A * y + B) ** 2) ** 1.5 / abs(2 * A)


def car_position_px(M, image_size=IMAGE_SIZE):
    """x position of the car in the birds-eye view.

    The camera is assumed to sit on the car's centerline and look straight ahead, so the car
    is at the horizontal center of the camera image. The trapezoid SRC is not symmetric about
    that center, so the point is mapped through M instead of assuming x = width / 2.
    """
    point = np.float32([[[image_size[0] / 2, SRC[1, 1]]]])
    return float(cv2.perspectiveTransform(point, M)[0, 0, 0])


def vehicle_offset(left_fit, right_fit, car_x, y_eval, xm=XM_PER_PIX):
    """Distance in meters of the car from the lane center; positive = right of center."""
    lane_center = (np.polyval(left_fit, y_eval) + np.polyval(right_fit, y_eval)) / 2
    return (car_x - lane_center) * xm


def lane_measurements(left_fit, right_fit, car_x, y_eval):
    """(radius in m, offset in m): the lane's radius is the mean of both lines' radii."""
    radius = (radius_of_curvature(left_fit, y_eval) + radius_of_curvature(right_fit, y_eval)) / 2
    return radius, vehicle_offset(left_fit, right_fit, car_x, y_eval)


if __name__ == "__main__":
    mtx, dist, _ = load_calibration()
    M, _ = get_perspective_matrices()
    car_x = car_position_px(M)
    y_eval = IMAGE_SIZE[1] - 1  # bottom of the warped image, closest to the car
    print(f"Car position in the birds-eye view: x = {car_x:.1f} px")
    print(f"{'image':20} {'left R':>9} {'right R':>9} {'lane R':>9} {'offset':>8}")
    for path in sorted((ROOT / "test_images").iterdir()):
        img = cv2.imread(str(path))
        if img is None or img.shape[1::-1] != IMAGE_SIZE:
            continue
        binary_warped = warp(threshold_binary(correct_distortion(img, mtx, dist))[0], M)
        leftx, lefty, rightx, righty, _ = find_lane_pixels(binary_warped)
        left_fit, right_fit = fit_lane(leftx, lefty, rightx, righty, binary_warped.shape[0])
        radius, offset = lane_measurements(left_fit, right_fit, car_x, y_eval)
        print(f"{path.name:20} {radius_of_curvature(left_fit, y_eval):8.0f}m "
              f"{radius_of_curvature(right_fit, y_eval):8.0f}m {radius:8.0f}m {offset:+7.2f}m")
