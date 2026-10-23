**Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image0]: ./output_images/corners_calibration2.jpg "Chessboard corners"
[image1]: ./output_images/undistort_calibration1.jpg "Undistorted"
[image2]: ./output_images/undistort_test1.jpg "Road Transformed"
[image3]: ./output_images/binary_test5.jpg "Binary Example"
[image3b]: ./output_images/binary_test4.jpg "Binary Example (shadows)"
[image4]: ./output_images/warped_straight_lines1.jpg "Warp Example"
[image4b]: ./output_images/warped_binary_test5.jpg "Warped binary"
[image5]: ./output_images/fit_test5.jpg "Fit Visual"
[image5a]: ./output_images/histogram_test5.jpg "Histogram"
[image5b]: ./output_images/search_around_test5.jpg "Search around previous fit"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

---

### Writeup / README

### Camera Calibration

The code is in `calibrate_camera()` in `src/calibration.py`. Run `python src/calibration.py` to regenerate the example images, adding `--recalibrate` to recompute the calibration itself.

`cv2.findChessboardCorners()` is used to look for the 9x6 inner corners for each chessboard image in `camera_cal/` and refine them to sub-pixel accuracy with `cv2.cornerSubPix()`. Each successful detection adds two things:

* the corners' image coordinates (`imgpoints`);
* the same fixed grid of "object points" `(0,0,0), (1,0,0), ..., (8,5,0)` (`objpoints`). The board is assumed to lie in the z=0 plane, so this grid is identical for every image.

Corners were found in 17 of the 20 images. In `calibration1.jpg`, `calibration4.jpg` and `calibration5.jpg` part of the board is outside the frame, so they are skipped. Two images (`calibration7.jpg`, `calibration15.jpg`) are 1281x721 instead of 1280x720. They are cropped to 1280x720, which leaves the pixel coordinates unchanged.

`cv2.calibrateCamera()` then computes the camera matrix and the distortion coefficients from all point pairs. The RMS reprojection error is 1.00 px. The result is saved to `camera_cal/calibration.npz`, so later steps load it with `load_calibration()` instead of recalibrating.

![Chessboard corners][image0]

Applying the result with `cv2.undistort()` (`undistort()` in `src/calibration.py`) straightens the strong barrel distortion in `calibration1.jpg`. That image was not used for calibration:

![Undistorted][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Each road image is first corrected with `correct_distortion()` in `src/calibration.py`. It applies `cv2.undistort()` with the saved camera matrix and distortion coefficients. Passing the original camera matrix as the new one keeps the focal length, so the result is slightly zoomed in and has no black borders.

On a road image the effect is subtle. It is easiest to see at the image edges: the car hood at the bottom and the white car on the right are pushed outward. The third panel shows the amplified per-pixel difference. It is mostly near zero in the middle of the image and grows toward the borders, as expected for radial distortion.

![Distortion-corrected road image][image2]

The calibration describes the camera's full 1280x720 frame. Some inputs are 960x540: `challange*.jpg`, `solid*.jpg`, `whiteCarLaneSwitch.jpg` and `project_video01/02.mp4`. A 960x540 image can be a scaled-down or a cropped version of the frame, and the pixels alone can't tell which. The camera matrix differs between the two cases, so the calibration can't be applied reliably. `correct_distortion()` leaves images at any other resolution than the calibrated one unchanged.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The code is in `threshold_binary()` in `src/thresholding.py`. Run `python src/thresholding.py` to regenerate the example images. The undistorted image is converted to HLS, and two masks are combined with a logical OR:

* **Gradient mask** (`sobel_x_mask()`): Sobel derivative in x on the lightness channel, scaled to 0-255 and kept between 20 and 100. Lane lines run roughly vertically in the image, so their edges have a strong horizontal gradient. This finds lines regardless of their color, including faint or worn ones.
* **Color mask** (`color_mask()`), which is the union of:
  * saturation between 170 and 255, with a lightness floor of 60. Saturation picks out yellow and white paint under changing light. The floor removes dark shadow pixels, which can also be highly saturated.
  * yellow: hue 15-35 (OpenCV's 0-180 scale) with saturation >= 100 and lightness >= 80. This catches the yellow line on pale concrete, where saturation alone is weak.
  * white: lightness >= 205.

The middle panel shows which mask found each pixel: green for the gradient, blue for the color mask. Both lines are found in full, in the shade of the trees as well as on the pale concrete:

![Binary Example][image3]
![Binary Example (shadows)][image3b]

Some noise remains, mostly trees, sky and cars outside the road area. The perspective transform in the next step keeps only the road in front of the car, which removes most of it. The scattered gradient pixels from the concrete texture are much sparser than the lane lines, so the histogram-based lane-pixel search (section 4) is not misled by them.

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code is in `src/perspective.py`. `get_perspective_matrices()` computes the matrix `M` from the camera view to the birds-eye view with `cv2.getPerspectiveTransform()`, and the inverse `Minv` for warping results back (section 6). `warp()` applies a matrix with `cv2.warpPerspective()`.

The source points form a trapezoid on the road that becomes a rectangle in the birds-eye view. They were measured once on the straight-road images, by fitting a line to each lane line. The bottom edge sits just above the hood, so the hood stays out of the birds-eye view. Like the camera calibration, the points are a fixed setting of the camera mounting, and every image is warped with the same matrix.

| Source | Destination |
|:-:|:-:|
| 582, 460 | 320, 0 |
| 269, 680 | 320, 720 |
| 1044, 680 | 960, 720 |
| 702, 460 | 960, 0 |

On a straight road the lane lines come out vertical and parallel:

![Warp Example][image4]

On a curved road they curve together. This warped binary image is the input for the lane-pixel search:

![Warped binary][image4b]

The transform assumes a flat road and a fixed camera pitch. On hills, or when the car pitches while braking, the lines are no longer exactly parallel in the birds-eye view.

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code is in `src/lane_detection.py`. The input is the warped binary image from section 3.

**Histogram.** `lane_histogram()` sums the binary pixels in each column of the bottom half of the image, where the lines are closest to the car and most vertical. The highest peak on each side of the image center gives each line's starting position:

![Histogram][image5a]

**Sliding windows.** `find_lane_pixels()` places a window on each starting position and assigns the binary pixels inside it to that line. The window then moves up one step at a time (9 steps over the image height). When it contains enough pixels, it is re-centred on their mean x, so the windows follow the line as it curves. In the gaps of a dashed line the window keeps its position.

**Polynomial fit.** `fit_polynomial()` fits `x = A*y^2 + B*y + C` to each line's pixels with `np.polyfit()`. x is a function of y because the lines are nearly vertical in the image.

A second-order fit through a short stretch of pixels, such as a single dash, can bend in any direction. `fit_lane()` therefore checks how much of the image height each line's pixels cover. If one line covers too little and the other enough, the weak line takes the other line's curvature (A and B), and only its position (C) is fitted from its own pixels. This assumes the two lane lines are parallel, which holds except where lanes merge or split. Those cases are outside the scope of this project.

Left-line pixels are red, right-line pixels blue, the windows green and the fits yellow:

![Fit Visual][image5]

**Search around the previous fit.** In a video the lines move little from one frame to the next. Once both lines have been found, `search_around_poly()` takes all pixels within a band around the previous frame's fits (green) instead of running the sliding windows again. That is faster, and it is less likely to jump to noise. This solution has an obvious issue though: what happens when a line moves outside of the +-100px range? This will be explained in the tracking section of the video pipeline below.

![Search around previous fit][image5b]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The code is in `src/curvature.py`.

**Pixels to meters.** The birds-eye view has a scale in each direction (`XM_PER_PIX`, `YM_PER_PIX`). Some known distance on the road is needed to express the result in meters. The scale depends only on how the camera is mounted and on the perspective transform, not on the road being driven. Like the calibration, it is a setup parameter, estimated once for this camera: about 3.7 m across the 640 px between the lane lines, and about 24 m of road over the 720 px image height.

**Radius of curvature.** `fit_to_meters()` converts a fit from pixels to meters by scaling its coefficients (`A*xm/ym^2`, `B*xm/ym`, `C*xm`), so no refit is needed. `radius_of_curvature()` evaluates

R = (1 + (2*A*y + B)^2)^(3/2) / |2*A|

at the bottom of the image, closest to the car. The lane's radius is the mean of both lines' radii. A straight road gives a very large radius.

**Vehicle position.** The camera is assumed to sit on the car's centerline, so the car is at the horizontal center of the camera image. `car_position_px()` maps that point through the perspective transform, because the source trapezoid is not symmetric about the image center. `vehicle_offset()` returns the distance between the car and the midpoint of the two lines at the bottom of the image, in meters. Positive means right of the lane center.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

TODO: Add your text here!!!

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

TODO: Add your text here!!!

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

TODO: Add your text here!!!

