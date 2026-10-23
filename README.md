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
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
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

The code is in `src/perspective.py`. Run `python src/perspective.py` to regenerate the example images. `get_perspective_matrices()` computes the matrix `M` from the camera view to the birds-eye view with `cv2.getPerspectiveTransform()`, and the inverse `Minv` for warping results back in section 6. `warp()` applies a matrix with `cv2.warpPerspective()`.

The source points are measured rather than picked by eye. On the undistorted, thresholded `straight_lines1.jpg` and `straight_lines2.jpg`, a straight line is fitted to each lane line (`x = a*y + c`). Both images put the vanishing point at about (639, 420). The source trapezoid uses the lines' average positions at y=460 (far ahead) and y=680 (just above the hood, so the hood stays out of the birds-eye view). It is mapped to a rectangle with the lane lines at x=320 and x=960.

The source points are a fixed setting, like the camera calibration: they were measured once and every image is warped with the same matrix. Measured on their own, the two images disagree by 5-10 px, because the road's slope and the car's pitch differ slightly between them. No single flat-road transform fits both exactly, so the average is used. Fitting lines to the warped binary images shows the result: on `straight_lines1.jpg` both lane lines are vertical to within 1 px over the full image height, and on `straight_lines2.jpg` they drift by 10-17 px (about 2-3% of the lane width). These two images were also used for the measurement, so this check only shows that the measurement is consistent. The independent test comes later, with the lane width and curvature measured on the other test images and the videos.

![Warp Example][image4]

On a curved road the lines stay parallel and curve together. This warped binary image is the input for the lane-pixel search:

![Warped binary][image4b]

The transform assumes a flat road and a fixed camera pitch. On hills, or when the car pitches while braking, the lines will no longer be exactly parallel in the birds-eye view.

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

TODO: Add your text here!!!

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

TODO: Add your text here!!!

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

TODO: Add your text here!!!

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

TODO: Add your text here!!!

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

TODO: Add your text here!!!

