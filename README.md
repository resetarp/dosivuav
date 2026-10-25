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

[image0]: ./docs/corners_calibration2.jpg "Chessboard corners"
[image1]: ./docs/undistort_calibration1.jpg "Undistorted"
[image2]: ./docs/undistort_test1.jpg "Road Transformed"
[image3]: ./docs/binary_test5.jpg "Binary Example"
[image3b]: ./docs/binary_test4.jpg "Binary Example (shadows)"
[image4]: ./docs/warped_straight_lines1.jpg "Warp Example"
[image4b]: ./docs/warped_binary_test5.jpg "Warped binary"
[image5]: ./docs/fit_test5.jpg "Fit Visual"
[image5a]: ./docs/histogram_test5.jpg "Histogram"
[image5b]: ./docs/search_around_test5.jpg "Search around previous fit"
[image6]: ./docs/final_test5.jpg "Output"
[video1]: ./output_videos/project_video03_output.mp4 "Video"

---

### Setup and usage

Requirements: Python 3 with OpenCV, NumPy, matplotlib and imageio-ffmpeg (used to write H.264 videos).

```bash
# one-time setup
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
source .venv/bin/activate

# full pipeline: any mix of images, videos and directories
python src/pipeline.py test_images test_videos     # -> output_images/, output_videos/
python src/pipeline.py test_videos/challenge01.mp4 # a single file

# figures of the individual steps used in this writeup (-> docs/)
python src/calibration.py     # add --recalibrate to recompute camera_cal/calibration.npz
python src/thresholding.py
python src/perspective.py
python src/lane_detection.py
python src/curvature.py       # prints the curvature and offset of the test images
python src/visualization.py
```

The camera calibration is already stored in `camera_cal/calibration.npz`, so the pipeline runs without recalibrating. Inputs at a different resolution than the calibration (1280x720) are skipped, because the calibration and the perspective transform only apply to that camera.

The figures in this writeup are in `docs/`, and each was chosen to illustrate its step. `output_images/` and `output_videos/` contain the pipeline's results on the provided test material. All of these are examples: the code is not tied to these inputs.

### Applying the calibration to the 960x540 inputs

The pipeline only processes inputs at the calibrated resolution (1280x720) and skips the 960x540 ones. Processing them is outside the scope of this project, but the calibration can be applied to them by hand.

All 960x540 inputs come from the same camera as the rest of the material. When all frames of a video are averaged, the moving road blurs away, but dust specks and a reflection on the windshield stay sharp at fixed positions. They appear in the 960x540 videos too, at positions that one scale and one shift map onto the 1280x720 frames. The 960x540 files are re-framed versions of the original recordings, in one of two ways:

* **The whole frame at 75% size**: `challange*.jpg`, with the car hood visible. Resizing them back to 1280x720 restores the original geometry, so the calibration applies unchanged: `cv2.undistort(cv2.resize(img, (1280, 720)), mtx, dist, None, mtx)`.
* **A crop of the frame, resized to 960x540**: `solid*.jpg`, `whiteCarLaneSwitch.jpg` and `project_video01/02.mp4`. The crop is about x 144-1135, y 102-660 of the 1280x720 frame: the middle part, with the hood cut off.

For the cropped inputs, the distortion coefficients stay the same, because they describe the lens. The camera matrix has to be adapted: cropping moves the principal point by the crop's origin, and resizing scales the focal lengths and the principal point. After resizing a cropped frame to 1280x720:

```python
x0, y0, crop_w, crop_h = 143.7, 102.3, 991.5, 557.7   # crop window in the 1280x720 frame
k = 1280 / crop_w                                      # = 720 / crop_h
mtx_crop = mtx.copy()
mtx_crop[0, 0] *= k                                    # focal lengths
mtx_crop[1, 1] *= k
mtx_crop[0, 2] = (mtx[0, 2] - x0) * k                  # principal point
mtx_crop[1, 2] = (mtx[1, 2] - y0) * k
frame = cv2.resize(frame, (1280, 720))
undistorted = cv2.undistort(frame, mtx_crop, dist, None, mtx_crop)
```

The rest of the pipeline would need the same adaptation for the cropped frames: the perspective points map into the crop with `x' = (x - x0) * k`, `y' = (y - y0) * k`, with the bottom edge raised so that it stays inside the crop. Once again this is out-of-scope for this project, but given the input material an explanation was due. In the rest of the README only 1280x720 images will be considered.

### Project layout

| Path | Content |
|---|---|
| `src/calibration.py` | Chessboard calibration and distortion correction |
| `src/thresholding.py` | Binary image of lane-line pixels: narrow white or yellow stripes |
| `src/perspective.py` | Perspective transform to the birds-eye view and back |
| `src/horizon.py` | Estimating the horizon of a video from its lane lines |
| `src/lane_detection.py` | Histogram, sliding windows, search around the previous fit, polynomial fit |
| `src/curvature.py` | Radius of curvature and vehicle offset in meters |
| `src/visualization.py` | Drawing the lane onto the image, text overlay, writeup figures |
| `src/pipeline.py` | `LanePipeline`, which chains all steps; lane tracking for videos; command line |
| `camera_cal/` | Chessboard images and the stored calibration |
| `docs/` | Figures used in this writeup |

### Pipeline overview

`LanePipeline.process()` in `src/pipeline.py` applies the steps to every image or video frame in this order. The sections below describe each step.

| # | Step | Function | Output |
|---|---|---|---|
| 0 | Camera calibration (once, in advance) | `calibrate_camera()` | camera matrix, distortion coefficients |
| 1 | Distortion correction | `correct_distortion()` | undistorted image |
| 2 | Thresholding | `threshold_binary()` | binary image of lane-line pixels |
| 3 | Perspective transform | `warp()` | birds-eye binary image |
| 4 | Lane-line pixels and polynomial fit | `find_lane_pixels()` / `search_around_poly()`, `fit_lane()` | one polynomial per line |
| 5 | Curvature and vehicle position | `lane_measurements()` | radius and offset in meters |
| 6 | Drawing the result | `draw_lane()`, `annotate()` | output image |

For videos, step 3 uses the horizon estimated from the frames so far, step 4 uses the lane of the previous frames, and the result goes through a plausibility check and smoothing before steps 5 and 6 (see "Pipeline (video)").

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

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The code is in `threshold_binary()` in `src/thresholding.py`. Run `python src/thresholding.py` to regenerate the example images.

A painted lane line is a **narrow stripe that is lighter or more colorful than the road on both of its sides**. The binary image is built from that property:

* **Two HLS channels.** White paint stands out in the lightness channel. Yellow paint stands out in the saturation channel: asphalt, concrete and white paint are nearly grey, so a yellow line is visible there even on pale concrete, where it is not lighter than the road. Dark shadow pixels, foliage and sky can be saturated as well, so of the saturated stripes only those with a yellow hue (15-35 on OpenCV's 0-179 scale) are kept.
* **Stripe response** (`stripe_response()`): a morphological top-hat with a horizontal structuring element 31 px wide. The opening removes every bright structure narrower than the element, and subtracting it from the channel leaves only those narrow structures. Flat areas give 0, and so do edges such as asphalt seams, shadow borders and the road edge, because they are darker on one side only.
* **Threshold chosen per image** (`stripe_mask()`): Otsu's method splits the stripe responses on the road into two groups, road texture and stripes, so the threshold follows the light and contrast of every frame. Lines far ahead are thinner, blurrier and stand out less, so the road area is divided into 4 horizontal bands with a threshold each. A minimum contrast applies when a band contains no line and Otsu would split pure texture. Rows outside the road area (sky, trees, hood) stay empty.

The two masks are combined with a logical OR. The middle panel shows which channel found each pixel: green for white stripes, blue for yellow ones. Both lines are found in full, in the shade of the trees as well as on the pale concrete:

![Binary Example][image3]
![Binary Example (shadows)][image3b]

An earlier version of this step used fixed thresholds on a Sobel gradient and on HLS colors. It worked on the project video, but on the challenge videos asphalt seams and shadow borders were detected as lines, and fixed thresholds failed in deep shade and glare. Take a look at the discussion section for more info.

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code is in `src/perspective.py`. `get_perspective_matrices()` computes the matrix `M` from the camera view to the birds-eye view with `cv2.getPerspectiveTransform()`, and the inverse `Minv` for warping results back (section 6). `warp()` applies a matrix with `cv2.warpPerspective()`.

The source points form a trapezoid on the road that becomes a rectangle in the birds-eye view. They were measured once on the straight-road images, by fitting a line to each lane line: where the lines cross the row just above the hood, and their vanishing point, where they meet on the horizon (row 420). `source_points()` builds the trapezoid from these: the bottom corners lie on the lane lines just above the hood, so the hood stays out of the birds-eye view, the sides point at the vanishing point, and the top edge lies 40 rows below the horizon. Like the camera calibration, these measurements are a fixed setting of the camera mounting. For the measured horizon this gives:

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

**The horizon moves.** The row of the horizon depends on the camera's pitch and on the slope of the road, so it differs between videos and changes on hills. With a wrong horizon, parallel lane lines converge or diverge in the birds-eye view. In videos the horizon is therefore estimated from the frames (`src/horizon.py`), and the trapezoid is rebuilt for it with `source_points()`. Keeping the top edge at the same distance below the horizon keeps the view looking the same distance ahead.

* `vanishing_row()` finds line segments in the binary image with the probabilistic Hough transform (`cv2.HoughLinesP()`), using only the rows near the car. Segments leaning to the right on the left half form the left lane line, segments leaning to the left on the right half the right one. Each side's segments are averaged into one straight line, weighted by their length, and the two lines are intersected.
* `HorizonEstimator` keeps the median of the last 3 seconds of these rows, which ignores single bad frames. Rows more than 60 px from the measured horizon are not a plausible pitch or slope change and are dropped.
* Line segments near the car only point at the vanishing point on a nearly straight road, so the estimate is only updated while the tracked lane's radius is at least 500 m.

Single images have no history, so they use the measured horizon. Both straight-road images and the project video agree with it within a few rows. In `challenge02.mp4`, the horizon lies about 25 rows lower, so the lines converged strongly in the fixed birds-eye view.

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

at the bottom of the image, closest to the car. The lane's radius is that of its center line, the mean of the two polynomials. Averaging the two values instead would let one nearly straight line, with a radius close to infinity, dominate the result.

**Vehicle position.** The camera is assumed to sit on the car's centerline, so the car is at the horizontal center of the camera image. `car_position_px()` maps that point through the perspective transform, because the source trapezoid is not symmetric about the image center. `vehicle_offset()` returns the distance between the car and the midpoint of the two lines at the bottom of the image, in meters. Positive means right of the lane center.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The code is in `draw_lane()` in `src/visualization.py`. Run `python src/visualization.py` to regenerate the example images. The lane is drawn in the birds-eye view, where the fits are defined: the area between the two fitted polynomials is filled green, and the detected line pixels are colored red (left) and blue (right). `cv2.warpPerspective()` with the inverse matrix `Minv` from section 3 warps the drawing back into the camera view. The lane area is blended onto the undistorted image at 30% opacity, and the line pixels are drawn on top at full opacity.

Finally, `annotate()` writes the lane's radius of curvature and the vehicle's offset from the lane center (section 5) onto the image. `LanePipeline` in `src/pipeline.py` chains all steps:

![Output][image6]

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

An example of a video output can be found [here](./output_videos/project_video03_output.mp4). The other videos at the calibrated resolution can be processed and saved in `output_videos/` as well.

For videos, `LanePipeline` in `src/pipeline.py` keeps track of the lane between frames:

* **Horizon.** The birds-eye view follows the estimated horizon of the video (section 3). When it moves, the tracked lane is converted into the new view.
* **Search.** If the previous frame had a lane, the lines are searched around it (`search_around_poly()`, section 4). If that result is not plausible, the sliding-window search runs on the same frame.
* **Sanity check.** `is_plausible()` accepts a detection only if both lines were found, the lane is about as wide as in the birds-eye view (within 30%), and the lines are roughly parallel. The radius of curvature is not limited, because tight curves are legitimate on winding roads.
* **Smoothing.** Each `Line` keeps its last 5 accepted fits, and their average is drawn. This removes jitter between frames.
* **Recovery.** If no plausible lane is found, the last good lane is still shown. After 5 failed frames in a row, the history is cleared, so a stale lane is not shown and the next frame starts over with the sliding windows.

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

**Inputs from a different framing.** The 960x540 inputs are either scaled-down or cropped versions of the calibrated camera's frame, and it would be complicated to try to tell which. The pipeline therefore skips them. Averaging the frames of the videos revealed the crop window, and with it the camera matrix (see "Applying the calibration to the 960x540 inputs"). Passing the framing as a camera setting would let the pipeline process them as well.

**Edges that look like lines.** The first version of the thresholding combined a Sobel gradient mask and fixed color thresholds with a logical OR. The gradient mask finds any strong vertical edge, not only paint: in `challenge02.mp4` a seam between two kinds of asphalt runs along the lane, and the concrete barrier casts a sharp shadow next to the yellow line, and both were detected as lane lines. Requiring an edge to also be paint-colored (a logical AND of the two masks) removed them, but also removed real lines: faded or shaded paint failed the color thresholds, and a yellow line on pale concrete has almost no lightness edge. Detecting narrow stripes instead of edges, with thresholds chosen per image (section 2), solved both problems. The lane is now found in most of `challenge02.mp4` and `challenge03.mp4`.

**A different horizon.** Even with clean line detection, `challenge02.mp4` mostly failed the sanity check: throughout the video, the lines converged strongly in the birds-eye view instead of staying parallel. The horizon of that video lies about 25 rows lower than in the images the transform was measured on, because of the camera's pitch or the slope of the road. Estimating the horizon from the frames (section 3) fixed this, apart from the first seconds, while the estimate is still forming. In bends the estimate is paused, so on winding roads it reacts slowly to slope changes.

**Tight curves and changing light.** In the hairpins of `challenge03.mp4` the lines leave the fixed birds-eye trapezoid sideways within a few meters, and the polynomial can't describe them. The thresholds adapt to each frame, but in strong glare the lines are washed out and can't be separated from the road at all. A shorter look-ahead in tight curves would help with the hairpins.

**Assumptions.** The perspective transform assumes a flat road between the car and the far edge of the view. The horizon estimate follows slow changes of pitch and slope, but not quick ones such as hard braking. `fit_lane()` and the sanity check assume parallel lane lines, which does not hold where lanes merge or split. These cases are outside the scope of this project.

