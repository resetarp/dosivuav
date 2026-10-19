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
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

---

### Writeup / README

### Camera Calibration

The code is in `calibrate_camera()` in `src/calibration.py`. Run `python src/calibration.py` to recompute the calibration and regenerate the example images.

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

TODO: Add your text here!!!

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

TODO: Add your text here!!!

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

TODO: Add your text here!!!

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

