**Advanced Lane Finding Project**

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

[image1]: ./output_images/calibration_example.jpg "Undistorted"
[image2]: ./output_images/undistorted_test.png "Undistorted"
[image3]: ./output_images/channel_dmensions.png "LHS dimension"
[image4]: ./output_images/gradient_and_color.jpg "Binary Example"
[image5]: ./output_images/warping.png "Warp Example"
[image6]: ./output_images/unwarping_polynomial.png "Fit Visual"
[image7]: ./output_images/curvature.png "Curvature"
[video1]: ./final.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the code cell of the IPython notebook.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Once the distortion matrix is calculated, it is easy to undistort an image. Simple call
`undistort_image(image)`

![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image.  I analyzed LHS transformations, grayscale, and sobel gradient methods of finding image portions. To see this in action, you can run the `make_test_gray_S_gradient_movies.py` and `make_test_gray_S_H_movies.py` scripts that will convert the project movie to a binary video. A sample of these outputs can be seen in the image below.

![alt text][image3]
![alt text][image4]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a section called `Road Warping`, which appears in the python notebook.  The `warp_lines()` function takes as inputs an image and a transformation matrix that defines the warping transformation. This matrix can be retrieved from a `cv2.getPerspectiveTransform` function. as a default, and for this project's implementation, I chose the following points of transformation:
- top left: (541, 493)
- top right: (759, 493)
- bottom left: (250, 687)
- bottom right: (1052, 687)

After further refinement and experimentation shown below, I found that the following transformations are the best:
- top left: (529, 493) --> (250, 450)
- top right: (759, 493) --> (1080, 450)
- bottom left: (235, 687) --> (250, 685)
- bottom right: (1072, 687) --> (1080, 685)

I verified that my perspective transform was working as expected by drawing the source and destination points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

After warping the road image, I divided the image vertically into 15 windows, and calculated the most dense areas of both the right and left side of the image. Then I flip the most dense points to be horizontal and fit a polynomial through them. Afterwards, I unwarp the polynomial and overlay it back onto the original image. Examples of this can be seen in the image below.

The code for this is in the **Line angle measurement and detection** section of the notebook.
![alt text][image6]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in section titled **Line angle measurement and detection** in the notebook.  Here is a simple example:
![alt text][image7]

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I did this in section titled **Line angle measurement and detection** in the notebook. Please see section 4 above for this example.

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./final.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The program I developed works surprisingly well, but suffers from jitteriness with the striped portion of the lane marking. To allay this problem, we can keep the polynomial measurements from the previous steps, and adjust it slowly instead of throwing it out all together and calculating a new one. This will in a sense be a running average of the polynomial, which will help with the smoothness of the polynomial and avoiding large changes due to abnormal behavior.
