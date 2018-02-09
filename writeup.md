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

[image1]: ./writeup_calibration1_undist_before.jpg "original"
[image1_1]: ./writeup_calibration1_undist_after.jpg "Undistorted"

[image2]: ./writeup_undisorted_test_image.jpg   "undistorted test image"
[image3]: ./writeup_combined_binary.jpg         "Binary Example"
[image4]: ./writeup_perspected_transform_before.jpg "perspective transform before"
[image4_1]: ./writeup_perspected_transform_after.jpg "perspective transform after"
[image5]: ./test_images/test4.jpg                   "original"
[image5_1]: ./test_images_output/test4.jpg           "output"
[video1]: ./project_video_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

It can be found in 474 lines to 532 lines of final.py   
First of all, I need to have object points and image points about all calibration files. It can be done by cv2.findChessBoardCorners().    
After That,  cv2.callibrateCamera() is called with objpoints and imgpoints.  The function returns 5 values.But all we need to use is mtx and dist.    
Finally, cv2.undistort() is called with mtx ,dist. The function returns an undistorted image.    

In addition, I used python pickle  to save time. After executing one time, PICKLE_READY can be set to True.


![alt text][image1]
![alt text][image1_1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

I'll show one of undistorted test images.              
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I tried to get the best result using a combination of color and gradient thresholds.  The code can be found from 175 to 265 lines of final.py     

One of example is below. Actually,I eliminate other scene before processing. that's the reason why the image only has lane lines.    


![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

You can find it from line 228 to line 244.     
I need to make source/destination rectangles.    
I got source values by hand using image viewers.   
I got destination values using trial & error.  


```python
    #I got these values directly by an image viewer.
    src = np.float32( [[540,495],
                      [750,495],
                      [376,603],
                      [922,603]])
    
    #I got these values by trial & error.
    dst = np.float32( [[400,400],
                      [600,400],
                      [400,603],
                      [600,603]])
```


I tested it on one of test images.     
![alt text][image4]

I got following result.     
![alt text][image4_1]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I found the lane lines by windowing. I tried to avoid redundant windowing calculation. The variable need_windowing is for it.
I fit the lane lines with a 2nd order polynomial using np.polyfit()


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

you can find the code  in line 108 to line 121. I wrote this code from the radious of curvature fomular.   
I calculated this pixel based. I thought it's ok because I can convert it to meter based when I need to put some text on frames.    

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.
I'll test it on one of test images(test4.jpg).

![alt text][image5]


I got this result from upper image.
![alt text][image5_1]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?





