## Experiment Steps

Study we will conduct

1. Effect of features detected [using NotreDame dataset]

* points/sec between two methods using SIFT features
* points/sec between two methods using ORB features
From the above we chooose the best feature to use with the next study

2. Convergence of translation and rotation estimation: 

On the house dataset compare against the ground truth
    * R and t found by DLT
    * R and t found by niter2


### Steps:

1. Pull pairwise images $I_t$ and $I_{t+1}$ from Notre Dame dataset

2. Compute feature points (at least 8) $m_t$ and $m_{t+1}$ for each image using
    * SIFT
    * ORB

3. Find pairwise matching points $<x, x'>$ using optical flow algorithm, OpenCV

4. Estimate Fundamental matrix

5. Compute Essential matrix

6. Compute 2nd camera position w.r.t first by decomposing essential matrix

7. triangulate points using openCV [DLT](https://docs.opencv.org/4.5.5/d9/d0c/group__calib3d.html#gad3fc9a0c82b08df034234979960b778c)

8. Apply Lindstorms's code and compute map points

8b. Compute $X\in R^3$ position by solving $x = P.X$ problem using non linear
least squares approach. Based on this [gist](https://gist.github.com/davegreenwood/e1d2227d08e24cc4e353d95d0c18c914)

9. Record stats for point/sec
10. Record RMSE of triangulated points

# Important resources for implementation

1. opencv_epipolar: https://docs.opencv.org/4.x/da/de9/tutorial_py_epipolar_geometry.html

2. [OpenCV and Sift](https://docs.opencv.org/4.x/da/de9/tutorial_py_epipolar_geometry.html) or 

[1] OpenCV, "Feature Matching with FLANN", URL:https://docs.opencv.org/4.x/d5/d6f/tutorial_feature_flann_matcher.html

[1] OpenCV, "Feature Matching", URL: https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html 

3. [OpenCV with ORB](https://towardsdatascience.com/improving-your-image-matching-results-by-14-with-one-line-of-code-b72ae9ca2b73)

4. CMU, "Triangulation", URL: http://www.cs.cmu.edu/~16385/s17/Slides/11.4_Triangulation.pdf

5. Finding R and t from Essential matrix: https://inst.eecs.berkeley.edu/~ee290t/fa19/lectures/lecture10-3-decomposing-F-matrix-into-Rotation-and-Translation.pdf

5. Using opencv: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga54a2f5b3f8aeaf6c76d4a31dece85d5d

6. More useful information on point triangulation: https://inst.eecs.berkeley.edu/~ee290t/fa19/lectures/lecture10-3-decomposing-F-matrix-into-Rotation-and-Translation.pdf

8. Another note on Triangualation, [SfM](https://www.youtube.com/watch?v=JlOzyyhk1v0)

9. [Project 3 / Camera Calibration and Fundamental Matrix Estimation with RANSAC](https://sites.cc.gatech.edu/classes/AY2016/cs4476_fall/results/proj3/html/euzun3/index.html) is a good reference that uses Notre Dame dataset

10. Converting from 4D homogenenous to 3d coordinate: https://stackoverflow.com/questions/58543362/determining-3d-locations-from-two-images-using-opencv-traingulatepoints-units

### Obtaining ground truth

* Extract data from [notredame.out](http://www.cs.cornell.edu/~snavely/bundler/bundler-v0.3-manual.html#S6) DEPRICATED, very difficult and Blunder does not install correctly, last updated over 10 years ago