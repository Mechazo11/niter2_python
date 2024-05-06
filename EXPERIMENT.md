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

2. Compute ffeature points (at least 8) $m_t$ and $m_{t+1}$ for each image using
    * SIFT
    * ORB

3. Find pairwise matching points $<x, x'>$ using optical flow algorithm, OpenCV uses KTL

4. As the camera matrix K is not known for Notre Dame dataset, we need to estimate Fundamental matrix F.

[Project 3 / Camera Calibration and Fundamental Matrix Estimation with RANSAC](https://sites.cc.gatech.edu/classes/AY2016/cs4476_fall/results/proj3/html/euzun3/index.html) is a good reference that uses Notre Dame dataset


# Important resources for implementation

1. opencv_epipolar: https://docs.opencv.org/4.x/da/de9/tutorial_py_epipolar_geometry.html

2. [OpenCV and Sift](https://docs.opencv.org/4.x/da/de9/tutorial_py_epipolar_geometry.html) or 

3. [OpenCV with ORB](https://towardsdatascience.com/improving-your-image-matching-results-by-14-with-one-line-of-code-b72ae9ca2b73)

4. Worked out example of triangulation using OpenCV: https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html

5. Finding R and t from Essential matrix: https://inst.eecs.berkeley.edu/~ee290t/fa19/lectures/lecture10-3-decomposing-F-matrix-into-Rotation-and-Translation.pdf
Using opencv: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga54a2f5b3f8aeaf6c76d4a31dece85d5d

6. More useful information on point triangulation: https://inst.eecs.berkeley.edu/~ee290t/fa19/lectures/lecture10-3-decomposing-F-matrix-into-Rotation-and-Translation.pdf

### Obtaining ground truth

* Extract data from [notredame.out](http://www.cs.cornell.edu/~snavely/bundler/bundler-v0.3-manual.html#S6) DEPRICATED, very difficult and Blunder does not install correctly, last updated over 10 years ago