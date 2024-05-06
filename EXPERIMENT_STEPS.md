## Experiment Steps

From [mono-odometry](https://github.com/alishobeiri/Monocular-Video-Odometery)
Capture images: It and It + 1
Undistort the captured images (not necessary for KITTI dataset)
Use FAST algorithm to detect features in image It. Track these features using an optical flow methodology, remove points that fall out of frame or are not visible in It + 1. Trigger a new detection of points if the number of tracked points falls behind a threshold. Set to 2000 in this implementation.
Apply Nister's 5-point algorithm with RANSAC to find the essential matrix.



### Steps:

1. Pull pairwise images $I_t$ and $I_{t+1}$ from Notre Dame dataset

2. Compute ffeature points (at least 8) $m_t$ and $m_{t+1}$ for each image

3. Find pairwise matching points $<x, x'>$ using optical flow algorithm, OpenCV uses KTL

4. As the camera matrix K is not known for Notre Dame dataset, we need to estimate it. This gives us the Fundamental Matrix F that can be found using OpenCV 

[Project 3 / Camera Calibration and Fundamental Matrix Estimation with RANSAC](https://sites.cc.gatech.edu/classes/AY2016/cs4476_fall/results/proj3/html/euzun3/index.html) is a good reference that uses Notre Dame dataset




### Obtaining ground truth

* Extract data from [notredame.out](http://www.cs.cornell.edu/~snavely/bundler/bundler-v0.3-manual.html#S6) 