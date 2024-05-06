## Niter2 Python

This is the python port of Dr. Lindstorm's ```niter2``` triangulation method from his 2010 paper[1]. 

The key problem this method solves is recovering 3D scene given two or more images are obtained from a calibrated(or uncalibrated) camera.

TODO complete this write up once the project is ready.

### Requirements
    * OpenCV>= 4.2
    * Numpy
    * Matplotlib
    * Numba
    * Ruff [optional for linting]

### Helpful tutorials
* 

### Some methodology writing points

1. Use images from [Simple Stereo](https://www.youtube.com/watch?v=hUVyDabn1Mg&list=PL2zRqk16wsdoCCLpou-dGo7QQNks1Ppzo&index=5) video to write draw the epipolar geometry like figure

2. Epipolar geometry: The mathematical model that concisely describes how points in the left and right images are related to each other through a 3x3 matrix called the Fundamental matrix. If Fundamental matrix is found then we can find the rotation and translation of one camera with respect to the other

3. Excellent video of [epipolar geometry](https://www.youtube.com/watch?v=6kpBqfgSPRc&list=PL2zRqk16wsdoCCLpou-dGo7QQNks1Ppzo&index=8)

### TODOs

- [ ] If time permits, figure out a way to access the ground-truth camera poses from ```notredame.out``` and then compare it with the Opencv.pose_estimate() based on the 3D points tracked by this method 


### References
[1] Peter Lindstorm, (2010), "Triangulation Made Easy"

[2] Stachniss, (2020), "Triangualtion from Image Pairs", URL: https://www.youtube.com/watch?v=UZlRhEUWSas&t=143s

[3] ???, (2019), "Closed Form Optimal Two-View Triangulation Based on Angular Errors"

[4] Lee and Civera, "Triangulation: Why optimize?"

[5] OpenCV, "Camera Calibration and 3D Reconstruction" URL: https://docs.opencv.org/4.5.5/d9/d0c/group__calib3d.html#gad3fc9a0c82b08df034234979960b778c

[6] OpenCV, "Epipolar Geometry", URL: https://docs.opencv.org/4.x/da/de9/tutorial_py_epipolar_geometry.html

[7] Shree Nayar, "Camera Calibration", URL: https://www.youtube.com/watch?v=S-UHiFsn-GI&list=PL2zRqk16wsdoCCLpou-dGo7QQNks1Ppzo&index=1

[8] Stanchiss, "Epipolar Geometry Basics", URL:https://www.youtube.com/watch?v=cLeF-KNHgwU&pp=ygUQZXNzZW50aWFsIG1hdHJpeA%3D%3D

[9] CMU, "Two view geometry", URL: http://16720.courses.cs.cmu.edu/lec/two-view2.pdf

[] Unknown, "Project 3: Fundamental Matrix estimation", URL: https://sites.cc.gatech.edu/classes/AY2016/cs4476_fall/results/proj3/html/euzun3/index.html