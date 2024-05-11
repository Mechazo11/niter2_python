## Niter2 Python

This is the python port of Dr. Lindstorm's ```niter2``` triangulation method from his 2010 paper[1]. 

The key problem this method solves is recovering 3D scene given two or more images are obtained from a calibrated(or uncalibrated) camera.

TODO complete this write up once the project is ready.

TODO the commands to setup enviornment

### Requirements
    * ```mamba create --name niter2 python=3.9
    * ```micromamba activate niter2```
    * OpenCV= 4.5.5
        * ```mamba install -c conda-forge opencv=4.5.5=py39hf3d152e_9```
    * Numpy
    * Matplotlib
    * Numba
    * Natsort
    * Ruff [optional for linting]
    * PyYAML

### Report write ups

* State that ```Triangulation``` is also called the ```Structure-from-Motion (SfM)``` problem

* In methodology secion, in short show how the math for $x^TFx' = 0$ comes out

* Also show $E = K_l^T F K_r$ equation to explain why we unable to use Essential matrix for the NotreDam dataset

* From paper, show the key steps for the non-iterative niter2 method


### Some methodology writing points

1. Use images from [Simple Stereo](https://www.youtube.com/watch?v=hUVyDabn1Mg&list=PL2zRqk16wsdoCCLpou-dGo7QQNks1Ppzo&index=5) video to write draw the epipolar geometry like figure

2. Epipolar geometry: The mathematical model that concisely describes how points in the left and right images are related to each other through a 3x3 matrix called the Fundamental matrix. If Fundamental matrix is found then we can find the rotation and translation of one camera with respect to the other

3. Excellent video of [epipolar geometry](https://www.youtube.com/watch?v=6kpBqfgSPRc&list=PL2zRqk16wsdoCCLpou-dGo7QQNks1Ppzo&index=8)

### TODOs

- [ ] If time permits, figure out a way to access the ground-truth camera poses from ```notredame.out``` and then compare it with the Opencv.pose_estimate() based on the 3D points tracked by this method 


### References

[0] Oxford Univ., "Multiview dataset", URL: https://www.robots.ox.ac.uk/~vgg/data/mview/

[0] D Lowe, (2004) "Scale-Invariant Feature Transform", URL: https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html

[1a] Lowe,(2004), "Distinctive Image Features from Scale-Invariant Keypoints"

[1] Peter Lindstorm, (2010), "Triangulation Made Easy"

[2] Stachniss, (2020), "Triangualtion from Image Pairs", URL: https://www.youtube.com/watch?v=UZlRhEUWSas&t=143s

[3] ???, (2019), "Closed Form Optimal Two-View Triangulation Based on Angular Errors"

[4] Lee and Civera, "Triangulation: Why optimize?"

[5] OpenCV, "Camera Calibration and 3D Reconstruction" URL: https://docs.opencv.org/4.5.5/d9/d0c/group__calib3d.html#gad3fc9a0c82b08df034234979960b778c

[6] OpenCV, "Epipolar Geometry", URL: https://docs.opencv.org/4.x/da/de9/tutorial_py_epipolar_geometry.html

[7] Shree Nayar, "Camera Calibration", URL: https://www.youtube.com/watch?v=S-UHiFsn-GI&list=PL2zRqk16wsdoCCLpou-dGo7QQNks1Ppzo&index=1

[8] Stanchiss, "Epipolar Geometry Basics", URL:https://www.youtube.com/watch?v=cLeF-KNHgwU&pp=ygUQZXNzZW50aWFsIG1hdHJpeA%3D%3D

[9] CMU, "Two view geometry", URL: http://16720.courses.cs.cmu.edu/lec/two-view2.pdf

[10] BigSFM: Reconstructing the World from Internet Photos, URL: https://www.cs.cornell.edu/projects/bigsfm/

[11] Li, Snavely, Huttenlocher, "Location Recognition using Prioritized Feature Matching" URL: https://www.cs.cornell.edu/projects/p2f/

[12] PyYML, "PyYML", URL: https://python.land/data-processing/python-yaml

[13] Numba, "A ~5 minutes guide to Numba", URL: https://numba.pydata.org/numba-doc/dev/user/5minguide.html

[14] Amy Tabb, "N-view triangulation: DLT method", URL: https://amytabb.com/tips/tutorials/2021/10/31/triangulation-DLT-2-3/

[15] R. Sara, (2012), "The Triangulation Problem", URL: http://cmp.felk.cvut.cz/cmp/courses/TDV/2012W/lectures/tdv-2012-07-anot.pdf