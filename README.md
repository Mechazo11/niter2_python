## Niter2 Python

### Preamble
This is the python port of Dr. Lindstorm's ```niter2``` image-space triangulation method from his [2010 paper](https://ieeexplore.ieee.org/document/5539785). This was done as part of the Final Project for ME 7953 Applied Computational Methods taught by Dr. Ingmar Schoegl at Louisiana State University.

### Implementation notes
```niter2``` primarily optimizes matched keypoints by moving them on the epipolar line in such a way that intermediate steps obeys epipolar constraint. Lindstorm later demonstrated that with only two steps, reprojection error may reach as low as machine precision. Thus, ```niter2``` is a two interation process. Please refer to the original paper (given in the repository) for further details.

```Niter2``` is the class that implements the ```niter2``` algorithm. A detail description of how it was implemented here may be found in the ```final.pdf``` report or in ```final.ipynb``` Secion 3. Hence, a method to solve the $x_i = P_i X$ is required. In ```niter2```, ```linear-triangulation``` based on this [lecture](http://cmp.felk.cvut.cz/cmp/courses/TDV/2012W/lectures/tdv-2012-07-anot.pdf) is given. Here are the key function calls to use this method

### Setup

It is highly recommended that either ```anaconda``` or ```miniforge3``` with the [mamba](https://github.com/mamba-org/mamba) package manager instead of [conda](https://en.wikipedia.org/wiki/Conda_(package_manager)). If you are using ruff, my configurations for it are given in the ```pyproject.toml``` file.

* OpenCV >= 4.9.0
* Numpy
* Matplotlib
* Numba
* Natsort
* PyYAML
* Ruff [optional for linting]

### Usage

* To optimize matched keypoints: ```Niter2.triangulate_niter2(args**)->Tuple[np.ndarray, np.ndarray]```
* To use the provided ```linear-triangulation```: ```Niter2.linear_triangulate()->np.ndarray```
* Check ```final.ipynb``` and ```test.ipynb``` to see how ```Niter2``` can be used in computer vision project.

### TODOs

* [x] Convert Niter2 into a standalone module, make numba accelerated functions staticmethods and pin the project.
* [x] Release version 1.0 of ```Niter2```
* [] Use ```CuPy``` to convert ```niter2_triangulate()``` into a GPU accelerated method

### Future Upgrade?

* [] Use the Dinosaur dataset to test full 3d point cloud generation https://vision.middlebury.edu/mview/data/
* [] Find out how to do Bundle adjustment to reduce error.
* [] Go thru this project to add chilrety test https://github.com/sakshikakde/SFM
* [] https://unisvalbard.github.io/Geo-SfM/content/lessons/l1/sfm_photogrammetry.html