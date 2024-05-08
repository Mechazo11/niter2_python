"""
Project - Python port of niter2 algorithm from "Triangulation Made Easy" paper.

Created in partial submission to the requirements of
ME7953 Applied Computational Methods, Spring 2024

Author:
Azmyin Md. Kamal,
Ph.D. student in MIE,
Louisiana State University,
Louisiana, USA

Date: May 5th, 2024
Version: 0.1

AI: ChatGPT

"""

#Imports
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import time
from typing import Tuple, List
import yaml
from pathlib import Path
import natsort
import cv2


class DataSetLoader:
    """
    Compile image paths and camera intrinsic parameters for chosen dataset.
    
    Constructor only, all processing done in this method.
    """

    def __init__(self, dataset_name:str = "") -> None:
        """Class constructor."""
        # Initialize variables
        self.parent_dir = ""
        self.dataset_path = ""
        self.image_path = ""
        self.num_images = 0
        self.image_path_lss = [] # List[str,str.....]
        self.dataset_yaml_file = "" # Path to dataset.yaml
        self.calibration_matrix = np.zeros((3,3), dtype=float) # K matrix, 3x3
        if not isinstance(dataset_name, str):
            err_msg = "Dataset name must be a string!"
            raise TypeError(err_msg)
        if dataset_name == "":
            err_msg = "Dataset name must be a string!"
            raise ValueError(err_msg)
        if not Path("./config.yaml").is_file():  # noqa: PTH113, PTH118
            err_msg = "config.yaml not found in root directory!"
            raise FileNotFoundError(err_msg)
        with Path('config.yaml').open() as file:
            config_data = yaml.safe_load(file)
        self.parent_dir = config_data["Directory"]["parent_dir"]
        self.dataset_path = self.parent_dir + dataset_name + "/"
        self.image_path = self.dataset_path + "images/"
        self.dataset_yaml_file = self.dataset_path + "dataset.yaml"
        # Check if /image subdirectory exsists
        if not Path(self.image_path).is_dir():
            err_msg = "Dataset does not have the /image directory!"
            raise FileNotFoundError(err_msg)
        self.build_image_paths() # Build paths to all images
        # Watchdog, check if dataset.yaml exsists
        if not Path(self.dataset_yaml_file).is_file():  # noqa: PTH113, PTH118
            err_msg = "dataset.yaml not found in the dataset directory!"
            raise FileNotFoundError(err_msg)
        self.extract_camera_params() # Retrieve camera matrix
        # DEBUG test if pairwise images can be loaded correctly
        #self.test_pairwise_images()

    def build_image_paths(self)->None:
        """Build path to all images in /image directory."""
        im_pth = self.image_path
        self.image_path_lss = natsort.natsorted(list(Path(im_pth).iterdir()),
                                              reverse=False)
        # From pathlib.PosixPath to str
        self.image_path_lss = [str(path) for path in self.image_path_lss]
        self.num_images = len(self.image_path_lss)
    
    def extract_camera_params(self):
        """Extract camera configuration parameters."""
        with Path(self.dataset_yaml_file).open() as file:
            dataset_data = yaml.safe_load(file)
        self.calibration_matrix[0,0] = float(dataset_data["Camera"]["fx"])
        self.calibration_matrix[0,2] = float(dataset_data["Camera"]["cx"])
        self.calibration_matrix[1,1] = float(dataset_data["Camera"]["fy"])
        self.calibration_matrix[1,2] = float(dataset_data["Camera"]["cy"])
        self.calibration_matrix[2,2] = 1.0
        #DEBUG
        #print(f"calibration matrix: {self.calibration_matrix}")
    
    def test_pairwise_images(self):
        """Test if images are loaded in correct sequence."""
        for i in range(self.num_images - 1):
            left_img = cv2.imread(self.image_path_lss[i],cv2.IMREAD_UNCHANGED)
            right_img = cv2.imread(self.image_path_lss[i+1],cv2.IMREAD_UNCHANGED)
            cv2.imshow("Left image", left_img)
            cv2.imshow("Right image", right_img)
            cv2.waitKey(100)
        cv2.destroyAllWindows()

class Niter2:
    """Non-iterative niter2 triangulation algorthm."""

    pass

# def drawlines(img1,img2,lines,pts1,pts2):
#     ''' img1 - image on which we draw the epilines for the points in img2
#     lines - corresponding epilines '''
#     r,c = img1.shape
#     img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
#     img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
#     for r,pt1,pt2 in zip(lines,pts1,pts2):
#     color = tuple(np.random.randint(0,255,3).tolist())
#     x0,y0 = map(int, [0, -r[2]/r[1] ])
#     x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
#     img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
#     img1 = cv.circle(img1,tuple(pt1),5,color,-1)
#     img2 = cv.circle(img2,tuple(pt2),5,color,-1)
#     return img1,img2

def lowe_ratio_test(matches:tuple, kp1:tuple, kp2:tuple)->Tuple[List, List]:
    """Do Lowe's ratio test to return best matching keypoints."""
    # Initialize variables and constants
    pts1 = []
    pts2 = []
    dist_thres = 0.8
    for _,(m,n) in enumerate(matches):
        if m.distance < dist_thres*n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    # DEBUG
    # print(f"Num keypoints in pts1: {len(pts1)}")
    # print(f"Num keypoints in pts2: {len(pts2)}")
    # Convert to 32-bit integers
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    return pts1, pts2

def find_matching_keypoints(feature_detector:str, img1:np.ndarray,
                            img2:np.ndarray)->None:
    """Compute and match keypoints in a given set of images."""
    # Initilize work variables
    kp1, des1 = None, None
    kp2, des2 = None, None
    pts1, pts2 = [],[]
    if(feature_detector == "SIFT"):
        # Setup FLANN, as suggested in https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
        FLANN_INDEX_KDTREE = 1  # noqa: N806
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)  # noqa: C408
        search_params = dict(checks=50)  # noqa: C408
        flann = cv2.FlannBasedMatcher(index_params,search_params) # FLANN object
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)
        # print(type(kp1)) # class tuple
        # print(type(des1))# class numpy.ndarray
        matches = flann.knnMatch(des1,des2,k=2) # Find an initial matches of keypoints
    else:
        # ORB
        pass
    # Perform Lowe's ratio test and return best matching points
    pts1,pts2 = lowe_ratio_test(matches, kp1, kp2)
    return pts1, pts2

def compute_fundamental_matrix(pts1:List, pts2:List)->None:
    """Call cv2.finFundamentalMat() and return fundamental matrix."""
    # Initialize work variables
    F_mat, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
    print(type(F_mat))
    # Select inliners only
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]

def test_dev(dataset_name: str, feature_detector:str,
             show_verbose:bool = True)->None:
    """TODO."""
    if feature_detector.strip().upper() not in ["ORB", "SIFT"]:
        err_msg = "choose either 'ORB' or 'SIFT' feature."
        raise ValueError(err_msg)
    # Intialize variables
    left_img = None
    right_img = None
    pts1, pts2 = [],[]
    F_mat, mask = None, None

    # Define objects
    dataloader = DataSetLoader(dataset_name)
    lp = dataloader.image_path_lss[0]
    rp = dataloader.image_path_lss[1]
    # Load images and make them grayscale
    left_img = cv2.imread(lp,cv2.COLOR_BGR2GRAY)
    right_img = cv2.imread(rp,cv2.COLOR_BGR2GRAY)
    pts1, pts2 = find_matching_keypoints(feature_detector,left_img, right_img)
    compute_fundamental_matrix(pts1, pts2)

    
    # DEBUG print stats
    print()
    print(f"Number of images in dataset: {dataloader.num_images}")
    print(f"Feature detector selected: {feature_detector}")
    print()
    