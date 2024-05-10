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
from numba import jit, prange
import time
from typing import Tuple, List
import yaml
from pathlib import Path
import natsort
import cv2

# Utility functions
def curr_time():
    """Return the time tick in milliseconds."""
    return time.monotonic() * 1000

def debug_lock():
    """Locks system in an infinite loop for debugging."""
    print("LOCK")
    while (1):
        pass

@jit(nopython = True, parallel = False)
def _non_iter_update(algorithm:np.ndarray,left_pts:np.ndarray, right_pts:np.ndarray,
                           e_mat:np.ndarray, s_mat:np.ndarray)->Tuple[np.ndarray,
                                                                      np.ndarray]:
        """Perform non-iterative update as shown in Section 5."""
        """
        Algorithm is shown in Listing 3 in the paper

        Args:
        left_pts:[Kx2], keypoint from left camera. Each row corresponds to x in paper
        right_pts:[Kx2], keypoint from right camera. Each row corresponds to x' in paper
        e_mat: [3x3]: essential matrix
        s_mat: [2x3]: matrix defined in Equation 4
        rot: [3x3]: Rotation matrix (obtained by decomposing essential matrix)

        Output:
        x_hat: [Kx3] updated left points in homogeneous coord
        x_hat_prime: [Kx3] updated right points in homogeneous coord
        """
        # Initialize constant variables
        e_tildae = np.zeros((2,2), dtype=float)
        e_tildae = np.dot(np.dot(s_mat, e_mat),s_mat.T)
        # Initialize work variables
        x = np.zeros(0, dtype=float) # in paper x \in R^3, homogeneous coord
        x_prime = np.zeros(0, dtype=float) # in paper x' \in R^3, homogeneous coord
        n = np.zeros(2, dtype=float) # [1x2] step direction for x
        n_prime = np.zeros(2, dtype=float) # [1x2] step direction for x'
        a = np.zeros(1, dtype=float) # [1x1] scalar for computing \lambda
        b = np.zeros(1, dtype=float) # [1x1] scalar for computing \lambda
        b_rhs = np.zeros(1, dtype=float) # [1x1] scalar for computing \lambda
        c = np.zeros(1, dtype=float) # [1x1], x.T * E * x', scalar
        d = np.zeros(1, dtype=float) # [1x1], scalar
        lambda_ = np.zeros(1, dtype=float) # [1x1], scalar, step size
        del_x = np.zeros(3, dtype=float) # in paper delta_x \in R^3, hom coord
        del_x_prime = np.zeros(3, dtype=float) # in paper delta_x' \in R^3, hom coord
        x_hat = np.zeros((0,3), dtype=float) # [Kx3]
        x_hat_prime = np.zeros((0,3), dtype=float) # [Kx3]

        # Primary loop
        for i in range(left_pts.shape[0]):
            # Initialize variables for these keypoints pairs
            x = np.append(left_pts[i], 1)  # [1x3]
            x_prime = np.append(right_pts[i], 1)  # [1x3]
            n = np.dot(np.dot(s_mat, e_mat),x_prime) # n = S.E.x'
            n_prime = np.dot(np.dot(s_mat, e_mat.T),x) # n'= S.(E.T).x
            a = np.dot(np.dot(n.T, e_tildae),n_prime) # a = (n.T).E_tildae.n'
            b_rhs = (np.dot(n.T,n) + np.dot(n_prime.T, n_prime)) # L_2(n,n') norm
            b = 0.5 * b_rhs # b = 0.5*((n.T).n + (n'.T).n')  # noqa: E501
            c = np.dot(np.dot(x.T, e_mat),x_prime) # c = (x.T).E.x', epipolar constraint
            d = np.sqrt(b**2 - a * c) # d = sqrt(b**2 - a*c)
            lambda_ = c / (b + d) # lambda_ = c / (b+d)
            del_x = lambda_ * n # [1x3]
            del_x_prime = lambda_ * n_prime #[1x3]
            n = n - np.dot(e_tildae,del_x_prime) # n = n-E_tildae.delta_x'
            n_prime = n_prime - np.dot(e_tildae.T, del_x) # n' = n'-(E_tildae.T).delta_x
            if np.array_equal(algorithm, np.array([[2]], dtype=np.int8)):
                # niter2
                lambda_ = lambda_ * ((2*d)/b_rhs)
                del_x = lambda_ * n # [1x3]
                del_x_prime = lambda_ * n_prime #[1x3]
            else:
                # niter1
                # FUTURE WORK
                pass
            # Corrected points <x_hat, x_hat_prime> in homogeneous coord
            x = x - np.dot(s_mat.T, del_x) # x_hat
            x_prime = x_prime - np.dot(s_mat.T, del_x_prime) # x_hat_prime
            # Reshape to correct dimension, [1x3] from [3]
            x = np.reshape(x, (-1,3))
            x_prime = np.reshape(x_prime, (-1,3))
            # Push to matrix
            x_hat = np.vstack((x_hat, x))
            x_hat_prime = np.vstack((x_hat_prime, x_prime))

        # Breaks with Numba
        x_hat = np.floor(x_hat).astype(np.int32)
        x_hat_prime = np.floor(x_hat_prime).astype(np.int32)
        return x_hat, x_hat_prime

def _non_iter_vectorized(algorithm:np.ndarray,left_pts:np.ndarray, right_pts:np.ndarray,
                           e_mat:np.ndarray, s_mat:np.ndarray)->Tuple[np.ndarray,
                                                                      np.ndarray]:
    """Vectorized version of _non_iter_update()."""
    #! THIS VERSION DOES NOT WORK, FUTURE WORK
    e_tildae = np.dot(np.dot(s_mat, e_mat),s_mat.T) # [2x2]
    arr_add = np.ones(left_pts.shape[0], dtype=np.float32).reshape(-1,1) # [Kx1]
    x_mat = np.hstack((left_pts, arr_add)) # [Kx3]
    x_mat_prime = np.hstack((right_pts, arr_add)) # [Kx3]
    # Find directions
    n_mat = np.dot(x_mat_prime,np.dot(s_mat, e_mat).T) # [Kx2]
    n_mat_prime = np.dot(x_mat,np.dot(s_mat, e_mat.T).T) # [Kx2]
    # Compute a,b,c,d
    a = np.dot(np.dot(n_mat, e_tildae),n_mat_prime.T)
    b = 0.5 * (np.dot(n_mat, n_mat.T) + np.dot(n_mat_prime, n_mat_prime.T))
    c = np.dot(np.dot(x_mat, e_mat), x_mat_prime.T)
    d = np.sqrt(b**2 - a @ c)
    # lambda
    lambda_mat = c / (b + d)
    # compute delta_x and delta_x_prime
    delta_x = np.dot(lambda_mat, n_mat)
    delta_x_prime = np.dot(lambda_mat, n_mat_prime)
    n_mat = n_mat - np.dot(e_tildae, delta_x_prime.T).T
    n_mat_prime = n_mat_prime - np.dot(e_tildae.T, delta_x.T).T
    # Update step size
    lambda_mat = lambda_mat * (d/b)
    delta_x = np.dot(lambda_mat, n_mat)
    delta_x_prime = np.dot(lambda_mat, n_mat_prime)
    # Corrected x_mat and x_mat_prime
    x_mat = x_mat - np.dot(s_mat.T, delta_x.T).T # [Kx3] - [Kx3]
    x_mat_prime = x_mat_prime - np.dot(s_mat.T, delta_x_prime.T).T # [Kx3] - [Kx3]
    # x_mat = np.floor(x_mat).astype(np.int16)
    # x_mat_prime = np.floor(x_mat_prime).astype(np.int16)
    return x_mat, x_mat_prime

#@jit(nopython = True, parallel = False)
# def _triangulate_nviews(p_lss:List[np.ndarray],ip_lss:List[np.ndarray])->np.ndarray:
#         """
#         Triangulate a point visible in n camera views.

#         Numba accelerated

#         p_lss is a list of camera projection matrices.
#         ip_lss is a list of homogenised image points. eg [ [x, y, 1], [x, y, 1] ], OR,
#         ip_lss is a 2d array - shape nx3 - [ [x, y, 1], [x, y, 1] ]
#         len of ip must be the same as len of P
#         """
#         # if not len(ip_lss) == len(p_lss):
#         #     err_msg = "Number of points and number of cameras not equal."
#         #     raise ValueError(err_msg)
#         n = len(p_lss)
#         m_mat = np.zeros([3*n, 4+n])
#         for i, (x, p) in enumerate(zip(ip_lss, p_lss)):
#             m_mat[3*i:3*i+3, :4] = p
#             m_mat[3*i:3*i+3, 4+i] = -x
#         v = np.linalg.svd(m_mat)[-1]
#         x_3d = v[-1, :4]
#         return x_3d / x_3d[3] # 4D coordinate

@jit(nopython = True, parallel = False)
def _triangulate_nviews(p1: np.ndarray, p2: np.ndarray,
                        x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """
    Triangulate a point visible in two camera views.

    p1, p2: 3x4 projection matrices.
    x1, x2: 3x1 homogeneous coordinates (2D points).

    Based on,
    Hartley, Richard, and Andrew Zisserman. 
    Multiple View Geometry in Computer Vision. 2nd ed.,
    Cambridge University Press, 2004.
    Chapter 12.2: "Triangulation" (especially pages 312–317).
    """
    # Prepare the matrix for two views
    # 3 rows per view, 4 for the projection matrix + 1 for the homogeneous coordinates
    m_mat = np.zeros((6,5), dtype=np.float32)
    # Fill in the matrix with data for the first view
    m_mat[0:3, :4] = p1
    m_mat[0:3, 4] = -x1
    # Fill in the matrix with data for the second view
    m_mat[3:6, :4] = p2
    m_mat[3:6, 4] = -x2
    # Perform SVD
    _, _, v = np.linalg.svd(m_mat)
    x_3d = v[-1, :4]
    # Return homogeneous 3D coordinate
    return x_3d / x_3d[3]


class Niter2:
    """
    Non-iterative niter2 triangulation algorthm.
    
    Based on 
    triangulate.h by 
    Peter Lindstrom
    Triangulation Made Easy
    IEEE Computer Vision and Pattern Recognition 2010, pp. 1554-1561

    Python implementation by
    Azmyin Md. Kamal
    Ph.d. student, MIE,
    Louisiana State University
    
    The function triangulate performs optimal two-view triangulation
    of a pair of point correspondences in calibrated cameras. Either niter1
    or niter2 can be used, with system defaulting to niter2 
    
    Given measured
    projections u = (u1, u2, -1) and v = (v1, v2, -1) of a 3D point, u and v
    are minimally corrected so that they (to near machine precision) satisfy
    the epipolar constraint u' E v = 0.  The corrected points on the image
    plane are returned as x = (x1, x2, -1) and y = (y1, y2, -1).

    A right-handed coordinate system is assumed, with the image plane at
    z = -1.  If a left-handed coordinate system is used, where the image plane
    is at z = +1, then the argument f should be negated (alternatively, all
    subtractions of f in the function could be changed to additions).

    """

    def __init__(self, algorithm:str = "niter2") -> None:
        # Initialize variables
        if algorithm == "niter1":
            self.algorithm = np.array([[1]], dtype=np.int8) # 1 --> niter1
        else:
            self.algorithm = np.array([[2]], dtype=np.int8) # 2 --> niter2, default
        self.s_mat = np.array([[1,0,0], [0,1,0]], dtype=float) # from Eqn 4
        print("running warmup....")
        self.warm_up_numba_fns()
        print("warmup complete")

    def warm_up_numba_fns(self):
        """Run warmup routine to compile numba methods."""
        lp = np.array([[1,1]],dtype=float)
        rp = np.array([[2,2]],dtype=float)
        ep = np.ones([3,3], dtype=float)
        sp = self.s_mat
        _non_iter_update(self.algorithm, lp, rp, ep,sp)
        p1 = np.array([[500, 0, 320, 0], [0, 500, 240, 0], [0, 0, 1, 0]],
                      dtype=np.float32)
        p2 = np.array([[500, 0, 320, -100], [0, 500, 240, 0], [0, 0, 1, 0]],
                      dtype=np.float32)
        x1 = np.array([400, 300, 1], dtype=np.float32)
        x2 = np.array([420, 310, 1], dtype=np.float32)
        _triangulate_nviews(p1,p2,x1,x2)


    def non_iter_update(self,left_pts:np.ndarray, right_pts:np.ndarray,
                           e_mat:np.ndarray, s_mat:np.ndarray)->Tuple[np.ndarray,
                                                                      np.ndarray]:
        """
        Perform non-iterative update as shown in Section 5.

        Output:
        x_hat: [Kx3] updated left points in homogeneous coord
        x_hat_prime: [Kx3] updated right points in homogeneous coord
        """
        # Initialize constant variables
        # Accelerated by numba
        x_hat, x_hat_prime = _non_iter_update(self.algorithm, left_pts, right_pts,
                                              e_mat,s_mat)

        return x_hat, x_hat_prime

    def triangulate(self, left_pts:np.ndarray, right_pts:np.ndarray,
                           e_mat:np.ndarray, s_mat:np.ndarray,
                           rot:np.ndarray, p_left:np.ndarray,
                           p_right:np.ndarray,
                           show_time_stat:bool = False)->np.ndarray:
        """

        Perform triangulation using niter2 algorithm for all 3D points.

        Output: x_mat: [Kx3], 3D triangulated points
        """
        # Type checks, value checks
        if not isinstance(left_pts, np.ndarray):
            err_msg = "left_pts must be a Numpy array."
            raise TypeError(err_msg)
        if not isinstance(right_pts, np.ndarray):
            err_msg = "right_pts must be a Numpy array."
            raise TypeError(err_msg)
        if not isinstance(e_mat, np.ndarray):
            err_msg = "Essential matrix must be a Numpy array."
            raise TypeError(err_msg)
        if (left_pts.shape[0]!=right_pts.shape[0]):
            err_msg = "keypoints matched between left and right images must be same!"
            raise ValueError(err_msg)
        if left_pts.shape[1]!=2 or right_pts.shape[1]!=2:
            err_msg = "keypoints must be passed as a Nx2 numpy array"
            raise ValueError(err_msg)

        out_pts3d = np.zeros((0,3), dtype=float)
        t00 = curr_time()
        left_pts_updated, right_pts_updated = self.non_iter_update(np.copy(left_pts),
                                                                   np.copy(right_pts),
                                                                   e_mat, s_mat)
        t_iter_up = curr_time() - t00
        
        # In homogeneous coordinates converted to floats
        left_pts_updated = left_pts_updated.astype(float)
        left_pts_updated = right_pts_updated.astype(float)

        # Use DLT to triangulate 3D points
        t00 = curr_time()
        out_pts3d = self.two_view_triangulate(left_pts_updated,right_pts_updated,
                                              p_left, p_right) # [Kx3]
        t_triangulate = curr_time() - t00

        if show_time_stat:
            print(f"non_iter_update: {t_iter_up} ms")
            print(f"triangulation: {t_triangulate} ms")

        return out_pts3d

    def triangulate_nviews(self,p_lss:List[np.ndarray],
                           ip_lss:List[np.ndarray])->np.ndarray:
        """
        Triangulate a point visible in n camera views.

        p_lss is a list of camera projection matrices.
        ip_lss is a list of homogenised image points. eg [ [x, y, 1], [x, y, 1] ], OR,
        ip_lss is a 2d array - shape nx3 - [ [x, y, 1], [x, y, 1] ]
        len of ip must be the same as len of P
        TODO this method may need acceleration
        """
        x_3d = _triangulate_nviews(p_lss[0], p_lss[1], ip_lss[0], ip_lss[1])

        return x_3d / x_3d[3]

    def two_view_triangulate(self, x1:np.ndarray, x2:np.ndarray,
                        p1:np.ndarray, p2:np.ndarray)->np.ndarray:
        """
        Perform two-view triangulation.

        x1,x2: [Kx3] homog. coordinates.
        p1, p2: projection matrices
        """
        if not len(x2) == len(x1):
            err_msg = "Number of points don't match."
            raise ValueError(err_msg)
        # x_3d = [self.triangulate_nviews([p1, p2], [x[0], x[1]]) for x in zip(x1, x2)]

        x_3d = []
        for i in range(x1.shape[0]):
            point_3d = self.triangulate_nviews([p1, p2], [x1[i], x2[i]])
            x_3d.append(point_3d)
        x_3d = np.array(x_3d) # [Kx4]
        x_3d = x_3d[:,0:3] # Drop the last column
        return x_3d

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

def detect_features_and_track(feature_detector:str, img1:np.ndarray,
                            img2:np.ndarray)->Tuple[List, List]:
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
        
    else:
        # Setup FLANN for ORB, as suggested
        # https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
        FLANN_INDEX_LSH = 6  # noqa: N806
        index_params= dict(algorithm = FLANN_INDEX_LSH,  # noqa: C408
        # table_number = 12, # 6
        # key_size = 20, # 12
        # multi_probe_level = 2) #1
        table_number = 6, # 6
        key_size = 12, # 12
        multi_probe_level = 1) #1
        #search_params = dict(checks=50)  # noqa: C408
        search_params = {}  # noqa: C408
        flann = cv2.FlannBasedMatcher(index_params,search_params) # FLANN object
        # ORB feature detector documentation
        orb = cv2.ORB_create(nfeatures=500)
        kp1, des1 = orb.detectAndCompute(img1,None)
        kp2, des2 = orb.detectAndCompute(img2,None)
        
    # Perform Lowe's ratio test and return best matching points
    if (feature_detector == "SIFT"):
        matches = flann.knnMatch(des1,des2,k=2) # Find an initial matches of keypoints
        pts1,pts2 = lowe_ratio_test(matches, kp1, kp2)
    else:
        # ORB
        # Brute force matcher and hamming distance to find matching points
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1,des2)
        matches = sorted(matches, key = lambda x:x.distance) # Sort them in the order of their distance.
        pts1 = np.int32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        pts2 = np.int32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
        #pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        #pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
    return pts1, pts2

def compute_fundamental_matrix(pts1:List, pts2:List)->Tuple[np.ndarray, List, List]:
    """Call cv2.finFundamentalMat() and return fundamental matrix and inliner points."""
    # Initialize work variables
    fundamental_mat, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
    # Select inliners only
    in_pts1 = pts1[mask.ravel()==1]
    in_pts2 = pts2[mask.ravel()==1]
    return fundamental_mat, in_pts1, in_pts2

def drawlines(img1:np.ndarray,img2:np.ndarray,lines,pts1,pts2)->Tuple[np.ndarray, np.ndarray]:
    """Draw epilines on img2 w.r.t img1."""
    # Adopted from https://docs.opencv.org/4.x/da/de9/tutorial_py_epipolar_geometry.html
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

def generate_epipline_imgs(left_img:np.ndarray, right_img:np.ndarray,
                   f_mat:np.ndarray, left_pts:List,right_pts:List)->None:
    """
    Draw epilines between the two images.
    
    Serves as a visual test to check if Fundamental matrix is correct or not
    f_mat: 3x3 fundamental matrix
    """
    # Find epilines correspinding to points in the right image
    # These lines are drawn on the left image
    lines1 = cv2.computeCorrespondEpilines(right_pts.reshape(-1,1,2), 2,f_mat)
    lines1 = lines1.reshape(-1,3)
    left_epliline_img,_ = drawlines(left_img,right_img,lines1,left_pts,right_pts)
    # Find epilines correspinding to points in the left image
    # These lines are drawn on the right image
    lines2 = cv2.computeCorrespondEpilines(left_pts.reshape(-1,1,2), 1,f_mat)
    lines2 = lines2.reshape(-1,3)
    right_epiline_img,_ = drawlines(right_img,left_img,lines2,right_pts,left_pts)
    plt.subplot(1,2,1),plt.imshow(left_epliline_img), plt.title('Left Epiline Image')
    plt.subplot(1,2,2),plt.imshow(right_epiline_img), plt.title('Right Epiline Image')
    plt.tight_layout()
    plt.show()

def show_epilines(dataset_name: str, feature_detector:str,
             show_verbose:bool = False)->None:
    """Demonstrate correct setup of SIFT and ORB detectors using epiline images."""
    if feature_detector.strip().upper() not in ["ORB", "SIFT"]:
        err_msg = "choose either 'ORB' or 'SIFT' feature."
        raise ValueError(err_msg)
    # Intialize variables
    left_img = None
    right_img = None
    left_pts, right_pts = [],[]
    f_mat = np.zeros((0),dtype=float)

    # Define objects
    dataloader = DataSetLoader(dataset_name)
    lp = dataloader.image_path_lss[0]
    rp = dataloader.image_path_lss[1]
    # Load images and make them grayscale
    left_img = cv2.imread(lp,cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread(rp,cv2.IMREAD_GRAYSCALE)
    left_pts, right_pts = detect_features_and_track(feature_detector,left_img, right_img)
    # pts1, pts2 updated in place through return
    f_mat, left_pts, right_pts = compute_fundamental_matrix(left_pts,right_pts)
    # DEBUG
    generate_epipline_imgs(left_img, right_img,f_mat,left_pts, right_pts)

    # DEBUG print stats
    if show_verbose:
        print()
        print(f"Number of images in dataset: {dataloader.num_images}")
        print(f"Feature detector selected: {feature_detector}")
        print(f"Pts matched: {len(left_pts)}")
        print(f"Fundamental matrix computed: {f_mat}")
        print()

def compute_essential_matrix(f_mat:np.ndarray, k_mat:np.ndarray)->np.ndarray:
    """Compute essential matrix given K and F."""
    e_mat = np.zeros((3,3), dtype=float)
    e_mat = np.dot(np.dot(k_mat.T, f_mat),k_mat)
    return e_mat

def generate_projection_matrix(k_mat:np.ndarray, rot:np.ndarray,
                               tvec:np.ndarray)->np.ndarray:
    """
    Compute 3x4 Projectio Matrix P.
    
    k_mat: 3x3 calibration matrix K
    rot: 3x3 rotation matrix R
    tvec: 3x1 translation vector t
    p_mat: 3x4 projection matrix P
    m_intrin_mat and m_extrin_mat defined based on
    https://www.youtube.com/watch?v=S-UHiFsn-GI&list=PL2zRqk16wsdoCCLpou-dGo7QQNks1Ppzo&index=1
    m_intrin: 3x4 intrinsic matrix M_int
    m_extrin: 4x4 extrinsic matrix M_ext
    """
    # Initialize work variables
    m_intrin = np.zeros((3,4), dtype=float)
    m_extrin = np.zeros((4,4), dtype=float)
    p_mat = np.zeros((3,4), dtype=float)
    # Populate Mint, and Mext matrices
    m_intrin[:3,:3] = np.copy(k_mat)
    m_extrin[-1,-1] = 1
    m_extrin[0:3, -1] = tvec.T
    m_extrin[0:3, 0:3] = rot
    p_mat = np.dot(m_intrin, m_extrin)
    # DEBUG
    # print(f"m_intrin: \n{m_intrin}")
    # print()
    # print(f"rot: \n{rot}")
    # print(f"tvec: {tvec}")
    # print()
    # print(f"m_extrin: \n{m_extrin}")
    # print()
    # print(f"proj_mat: {p_mat}")
    return p_mat

def triangualte_hs(pts1:np.ndarray, pts2:np.ndarray, 
                   proj1:np.ndarray, proj2:np.ndarray)->np.ndarray:
    """
    Triangulate 2D keypoints in world coordinate using hs method.

    Uses Hartley and Zisserman's optimal triangulation method
    _1, _2: left and right camera
    pts1, pts2: 2D matched keypoints [Kx2]
    proj1, proj2: projection matrices of left and right camera respectively, [3x4]
    Call opencv's triangulatePoints()
    https://docs.opencv.org/4.x/d0/dbd/group__triangulation.html
    """
    hs_pts3d = np.zeros(0, dtype=float)
    hs_pts4d_hom = np.zeros(0, dtype=float)
    hs_pts4d_hom = cv2.triangulatePoints(proj1, proj2, pts1.T, pts2.T)
    ##Convert 4d homogeneous coordinates to 3d coordinate
    hs_pts4d_hom = hs_pts4d_hom / np.tile(hs_pts4d_hom[-1, :], (4, 1))
    hs_pts3d = hs_pts4d_hom[:3, :].T # [Nx3], [x,y,z]
    return hs_pts3d

def show_stereo_images(left_img:np.ndarray, right_img:np.ndarray):
    """Call imshow() to show left right image pairs."""
    # Only needed for debugging purpose
    cv2.imshow("left_img", left_img)
    cv2.imshow("right_img", right_img)
    cv2.waitKey(1)

def plot_on_3d(hs_pts3d: np.ndarray, niter2_pts3d: np.ndarray) -> None:
    """Plot 3D points for both methods using an isometric plot."""
    # Configure plot
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Plot hs_pts3d points
    xs = hs_pts3d[:, 0]
    ys = hs_pts3d[:, 1]
    zs = hs_pts3d[:, 2]
    ax.scatter(xs, ys, zs, label='hs', color='black', alpha=0.3)

    # Plot niter2_pts3d points in orange
    xs_niter2 = niter2_pts3d[:, 0]
    ys_niter2 = niter2_pts3d[:, 1]
    zs_niter2 = niter2_pts3d[:, 2]
    ax.scatter(xs_niter2, ys_niter2, zs_niter2, label='niter2', color='red',
               marker='+')

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

def compute_points_per_sec(lss_pts:List[np.ndarray], lss_time:List[float])->float:
    """Compute points/sec metric."""
    _total_pts = np.sum(np.asarray(lss_pts))
    _total_time = np.sum(np.asarray(lss_time))
    _pts_sec = _total_pts / _total_time
    return _pts_sec

def test_pipeline(dataset_name: str, feature_detector:str,
                  full_verbose:bool = False,
                  short_verbose:bool = False)->None:
    """Perform full experiment."""
    # Intialize variables
    pair_processed = 1
    # Triangulate with niter2
    niter2 = Niter2()
    dataloader = DataSetLoader(dataset_name)
    k_mat = np.copy(dataloader.calibration_matrix) # camera calibration matrix
    p_mat_left = np.zeros((3,4), dtype=float) # P1
    p_mat_left[0:3, 0:3] = np.eye(3, dtype=float)
    # Result variables
    triangualted_pts_hs = [] # List[int]
    hs_time = [] # List[float]
    triangulated_pts_niter2 = [] # List[int]
    niter2_time = []

    # Cycle through pairwise images
    # TODO add start_idx to be read from dataset.yaml file
    start_idx = 15 # Trial and error, the scale is mostly stable now
    for i in range(start_idx, dataloader.num_images - 1):
        # Initialize variables
        left_img = None
        right_img = None
        left_pts, right_pts = [],[]
        f_mat = np.zeros((0),dtype=float)
        e_mat = np.zeros((0), dtype=float)
        s_mat = np.array([[1,0,0], [0,1,0]], dtype=float)
        p_mat_right = np.zeros((3,4), dtype=float)
        rot1 = np.zeros((3,3), dtype=float)
        #rot2 = np.zeros((3,3), dtype=float)
        tvec = np.zeros(3, dtype=float)
        hs_pts3d = np.zeros((0,3), dtype=float) # [Kx4] 3d poits in homogeneous coord
        niter2_pts3d = np.zeros((0,3), dtype=float) # [Kx4] 3d points in homogeneous coord

        # Paths to left and right image
        lp = dataloader.image_path_lss[i]
        rp = dataloader.image_path_lss[i+1]
        left_img = cv2.imread(lp,cv2.IMREAD_GRAYSCALE)
        right_img = cv2.imread(rp,cv2.IMREAD_GRAYSCALE)
        #show_stereo_images(left_img, right_img) # DEBUG

        left_pts, right_pts = detect_features_and_track(feature_detector,left_img,
                                                      right_img)
        f_mat, left_pts, right_pts = compute_fundamental_matrix(left_pts,right_pts)
        e_mat = compute_essential_matrix(f_mat, k_mat)
        rot1, _, tvec = cv2.decomposeEssentialMat(e_mat)

        # If not converted to float32, cv2.triangulate points crashes kernel
        # All points [Nx2] here
        left_pts = left_pts.astype(np.float32)
        right_pts = right_pts.astype(np.float32)
        p_mat_right = generate_projection_matrix(k_mat, rot1, tvec) # outputs P2

        ## Triangulate using optimal triangulation method
        t0 = curr_time()

        hs_pts3d = triangualte_hs(left_pts, right_pts, p_mat_left, p_mat_right)

        t_hs = (curr_time() - t0)/1000 # seconds
        hs_time.append(t_hs) # seconds
        triangualted_pts_hs.append(hs_pts3d.shape[0]) # int

        ## Triangulate using non-iterative niter2 method
        t1 = curr_time()

        niter2_pts3d = niter2.triangulate(left_pts, right_pts, e_mat, s_mat, rot1,
                                         p_mat_left, p_mat_right,
                                         show_time_stat=True)

        triangulated_pts_niter2.append(niter2_pts3d.shape[0])
        t_niter2 = (curr_time() - t1)/1000 # seconds
        niter2_time.append(t_niter2) # seconds

        if full_verbose:
            #print(f"Processed image pair: {pair_processed}")
            print(f"hs: triangulated points: {hs_pts3d.shape[0]}")
            print(f"niter2: triangulated points: {niter2_pts3d.shape[0]}")
            print(f"t_hs: {t_hs} s")
            print(f"t_niter2: {t_niter2} s")
        pair_processed+=1
        p_mat_left = np.copy(p_mat_right) # Update for next round
        # short verbose message
        if short_verbose:
            print(f"Image pair: {pair_processed} contains {left_pts.shape[0]} kp pts.")
        
        if i == 1:
            plot_on_3d(hs_pts3d, niter2_pts3d) # Test
            break # Only do one image pair

    # Print statistics
    hs_pts_sec = compute_points_per_sec(triangualted_pts_hs, hs_time)
    niter_pts_sec = compute_points_per_sec(triangulated_pts_niter2, niter2_time)
    # TODO a separate class to store and process results
    print()
    print(f"points/sec by hs method: {int(hs_pts_sec)}")
    print(f"points/sec by niter2 method: {int(niter_pts_sec)}")
    # TODO mean relative error
    print()
    # TODO this needs to be corrected, comes out all wrong
    # TODO save that plot where agreement was best
    #plot_on_3d(hs_pts3d, niter2_pts3d) # Only generate image once?
    #cv2.destroyAllWindows()