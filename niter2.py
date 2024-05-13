"""
Module that implements Lindstorm`s ```niter2``` method.

Author:
Azmyin Md. Kamal,
Ph.D. student in MIE,
Louisiana State University,
Louisiana, USA

Date: May 12th, 2024
Version: 0.5
"""

#Imports
import numpy as np
# TODO cupy?
from numba import jit
import time
from typing import Tuple, List

np.set_printoptions(suppress=True)
np.seterr(all='ignore') #! BAD IDEA TODO WILL BE REMOVED

# Utility functions
def curr_time():
    """Return the time tick in milliseconds."""
    return time.monotonic() * 1000

# @jit(nopython = True, parallel = False)
# def _non_iter_update(algorithm:np.ndarray,left_pts:np.ndarray, right_pts:np.ndarray,
#                            e_mat:np.ndarray, s_mat:np.ndarray)->Tuple[np.ndarray,
#                                                                       np.ndarray]:
#         """Perform non-iterative update as shown in Section 5."""
#         """
#         Algorithm is shown in Listing 3 in the paper

#         Args:
#         left_pts:[Kx2], keypoint from left camera. Each row corresponds to x in paper
#         right_pts:[Kx2], keypoint from right camera. Each row corresponds to x' in paper
#         e_mat: [3x3]: essential matrix
#         s_mat: [2x3]: matrix defined in Equation 4
#         rot: [3x3]: Rotation matrix (obtained by decomposing essential matrix)

#         Output:
#         x_hat: [Kx3] updated left points in homogeneous coord
#         x_hat_prime: [Kx3] updated right points in homogeneous coord

#         TODO this function needs another version that runs on the GPU
#         """
#         # Initialize constant variables
#         e_tildae = np.zeros((2,2), dtype=float)
#         e_tildae = np.dot(np.dot(s_mat, e_mat),s_mat.T)
#         # Initialize work variables
#         x = np.zeros(0, dtype=float) # in paper x \in R^3, homogeneous coord
#         x_prime = np.zeros(0, dtype=float) # in paper x' \in R^3, homogeneous coord
#         n = np.zeros(2, dtype=float) # [1x2] step direction for x
#         n_prime = np.zeros(2, dtype=float) # [1x2] step direction for x'
#         a = np.zeros(1, dtype=float) # [1x1] scalar for computing \lambda
#         b = np.zeros(1, dtype=float) # [1x1] scalar for computing \lambda
#         b_rhs = np.zeros(1, dtype=float) # [1x1] scalar for computing \lambda
#         c = np.zeros(1, dtype=float) # [1x1], x.T * E * x', scalar
#         d = np.zeros(1, dtype=float) # [1x1], scalar
#         lambda_ = np.zeros(1, dtype=float) # [1x1], scalar, step size
#         del_x = np.zeros(3, dtype=float) # in paper delta_x \in R^3, hom coord
#         del_x_prime = np.zeros(3, dtype=float) # in paper delta_x' \in R^3, hom coord
#         x_hat = np.zeros((0,3), dtype=float) # [Kx3]
#         x_hat_prime = np.zeros((0,3), dtype=float) # [Kx3]

#         # Primary loop
#         for i in range(left_pts.shape[0]):
#             # Initialize variables for these keypoints pairs
#             x = np.append(left_pts[i], 1)  # [1x3]
#             x_prime = np.append(right_pts[i], 1)  # [1x3]
#             n = np.dot(np.dot(s_mat, e_mat),x_prime) # n = S.E.x'
#             n_prime = np.dot(np.dot(s_mat, e_mat.T),x) # n'= S.(E.T).x
#             a = np.dot(np.dot(n.T, e_tildae),n_prime) # a = (n.T).E_tildae.n'
#             b_rhs = (np.dot(n.T,n) + np.dot(n_prime.T, n_prime)) # L_2(n,n') norm
#             b = 0.5 * b_rhs # b = 0.5*((n.T).n + (n'.T).n')  # noqa: E501
#             c = np.dot(np.dot(x.T, e_mat),x_prime) # c = (x.T).E.x', epipolar constraint
#             d = np.sqrt(b**2 - a * c) # d = sqrt(b**2 - a*c)
#             lambda_ = c / (b + d) # lambda_ = c / (b+d)
#             del_x = lambda_ * n # [1x3]
#             del_x_prime = lambda_ * n_prime #[1x3]
#             n = n - np.dot(e_tildae,del_x_prime) # n = n-E_tildae.delta_x'
#             n_prime = n_prime - np.dot(e_tildae.T, del_x) # n' = n'-(E_tildae.T).delta_x
#             if np.array_equal(algorithm, np.array([[2]], dtype=np.int8)):
#                 # niter2
#                 lambda_ = lambda_ * ((2*d)/b_rhs)
#                 del_x = lambda_ * n # [1x3]
#                 del_x_prime = lambda_ * n_prime #[1x3]
#             else:
#                 # niter1
#                 # FUTURE WORK
#                 pass
#             # Corrected points <x_hat, x_hat_prime> in homogeneous coord
#             x = x - np.dot(s_mat.T, del_x) # x_hat
#             x_prime = x_prime - np.dot(s_mat.T, del_x_prime) # x_hat_prime
#             # Reshape to correct dimension, [1x3] from [3]
#             x = np.reshape(x, (-1,3))
#             x_prime = np.reshape(x_prime, (-1,3))
#             # Push to matrix
#             x_hat = np.vstack((x_hat, x))
#             x_hat_prime = np.vstack((x_hat_prime, x_prime))

#         # Breaks with Numba
#         x_hat = np.floor(x_hat).astype(np.int32)
#         x_hat_prime = np.floor(x_hat_prime).astype(np.int32)
#         return x_hat, x_hat_prime



class Niter2:
    """
    Non-iterative niter2 triangulation algorthm.

    Based on
    niter2 algorithm by
    Peter Lindstrom
    Triangulation Made Easy
    IEEE Computer Vision and Pattern Recognition 2010, pp. 1554-1561

    Python implementation by
    Azmyin Md. Kamal
    Ph.d. student, MIE,
    Louisiana State University
    """

    def __init__(self, algorithm:str = "niter2") -> None:
        # Initialize variables
        if algorithm == "niter1":
            self.algorithm = np.array([[1]], dtype=np.int8) # 1 --> niter1
        else:
            self.algorithm = np.array([[2]], dtype=np.int8) # 2 --> niter2, default
        self.s_mat = np.array([[1,0,0], [0,1,0]], dtype=float) # from Eqn 4
        self.warm_up_numba_fns()

    def warm_up_numba_fns(self):
        """Run warmup routine to compile numba methods."""
        print()
        print("Niter2: compiling all numba accelerated methods.")
        lp = np.array([[1,1]],dtype=float)
        rp = np.array([[2,2]],dtype=float)
        ep = np.ones([3,3], dtype=float)
        sp = self.s_mat
        for _ in range(5):
            # run 5 times to stabilize?
            self._non_iter_update(self.algorithm, lp, rp, ep,sp)
        print("Niter2: Numba accelerated methods ready.")
        print()

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
        x_hat, x_hat_prime = self._non_iter_update(self.algorithm, left_pts, right_pts,
                                              e_mat,s_mat)

        return x_hat, x_hat_prime

    @staticmethod
    @jit(nopython = True)
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

        TODO this function needs another version that runs on the GPU
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

    def triangulate_niter2(self, left_pts:np.ndarray, right_pts:np.ndarray,
                           e_mat:np.ndarray, s_mat:np.ndarray,
                           rot:np.ndarray, p_left:np.ndarray,
                           p_right:np.ndarray,
                           show_time_stat:bool = False)->Tuple[np.ndarray, List]:
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
        t01 = curr_time()
        t_optimal_pts = (t01 - t00)/1000
        # In homogeneous coordinates converted to floats
        left_pts_updated = left_pts_updated.astype(float)
        left_pts_updated = right_pts_updated.astype(float)
        # Use DLT to triangulate 3D points
        t11 = curr_time()
        lp = left_pts_updated[:,:2]
        rp = right_pts_updated[:,:2]
        out_pts3d = self.linear_triangulate(p_left, lp, p_right, rp) # [Kx3]
        t12 = curr_time()
        t_triangulate = (t12 - t11)/1000 # seconds
        if show_time_stat:
            print(f"non_iter_update: {t01 - t00} ms")
            print(f"triangulation: {t12 - t11} ms")
        return out_pts3d, [t_optimal_pts, t_triangulate]
    
    def linear_triangulate(self, p1:np.ndarray, pts1:np.ndarray, p2:np.ndarray, pts2:np.ndarray):
        """
        Triangulate a set of 2D coordinates in the image to a set of 3D points.

        # Adopted from
        # https://github.com/laavanyebahl/3D-Reconstruction-and-Epipolar-Geometry/blob/aa68896b32f58eb6f028cdab632d196b191199e4/python/submission.py#L142

        Input:
        p1: [3x4] projection matrix for left image
        pts1: [Kx2] matches keypoints from left image
        p2: [3x4] projection matrix from rihgt image
        pts2: [Kx2] matched keypoints from right image

        Output:
        p_i,t[Nx3] matrix with the corresponding 3D points per row
        """
        p_i = []
        for i in range(pts1.shape[0]):
            a_mat = np.array([   pts1[i,0]*p1[2,:] - p1[0,:] ,
                            pts1[i,1]*p1[2,:] - p1[1,:] ,
                            pts2[i,0]*p1[2,:] - p1[0,:] ,
                            pts2[i,1]*p2[2,:] - p2[1,:]   ])
            _, _, vh = np.linalg.svd(a_mat)
            v = vh.T
            x_pt = v[:,-1]
            # NORMALIZING
            x_pt = x_pt/x_pt[-1]
            # print(X)
            p_i.append(x_pt)
        p_i = np.asarray(p_i)
        # MULTIPLYING TOGETHER WIH ALL ELEMENET OF Ps
        pts1_out = np.matmul(p1, p_i.T )
        pts2_out = np.matmul(p2, p_i.T )
        pts1_out = pts1_out.T
        pts2_out = pts2_out.T
        # NORMALIZING
        for i in range(pts1_out.shape[0]):
            pts1_out[i,:] = pts1_out[i,:] / pts1_out[i, -1]
            pts2_out[i,:] = pts2_out[i,:] / pts2_out[i, -1]
        # NON - HOMOGENIZING
        pts1_out = pts1_out[:, :-1]
        pts2_out = pts2_out[:, :-1]
        # NON-HOMOGENIZING
        p_i = p_i[:, :-1]
        return p_i # [Kx3]


# ---------------------------------- EOF --------------------------------------------------
# def _non_iter_vectorized(algorithm:np.ndarray,left_pts:np.ndarray, right_pts:np.ndarray,
#                            e_mat:np.ndarray, s_mat:np.ndarray)->Tuple[np.ndarray,
#                                                                       np.ndarray]:
#     """Vectorized version of _non_iter_update()."""
#     #! THIS VERSION DOES NOT WORK, FUTURE WORK
#     e_tildae = np.dot(np.dot(s_mat, e_mat),s_mat.T) # [2x2]
#     arr_add = np.ones(left_pts.shape[0], dtype=np.float32).reshape(-1,1) # [Kx1]
#     x_mat = np.hstack((left_pts, arr_add)) # [Kx3]
#     x_mat_prime = np.hstack((right_pts, arr_add)) # [Kx3]
#     # Find directions
#     n_mat = np.dot(x_mat_prime,np.dot(s_mat, e_mat).T) # [Kx2]
#     n_mat_prime = np.dot(x_mat,np.dot(s_mat, e_mat.T).T) # [Kx2]
#     # Compute a,b,c,d
#     a = np.dot(np.dot(n_mat, e_tildae),n_mat_prime.T)
#     b = 0.5 * (np.dot(n_mat, n_mat.T) + np.dot(n_mat_prime, n_mat_prime.T))
#     c = np.dot(np.dot(x_mat, e_mat), x_mat_prime.T)
#     d = np.sqrt(b**2 - a @ c)
#     # lambda
#     lambda_mat = c / (b + d)
#     # compute delta_x and delta_x_prime
#     delta_x = np.dot(lambda_mat, n_mat)
#     delta_x_prime = np.dot(lambda_mat, n_mat_prime)
#     n_mat = n_mat - np.dot(e_tildae, delta_x_prime.T).T
#     n_mat_prime = n_mat_prime - np.dot(e_tildae.T, delta_x.T).T
#     # Update step size
#     lambda_mat = lambda_mat * (d/b)
#     delta_x = np.dot(lambda_mat, n_mat)
#     delta_x_prime = np.dot(lambda_mat, n_mat_prime)
#     # Corrected x_mat and x_mat_prime
#     x_mat = x_mat - np.dot(s_mat.T, delta_x.T).T # [Kx3] - [Kx3]
#     x_mat_prime = x_mat_prime - np.dot(s_mat.T, delta_x_prime.T).T # [Kx3] - [Kx3]
#     # x_mat = np.floor(x_mat).astype(np.int16)
#     # x_mat_prime = np.floor(x_mat_prime).astype(np.int16)
#     return x_mat, x_mat_prime