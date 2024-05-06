"""Module for project 3."""

# Imports
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import time
from typing import Tuple, List

def celestial_object(
        shape: tuple[int, int], *,
        radius: float, position: tuple[float, float]=(0., 0.), corona: float=5.,
        ) -> np.ndarray:
    """
    Simulate image of celestial objects.

    Author: Dr. Schoegl
    Parameters:
        shape: shape of image
        radius: radius of celestial object in pixels
        corona: decay factor of corona
        position: position of celestial object
    Returns:
        image of celestial object
    """
    hh, ww = shape
    xx = np.linspace(-.5*(ww-1), .5*(ww-1), ww)
    yy = np.linspace(-.5*(hh-1), .5*(hh-1), hh)
    xx, yy = np.meshgrid(xx, yy)
    rr = np.sqrt((xx-position[1])**2 + (yy-position[0])**2)
    rr[rr<radius] = radius
    rr = np.exp(-corona*rr/radius)
    return rr / rr.max()

def generate_eclipse_image(m_img: int, celes_rad: int, pos: int,
                           show_debug: bool = False)->np.ndarray:
    """Generate an eclipse image."""
    shape = (m_img, m_img)
    radius = celes_rad
    sun = celestial_object(shape, radius=radius, position=(0, 0), corona=10)
    moon = 1 - celestial_object(shape, radius=radius, position=(pos, 0), corona=20)
    if show_debug:
        plt.imshow(sun * moon, origin='upper', cmap='inferno')
        plt.axis('off')
    return sun * moon

def curr_time():
    """Return the time tick in milliseconds."""
    return time.monotonic() * 1000

def debug_lock():
    """Locks system in an infinite loop for debugging."""
    print("LOCK")
    while (1):
        pass

@njit
def _normalize_vector_l2(vector):
    """Normalize vector using numpy linalg."""
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

@njit
def _hit_pinhole_plane(num_rays: int, obj_dist: float) -> np.ndarray:
    """Generate 3D coordinates of where light rays hit pinhole plane."""
    coords = np.zeros((num_rays,3), dtype=float)
    ll = 1.0
    r1 = np.random.uniform(0, ll, num_rays)  # Draw R1
    r2 = np.random.uniform(0, ll, num_rays)  # Draw R2
    theta = np.arcsin(np.sqrt(r1)) # theta between 0 to pi/2
    phi = 2*np.pi*r2 # 2 * pi * R2, radians, between 0 to 2pi
    rr = obj_dist*np.tan(theta)
    coords[:, 0] = rr*np.sin(phi)  # Assign values to first column
    coords[:, 1] = rr*np.cos(phi)  # Assign values to second column
    return coords

@njit
def _thru_pinhole(pin_arr:np.ndarray ,hit_mat:np.ndarray)->np.ndarray:
        """Find rays that passes thru this pinhole."""
        on_pinhole = np.zeros((0,3), dtype=float)
        o_x, o_y, o_radius = pin_arr
        dist = np.sqrt((hit_mat[:, 0] - o_x)**2 + (hit_mat[:, 1] - o_y)**2 +
                       (hit_mat[:, 2] - 0)**2)
        on_pinhole = hit_mat[(dist <= o_radius)]
        return on_pinhole

@njit
def _v0_tru_pinhole(p0, a0, n0, v0_mat, pin_arr):
        """Compute which rays passes thru a pinhole."""
        o_x, o_y, o_r = pin_arr
        num = np.dot((a0 - p0), n0)
        deno = np.dot(v0_mat, n0)
        t = num / deno
        pt_one = p0 + v0_mat * t[:, np.newaxis]
        #dist = np.sqrt((pt_one[:, 0] - o_x)**2 + (pt_one[:, 1] - o_y)**2)
        dist = np.sqrt((pt_one[:, 0] - o_x)**2 + (pt_one[:, 1] - o_y)**2 +
                       (pt_one[:, 2] - 0)**2)
        inside_pinhole = (dist <= o_r)
        v0_keep = v0_mat[inside_pinhole]
        return v0_keep

@njit
def _trace_to_proj_plane(p0: np.ndarray, pi_mat: np.ndarray,
                         a0: np.ndarray, n0: np.ndarray) -> np.ndarray:
    """Generate ray-traced projection through pinhole(s)."""
    v_bar = pi_mat - p0
    v_bar_normalized = np.empty_like(v_bar)
    for i in range(len(v_bar)):
        v_bar_normalized[i] = _normalize_vector_l2(v_bar[i])
    proj_img_xy = np.zeros((len(pi_mat), 3), dtype=np.float64)
    for i in range(len(pi_mat)):
        t = np.dot((a0 - p0), n0) / np.dot(v_bar[i], n0)
        proj_img_xy[i] = p0 + v_bar[i] * t
    return proj_img_xy

@njit
def _find_pixel_coord(plane_coord, image_size, plane_bound):
        """
        Find the corresponding pixel position on an MxM image for a given 2D coordinate.

        :param plane_coord: Tuple containing (x, y) coordinates on the centered plane
        :param image_size: Size of the MxM image
        :param plane_bound: Bound of the centered plane (e.g., -a to a)
        :return: Tuple containing (pixel_x, pixel_y) position on the MxM image
        """
        # https://stackoverflow.com/questions/58714618/numba-invalid-use-of-boundfunction-on-np-astype
        translated_x = plane_coord[:, 0]
        translated_y = plane_coord[:, 1]
        scaled_x = ((translated_x + plane_bound) / (2 * plane_bound) * (image_size - 1))
        scaled_y = ((plane_bound - translated_y) / (2 * plane_bound) * (image_size - 1))
        scaled_x = scaled_x.astype(np.int16)
        scaled_y = scaled_y.astype(np.int16)
        pixel_x = np.clip(scaled_x, 0, image_size - 1)
        pixel_y = np.clip(scaled_y, 0, image_size - 1)
        return pixel_x, pixel_y

class EclipsePinHoleCamera:
    """Pinhole camera model to view eclipse photos."""

    def __init__(self, image_2d: np.ndarray,
                screen_size: int,
                pinhole_dim: Tuple[int] = (1,1),
                pinhole_radius: float = 1.0,
                spacing_multiplier: int = 2,
                max_num_rays: int = 1000,
                focal_length: float = 1.0,
                obj_dist: float = 1.0,
                report_time: bool = False) -> None:
        # Initialize virtual image variables
        self.image_2d = np.copy(image_2d)    # Copy of the eclipse photo
        self.image_plane_bound = image_2d.shape[0]//2 # Divide with integral reminder
        self.m_img = image_2d.shape[0] # Dimension of virtual image
        # Geometric properties of camera
        self.D = (-1) * obj_dist   # Distance, as setup
        self.f = focal_length
        # Pinholes
        self.Or = pinhole_radius
        if spacing_multiplier>0.5:
            self.pinhole_spacing = spacing_multiplier * (2 * pinhole_radius)
        else:
            self.pinhole_spacing = 0.5 * (2 * pinhole_radius)
        self.pinholes = self.generate_pinholes(pinhole_dim, self.Or)
        # Ray tracing properties
        self.norm_apperture = np.array([0.0, 0.0, -1])
        self.norm_projection = np.array([0.0, 0.0, -1])
        self.a0_pinhole = np.array([0.0, 0.0, 0.0]) # Center of apperture plane
        self.a0_projection = np.array([0.0, 0.0, focal_length]) # Center of proj plane
        # Monte Carlo simulation parameters
        self.min_rays = 0
        self.max_rays = max_num_rays
        # Setup up screen size bounds [-a,a]
        self.screen_size = screen_size  # Same for both pinhole and projection plane
        # Monte Carlo Simluation
        self.points_2d = self.gen_points_2d()
        self.min_intensity, self.max_intensity = self.intensity_minmax() # floats
        # Visualization
        self.out_image_2d = np.zeros((self.m_img, self.m_img), dtype=int)
        self.proj_img_2d = np.zeros((self.m_img, self.m_img), dtype=float)
        self.proj_img_xy = np.zeros((0,3), dtype=float)
        # Results and Discussion
        self.report_time = report_time
        self.thru_pinhole_time = []
        self.time_to_proj_plane = []

    def gen_points_2d(self, show_debug = False):
        """
        Generate Nx2 points matrix.

        #Each row --> [y_cord, x_cord, D, intensity]
        Each row --> [x_cord, y_cord, D, intensity]
        First from the MxM discrete domain, we convert points to continous domain
        The coordinate frame is placed at center of the plane
        x points to left
        y points to top
        points are then accessed (y,x)
        numpy coordinate begins top-left
        u --> along colms
        v --> along rows
        elements are accessed (v, u)
        """
        # Initialize work variables
        m_rows = self.image_2d.shape[0]
        a = self.image_plane_bound
        d = self.D
        points_mat = np.zeros((m_rows, m_rows, 4))
        v,u = 0,0 # numpy coordinates starts from top-left
        for y_cord in range(a, -a-1, -1): # 100 -- -100
            for x_cord in range (-a,a+1, 1): # -100 -- 100
                pixel_inten = self.image_2d[u,v]
                x_cord = -1 * x_cord # To correct x direction
                #points_mat[v,u] = np.array([y_cord, x_cord, d, pixel_inten])
                points_mat[v,u] = np.array([x_cord, y_cord, d, pixel_inten])
                # print(f"pixel {u,v} --> {x_cord, y_cord, d, pixel_inten}")
                if (u < m_rows - 1): # Increment column index
                    u = u + 1
            u = 0 # Reset column
            if(v < m_rows - 1): # Increment row index
                v = v + 1
        # DEBUG
        m_last = m_rows - 1
        if show_debug:
            print()
            print(f"Center: {points_mat[m_rows//2,m_rows//2]}") # middle
            print(f"Top-left: {points_mat[0,0]}") # top-left
            print(f"Top-right: {points_mat[0,m_last]}") # top-right
            print(f"Bottom-left: {points_mat[m_last,0]}") # bottom-left
            print(f"Bottom-right: {points_mat[m_last,m_last]}") # bottom-right
            print()
        # Reshape to final matrix
        points_mat = points_mat.reshape(-1,4) # Nx4 where N = M*M
        if show_debug:
            print()
            print(f"points_mat shape: {points_mat.shape}")
            print(f"first ten rows: {points_mat[:10, :]}")
            print()
        return points_mat

    def visualize_pinhole_config(self):
        """Visualize pinhole arrangement."""
        # Generate image of a pinhole object
        _, ax = plt.subplots()
        a_size = self.screen_size
        for _, pinhole in enumerate(self.pinholes):
            # Create a circle at the (x, y) position with the given radius
            # print(f"idx: {(pinhole[0], pinhole[1]), pinhole[2]}")
            circle = plt.Circle((pinhole[0], pinhole[1]), pinhole[2],
                                color = "red", fill=True)
            ax.add_artist(circle) # Add the circle to the axes
        # Set the aspect of the plot to equal, so the circles appear as circles
        ax.set_aspect('equal')
        # Set the limits of x and y axes
        ax.set_xlim([-a_size, a_size])
        ax.set_ylim([-a_size, a_size])
        plt.title("Pinhole configuration")
        plt.show()

    def generate_pinholes(self, pinhole_dim: Tuple[int],
                          pinhole_radius: float) -> List[np.ndarray]:
        """
        Generate pinholes with (x,y) position and radius.

        Returns
        List[np.ndarrays,....,] where each np.ndarray is a pinhole
        """
        pinhole_lss = []
        if(pinhole_dim == (1,1) or pinhole_dim == (3,3) or pinhole_dim == (5,5)):
            # Initialize
            num_pinholes = pinhole_dim[0] * pinhole_dim[1]
            # Determine number of pinholes
            # Generate pinholes symmetrically around the center
            for i in range(-(pinhole_dim[0] // 2), pinhole_dim[0] // 2 + 1):
                for j in range(-(pinhole_dim[1] // 2), pinhole_dim[1] // 2 + 1):
                    px = i * self.pinhole_spacing
                    py = j * self.pinhole_spacing
                    pin_obj = np.array([px, py, pinhole_radius])
                    pinhole_lss.append(pin_obj)
                    # Break if reached the desired number of pinholes
                    if len(pinhole_lss) == num_pinholes:
                        break
        else:
            # Watchdog, only these configurations are supported
            err_str = "pinhole_dim can be (1,1), (3,3) or (5,5)"
            raise ValueError(err_str)
        # for x in pinhole_lss:
        #     print(x)
        return pinhole_lss

    def intensity_minmax(self):
        """Extract min-max intensity values from Nx4 points matrix."""
        min_val = np.min(self.points_2d[:, -1])
        max_val = np.max(self.points_2d[:, -1])
        return min_val, max_val

    def intensity_to_num_rays(self,x:float) -> int:
        """Return number of rays to generate, given intensity."""
        a = self.min_intensity
        b = self.max_intensity
        c = self.min_rays
        d = self.max_rays
        return int(c + (x - a) * (d - c) / (b - a))

    def rays_to_intensity(self, x:int, max_rays:int = 100)->float:
        """Return intensity proportional to the number of rays."""
        a = 0
        b = max_rays
        if (x>b):
            x = b
        c = 0.0
        d = self.max_intensity
        return c + ((x - a) * (d - c) / (b - a))

    def hit_pinhole_plane(self, num_rays: int, obj_dist: float) -> np.ndarray:
        """Generate 3D coordinates of where light rays hit pinhole plane."""
        return _hit_pinhole_plane(num_rays, obj_dist)

    def thru_pinhole(self, pin_arr:np.ndarray ,hit_mat:np.ndarray)->np.ndarray:
        """Find rays that passes thru this pinhole."""
        return _thru_pinhole(pin_arr, hit_mat)

    def v0_tru_pinhole(self, p0, a0, n0, v0_mat, pin_arr):
        """Compute which rays passes thru pinhole."""
        vv = _v0_tru_pinhole(p0, a0, n0, v0_mat, pin_arr)
        return vv

    def find_pixel_coord(self, plane_coord, image_size, plane_bound):
        """Find the corresponding pixel position for a given 2D coordinate."""
        return _find_pixel_coord(plane_coord, image_size, plane_bound)

    def trace_to_proj_image(self, p0, pi_mat, a0, n0)->None:
        """Generate ray-traced projection thru pinhole(s)."""
        proj_img_xy = np.zeros((len(pi_mat), 3), dtype=float) # Numba accelerated
        proj_img_xy = _trace_to_proj_plane(p0, pi_mat, a0, n0)
        u_pix, v_pix = self.find_pixel_coord(proj_img_xy, self.m_img,
                                             self.screen_size)
        self.out_image_2d[u_pix, v_pix] += 1

    def monte_carlo_camera(self):
        """Trace diffused sunglight through pinhole(s)."""
        t00 = curr_time() # Total simulation time
        pts_pros = self.points_2d.shape[0]
        for row in self.points_2d:
            num_rays = self.intensity_to_num_rays(row[-1])
            po = row[:3] # 1x3, px,py,pz
            if num_rays != 0:
                # 3D coordinate of where rays hit pinhole camera
                hit_mat = self.hit_pinhole_plane(num_rays, self.D)
                for pin_arr in self.pinholes:
                    # Filter those rays that are within radius of this pinhole
                    t1 = curr_time()
                    pts_thru = self.thru_pinhole(pin_arr, hit_mat)
                    t2 = curr_time() - t1
                    self.thru_pinhole_time.append(t2)
                    if pts_thru.shape[0] > 2:
                        # Trace these bundles to projection plane
                        a0 = self.a0_projection # 1x3
                        n0 = self.norm_projection # 1x3
                        t1 = curr_time()
                        self.trace_to_proj_image(po, pts_thru, a0, n0)
                        self.time_to_proj_plane.append(t2)
                    else:
                        pass
                        #print(f"No rays thru pinhole: {idx}")
            else:
                #print("No rays to generate.")
                pass
        # Time statistics
        avg_time_to_pinhole = np.mean(self.thru_pinhole_time)
        avg_time_to_proj_plane = np.mean(self.time_to_proj_plane)
        if self.report_time:
            print()
            print(f"Processed {pts_pros} points in {(curr_time() - t00)/1000:.3f} s")
            print(f"Avg time thru pinholes: {avg_time_to_pinhole} s")
            print(f"Avg time to project: {avg_time_to_proj_plane} s")
            print()

    def generate_projected_image(self, show_img :bool = False)->None:
        """Generate projected image."""
        self.proj_img_2d = self.out_image_2d/np.max(self.out_image_2d) # Works
        if show_img:
            plt.imshow(self.proj_img_2d, origin='lower', cmap='inferno')
            plt.axis('off')
        return self.proj_img_2d

def sec_3_1():
    """Perform Section 3.1 experiment."""
    imgz = generate_eclipse_image(200, 40, 0, False)
    time_report = True
    obj = EclipsePinHoleCamera(imgz, 200, (5,5), 2.5, 2.0,
                               10000, 4.0, 2.0, time_report)
    obj.monte_carlo_camera()

def sec_3_2(rays_max:int)->None:
    """Perform Section 3.2 experiment."""
    m_img = 200
    projected_images = [] # List[np.ndarray]
    pin_h = (1,1)
    pin_rad = [0.25,2.5,25.0]
    celes_rad = 40
    eclipse_phases = [60,30,15,0]
    eclipse_images = [generate_eclipse_image(m_img, celes_rad, ec, False)
                      for ec in eclipse_phases]
    plot_h = (4,4) # Shape of the subplot matrix
    fig_size = (10,8)
    t1 = curr_time()
    for pin_r in pin_rad:
        t00 = curr_time()
        print(f"Generating 1x4 images for pinhole :{pin_r}")
        for ec_img in eclipse_images:
            proj_img = np.zeros((m_img, m_img), dtype=float)
            obj = EclipsePinHoleCamera(ec_img, m_img, pin_h, pin_r,
                                       2.0,rays_max, 4.0, 2.0, False)
            obj.monte_carlo_camera()
            proj_img = obj.generate_projected_image()
            projected_images.append(proj_img)
        print(f"Pinhole {pin_r} took simulation: {(curr_time() - t00)/1000:.3f} s")
    print(f"Time to finish simulation: {(curr_time() - t1)/1000:.3f} s")
    generate_matrix_plots(eclipse_images, projected_images, plot_h,fig_size)

def sec_3_3(pin_h, ray_count:int = 5000, pin_rad = 2.5):
    """Perform section 3.3 experiment."""
    eclipse_phases = [60,30,15,0]
    visualize_eclipse(pin_h, pin_rad ,eclipse_phases, ray_count)

def sec_3_4(pin_h:Tuple, ray_count:int = 5000,
            pin_rad:float = 2.5,
            pin_spacing:float = 2.0,
            visual_pinhole:bool = False):
    """Perform section 3.4 experiment."""
    eclipse_phases = [60,30,15,0]
    visualize_eclipse(pin_h, pin_rad,
                      eclipse_phases,
                      ray_count,
                      pin_spacing,
                      visual_pinhole)

def visualize_eclipse(pin_dim:List[Tuple], pin_rad:float,
                      eclipse_phases:List[int],
                      max_rays:int,
                      pin_spacing:float = 2.0,
                      visualize_pinhole:bool = False)-> None:
    """Generate phase images for chosen pinhole configuration."""
    projected_images = [] # List[np.ndarray]
    m_img = 200
    celes_rad = 40
    eclipse_images = [generate_eclipse_image(m_img, celes_rad, ec, False)
                      for ec in eclipse_phases]
    done_once = True
    t00 = curr_time()
    # Run through simulation
    idx = 0
    for ec_imgz in eclipse_images:
        proj_img = np.zeros((m_img, m_img), dtype=float)
        obj = EclipsePinHoleCamera(ec_imgz, m_img, pin_dim, pin_rad,
                                   pin_spacing,max_rays,
                                   4.0, 2.0, False)
        if visualize_pinhole and done_once:
            obj.visualize_pinhole_config()
            done_once = False
        t1 = curr_time()
        print(f"Processing for phase {idx} .....")
        obj.monte_carlo_camera()
        proj_img = obj.generate_projected_image()
        print(f"Phase {idx} finished in {(curr_time() - t1)/1000} s")
        projected_images.append(proj_img)
        idx+=1
    print(f"Time to finish simulation: {(curr_time() - t00)/1000} s")
    print()

    generate_matrix_plots(eclipse_images,projected_images, (2,4))

def generate_matrix_plots(ec_lss:List[np.ndarray],
                          proj_lss:List[np.ndarray],
                          plot_dim:Tuple,
                          fig_size:Tuple = (12,6))->None:
    """Generate Mx4 subplots."""
    #print(f"Number of images: {len(proj_lss)}")
    _, axs = plt.subplots(plot_dim[0], plot_dim[1], figsize=fig_size)
    axs = axs.flatten() # Flatten the axs array for easier indexing
    plt_idx = 0 # Global counter
    for _, img in enumerate(ec_lss):
        axs[plt_idx].imshow(img, origin='upper', cmap='inferno')
        axs[plt_idx].axis('off')
        plt_idx+=1
    for _, img in enumerate(proj_lss):
        axs[plt_idx].imshow(img, origin='lower', cmap='inferno')
        axs[plt_idx].axis('off')
        plt_idx+=1
    # Hide any extra subplots
    for j in range(len(ec_lss), len(axs)):
        axs[j].axis('off')
    plt.tight_layout()
    plt.show()
