from PIL import Image
from PIL.ExifTags import TAGS
import PIL.ImageOps as img_ops
from typing import Any

import numpy as np
from numpy import float64, int_
from numpy.typing import NDArray

import scipy.constants as con
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter

import cv2
from cv2.typing import MatLike

""" 
    This package is meant to extract ion energy spectra from data
    collected by a Thomson Parabola Spectrometer (TPS). A TPS works by using 
    electric and magnetic fields to separate particles based on their energy and
    charge-to-mass ratio. This allows simultaneous measurement of energy 
    spectra for several different positively charged ion species. This code 
    uses solutions to the nonrelativistic equation of motion for
    charged particles in piecewise-constant electromagnetic fields to calculate
    their positions at the detector. We typically measure energies to tens of 
    MeV. 

    All default values are specific to OSU's home-built TPS. More information
    can be found in Connor Winter's bachelor's thesis 'Development of a new 
    Thomson parabola spectrometer for analysis of laser accelerated ions'.
"""


class TPSData(object):
    """ 
        A class to load, analyze, save, and plot ion energy spectra from a
        Thomson Parabola Spectrometer. In order to extract the energy spectrum
        for an ion species with charge q and mass m, the user should call the
        tps_counts method.

    Parameters:
    --------------------------------------
    path: The file path to the desired tps image.

    bkgrd: The amount of background to subtract. This is a simple
        background subtraction, meaning the the result is px_value - background
        for all pixels in img. By default, no subtraction is performed.
    grayscale: Whether or not to convert to grayscale. By default this is True.
        Note that the imgage must be converted to grayscale prior to spectrum 
        retrieval.
    processed: Whether the loaded image is pre-processed or not. By default, 
        this parameter is False, meaning the user is passing a raw TPS
        image that will be handled by the class's internal image processing
        routine. If the image is pre-processed (meaning the user has already
        done e.g. a background subtraction), set this value to True.
    smooth: Whether to apply a Gaussian filter to the image after background 
        subtraction. By default, smooth = True.

    spec_len: The total length of the spectrometer in the direction of particle
        motion. (in m)
    B_start: The distance from the pinhole to the start of the magnetic field 
        region. (in m)
    B_end: The ending location of the magnetic field region. (in m)
    B: The magnetic field strength in T.
    E_start: The distance from pinhole to the start of the electric field 
        region. (in m)
    E_end: The ending location of the electric field region. (in m)
    E: The electric field strength in V/m.

    """

    def __init__(self, path: str, 
                    x0: int, y0: int, ang: float,
                    bkgrd: int = 0, 
                    grayscale: bool = True, processed: bool = False,
                    smooth: bool = True,
                    px_dens: float = 119.6, pinhole_sz = 750e-6,
                    part_cal: float = 1,
                    spec_len: float = 0.85, 
                    B_start: float = 0, B_end: float = 0.0508, B: float = .2055,
                    E_start: float = 0.0762, E_end: float = 0.1778, 
                    E: float = 5000/.01): 
         
        # Load and process (if necessary) image
        self.raw = Image.open(path)
        if processed:
            self.img = np.array(self.raw)
        else:
            if grayscale:
                self.img = np.array(self.raw).clip(bkgrd) - bkgrd
            else:
                self.img = np.array(self.raw).clip(bkgrd) - bkgrd

            if smooth:
                self.img = gaussian_filter(self.img, sigma = 2)

        # Store image metadata.
        self.path = path
        self.metadata = {'Filepath': self.path}
        self.metadata.update(self.read_img_metadata())

        # self.raw = cv2.imread(path)
        # self.img = self.raw

        # Rotate image and prepare for spectrum recovery
        self.netural = (x0, y0)
        self.rot_img = self.rotate_img(x0, y0, ang)

        # Store tps geometry
        self.px_dens = px_dens
        self.hole_sz = pinhole_sz
        self.part_cal = part_cal
        self.spec_len = spec_len
        self.B_start = B_start
        self.B_end = B_end
        self.B = B
        self.E_start = E_start
        self.E_end = E_end
        self.E = E
        # self.metadata.update(self.get_geometry())


    def read_img_metadata(self) -> dict[str, Any]:
        """ Load the metadata stored in self.img and create a dictionary to
            store it.
        """

        meta = {}
        exif = self.raw.getexif()

        # Loop through all tags in metadata
        for tag_id in exif:
            tag = TAGS.get(tag_id, tag_id)
            content = exif.get(tag_id)

            meta[tag] = content

        return meta
    
    def print_metadata(self):
        """ Format and print metadata to the console."""

        for entry in self.metadata.keys():
            print(f'{entry:<35}:', self.metadata[entry])
    
    def get_geometry(self)-> dict[str, str]:
        """ 
            Copy the TPS geometry into a dictionary for saving. The SI units for
            each value are included.
        """
        geom = {'spectrometer_length': f'{self.spec_len} m',
                'Bfield_Starting_loc': f'{self.B_start} m',
                'Bfield_End_loc': f'{self.B_end} m',
                'Bfield_Strength': f'{self.B} T',
                'E_Start_loc': f'{self.E_start} m',
                'E_End_loc': f'{self.E_end} m',
                'Efield_Strangth': f'{self.E} V/m',
                'Linear_Pixel_density': f'{self.px_dens} lines/cm',
                'Pinhole Size': f'{self.hole_sz} m',
                'Signal-to_Particle Calibration': f'{self.part_cal}'
                }
        return geom

    def rotate_img(self, x0: int, y0: int, ang: float) -> MatLike:
        """ Rotate self.raw around the point (x0, y0) by an angle ang.
        
        Parameters:
        x0, y0: The pixel coordinates about which the image is rotated.
        ang: The rotation angle in degrees.
        
        Returns:
        rot_img: The rotated image. It has the same size as self.raw.
        """

        rot_mat = cv2.getRotationMatrix2D((x0, y0), ang, 1)
        rot_img = cv2.warpAffine(self.img, rot_mat, self.img.shape[::-1])

        return rot_img
    
    def tps_traj_real(self, q: float, m: float, 
                        U: NDArray[float64]) -> NDArray[float64]:
        """ 
            Calculate the trajectory in the detector plane (in physical units)
            of an ion species with charge q, mass m and energy U in a Thomson 
            Parabola spectrometer.

        Parameters:
        q: The ion charge, in units of the elementary charge.
        m: The ion mass, in atomic mass units.
        U: The energy range (in MeV) over which to determine the particle 
            trajectory. 

        Returns:
        traj: The (x, y) coordinates in the detector plane for each energy U.
            Note that these are in physical units, and must be converted to
            pixel coordinates before extracting the ion energy spectrum.
        """
        len_B = self.B_end - self.B_start
        M_p = con.u*m           # Ion mass in kg
        Q_p = q*con.e           # Ion charge in C

        gamma = 1 + U*con.e*1e6/(M_p*con.c**2)
        beta = np.sqrt(1 - 1/gamma**2)
        vel = beta*con.c

        # Magnetic field deflection
        vz_0 = vel
        LR = gamma*M_p*vel/(np.abs(Q_p)*self.B)                 # Larmor Radius
        x_B = (LR - np.sqrt(LR**2 - len_B**2))
        theta_B = np.arcsin(len_B/LR)
        vz_B = vz_0*np.cos(theta_B)
        vx_B = vz_0*np.sin(theta_B)
        t_B_end = (self.spec_len - self.B_end)/vz_B             # Travel time after leaving B

        # Electric field deflection
        t_E = (self.E_end - self.E_start)/vz_B                  # Time particle enters E
        vy_E = Q_p*self.E*t_E/M_p
        y_E = 0.5*Q_p*self.E*t_E**2
        t_E_end = (self.spec_len - self.E_end)/vz_B             # Travel time after leaving E

        # Total deflection
        xF = x_B + t_B_end*vx_B
        yF = y_E + t_E_end*vy_E

        return np.array([xF, yF])
    
    def traj_to_px_coords(self, x: NDArray[float64],
                             y: NDArray[float64]) -> NDArray[float64]:
        """ Convert physical coordinates to pixel coordinates using the known
            geometry of the system.
            
        Parameters:
        x, y: The x, y coordinates of the tps trajectory in the detector plane.
            These should almost always be the result of tps_trajectory.
        
        Returns:
        traj_px: The locations in traj are converted to pixel coordinates.
        """
        x0, y0 = self.netural
        x_px = x0 + 100*x*self.px_dens       # Center line in pixel coordinates
        y_px = y0 + 100*y*self.px_dens
        rc = np.column_stack((x_px, y_px))
        return rc
    
    def get_roi(self, x_px, y_px) -> tuple[NDArray[float64], MatLike]:
        # Get boundaries of the region of interest surrounding the tps trace.
        x_bot = x_px - 100*self.hole_sz*self.px_dens/2
        y_bot = y_px + 100*self.hole_sz*self.px_dens/2
        x_top = x_px + 100*self.hole_sz*self.px_dens/2
        y_top = y_px - 100*self.hole_sz*self.px_dens/2
        x_bnd = np.concatenate((x_top, x_bot[::-1], [x_top[0]]))
        y_bnd = np.concatenate((y_top, y_bot[::-1], [y_top[0]]))
        r_bnd = np.column_stack([x_bnd, y_bnd])

        # Handle boundaries outside image
        max_x = self.rot_img.shape[1]
        max_y = self.rot_img.shape[0]
        r_bnd[:, 0] = r_bnd[:, 0].clip(max = max_x)
        r_bnd[:, 1] = r_bnd[:, 1].clip(max = max_y)

        # Create mask
        mask = np.zeros(self.rot_img.shape, dtype = self.rot_img.dtype)
        cv2.fillPoly(mask, [np.rint(r_bnd).astype(np.int32)], (255, 255, 255))
        masked_img = cv2.bitwise_and(self.rot_img, mask)

        return r_bnd, masked_img

    def dump_trajectory(self, q: float, m: float, 
                            U:NDArray[float64]) -> NDArray[float64]:
        """ 
            Calculate the ion species' trajectory, convert it to pixel
            coordinates, and dump for the user to plot.
        """
        x_si, y_si = self.tps_traj_real(q, m, U)
        return self.traj_to_px_coords(x_si, y_si)
    
    def tps_counts(self, q: float, m: float, 
                    U: NDArray[float64]) -> NDArray[float64]:
        """ 
            Extract the ion energy spectrum over U from self.rot_img for an 
            species with charge q and mass m. 
        """
        
        x, y = self.tps_traj_real(q, m, U)
        rc = self.traj_to_px_coords(x, y)

        _, masked_img = self.get_roi(*rc.T)
        roi_pixels = np.argwhere(masked_img)         # Indeces of nonzero pixels

        # For every pixel in the region of interest, find the closest point along rc
        # and add its value to the index in spec corresponding to rc_closest.
        dist = cdist(rc, roi_pixels[:, ::-1], 'euclidean')
        rc_index = np.argmin(dist, axis = 0)
        spec = np.zeros(len(rc))
        np.add.at(spec, rc_index, self.rot_img[roi_pixels[:, 0], 
                                                roi_pixels[:, 1]])
        
        return spec



#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Same implementation as above in individual functions. 
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

def load_img(path: str, background: int = 0, 
                grayscale: bool = True) -> NDArray[int_]:
    
    """ Load a TPS trace image specified by path, convert to grayscale if
        necessary, perform a background subtraction, and convert to a numpy 
        NDArray.

    Parameters:
    --------------------------------------
    path: The file path to the desired tps image.
    background: The amount of background to subtract. This is a simple
        background subtraction, meaning the the result is px_value - background
        for all pixels in img. By default, no subtraction is performed.
    grayscale: Whether or not to convert to grayscale. By default this is True.

    Returns:
    --------------------------------------
    img: The processed image. 
    """

    img = Image.open(path)

    # if grayscale:
    #     img = img.convert('L')

    return np.array(img).clip(background) - background



def tps_trajectory(m: float, q: float, U: NDArray[float64], 
                    spec_len: float = 0.85, B_start: float = 0, 
                    B_end: float = 0.0508, B: float = .2055,
                    E_start: float = 0.0762, E_end: float = 0.1778, 
                    E: float = 5000/.01) -> NDArray[float64]:
    
    """ Compute the shape of a TPS trace for a particle with mass m and charge
        q provided the field configuration of the TPS. 
        
    Parameters:
    --------------------------------------
    m: The ion mass in atomic mass units.
    q: The ion charge in elementary charge units.
    U: The energy range (in MeV) over which to calculate the ion trajectories.
    spec_len: The total length of the spectrometer in the direction of particle
        motion. (in m)
    B_start: The distance from the pinhole to the start of the magnetic field 
        region. (in m)
    B_end: The ending location of the magnetic field region. (in m)
    B: The magnetic field strength in T.
    E_start: The distance from pinhole to the start of the electric field 
        region. (in m)
    E_end: The ending location of the electric field region. (in m)
    E: The electric field strength in V/m.
    
    Returns:
    --------------------------------------
    trace: The x, y coordinates of the species' TPS trace in the plane of the 
        detector (in m).
    """

    len_B = B_end - B_start
    M_p = con.u*m           # Ion mass in kg
    Q_p = q*con.e           # Ion charge in C

    gamma = 1 + U*con.e*1e6/(M_p*con.c**2)
    beta = np.sqrt(1 - 1/gamma**2)
    vel = beta*con.c

    # Magnetic field deflection
    vz_0 = vel
    LR = gamma*M_p*vel/(np.abs(Q_p)*B)          # Larmor Radius
    x_B = (LR - np.sqrt(LR**2 - len_B**2))
    theta_B = np.arcsin(len_B/LR)
    vz_B = vz_0*np.cos(theta_B)
    vx_B = vz_0*np.sin(theta_B)
    t_B_end = (spec_len - B_end)/vz_B           # Travel time after leaving B

    # Electric field deflection
    t_E = (E_end - E_start)/vz_B                # Time particle enters E
    vy_E = Q_p*E*t_E/M_p
    y_E = 0.5*Q_p*E*t_E**2
    t_E_end = (spec_len - E_end)/vz_B           # Travel time after leaving E

    # Total deflection
    xF = x_B + t_B_end*vx_B
    yF = y_E + t_E_end*vy_E

    return np.array([xF, yF])



def tps_counts(img: NDArray[int_], m: float, q: float, x0: float, y0: float, 
               rot_ang: float, px_dens: float = 119.6, pinhole_sz = 750e-6,
               part_cal: float = 1, U_min: float = 1, U_max: float = 30, 
               dU: float = 0.1
               ) -> tuple[NDArray[float64], NDArray[float64], MatLike]:

    """ Retrieve the number of particles per energy bin from a TPS trace.
        
    Parameters:
    --------------------------------------
    img: The image containing the Thomson traces.
    m: The ion mass in multiples of the proton mass.
    q: The ion charge in elementary charge units.
    x0, y0: The pixel location of the neutral spot. This acts as the origin of 
        detector coordinates.
    rot_ang: The rotation angle required to align the calculated trajectory with 
            the trajectory shown on the detector. (in deg)
    px_dens: The number of pixels per cm. 
    pinhole_sz: The diameter of the TPS pinhole in m.
    part_cal: This a conversion factor to convert from pixel value (usually an 
        integer 0 - 255) to number of particles. The default value gives the 
        ion energy spectrum in arbitrary units.
    U_min: The minimum energy to calculate the trajectories. (in MeV)
    U_max: The maximum energy to calculate trajectories. (in MeV)
    dU: The energy resolution used to calculate the TPS trace. (in MeV)
    
    Returns:
    --------------------------------------
    spec: The number of particles at each energy along the TPS trace.
    """

    # Get energies
    U = np.arange(U_min, U_max, dU)

    # Get particle trajectories
    x, y = 100*tps_trajectory(m, q, U)
    x_px = x0 + x*px_dens       # Center line in pixel coordinates
    y_px = y0 + y*px_dens
    rc = np.column_stack((x_px, y_px))

    # Get boundaries of the region of interest surrounding the tps trace.
    x_bot = x_px - 100*pinhole_sz*px_dens/2
    y_bot = y_px + 100*pinhole_sz*px_dens/2
    x_top = x_px + 100*pinhole_sz*px_dens/2
    y_top = y_px - 100*pinhole_sz*px_dens/2
    x_bnd = np.concatenate((x_top, x_bot[::-1], [x_top[0]]))
    y_bnd = np.concatenate((y_top, y_bot[::-1], [y_top[0]]))
    r_bnd = np.column_stack([x_bnd, y_bnd])

    # Rotate img
    rot_mat = cv2.getRotationMatrix2D((x0, y0), rot_ang, 1)
    rot_img = cv2.warpAffine(img, rot_mat, img.shape[::-1])

    # Create mask
    mask = np.zeros(rot_img.shape, dtype = rot_img.dtype)
    cv2.fillPoly(mask, [np.rint(r_bnd).astype(np.int32)], (255, 255, 255))
    masked_img = cv2.bitwise_and(rot_img, mask)
    roi_pixels = np.argwhere(masked_img)         # Indeces of nonzero pixels

    # For every pixel in the region of interest, find the closest point along rc
    # and add its value to the index in spec corresponding to rc_closest.
    dist = cdist(rc, roi_pixels[:, ::-1], 'euclidean')
    rc_index = np.argmin(dist, axis = 0)
    spec = np.zeros(len(rc))
    np.add.at(spec, rc_index, rot_img[roi_pixels[:, 0], roi_pixels[:, 1]])
    
    return part_cal*spec, r_bnd, rot_img



def energy_marker(U: float, m: float, q: float, x0: float, y0: float, 
                    px_dens: float = 119.6) -> tuple[int, int]:
    """ Return the pixel coordinates of the location along the TPS trace with
        energy U..
        
    Parameters:
    --------------------------------------
    U: The ion energy in MeV.
    m: The ion mass in multiples of the proton mass.
    q: The ion charge in elementary charge units.
    x0, y0: The pixel location of the neutral spot. This acts as the origin of 
        detector coordinates.
    px_dens: The number of pixels per cm. 
    
    Returns:
    --------------------------------------
    (x, y): The pixel coordinates of the location along the TPS trace with
        energy U.
    """
    x, y = 100*tps_trajectory(m, q, U)  # type: ignore
    x_px = x0 + x*px_dens
    y_px = y0 + y*px_dens

    return (x_px, y_px)



def circ_from_pts(p1: tuple[float, float], 
                    p2: tuple[float, float], 
                    p3: tuple[float, float]
                ) -> tuple[tuple[float, float]|None, float]:
    """ Return the center and radius of the circle passing through any 3 points.
        In case the 3 points form a line, returns (None, infinity).

    Parameters:
    --------------------------------------
    p1, p2, p3: Tuples containing (x, y) pairs for three points on the circle.

    Returns:
    --------------------------------------
    center = (cx, cy) containing the center of the circle
    radius: The radius of the circle.
    """
    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])
    
    if abs(det) < 1.0e-6:
        return (None, np.inf)
    
    # Center of circle
    cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det
    
    radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
    return (cx, cy), radius