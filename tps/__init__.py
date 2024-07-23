from pathlib import Path
from PIL import Image

import numpy as np
from numpy import float64, int_
from numpy.typing import NDArray

import scipy.constants as con
from scipy.spatial import KDTree

import cv2
from cv2.typing import MatLike

""" 
    This package is meant to analyze extract ion energy energy spectra from data
    collected by a Thomson Parabola Spectrometer (TPS). A TPS works by using 
    electric and magnetic fields to separate particles based on their energy and
    charge-to-mass ratio. This allows simultaneous mearsurement of energy 
    spectra for several different positively charged ion species. This codes 
    uses solutions to the nonrelativistic equation of motion for
    charged particles in piecewise-constant electromagnetic fields to calculate
    their positions at the detector. 

    All default values are specific to OSU's home-built TPS. More information
    can be found in Connor Winter's bachelor's thesis 'Development of a new 
    Thomson parabola spectrometer for analysis of laser accelerated ions'.
"""

def load_img(path: str | Path, background: int = 0, 
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

    if grayscale:
        img = img.convert('L')

    return np.array(img) - background


def tps_trajectory(m: float, q: float, spec_len: float = 0.85,
                    B_start: float = 0, B_end: float = 0.0508, B: float = .2055,
                    E_start: float = 0.0762, E_end: float = 0.1778, 
                    E: float = 5000/.01,
                    U_min: float = 1, U_max: float = 30,
                    dU: float = 1e-1) -> NDArray[float64]:
    
    """ Compute the shape of a TPS trace for a particle with mass m and charge
        q provided the field configuration of the TPS. 
        
    Parameters:
    --------------------------------------
    m: The ion mass in multiples of the proton mass.
    q: The ion charge in elementary charge units.
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
    U_min: The minimum energy to calculate the trajectories. (in MeV)
    U_max: The maximum energy to calculate trajectories. (in MeV)
    dU: The energy resolution used to calculate the TPS trace. (in MeV)
    
    Returns:
    --------------------------------------
    trace: The x, y coordinates of the species' TPS trace in the plane of the 
        detector (in m).
    """

    len_B = B_end - B_start
    M_p = con.m_p*m         # Ion mass in kg
    Q_p = q*con.e           # Ion charge in C

    U = np.arange(U_min, U_max, dU)

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

    # Get particle trajectories
    x, y = 100*tps_trajectory(m, q, U_min = U_min, U_max = U_max, dU = dU)

    # Get boundaries of the region of interest.
    x_px = x0 + x*px_dens       # Center line
    y_px = y0 + y*px_dens 

    x_bot = x_px - 100*pinhole_sz*px_dens/2
    y_bot = y_px + 100*pinhole_sz*px_dens/2
    x_top = x_px + 100*pinhole_sz*px_dens/2
    y_top = y_px - 100*pinhole_sz*px_dens/2
    x_bnd = np.concatenate((x_top, x_bot[::-1], [x_top[0]]))
    y_bnd = np.concatenate((y_top, y_bot[::-1], [y_top[0]]))
    r_bnd = np.array([x_bnd, y_bnd]).T

    # Rotate img
    rot_mat = cv2.getRotationMatrix2D((x0, y0), rot_ang, 1)
    rot_img = cv2.warpAffine(img, rot_mat, img.shape[::-1])

    # # # Interpolate to get pixel coordinates of x_bnd, y_bnd
    # Nx = len(img[0])
    # Ny = len(img)
    # x_det = np.linspace(1, Nx, Nx) - x0
    # y_det = np.linspace(1, Ny, Ny) - y0

    # # Get closest points in rot_img for each value in (x_px, y_px).
    # tree = KDTree(rot_img)
    # nearest_neighbors, nn_index = tree.query(r_bnd)

    return np.array([x_px, y_px]), r_bnd.T, rot_img
    # return spec, np.array([x_px, y_px]), rot_img

# Write a function to rotate the coordinates of the tps trace instead of the 
# rotating the image. This will be much computationaly cheaper.
def rotate_coordinates(x, y, ang, x0, y0) -> NDArray[float64]:
    """ Rotate coordinates around the point(x0, y0).
    
    Parameters:

    Returns:
    r_rot:
    """



    return np.zeros(100)