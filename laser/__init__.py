import numpy as np
from numpy.typing import NDArray
from numpy import float_, complex_
import scipy.constants as con
import scipy.special as spec

def scalar_vortex_field(x: float|NDArray[float_], 
                        y: float|NDArray[float_], 
                        z: float|NDArray[float_], 
                        wt: float|NDArray[float_],
                        amp: float, spotsize: float, pulse_dur: float, 
                        p: int, l: int|float, 
                        w_t0: float = 0.0) -> NDArray[complex_]:
    """ 
    Create a Laguerre-Gaussian laser defined at position (x, y, z) at time t. 
    This function assumes the laser propagates in the +z direction and that all 
    parameters are given in dimensionless units.

    -----------------------------------------
    Parameters:
    x, y, z: float, The position (in units of kL*r) of the laser.
    wt: float, The time (in units wL*t) to calculate the laser.
    amp: float, The normalized electric field strength (a0).
    spotsize: float, The size of the beam waist (in units kL*w0).
    pulse_dur: float, The intensity FWHM of the laser (in units wL*t).
    p: int, The radial index of the vortex beam
    l: int, the laser twist index
    w_t0: float, The time (in units wL*t) corresponding to the maximum intensity
            at z = 0

    Returns:
    field: float, The electric field defined at the time/position given by t,
        (x, y, z) and characterized by the other arguments.
    """

    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    zr = spotsize**2/2
    norm = np.sqrt(spec.factorial(p)/spec.factorial(p + np.abs(l)))

    # Laser phase
    guoy = (2*p + np.abs(l))*np.arctan(z/zr)        # Guoy phase
    Rz_inv = z/(z**2 + zr**2)                       # Inverse radius of curvature
    phase = np.exp(1j*(z - wt - 0.5*r**2*Rz_inv + guoy - l*phi))

    # Spatial profile
    wz = spotsize * np.sqrt(1 + (z/zr)**2)          # Spotsize
    lg_term = spec.genlaguerre(p, np.abs(l))(2*r**2/wz**2)
    spat_prof = (r*np.sqrt(2)/wz)**np.abs(l)*np.exp(-r**2/wz**2)*lg_term

    # Temporal profile
    sigma = np.sqrt(2)*pulse_dur/(2*np.sqrt(np.log(2)))
    temp_prof = np.exp(-(wt - w_t0)**2/(sigma**2))

    field = amp*spotsize/wz*norm*spat_prof*temp_prof*phase

    return field

def scalar_vortex_field_real_args(x: float|NDArray[float_], 
                                  y: float|NDArray[float_], 
                                  z: float|NDArray[float_], 
                                  t: float|NDArray[float_], 
                                  wavelength: float, 
                                  spotsize: float, pulse_dur: float, 
                                  p: int, l: int|float, 
                                  t0: float = 0.0) -> NDArray[complex_]:
    """
    This function is a wrapper for scalar_vortex_field that automatically
    converts parameters into dimensionless units and returns a normalized scalar
    field. 

    Parameters:
    x, y, z: The position (in m) of the laser.
    wt: The time (in s) to calculate the laser.
    spotsize: The size of the beam waist (in m).
    pulse_dur: The intensity FWHM of the laser (in s).
    p: The radial index of the vortex beam
    l: The laser twist index
    t0: The time (in units s) corresponding to the maximum intensity
        at z = 0

    Returns:
    field: float, The electric field in dimensionless units defined at the 
        time/position given by t, (x, y, z) and characterized by the other 
        arguments.
    """

    # Normalization factors
    kL = 2*np.pi/wavelength     # Wavenumber
    wL = con.c*kL               # Angular frequency

    # Get dimensionless params
    x_norm = kL*x
    y_norm = kL*y
    z_norm = kL*z
    t_norm = wL*t
    spot_norm = kL*spotsize
    pulse_dur_norm = wL*pulse_dur
    t0_norm = wL*t0

    norm_field = scalar_vortex_field(x_norm, y_norm, z_norm, t_norm, 1, 
                                     spot_norm, pulse_dur_norm, p, l, 
                                     w_t0 = t0_norm)
    return norm_field

def get_vector_E(scalar_field: NDArray[float_|complex_], 
                 x: NDArray[float_], y: NDArray[float_], 
                    kL: float = 1) -> NDArray[float_]:
    """ Numerically calculate the electric field components from a scalar beam 
        propagating in the z_hat direction using the approximation developed by
        Erikson and Singh in Phys. Rev. E 49, 5778 (1994). 

    Parameters:
    scalar_field: 2d array, The scalar field describing the beam in the xy 
        plane. It should have shape (N, M).
    x, y: 1d array, The x and y coordinates over which scalar_field is defined. 
        x should have length N, while y should have length M.
    kL: float, The laser's wavenumber (in rad/m). By default, this function
        assumes kL = 1, meaning x and y are given in dimensionless units 
        (x = kL*x_real).

    Returns:
    vector_E: 3d array of shape (N, M, 3), The vector components of the electric
        field accurate to second order.
    """

    Ex = scalar_field
    Ey = -0.5*np.gradient(np.gradient(scalar_field, y, axis = 0), x, axis = 1)/kL**2
    Ez = 1j*np.gradient(scalar_field, x, axis = 1)/kL

    return np.transpose([Ex, Ey, Ez])

def get_vector_B(scalar_field: NDArray[float_|complex_], 
                 x: NDArray[float_], y: NDArray[float_], 
                    kL: float = 1) -> NDArray[float_]:
    """ Numerically calculate the magnetic field components from a scalar beam 
        propagating in the z_hat direction using the approximation developed by
        Erikson and Singh in Phys. Rev. E 49, 5778 (1994). 
    
    Parameters:
    scalar_field: 2d array, The scalar field describing the beam in the xy 
        plane. It should have shape (N, M).
    x, y: 1d array, The x and y coordinates over which scalar_field is defined. 
        x should have length N, while y should have length M.
    kL: float, The laser's wavenumber (in rad/m). By default, this function
        assumes kL = 1, meaning x and y are given in dimensionless units 
        (x = kL*x_real).

    Returns:
    vector_E: 3d array of shape (N, M, 3), The vector components of the electric
        field accurate to second order.
    """

    Bx = -0.5*np.gradient(np.gradient(scalar_field, y, axis = 0), x, axis = 1)/kL**2
    By = scalar_field
    Bz = 1j*np.gradient(scalar_field, y, axis = 0)/kL

    return np.transpose([Bx, By, Bz])


def get_oam(field: NDArray[complex_], x:  NDArray[float_], 
            y: NDArray[float_]) -> float:
    """ Calculate the amount of orbital angular momentum (OAM) in a paraxial 
        laser field. This functions utilizes the approach described in Zangwill 
        Ch. 16.7.5. 
    
    Parameters:
    field: The scalar electric field used to calculate the OAM. Must be two 
        dimensional.
    x, y: The data coordinates along each axis of field.
    
    Returns:
    l_retreived: float, The measured OAM.
    """
    norm = np.real(np.trapz(np.trapz(
                np.conjugate(field)*field, 
            y, axis = 1),
            x, axis = 0))


    dEdx = np.gradient(field, x, axis = 0)
    dEdy = np.gradient(field, y, axis = 1)
    dEdTheta = -y[None, :]*dEdx + x[:, None]*dEdy
    integrand = np.conjugate(field)*dEdTheta

    # Use abs instead of real to make positive
    Lz = np.abs(1j*np.trapz(np.trapz(
                            integrand, 
                        y, axis = 1), 
                        x, axis = 0))

    l_retreived = Lz/norm

    return l_retreived