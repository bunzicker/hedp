import os

import ctypes as ct
from ctypes import c_double, c_int

import numpy as np
import numpy.ctypeslib as np_ct
from numpy.typing import NDArray
from numpy import float_


"""
    This package calculates high frequency radiation at a virtual detector using
    the algorithm developed in Pardal, et. al. Computer Physics Communication 
    285 (2023) 108634.
"""

#-------------------------------------------------------------------------------
# Load dll 
folder = os.path.dirname(os.path.abspath(__file__))
filepath = folder + r'/radiation_calculator.dll'
lib = ct.CDLL(filepath, winmode = 0)


# Create types for use in argument types
vec3_ptr = np_ct.ndpointer(c_double, shape = (3))
vec_1d_ptr = np_ct.ndpointer(c_double, ndim = 1, flags = 'F')
vec_2d_ptr = np_ct.ndpointer(c_double, ndim = 2, flags = 'F')
generic_vec_ptr = np_ct.ndpointer(c_double)

# Set argtypes
lib.cross.argtypes = [vec3_ptr, vec3_ptr, vec3_ptr]
lib.field.argtypes = [vec3_ptr, vec3_ptr, vec3_ptr, vec3_ptr, vec3_ptr]
lib.interpolator.argtypes = [c_double, c_double, vec_1d_ptr, vec3_ptr, 
                                c_int, vec_2d_ptr]
lib.field_over_time.argtypes = [generic_vec_ptr, generic_vec_ptr, 
                                generic_vec_ptr, c_int, generic_vec_ptr, 
                                c_int, vec3_ptr, generic_vec_ptr]


#-------------------------------------------------------------------------------
# Python wrappers for the Fortran functions contained in lib
def cross(a: NDArray[float_], b:NDArray[float_]) -> NDArray[float_]:
    """
    Calculate the cross product to vectors a and b. 

    Parameters:
    a, b: NDArrays of shape (3,). Must have dtype = float

    Returns:
    c: The cross product \vec{c} = \vec{a} \times \vec{b}
    """
    c = np.zeros(3)
    lib.cross(a, b, c)
    return c
    

def calculate_field(r_part: NDArray[float_], 
                    r_det: NDArray[float_], 
                    beta: NDArray[float_], 
                    beta_dot: NDArray[float_]
                    ) -> NDArray[float_]:
    """ Calculate the acceleration field at r_det given the position, velocity, 
     and acceleration of the source particle. 
      
    Parameters:
    r_part: The particle position (in m.) 
    r_det: The position of the virtual detector (in m).
    beta: The particle's velocity, normalized to the speed of light.
    beta_dot: The particle's normalized acceleration in units of s^-1.
    
    Returns:
    E_field: The radiation field at the virtual detector (in V/m).
    """
    E_field = np.zeros(3, dtype = np.float64)
    lib.field(r_part, r_det, beta, beta_dot, E_field)
    return E_field

def interpolator(t: float, t_prev: float, 
                 t_det_array: NDArray[float_], 
                 field: NDArray[float_]) -> NDArray[float_]:
    
    nt_det = len(t_det_array)
    output = np.zeros((nt_det, 3), order = 'F')
    lib.interpolator(t, t_prev, t_det_array, field, nt_det, output)

    return output

def field_over_time(r_over_time: NDArray[float_],
                    beta_over_time: NDArray[float_], 
                    sim_times: NDArray[float_],
                    det_times: NDArray[float_],
                    det_pos: NDArray[float_]
                    ) -> NDArray[float_]:
    
    # Convert all arrays to Fortran contiguous
    r_over_time = np.asfortranarray(r_over_time)
    beta_over_time = np.asfortranarray(beta_over_time)
    sim_times = np.asfortranarray(sim_times)
    det_times = np.asfortranarray(det_times)

    # Start calculation
    nt_sim = len(sim_times)
    nt_det = len(det_times)
    all_radiation = np.zeros((nt_det, 3), order = 'F', dtype = np.float64)

    # Call fortran
    lib.field_over_time(r_over_time, beta_over_time, 
                        sim_times, c_int(nt_sim),
                        det_times, c_int(nt_det),
                        det_pos, all_radiation)

    return all_radiation


    






