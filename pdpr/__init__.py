import os

import ctypes as ct
from ctypes import c_double, c_int

import numpy as np
import numpy.ctypeslib as np_ct
from numpy.typing import NDArray
from numpy import float64, complex128

"""
    This package implements the phase diversity-phase retrieval algorithm 
    detailed in Section 8.6.3 in 'Intense Laser-Plasma Interactions in Ultrathin 
    Films: Plasma Mirrors, Relativistic Effects, and Orbital Angular Momentum'
    by Dr. Nicholas Czapla.

    The computationally expensive functions are written in Fortran90 for speed. 
    The majority of functions in this file are wrappers to facilitate calling 
    these fortran functions in python.

    author = B. Unzicker (5-27-2024)
"""

#-------------------------------------------------------------------------------
# Class for passing complex numbers to fortran arrays
class c_double_complex(ct.Structure): 
    """complex is a c structure
    https://docs.python.org/3/library/ctypes.html#module-ctypes suggests
    to use ctypes.Structure to pass structures (and, therefore, complex)
    """
    _fields_ = [("real", ct.c_double),("imag", ct.c_double)]
    @property
    def value(self):
        return self.real+1j*self.imag # fields declared above

#-------------------------------------------------------------------------------
# Load dll
Folder = os.path.dirname(os.path.abspath(__file__))
# lib = ct.CDLL(Folder + r'\functions.dll', winmode=0)
lib = ct.CDLL(Folder + r'\pdpr_parallel.dll', winmode=0)


# Create argument types
vec_1d_ptr = np_ct.ndpointer(c_double, ndim = 1)
gen_vec_ptr_dbl = np_ct.ndpointer(c_double)
gen_vec_ptr_cplx = np_ct.ndpointer(np.complex128)

c_complex128ptr = ct.POINTER(c_double_complex)

# Set argtypes
lib.propagator.argtypes = [gen_vec_ptr_cplx,    # U1
                           vec_1d_ptr, c_int,   # x1, n_x1
                           vec_1d_ptr, c_int,   # y1, n_y1
                           c_double,            # z1  
                           vec_1d_ptr, c_int,   # x2, n_x2
                           vec_1d_ptr, c_int,   # y2, n_y2
                           c_double,            # z2
                           gen_vec_ptr_cplx,    # U2
                           c_double,             # k
                           c_int                # n_threads
                           ]
#-------------------------------------------------------------------------------
# Wrappers
def propagator(U1: NDArray[complex128], 
               x1: NDArray[float64], y1: NDArray[float64], z1: float, 
               x2: NDArray[float64], y2: NDArray[float64], z2: float,
               k: float, n_threads: int|None = os.cpu_count()
               ) -> NDArray[complex128]:
    """
        Propagate U1 that is defined in the z1 plane to the z2 plane.

    Parameters:


    Returns:
    U2: The complex electric field in the z2 plane. 
    """

    n_x1 = len(x1)
    n_y1 = len(y1)
    n_x2 = len(x2)
    n_y2 = len(y2)

    # U1 = np.asfortranarray(U1)
    U2 = np.zeros((n_x2, n_y2), dtype = complex)

    # Call fortran
    lib.propagator(U1, x1, n_x1, y1, n_y1, z1, x2, n_x2, y2, n_y2, z2, 
                    U2, k, n_threads)

    return U2
