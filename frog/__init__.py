import numpy as np
from numpy.typing import NDArray
from numpy import float64, int_, complex128
import scipy.constants as con

def gaussian_laser(t: NDArray, wavelength: float, pulse_dur: float,
                     A: float = 1.0, t0: float = 0.0
                    ) -> NDArray[complex128]:
    """ Output the laser field over time for a laser with a Guassian temporal
        envelope. 
        
    Parameters:
    t: The times over which the pulse shape is calculated.
    wavelength: The laser wavelength (in m).
    pulse_dur: The FWHM of the intensity. (in s)
    A: Laser ampiltude in arbitrary units. By default, this is set to 1. 
    t0: Time delay for the Gaussian envelope. (in s).
    """

    wL = 2*np.pi*con.c/wavelength

    # This lets pulse_dur be the intensity FWHM.
    sigma_t = pulse_dur/(np.sqrt(2*np.log(2)))

    field = A*np.exp(-(t - t0)**2/sigma_t**2)*np.exp(1j*wL*t)

    return field

def get_pulse_duration(inten: NDArray[float64], t: NDArray[float64], 
                        background: float = 0) -> float:
    """ Return the FWHM pulse duration given the intensity curve. 
    
    Parameters:
    
    Returns:
    pulse_dur: The full-width at half maximum pulse duration of inten.
    """
    N = len(inten)
    inten = inten - background      # Background Subtraction
    half = 0.5*np.max(inten)

    i_left = np.argmin(np.abs(inten[:int(N/2)] - half))
    left_bnd = t[i_left]
    i_right = np.argmin(np.abs(inten[int(N/2):] - half))
    right_bnd = t[int(N/2):][i_right]

    return np.abs(right_bnd - left_bnd)
    

#-------------------------------------------------------------------------------
""" 
    The functions below are modified versions of Christopher Mahnke's froglib
    library. The original code can be found on Github 
    (https://github.com/xmhk/froglib/tree/master). Most of my changes involve 
    adding typehints to the functions and renaming variables so that they're 
    clearer.
"""

def gaussianrandomphase(n):
    """Return a gaussian pulse with random phase.

    Arguments:
        n : length of desired gaussian pulse array

    Returns:
        field : gaussian pulse with random phase         
    """
    tvec = np.arange(n) - int(n / 2)
    field = np.exp(-tvec**2/(n/16)**2)*np.exp(2.0j*np.pi*np.random.rand(n))
    return field


def calc_frog_trace( pulse: NDArray[float64|complex128], 
                     gate: NDArray[float64|complex128], 
                     mode:str ='shg'
                    ) -> NDArray[complex128]:
    """ Calculate the Frog Trace of given field(s). For a single-shot 
        GRENOUILLE, the gate is a replica of the pulse.
    
    Arguments:
        pulse: An (n, 1) array containing a time series representation of 
            the pulse.
        gate: An (n, 1) array containing a time series representation of 
            the gate.

    Optional Arguments:
        mode: determines whether to calculate the SHG or the Blind Frog trace.
            'shg'= SHG trace, 'blind'= Blind Frog trace

    Returns:
         frogtrace : Field of the frog trace, which in general is complex.
                     To compare with some experimental (intensity) trace,
                     the absolute value squared of Frog trace has to be taken.
    """    
    nn = len(pulse)
    n2 = int(nn / 2)

    # Set mode. Should almost always be 'shg'
    if mode == 'shg':
        ap = np.outer(pulse, gate) + np.outer(gate, pulse)  # Axis: time-time
    elif mode == 'blind':
        ap = np.outer(pulse, gate)
    else: 
        # Exit if given an invalid mode.
        print("Error: Please select set mode = 'shg' or 'blind'.")
        exit()

    m1 = np.zeros(np.shape(ap), dtype=np.complex128)
    frg_trc = np.zeros(np.shape(ap), dtype=np.complex128)

    for i in range(n2 - 1, -n2, -1):
        m1[i + n2, :] = np.roll(ap[i + n2, :], -i)
    m1 = np.transpose(m1)

    for i in range(nn):
        # Axis: time-freq
        frg_trc[i, :] = np.roll(np.fft.fft(np.roll(m1[i, :], n2)), -n2)

    frg_trc = frg_trc.T / np.max(np.abs(frg_trc))

    return frg_trc  # Axis: freq-time


def pcgp_step(exp_trc: NDArray[float64|complex128], 
              pulse: NDArray[float64|complex128], 
              gate:NDArray[float64|complex128], 
              mode:str = 'shg', svd:str = 'full'
            ) -> tuple[ NDArray[float64|complex128], 
                        NDArray[float64|complex128], 
                        NDArray[float64|complex128] ]:
    """Make a step in the pcgp algorithm.

    This function implements the Princple Components Generalized Projections
    (PCGP) Algorithm as describe in Trebino, Rick. Frequency-resolved 
    optical gating: the measurement of ultrashort laser 
    pulses. Springer Science & Business Media, 2012.

    Arguments:
        mexp: amplitude represenation of the experimental FROG trace, e.g.
              the square root of the measured intensity.
        pulse: input pulse to use as signal. When pulse==None, a gaussian with
               random phase is used.
        gatepulse: input pulse to use as gate. When pulse==None, a gaussian with
                    random phase is used.

    Optional arguments:
        mode: can either be 'shg' or 'blind'. Determines the kind of Frog trace 
                to calculate.
        svd: can either be 'full' or 'power'. When 'full' is used, the singular 
                value decomposition of numpy is used. For 'power', the (faster) 
                'power method' is used.

    Returns:
        pulse, gatepulse : iterated fields for signal and gate
        ferr : Frog error for the reconstructed trace
    """
    # Initialize pulse and gate is not given.
    if pulse is None:
        nn = np.shape(exp_trc)[0]
        pulse = gaussianrandomphase(nn)
    if gate is None:
        nn = np.shape(exp_trc)[0]
        gate = gaussianrandomphase(nn)

    nn = len(pulse)         # type:ignore
    n2 = int(nn / 2)
    guess_trc = calc_frog_trace(pulse, gate, mode=mode)    # type:ignore

    exp_inten = np.abs(exp_trc)**2/np.sum(np.sum(np.abs(exp_trc)**2))
    guess_inten = np.abs(guess_trc)**2/np.sum(np.sum(np.abs(guess_trc)**2))
    ferr = np.sqrt(1/nn**2*np.sum(np.sum(np.abs(exp_inten - guess_inten)**2)))

    m3 = np.abs(exp_trc) * np.exp(1.0j * np.angle(guess_trc))
    m3 = np.transpose(m3)  # zeit - freq
    m4 = np.zeros(np.shape(guess_trc), dtype=np.complex128)
    m5 = np.zeros(np.shape(guess_trc), dtype=np.complex128)

    for i in range(nn):
        m4[i, :] = np.roll(np.fft.ifft(np.roll(m3[i, :], -n2)), n2)
    for i in range(n2 - 1, -n2, -1):
        m5[i + n2, :] = np.roll(m4[:, i + n2], i)  # time-time
    if svd=='full':
        # full SVD
        u, w, v = np.linalg.svd(m5)
        pulse = u[:, 0]
        gate = v[0, :]
    else:
        #  power method
        pulse = np.dot(np.dot(m5, np.transpose(m5)), pulse)
        gate = np.dot(np.dot(np.transpose(m5), m5), gate)
        pulse = pulse / np.sqrt( np.sum( np.abs(pulse)**2))
        gate = gate / np.sqrt(np.sum(np.abs(gate) ** 2))
    return pulse, gate, ferr


def simplerec(exp_amp: NDArray[float64], pulse: NDArray[float64|complex128], 
              gatepulse:NDArray[float64|complex128], iterations: int = 10, 
              mode:str = 'shg', svd:str = 'full'
            ) -> dict:
    """Simple reconstruction loop for Frog traces.

    Arguments:
        exp_amp: The amplitude of the experimental FROG trace, e.g. the square 
            root of the measured intensity.

    Optional Arguments:
        iterations: number of PCGP steps to iterate (default = 10)
        mode: may be 'shg' or 'blind'  (default = 'shg')
        pulse: when given, this pulse will be feed into the PCGP loop 
                (default = None)
        gatepulse: when given, this gatepulse will be feed into the PCGP loop 
                (default = None)
        svd: method to calculate the singular value decomposition. 'full' :
                 use numpy's SVD, 'power' : use the power method.

    Returns:
        rdict : dictionary holding the values:
            errors : Array holding Frog error for each iteration
            gp, sp :  Arrays with signal and gate fields for each iteration
            minerror : minimal Frog error that occured during iterations
            min_sp, min_gp : signal and gate pulses for minimal Frog error
            mode : mode used ('shg' or 'blind')
            exp : experimental trace (amplitude)

    """
    errors = np.zeros(iterations)
    sps = np.zeros(iterations)
    gps = np.zeros(iterations)

    for i in range(iterations):
        pulse, gatepulse, frog_err = pcgp_step(exp_amp, pulse, gatepulse, 
                                                    mode=mode, svd=svd)
        errors[i] = frog_err
        sps[i] = pulse
        gps[i] = gatepulse

    # Store results in dictionary
    results = {'errors': errors, 'sp': sps, 'gp': gps, 'exp': exp_amp}
    minerr = np.min(results['errors'])
    indx = np.nonzero(minerr == results['errors'])[0][0]
    results['minerror'] = minerr                                   # type:ignore
    results['min_sp'] = results['sp'][indx]
    results['min_gp'] = results['gp'][indx]
    results['mode']=mode                                           # type:ignore

    return results
