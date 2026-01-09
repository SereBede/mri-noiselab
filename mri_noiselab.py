# -*- coding: utf-8 -*-
"""
The scope of this module is to reduce the noise in magnetic resonance images.
Functions:
subtract_noise

"""
import numpy as np
from scipy.ndimage import uniform_filter
import warnings

def subtract_noise(image, bg_area, b_tol=0.1, f_size=10):
    """
    Performs pixel-wise noise reduction on magnitude images with Rayleigh noise.
 
    This function reduces Rayleigh-distributed noise from magnitude images
    (e.g., MRI magnitude images) by estimating the noise level from a background
    region and applying a correction based on the Rician noise model. The algorithm
    computes corrected pixel intensities using the formula:
 
        A = sqrt(m_ave² + m_sd² - 2*sigma_r²)
 
    where sigma_r is the estimated Rayleigh noise parameter derived from the
    background standard deviation.
 
    Parameters
    ----------
    image : numpy.ndarray of positive values
         Input image to be corrected, typically a 2D array representing
         a uniform region or entire magnitude image affected by Rayleigh noise. 
    bg_area : numpy.ndarray of positive values
         Background region containing only noise (no signal). Can be a slice of
         the input image or a separate array. Shape may differ from `image`.
         Should represent a region where the true signal is zero or negligible,
         containing only noise to be characterized.
    b_tol : float, optional
         Bias tolerance parameter for validation checks. Default is 0,1.
    f_size: float, optional
        The size in pixels of local filter. Default is 10.
 
    Returns
    -------
     A : numpy.ndarray
         Noise-corrected image with the same shape as input `image`. All values
         are non-negative (>= 0) due to the positivity constraint applied to
         prevent unphysical negative intensities.
 
    Raises
    ------
    ValueError
         If background region has negative values.
    ValueError
         If input image has negative values.
    ValueError
         If background region is entirely zero (all pixels == 0).
    ValueError
         If input image is entirely zero (all pixels == 0).
    ValueError
         If background has zero standard deviation (constant values).
    ValueError
         If background standard deviation is negative (should never occur).
 
    Warns
    -----
    RuntimeWarning
         If image standard deviation is zero (uniform image).
    RuntimeWarning
        If ratio of background average and standard deviation differs from that 
        expected in a Rayleigh distribution.
    

    """
    # Checks on inputs
    
    if np.any(image < 0):
        raise ValueError("""Found negative values in the image.\n
                         Possible solutions: clip or set an offset before 
                         passing it to the function.""")
    
    if np.any(bg_area < 0):
        raise ValueError("""Found negative values in the background.\n
                         Possible solutions: clip or set an offset before 
                         passing it to the function.""")
    
    if np.all(bg_area == 0):
        raise ValueError("Background is totally dark, all pixel are 0")
        
    if np.all(image == 0):
        raise ValueError("Image is totally dark, all pixel are 0")
    
    sd_bg = np.std(bg_area)
    ave_bg = np.mean(bg_area)   

    if sd_bg == 0:
        raise ValueError("Unable to estimate noise from background (std = 0)")
    
    ratio = ave_bg / sd_bg

    if not np.isclose(ratio, 1.91, atol=b_tol):
        warnings.warn(
            f"Background may be biased: ave/std = {ratio:.3f},expected ~1.91 ± {b_tol}",
            RuntimeWarning)
    
    sd_img = np.std(image)
       
    if sd_img == 0:
        warnings.warn("No noise found in image (std = 0)", RuntimeWarning)
   
    
    # Computation
    
    m_ave = uniform_filter(image, size=f_size) #image local average magnitude
    m_ave_squared = np.square(m_ave) #square of local average magnitude
    squares_mean = uniform_filter(image**2, size=f_size) #local mean of the squared image 
    m_sd_squared = np.subtract(squares_mean,m_ave_squared) #local variance of the image

    m_sd_bg = np.full(np.shape(image), sd_bg) #background standard deviation
    sigma_r = np.divide(m_sd_bg, 0.655) #Rayleigh sigma parameter estimation 
    
    A_squared = m_ave_squared + m_sd_squared - 2*np.square(sigma_r)
    
    if np.any(A_squared < 0):
        warnings.warn("""Obtained at least one negative value,
                      changed into zero before final square root.""", RuntimeWarning)
        A_squared[A_squared < 0] = 0 # positivity requirement
    
    A = np.sqrt(A_squared)
    return A
