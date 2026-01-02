# -*- coding: utf-8 -*-
"""
The scope of this module is to reduce the noise in magnetic resonance images.
Functions:
subtract_noise

"""
import numpy as np
import warnings

def subtract_noise(image, bg_area):
    """
    Perform pixel-wise noise reduction on magnitude images with Rayleigh noise.
 
    This function removes Rayleigh-distributed noise from magnitude images
    (e.g., MRI magnitude images) by estimating the noise level from a background
    region and applying a correction based on the Rician noise model. The algorithm
    computes corrected pixel intensities using the formula:
 
        A = sqrt(m_ave² + m_sd² - 2*sigma_r²)
 
    where sigma_r is the estimated Rayleigh noise parameter derived from the
    background standard deviation.
 
    Parameters
    ----------
    image : numpy.ndarray
         Input image to be corrected, typically a 2D array representing a uniform
         region or entire magnitude image affected by Rayleigh noise. 
    bg_area : numpy.ndarray
         Background region containing only noise (no signal). Can be a slice of
         the input image or a separate array. Shape may differ from `image`.
         Should represent a region where the true signal is zero or negligible,
         containing only noise to be characterized.
 
 
     Returns
     -------
     A : numpy.ndarray
         Noise-corrected image with the same shape as input `image`. All values
         are non-negative (>= 0) due to the positivity constraint applied to
         prevent unphysical negative intensities.
 
    Raises
    ------
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
    UserWarning
         If background mean is exactly zero.
    RuntimeWarning
         If image mean is exactly zero.
    RuntimeWarning
         If image standard deviation is zero (uniform image).
    

    """
    
    if np.all(bg_area == 0):
        raise ValueError("Background is totally dark, all pixel are 0")
        
    if np.all(image == 0):
        raise ValueError("Image is totally dark, all pixel are 0")
    
    sd_bg = np.std(bg_area)
    ave_bg = np.mean(bg_area)   

    if sd_bg == 0:
        raise ValueError("Unable to estimate noise from background (std = 0)")
      
    if ave_bg == 0:
        warnings.warn("Background average results 0", UserWarning) 

    ave_img = np.mean(image)
    sd_img = np.std(image)
    
    if ave_img == 0:
        warnings.warn("Image average results 0", RuntimeWarning)
    
    if sd_img == 0:
        warnings.warn("No noise found in image (std = 0)", RuntimeWarning) 
    
    m_ave = np.full(np.shape(image),ave_img) #image average magnitude
    m_sd = np.full(np.shape(image),sd_img ) #image standard deviation
    m_sd_bg = np.full(np.shape(image), sd_bg) #background standard deviation
    sigma_r = np.divide(m_sd_bg, 0.655) #Rayleigh sigma parameter estimation 
    
    A2 = np.square(m_ave) + np.square(m_sd) - 2*np.square(sigma_r)
    A2[A2 < 0] = 0 # positivity requirement
    
    A = np.sqrt(A2)
    return A



    