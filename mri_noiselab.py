# -*- coding: utf-8 -*-
"""
The scope of this module is to reduce the noise in magnetic resonance images.
Functions:
subtract_noise

"""
import numpy as np

def subtract_noise(image, bg_area):
    """
    Performs a correction of the image, computing the standard deviation of a 
    background area, mean and the standard deviation of the region to be cleaned.

    Parameters
    ----------
    image : numpy array. Uniform image or region of an image
    bg_area : numpy array. It could be a sclicing of an image or another image
            containing only background pixel, shape can be different from image. 
    
    Returns
    -------
    A : numpy array. Same shape as image, positive or zero values 

    """
    m_ave = np.full(np.shape(image),np.mean(image)) #image average magnitude
    m_sd = np.full(np.shape(image), np.std(image)) #image standard deviation
    m_sd_bg = np.full(np.shape(image), np.std(bg_area)) #background standard deviation
    sigma_r = np.divide(m_sd_bg, 0.655) #Rayleigh sigma parameter estimation 
    
    A2 = np.square(m_ave) + np.square(m_sd) - 2*np.square(sigma_r)
    A2[A2 < 0] = 0 # positivity requirement
    A = np.sqrt(A2)
    return A