# mri-noiselab

## Description
A python library that estimates and reduces noise in magnetic resonance images.

This operation is particularly useful in quantitative MRI applications

The code follows the calculation presented in the article of 
R. Mark Henkelman, "Measument of signal intensities in the presence of 
noise in MR images", published in 1985.
Every voxel of a Nuclear Magnetic Resonance image has a measured magnitude M,
composed by the true signal A and the Rayleigh distributed noise, this latter 
resulting from white noise (gaussian noise with zero mean and a certain sigma) 
both on the real and complex components of NMR signal.

 The standard deviation of the measured magnitude follows the equation:
    
    (1) m_sd^2 = 2 * sigma^2 + A^2 - m_ave^2

and for A=0 (i.e. in the background) it results:
    
    (2) m_ave_bg = 1.253*sigma    and    (3) m_sd_bg = 0,655*sigma

The article -and so the code- relies on the the fact that by inferring the noise
level it is possible to subtract it and so approximate better A, the true 
value of signal.


## Installation

## Requirements

## Documentation