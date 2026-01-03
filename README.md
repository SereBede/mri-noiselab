# mri-noiselab

## Description
A python library that estimates and reduces noise in magnetic resonance images.

This operation is particularly useful in quantitative MRI applications

The subtract_noise function reduces Rayleigh-distributed noise in images 
(e.g., MRI magnitude images) by estimating the noise level from a background
region.



###Notes

    **Noise Model**
    
    Every voxel of a Nuclear Magnetic Resonance image has a measured magnitude M,
    composed by the true signal A and the noise. The noise results from the
    composition of white noise (gaussian noise with zero mean) both on the 
    real and complex components of NMR signal.
    
    The noise assumes a Rayleigh distribution in the magnitude image,
    which is characteristic of MRI and other coherent imaging systems where
    complex or bivariate data is converted to magnitude. 
    The Rayleigh distribution parameter sigma_r is estimated from background as:
    
        sigma_r = std(bg_area) / 0.655
    
    where 0.655 ≈ sqrt(2/π) relates the Rayleigh distribution's standard
    deviation to its scale parameter.
    
    By estimation of noise level it is possible to subtract it and so to better
    approximate A, the true value of signal.
    
    The relation between true signal A and the measured 
    magnitude average and standard deviation follows the equation:
        
        A^2 = m_ave^2 + m_sd^2 - 2 * sigma_r^2

    and for A=0 (i.e. in the background) it results:
        
        m_ave_bg = 1.253*sigma_r    and    m_sd_bg = 0,655*sigma_r
    
    Reference:
    R. Mark Henkelman, "Measument of signal intensities in the presence of 
    noise in MR images", published in 1985.
    
    **Cleaning Algorithm Steps**
    
    1. Validate inputs (non-zero images, valid background statistics)
    2. Compute image statistics: mean (m_ave) and std (m_sd)
    3. Estimate Rayleigh noise parameter from background: sigma_r
    4. Calculate corrected magnitude squared: A² = m_ave² + m_sd² - 2*sigma_r²
    5. Apply positivity constraint: A²[A² < 0] = 0
    6. Return corrected magnitude: A = sqrt(A²)
    
    
    **Assumptions**
    
    - Noise follows a Rayleigh distribution
    - Background region contains only noise (negligible true signal)
    - Image region is relatively uniform (constant or slowly varying signal)
    
    **Limitations**
    
    - Works on uniform or nearly uniform regions
    - Assumes spatially uniform noise characteristics
    - May overcorrect in regions with very low SNR

## Installation & Usage

From terminal inside your workig directory or your virtual env:

    git clone https://github.com/SereBede/mri-noiselab.git

Then inside your code:

    import mri_noiselab

Now you can simply use the function subtract_noise inside your code.

## Requirements

The module requires numpy and warnings

## Documentation

See the following examples in this same github repository:
 - example_uniform_image

## Developer
This python module is p