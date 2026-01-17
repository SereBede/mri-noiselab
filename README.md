# mri-noiselab

## Description
A python library that estimates and reduces noise in magnetic resonance images.

This operation is particularly useful in quantitative MRI applications

The subtract_noise function reduces Rayleigh-distributed noise in images 
(e.g., MRI magnitude images) by estimating the noise level from a background
region. Cleaning is performed locally and provides also numpy masked support.

## Installation
Clone the repository:
From terminal inside your workig directory or your virtual env:
```
    git clone https://github.com/SereBede/mri-noiselab.git
```
Then inside your code:
```python
    import mri_noiselab
```
Make sure the required dependencies are installed (see below).

## Requirements

The module requires:
 - numpy
 - warnings
 - uniform_filter from scipy.ndimage

## Documentation

###Basic Usage
⚠️ Important
Inputs must contain finite, non-negative values.
Background regions must contain noise only (no signal).

```python
import numpy as np
from mri_noiselab import subtract_noise

# example image and background
image = np.random.rayleigh(scale=20, size=(100, 100)) + 50
background = np.random.rayleigh(scale=20, size=(30, 30))

cleaned = subtract_noise(image, background, f_size=10)
```

Masked Array Support

The library supports NumPy masked arrays via a dedicated wrapper:

```python
import numpy as np
from mri_noiselab import subtract_noise_masked

image_ma = np.ma.masked_array(image, mask=image < 10)
bg_ma = np.ma.masked_array(background, mask=background < 5)

cleaned_ma = subtract_noise_masked(image_ma, bg_ma)
```

Masked pixels are excluded from validation and restored in the output.

### Examples

See the following examples in this same github repository:
 - example_uniform_image.py
 - example_3levels_image.py
 - example_3_ROIs.py
 
And how to guides:
 -  howto_clean_dicom working on the MR_femur_head.dcm file
  
 
## License

## Developer
This python module is p

## Notes

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

1. Validate inputs (non-zero or negative images, valid background statistics)
2. Compute image statistics: mean (m_ave) and std (m_sd)
3. Estimate Rayleigh noise parameter from background: sigma_r
4. Calculate corrected magnitude squared: A² = m_ave² + m_sd² - 2*sigma_r²
5. Apply positivity constraint: A²[A² < 0] = 0
6. Return corrected magnitude: A = sqrt(A²)


**Assumptions**

- Images and background are positive valued
- Noise follows a Rayleigh distribution 
- Background region contains only noise (negligible true signal)


**Limitations**

- Assumes spatially uniform noise characteristics
- The filter size affects the fine details and edges of the images