# mri-noiselab

A python library that estimates and reduces noise in magnetic resonance images.

mri-noiselab` is a Python library for estimating from a background region and 
reducing Rayleigh-distributed noise in magnetic resonance (MRI) magnitude images, with support for masked data.
Useful in image preprocessing for quantitative MRI studies.

## Installation

### From source

Clone the repository and install it.
...
Make sure the required dependencies are installed (see below).

## Dependencies

 - numpy
 - scipy

## Documentation
Full documentatio is provided at the page...

See the following examples in this same github repository:
 - example_uniform_image.py
 - example_3levels_image.py
 - example_3_ROIs.py
 
And how to guides:
 -  howto_clean_dicom working on the MR_femur_head.dcm file

### Basic Usage

Important
Inputs must contain finite, non-negative values, numpy array.
Background regions must contain noise only (no signal).

```python
import pydicom
import numpy as np
from mri_noiselab import subtract_noise

ds = pydicom.dcmread("example_mri.dcm")
image = ds.pixel_array.astype(np.float32)

# Select a background region (noise only)
background = image[0:50, 0:50]

corrected = subtract_noise(image, background)
```


### Masked Array Support

The library supports NumPy masked arrays via dedicated wrapper 
:func:subtract_noise_masked

```python
import numpy as np
from mri_noiselab import subtract_noise_masked

ds = pydicom.dcmread("example_mri.dcm")
image = ds.pixel_array.astype(np.float32)

threshold = 10

image_ma = np.ma.masked_array(image, mask = image >= threshold)
bg_ma = np.ma.masked_array(background, mask = background < threshold)

cleaned_ma = subtract_noise_masked(image_ma, bg_ma)
```

Masked pixels are excluded from validation and restored in the output.

## Author
Serena Bedeschi
email serena.bedeschi@studio.unibo.it
github SereBede

## License
This project is licensed under the Creative Commons
Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
for educational and research purposes. 

Commercial use is not permitted without explicit permission
from the author.

## Notes

**Noise Model**

Every voxel of a Nuclear Magnetic Resonance image has a measured magnitude M,
composed by the true signal A and the noise. The noise results from the
composition of white noise (gaussian noise with zero mean) both on the 
real and complex components of NMR signal.

The noise assumes a Rayleigh distribution in the magnitude image,
which is characteristic of MRI and other coherent imaging systems where
complex or bivariate data is converted to magnitude by quadrature sum. 
The Rayleigh distribution parameter sigma_r is estimated from background as:

    sigma_r = std(bg_area) / 0.655

where 0.655 ≈ sqrt(2/π) relates the Rayleigh distribution's standard
deviation to its scale parameter (sigma).

By estimation of noise level it is possible to subtract it and so to better
approximate A, the true value of signal.

The relationship between true signal A and the measured 
magnitude average and standard deviation follows the equation:
    
    A^2 = m_ave^2 + m_sd^2 - 2 * sigma_r^2

and for A=0 (i.e. in the background) it results:
    
    m_ave_bg = 1.253*sigma_r    and    m_sd_bg = 0,655*sigma_r

Reference:
R. Mark Henkelman, "Measument of signal intensities in the presence of 
noise in MR images", published in 1985.


**Cleaning Algorithm Steps**

1. Validate inputs (non-zero or negative images, valid background statistics)
2. Compute image local statistics: mean (m_ave) and std (m_sd)
    (already squared for computation convenience)
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