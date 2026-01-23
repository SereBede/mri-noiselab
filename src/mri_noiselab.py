# -*- coding: utf-8 -*-
"""
Rayleigh noise estimation and subtraction for MRI magnitude images.

This module provides utilities to estimate and remove Rayleigh-distributed
noise from magnetic resonance (MRI) magnitude images using a background-based
approach derived from the Rician noise model.

The public functions are:

- :func: subtract_noise
- :func:`subtract_noise_masked`

The module is designed for research and educational purposes.

"""

import numpy as np
from scipy.ndimage import uniform_filter
import warnings
from typing import Any
import numpy.typing as npt

#Real number array not bool or complex
NDArrayReal = npt.NDArray[np.floating[Any] | np.integer[Any] | np.unsignedinteger[Any]]
MaskedNDArray = np.ma.MaskedArray

# these factors are determined by the Raileygh probability distribution 
_RAYLEIGH_MEAN_STD_RATIO = 1.91
_RAYLEIGH_SIGMA_FACTOR = 0.655


def _validate_input(data:NDArrayReal, name:str) -> None:
    """Perform common validation checks on inputs.

    This function validates that the input is a NumPy array (not masked),
    contains only finite and non-negative values, and is not entirely zero.

    Parameters
    ----------
    data : numpy.ndarray 
        Input array to be validated.
    name : str
        Name of the input (used for informative error messages).


    Raises
    ------
    TypeError
        If the input is not a numpy.ndarray.
    ValueError
        If the input contains NaN or infinite values.
    ValueError
        If the input contains negative values.
    ValueError
        If the input is entirely zero.
    """
    # masked array handled with subtract_noise_masked
    if not isinstance(data, np.ndarray): 
        raise TypeError(f"{name} must be a numpy.ndarray")

    if not np.all(np.isfinite(data)): 
        raise ValueError(
            f"""{name} contains NaN or infinite values, 
            Possible solution: mask them and use subtract_noise_masked.""")

    if np.any(data < 0): # negative values handling not implemented in cleaning
        raise ValueError(
            f"""Found negative values in the {name}. \n
            Possible solutions: clip or mask or set an offset (equal for image 
            and background) before passing it to the function."""
        )

    if np.all(data == 0): # nothing to be done
        raise ValueError(f"{name} is totally dark, all pixels are 0.")

    return


def _estimate_rayleigh_noise(bg_area:NDArrayReal, b_tol:float = 0.1) -> float:
    """
    Estimate the Rayleigh noise sigma parameter from background data.

    The estimation is based on the standard deviation of the background
    and assumes a uniformly Rayleigh-distributed noise model.
    Before computation bias presence and null standard deviation are checked. 

    Parameters
    ----------
    bg_area : numpy.ndarray
        Background pixel values containing noise only. Must be non-negative
        and contain at least some variability.
    b_tol : float
        Absolute tolerance used to assess deviation of the background
        mean-to-standard-deviation ratio from 1.91, the expected Rayleigh value.

    Returns
    -------
    sigma_r : float
        Estimated Rayleigh distribution sigma parameter.

    Raises
    ------
    ValueError
        If the background standard deviation is zero (i.e uniform image).
    
    Warns
    -----
    RuntimeWarning
        If the background mean-to-standard-deviation ratio deviates from the
        expected Rayleigh distribution value within the given tolerance.
    
    """
     
    sd_bg = float(np.std(bg_area))
    ave_bg = float(np.mean(bg_area))

    if sd_bg == 0: # inform the user and avoid zero division
        raise ValueError("Unable to estimate noise from background (std = 0).")
    
    # anomalies in this ratio may be syntoms of bias, better tell the user
    ratio = ave_bg / sd_bg  

    if not np.isclose(ratio, _RAYLEIGH_MEAN_STD_RATIO, atol=b_tol):
        warnings.warn(
            f"""Background may be biased: ave/std = {ratio:.3f}, 
            expected ~{_RAYLEIGH_MEAN_STD_RATIO} ± {b_tol}""",
            RuntimeWarning
        )
    # estimate sigma_r 
    return sd_bg / _RAYLEIGH_SIGMA_FACTOR


def _clamp_negative_to_zero(x:NDArrayReal) -> NDArrayReal:
    """
    Clamp negative values to 0 to avoid invalid square root.
    
    If negative values are found, warns with the percentage of clipped value.
    
    Parameters
    ----------
    x : numpy.ndarray 
         Magnitude-corrected image to be clamped to non-negative values.
    
    Returns
    -------
    x : np.ndarray of non-negative values
        Magnitude-corrected image, clamped to non-negative values.
        
    Warns
    -----
    RuntimeWarning
        If negative values are found, a warning reports the percentage of
+        values clamped to zero.

    """
    n_neg = np.count_nonzero(x < 0)
    
    # in case too many negative values are found after cleaning algorithm
    # user can decide a data preprocess strategy
    if n_neg > 0 :
        warnings.warn(
            f"{(100 * n_neg / x.size):.2f}% negative values set to zero before square root.",
            RuntimeWarning
        )
        x[x < 0] = 0
    
    return x                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             


def subtract_noise(
        image:NDArrayReal,
        bg_area:NDArrayReal, 
        b_tol:float = 0.1, 
        f_size:int = 10, 
        np_type:npt.DTypeLike = np.float32
        ) -> NDArrayReal:
    """
    Perform pixel-wise Rayleigh noise reduction on MRI magnitude images.

    Noise is estimated from a background region assumed to contain
    Rayleigh-distributed noise and removed using a magnitude correction
    derived from the Rician noise model.
 
    Parameters
    ----------
    image : numpy.ndarray
         Input magnitude image. Values must be finite and non-negative.
        Can be N-dimensional as biomedical images.
    bg_area : numpy.ndarray 
         Background region containing only noise (no signal).
         Values must be finite and non-negative.
    b_tol : float, optional
         Absolute tolerance used to assess deviation of the background
         mean-to-standard-deviation ratio from 1.91, the expected Rayleigh value.
         Default is 0.1
    f_size : int, optional
        Size of the local window (in pixel) used to compute mean and variance.
        Default is 10.
    np_type : numpy.dtype, optional
        The numpy data type to be used during computation.
        Default is np.float32
 
    Returns
    -------
     A : numpy.ndarray
         Noise-corrected image with the same shape as `image`.
         Output values are non-negative.
 
    Raises
    ------
    TypeError
         If the input is not a numpy.ndarray (masked arrays must be handled via subtract_noise_masked)
    ValueError
         If background region has negative values.
    ValueError
         If input image has negative values.
    ValueError
         If background region is entirely zero (all pixels == 0).
    ValueError
         If input image is entirely zero (all pixels == 0).
    ValueError
         If background has zero standard deviation (noiseless values).
    RuntimeError
         If numpy encounters overflow, invalid, zero division, underflow.
 
    Warns
    -----
    RuntimeWarning
         If image standard deviation is zero (uniform noisless image).
    RuntimeWarning
        If ratio of background average and standard deviation differs from 1.91, 
        as expected in a Rayleigh distribution.
    RuntimeWarning
        If negative values are produced by the correction and are clamped to zero.
        
    Notes
    -----
    - The algorithm assumes spatially stationary Rayleigh noise.
    - Masked arrays must be handled via :func:subtract_noise_masked
    - Negative input values are not supported; they must be masked or removed
      prior to calling this function.
    - Background area can be a slice of the input image or a separate array.
    - Background size and shape may differ from `image`.
    - The choice of `f_size` affects the trade-off between noise reduction and
      spatial resolution.
      
    Examples
    --------
    Load an MRI magnitude image from a DICOM file, estimate noise from a
    background region, and apply noise correction:
    
    import pydicom
    import numpy as np
    from mri_noiselab import subtract_noise
    
    ds = pydicom.dcmread("example_mri.dcm")
    image = ds.pixel_array.astype(np.float32)
    
    # Select a background region (noise only)
    background = image[0:50, 0:50]
    
    corrected = subtract_noise(image, background)
    
    """
    # NOTE invalid pixel can be masked with subtract_noise_masked to pass through input checks
    _validate_input(image, "image")
    _validate_input(bg_area, "background")
           
    try:
        with np.errstate(over='raise',invalid='raise',divide='raise',under='raise'):
            
            # Numpy array cast so user can decide for data precision
            image = image.astype(np_type, copy=False)
            bg_area = bg_area.astype(np_type, copy=False)
                        
            # background global Rayleigh distribution sigma parameter
            sigma_r = _estimate_rayleigh_noise(bg_area, b_tol=b_tol)
            
            # physically uncommon a completely uniform image with no noise
            if float(np.std(image)) == 0:
                warnings.warn("No noise found in image (std = 0).", RuntimeWarning)
            
            # image stats locally computed avoiding manual windowing cycle
            m_ave = uniform_filter(image, size=f_size) #local average magnitude
            m_ave_sq = np.square(m_ave) #squared local average
            mean_sq = uniform_filter(np.square(image), size=f_size) #local average of squares
            m_var = mean_sq - m_ave_sq #local variance 
                        
            # Magnitude correction
                # NB sigma_r is still scalar: numpy broadcasting is more efficent than a full np.array
            A_squared = m_ave_sq + m_var - 2 * (sigma_r **2) 
                # positivity requirement to avoid Nan in sqrt
            A_squared = _clamp_negative_to_zero(A_squared)

            A = np.sqrt(A_squared)

    except FloatingPointError as e:
        raise RuntimeError(str(e) + " during noise subtraction.\n" + 
            "Tip: to circuvent overflow try np_type=np.float64 or scale inputs down and rescale outputs."
        )

    return A


def subtract_noise_masked(
        image_ma: MaskedNDArray,
        bg_ma: MaskedNDArray | NDArrayReal,
        *, # After * no positional argument
        fill_value:float = 0.0,
        return_masked:bool = True,
        **kwargs
        ) -> (MaskedNDArray | NDArrayReal):
    
    """
    Apply Rayleigh noise subtraction to a masked image.

    This wrapper enables the use of numpy masked arrays with the
    :func:`subtract_noise` function, which does not natively support masks.
    
    Useful in case segmentation is provided or to hide problematic pixels. 

    Masked pixels are temporarily filled with a constant value for the
    numerical computation and the original mask is restored on output.

    Parameters
    ----------
    image_ma : numpy.ma.MaskedArray
        Input image as a masked array. Masked pixels are excluded from
        validation checks and restored in the output.
    bg_ma : numpy.ndarray or numpy.ma.MaskedArray
        Background region used for noise estimation. If masked, only
        unmasked values are used to compute statistics.
    fill_value : float, optional
        Value used to fill masked pixels before computation.
        Default is 0.0 (it will affect local mean and std at borders).
    return_masked : bool, optional
        If True, return a masked array with the same mask as `image_ma`.
        If False, return a regular numpy.ndarray.
    **kwargs
        Additional keyword arguments forwarded to `subtract_noise`.

    Returns
    -------
    numpy.ndarray or numpy.ma.MaskedArray
        Noise-corrected image. Masked if `return_masked=True`.

    Raises
    ------
    TypeError
        If `image_ma` is not a numpy.ma.MaskedArray.
    ValueError
        If the background has no valid (unmasked) pixels.
    
    Notes
    -----
    - Noise correction is still computed over the full image extent.
    - Masking only affects validation, background statistics, and the output mask.
    - Use `fill_value` carefully to minimize boundary artifacts.
    
    Examples
    --------
    Apply noise correction to a DICOM image using a masked array to exclude
    invalid pixels:
    
    import pydicom
    import numpy as np
    import numpy.ma as ma
    from mri_noiselab import subtract_noise_masked
    
    ds = pydicom.dcmread("example_mri.dcm")  
    image = ds.pixel_array.astype(np.float32)
    
    mask = image <= 0
    image_ma = ma.masked_array(image, mask=mask)
    
    background = image[0:50, 0:50]
    corrected = subtract_noise_masked(image_ma, background)
    
    """

    if not np.ma.isMaskedArray(image_ma):
        raise TypeError("""image_ma must be a numpy.ma.MaskedArray,
                        otherwise use subtract_noise""")
       
    if np.ma.isMaskedArray(bg_ma):
        bg_valid = bg_ma.compressed()  # no need to preserve position in bg stats
    else:
        bg_valid = bg_ma # no need for the background to be masked too

    if bg_valid.size == 0:
        raise ValueError("Background has no valid (unmasked) pixels.")

    # user can better control borders effect in local image statistic computation 
    image_filled = image_ma.filled(fill_value)
    
    # NOTE computation is still performed for all the image size
    output = subtract_noise(image_filled, bg_valid, **kwargs)

    if return_masked: #if fill is zero maybe user has no need to mask the output
        return np.ma.array(output, mask=np.ma.getmaskarray(image_ma), copy=False)
    return output 