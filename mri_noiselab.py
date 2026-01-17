# -*- coding: utf-8 -*-
"""
The scope of this module is to reduce the noise in magnetic resonance images.
Functions:
 - subtract_noise
 - subtract_noise_masked

"""
import numpy as np
from scipy.ndimage import uniform_filter
import warnings

_RAYLEIGH_MEAN_STD_RATIO = 1.91
_RAYLEIGH_SIGMA_FACTOR = 0.655


def _validate_input(data, name):
    """Perform common validation checks on inputs.

    This function validates that the input is a NumPy array,
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
    
    if not isinstance(data, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")

    if not np.all(np.isfinite(data)):
        raise ValueError(f"{name} contains NaN or infinite values")

    if np.any(data < 0):
        raise ValueError(
            f"""Found negative values in the {name}. \n
            Possible solutions: clip or set an offset (equal for image and 
            background before passing it to the function."""
        )

    if np.all(data == 0):
        raise ValueError(f"{name} is totally dark, all pixels are 0.")

    return


def _estimate_rayleigh_noise(bg_area, b_tol=0.1):
    """
    Estimate the Rayleigh noise sigma parameter from background data.

    The estimation is based on the standard deviation of the background
    and assumes a uniformly Rayleigh-distributed noise model.

    Parameters
    ----------
    bg_data : numpy.ndarray
        Background pixel values containing noise only. Must be non-negative
        and contain at least some variability.
    b_tol : float
        Absolute tolerance used to assess deviation of the background
        mean-to-standard-deviation ratio from the expected Rayleigh value.

    Returns
    -------
    sigma_r : float
        Estimated Rayleigh distribution sigma parameter.

    Raises
    ------
    ValueError
        If the background standard deviation is zero (i.euniform image).
    
    Warns
    ------
    RuntimeWarning
        If the background mean-to-standard-deviation ratio deviates from the
        expected Rayleigh value within the given tolerance.
    
    """
    # background bias check 
    sd_bg = float(np.std(bg_area))
    ave_bg = float(np.mean(bg_area))

    if sd_bg == 0:
        raise ValueError("Unable to estimate noise from background (std = 0).")

    ratio = ave_bg / sd_bg

    if not np.isclose(ratio, _RAYLEIGH_MEAN_STD_RATIO, atol=b_tol):
        warnings.warn(
            f"Background may be biased: ave/std = {ratio:.3f}, expected ~{_RAYLEIGH_MEAN_STD_RATIO} ± {b_tol}",
            RuntimeWarning
        )
    # estimate sigma_r
    return sd_bg / _RAYLEIGH_SIGMA_FACTOR


def _clamp_negative_to_zero(x):
    """
    Count and clamp negative values to 0 to avoid invalid sqrt.
    
    Parameters
    ----------
    x : numpy.ndarray 
         Magnitude-corrected image to be clamped to non-negative values.
    
    Returns
    -------
    x : np.ndarray of non-negative values
        Magnitude-corrected image, clamped to non-negative values.
    """
    n_neg = np.count_nonzero(x < 0)
    
    if n_neg > 0 :
        warnings.warn(
            f"{(100 * n_neg / x.size):.2f}% negative values set to zero before square root.",
            RuntimeWarning
        )
        x[x < 0] = 0
    
    return x                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             


def subtract_noise(image, bg_area, b_tol=0.1, f_size=10, np_type=np.float32):
    """
    Performs pixel-wise noise reduction on magnitude images affected by 
    Rayleigh noise, estimating it from a background region.
 
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
    np_type: numpy.dtype, optional
        The numpy data type to be used during computation. Default is np.float32
 
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
    RuntimeError
         If numpy encounters overflow, invalid, zero division, underflow.
 
    Warns
    -----
    RuntimeWarning
         If image standard deviation is zero (uniform image).
    RuntimeWarning
        If ratio of background average and standard deviation differs from that 
        expected in a Rayleigh distribution.
    """
    # Inputs checks
    _validate_input(image, "image")
    _validate_input(bg_area, "background")
           
    try:
        with np.errstate(over='raise',invalid='raise',divide='raise',under='raise'):
            
            # Numpy array cast 
            image = image.astype(np_type, copy=False)
            bg_area = bg_area.astype(np_type, copy=False)
                        
            # background global Rayleigh distribution sigma parameter
            sigma_r = _estimate_rayleigh_noise(bg_area, b_tol=b_tol)
            
            # check image noise presence (warning only)
            if float(np.std(image)) == 0:
                warnings.warn("No noise found in image (std = 0).", RuntimeWarning)
            
            m_ave = uniform_filter(image, size=f_size) #image local average magnitude
            m_ave_sq = np.square(m_ave) #squared local average of the image
            mean_sq = uniform_filter(np.square(image), size=f_size) #local average of squared image
            m_var = mean_sq - m_ave_sq #local variance of the image
                        
            # Magnitude correction
                # NB sigma_r is still scalar: numpy broadcasting is more efficent than a full np.array
            A_squared = m_ave_sq + m_var - 2 * (sigma_r **2) 
                # positivity requirement to avoid Nan
            A_squared = _clamp_negative_to_zero(A_squared)

            A = np.sqrt(A_squared)

    except FloatingPointError as e:
        raise RuntimeError(str(e) + " during noise subtraction.\n" + 
            "Tip: to circuvent overflow try np_type=np.float64 or scale inputs down and rescale outputs."
        )

    return A


def subtract_noise_masked(image_ma, bg_ma, *, fill_value=0.0, return_masked=True, **kwargs):
    # After * no positional argument
    """
    Apply Rayleigh noise subtraction to a masked image.

    This wrapper enables the use of numpy masked arrays with the
    `subtract_noise` function, which does not natively support masks.

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
    """

    if not np.ma.isMaskedArray(image_ma):
        raise TypeError("""image_ma must be a numpy.ma.MaskedArray,
                        otherwise use subtract_noise""")

    # bg stats are based only on unmasked pixels, no need to preserve position
    if np.ma.isMaskedArray(bg_ma):
        bg_valid = bg_ma.compressed() #1D array with valid only
    else:
        bg_valid = bg_ma

    if bg_valid.size == 0:
        raise ValueError("Background has no valid (unmasked) pixels.")

    # fill masked pixels for computation
    image_filled = image_ma.filled(fill_value)

    output = subtract_noise(image_filled, bg_valid, **kwargs)

    if return_masked:
        return np.ma.array(output, mask=np.ma.getmaskarray(image_ma), copy=False)
    return output