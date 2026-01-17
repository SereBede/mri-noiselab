import pytest

import numpy as np
import mri_noiselab as mrnl
from mri_noiselab import subtract_noise, subtract_noise_masked


# For sake of simpicity images are generated with raileigh noise 
#  and a positive offset as signal
     

# Basic functionality tests

def test_return_type():
    """Test that it returns a numpy array
    Given: An image and background area
    When: The subtract_noise function is called
    Then: The result should be a numpy ndarray"""
    rng = np.random.default_rng(seed=42)
    image = rng.rayleigh(scale=10, size=(50, 50)) + 50
    bg_area = rng.rayleigh(scale=10, size=(20, 20))
    result = subtract_noise(image, bg_area)
    
    assert isinstance(result, np.ndarray)


# Size tests

def test_return_shape_matches_input():
    """Test that output has the same shape as input
    Given: Images of different shapes
    When: The subtract_noise function is called on each image
    Then: Each output should have the same shape as its corresponding input image"""
    shapes = [(50, 50), (500, 500), (200, 150), (10, 10), (5, 5, 5), (4, 4, 4, 4)]
    rng = np.random.default_rng(seed=42)
    
    for i, shape in enumerate(shapes):
        image = rng.rayleigh(scale=10, size=shape) + 50
        bg_area = rng.rayleigh(scale=10, size=(30, 30))
        result = subtract_noise(image, bg_area)
        
        assert result.shape == shape


def test_single_voxel_image():
    """Test with an image of only one pixel or voxel
    Given: An extended background and a single voxel image
    When: The subtract_noise function is called
    Then: A warning should be raised since std results 0"""
    rng = np.random.default_rng(seed=42)
    image = rng.rayleigh(scale=10, size=(1,1,1))+ 50
    bg_area = rng.rayleigh(scale=10, size=(100, 100)) 
    
    with pytest.warns(RuntimeWarning, 
                      match=r"No noise found in image \(std = 0\)"):
        subtract_noise(image, bg_area)


def test_bg_area_different_shapes():
    """Test with bg_area of different sizes
    Given: An image and background areas of various different sizes
    When: The subtract_noise function is called with each background size
    Then: The result should always have the same shape as the input image"""
    rng = np.random.default_rng(seed=42)
    image = rng.rayleigh(scale=10, size=(100, 100)) + 50
    
    bg_shapes = [(20, 20), (50, 50), (30, 40), (10, 100), (5, 5, 5)]
    
    for bg_shape in bg_shapes:
        bg_area = rng.rayleigh(scale=10, size=bg_shape)
        result = subtract_noise(image, bg_area)
        
        assert result.shape == image.shape
        

def test_bg_area_single_voxel():
    """Test with bg_area of only one pixel or voxel
    Given: An extended image and only one background point
    When: The subtract_noise function is called
    Then: ValueError should be raised since std results 0"""
    rng = np.random.default_rng(seed=42)
    image = rng.rayleigh(scale=10, size=(100, 100)) + 50
    bg_area = rng.rayleigh(scale=10, size=(1,1,1))
    
    with pytest.raises(ValueError,
            match=r"Unable to estimate noise from background \(std = 0\)"):
        subtract_noise(image, bg_area)


def test_bg_area_same_size_as_image():
    """Test when bg_area has same size as image
    Given: An image and background area with identical dimensions
    When: The subtract_noise function is called
    Then: The result should have the same shape as the input"""
    rng = np.random.default_rng(seed=42)
    size = (100, 100)
    image = rng.rayleigh(scale=10, size=size) + 50
    bg_area = rng.rayleigh(scale=10, size=size)
    
    result = subtract_noise(image, bg_area)
    
    assert result.shape == size


def test_bg_area_slice_of_image():
    """Test with bg_area as slice of the original image
    Given: An image and a background area that is a slice of the same image
    When: The subtract_noise function is called
    Then: The result should have the same shape as the input image"""
    rng = np.random.default_rng(seed=42)
    image = rng.rayleigh(scale=50, size=(100, 100))
    bg_area = image[0:20, 0:20]  # slice of top-left corner
    
    with pytest.warns(RuntimeWarning,
                      match=r"Obtained at least one negative value"): 
        result = subtract_noise(image, bg_area)
    
    assert result.shape == image.shape


def test_filter_different_sizes():
    """Test with filter of different sizes
    Given: A uniform image and a background area
    When: The subtract_noise function is called with each filter size
    Then: The result should always have the same shape as the input image
        """
    rng = np.random.default_rng(seed=42)
    image = rng.rayleigh(scale=10, size=(100, 100)) + 50
    bg_area = rng.rayleigh(scale=10, size=(50,50))
    
    filter_sizes = [1, 10, 20, 50, 100, 200]
    
    for fs in filter_sizes:
        
        result = subtract_noise(image, bg_area, f_size=fs)
        
        assert result.shape == image.shape


# Positivity,no Nan, no infinity requirement tests

def test_clamp_negative_to_zero_warns_and_clamps():
    """
    Given: an array containing at least one negative value
    When: _clamp_negative_to_zero is called
    Then: it emits a RuntimeWarning and returns an array with negatives set to 0
    """
    x = np.array([1.0, -0.5, 2.0], dtype=np.float32)

    with pytest.warns(RuntimeWarning, match="negative value"):
        y = mrnl._clamp_negative_to_zero(x)

    assert isinstance(y, np.ndarray)
    assert np.all(y >= 0)
    assert y[1] == 0


def test_result_positivity_requirement():
    """Test that A_squared[A_squared < 0] = 0 works correctly before square root is performed
    Given: An image with low intensity and background with high noise (A_squared might be negative)
    When: The subtract_noise function is called
    Then: The result should not contain any negative or NaN values"""
    # Create a case where A_squared might be negative
    rng = np.random.default_rng(seed=42)
    image = rng.rayleigh(scale=20, size=(100, 100)) + 1 # low signal compared to
    bg_area = rng.rayleigh(scale=20, size=(50, 50)) #  high background noise
    
    with pytest.warns(RuntimeWarning, 
                      match=r"Obtained at least one negative value"):
        result = subtract_noise(image, bg_area)
    
    # no square root of negative number performed
    assert not np.any(np.isnan(result))    
    assert np.all(result >= 0)

    
def test_image_positive_requirement():
    """Test with negative values in input
    Given: An image with negative values
    When: The subtract_noise function is called
    Then: ValueError is raised since computation does not allow negative values"""
    rng = np.random.default_rng(seed=42)
    image = - (rng.rayleigh(scale= 50, size=(100, 100))) + 25
    bg_area = rng.rayleigh(scale= 50, size=(50, 50))
    
    with pytest.raises(ValueError, 
                       match=r"Found negative values in the image"):
        subtract_noise(image, bg_area)

        
def test_bg_positive_requirement():
    """Test with negative values in input in the background
    Given: An background area containing negative values
    When: The subtract_noise function is called
    Then: ValueError is raised since computation does not allow negative values"""
    rng = np.random.default_rng(seed=42)
    image = rng.rayleigh(scale= 50, size=(100, 100))
    bg_area = - rng.rayleigh(scale= 50, size=(50, 50))
    
    with pytest.raises(ValueError, 
                       match=r"Found negative values in the background"):
        subtract_noise(image, bg_area)


def test_no_inf_values():
    """Test that there are no Inf values in the result, e.g. no division by zero has occurred
    Given: An image and background area
    When: The subtract_noise function is called
    Then: The result should not contain any infinite values"""
    rng = np.random.default_rng(seed=42)
    image = rng.rayleigh(scale=10, size=(100, 100)) + 50
    bg_area = rng.rayleigh(scale=10, size=(50, 50))
    
    result = subtract_noise(image, bg_area)
    
    # no division by zero has occurred
    assert not np.any(np.isinf(result))

# Overflow and underflow

def test_overflow_in_square_raises():
    """
    Force overflow during computation.
    Given: An image and a background area with very high values 
    When: The subtract_noise function is called
    Then: Numpy raises FloatingPointError and RuntimeError is raised
    """
    
    rng = np.random.default_rng(seed=42)
    bg_area = rng.rayleigh(scale=1e18, size=(50, 50)) # background valid
    image = np.add(np.full((50, 50), 1e20, dtype=np.float32),bg_area)  # square -> inf in float32

    with pytest.raises(RuntimeError, match=r"(overflow|Numerical overflow|invalid operation)"):
        subtract_noise(image, bg_area, f_size=3)
        

# Input validation

def test_validate_raises_on_not_ndarray_input():
    """
    Given: 2D data but not numpy array
    When: _validate_input is called
    Then: it raises TypeError
    """
    # array but not numpy array
    data = [[-1,0,1],
            [0,1,-2],
            [1,-3,4]]
    
    with pytest.raises(TypeError, match="must be a numpy.ndarray"):
        mrnl._validate_input(data,"input data")

        
def test_validate_raises_on_not_finite_elements():
    """
    Given: Input data containing inf and Nan
    When: _validate_input is called
    Then: it raises ValueError
    """
    # Generating data with inf and nan with a zero division
    data = [[-1,0,1],
            [0,1,-2],
            [1,-3,4]]
    data_in = np.array(data)/0
    
    with pytest.raises(ValueError, match="contains NaN or infinite"):
        mrnl._validate_input(data_in,"input data")


def test_validate_raises_on_negative_input():
    """
    Given: 2D numpy array with negative values
    When: _validate_inputs is called
    Then: it raises ValueError
    """
    # contains negatives
    data = [[-1,0,1],
            [0,1,-2],
            [1,-3,4]]
    data_in = np.array(data)
    
    with pytest.raises(ValueError, match="negative values"):
        mrnl._validate_input(data_in,"input data")


def test_validate_raises_on_all_zero_input():
    """
    Given: an all-zero input data
    When: _validate_inputs_and_estimate_noise is called
    Then: it raises ValueError
    """
    data_in = np.zeros((32, 32), dtype=np.float32)
    
    with pytest.raises(ValueError, match="totally dark"):
        mrnl._validate_input(data_in,"input data")


def test_zero_background():
    """Test with background containing only zeros, no noise is present
    Given: An image with noise and a background area containing only zeros
    When: The subtract_noise function is called
    Then: ValueError is raised"""
    rng = np.random.default_rng(seed=42)
    image = rng.rayleigh(scale=50, size=(100, 100)) + 50
    bg_area = np.zeros((50, 50))
    
    # Since background is uniformly 0, std(bg_area) = 0, so there is no noise  
    with pytest.raises(ValueError, match=r"background is totally dark"):
        subtract_noise(image, bg_area)


def test_all_zeros_image():
    """Test with image of all zeros
    Given: An image of all zeros and a random background
    When: The subtract_noise function is called
    Then: Value Error is raised, since there is nothing to be cleaned"""
    rng = np.random.default_rng(seed=42)
    image = np.zeros((50, 50))
    bg_area = rng.rayleigh(scale=10, size=(30, 30))
    
    # Since the image is uniformly 0,
    with pytest.raises(ValueError,match=r"image is totally dark"):
        subtract_noise(image, bg_area)


def test_uniform_image():
    """Test with uniform image (all pixels equal, meaning no noise)
    Given: A uniform noiseless image and a background affected by noise
    When: The subtract_noise function is called
    Then: A runtime warning is raised and the square(mean result) 
        should be close to square(m_ave) - 2*square(sigma_r) """
    rng = np.random.default_rng(seed=42)
    img_level = 50
    image = np.full((100, 100), img_level)
    bg_area = rng.rayleigh(scale=10, size=(50, 50))
    
    sigma_r = np.std(bg_area)/0.655
        
    # With uniform image, std(image) = 0
    with pytest.warns(RuntimeWarning, 
                      match=r"No noise found in image \(std = 0\)"):
        result = subtract_noise(image, bg_area)
            
    assert np.isclose((np.mean(result)**2),(img_level**2 - 2*(sigma_r**2))) 


def test_uniform_background_estimation_error():
    """
    Given: A uniform constant not-null noiseless background 
    When: The subtract_noise function is called
    Then: ValueError is raised since noise is estimated to be 0
    """
    bg_level= 30
    bg_area = np.full((50, 50), bg_level)
    
    with pytest.raises(ValueError,
                match=r"Unable to estimate noise from background \(std = 0\)"):
        mrnl._estimate_rayleigh_noise(bg_area, b_tol=0.1)
 

# Noise level tests

def test_noise_est_warns_on_biased_background():
    """
    Given: a background with a bias (mean/std ratio not Rayleigh-like)
    When: _estimate_noise is called
    Then: it emits a RuntimeWarning about biased background
    """
    rng = np.random.default_rng(0)
    biased_bg = rng.rayleigh(scale= 50, size=(100, 100)) + 100

    with pytest.warns(RuntimeWarning, match="Background may be biased"):
        mrnl._estimate_rayleigh_noise(biased_bg, b_tol=0.1)


def test_not_rayleigh_noise():
    """
    Given: A background with gaussian distributed noise
    When: The _estimate_rayleigh_noise function is called
    Then: RuntimeWarning is raised since the ratio of background average and 
        standard deviation differs from 1.91, expected in a Rayleigh distribution."""
    rng = np.random.default_rng(seed=42)
    bg_level= 30
    bg_area = rng.normal(loc=0.0, scale=bg_level, size=(50,50))
    
    with pytest.warns(RuntimeWarning, match="Background may be biased"):
        mrnl._estimate_rayleigh_noise(bg_area, b_tol=0.1)
        
    
def test_very_high_background():
    """Test with very high background standard deviation
    Given: An image with low noise and a background with very high standard deviation
    When: The subtract_noise function is called
    Then: The positivity constraint should handle negative A_squared values correctly"""
    rng = np.random.default_rng(seed=42)
    image = rng.rayleigh(scale=10, size=(50, 50)) + 50
    bg_area = rng.rayleigh(scale=100, size=(30, 30))  # very high sigma respect image
    
    with pytest.warns(RuntimeWarning, 
                      match="Obtained at least one negative value"):
        result = subtract_noise(image, bg_area)
    
    # With high bg std, A_squared might become negative, but must be handled correctly
    assert np.all(result >= 0)


def test_higher_bg_noise_gives_different_result():
    """Test that different background noise levels give different results
    Given: An image and two backgrounds with different noise levels
    When: The subtract_noise function is called with each background
    Then: The results should be different"""
    rng = np.random.default_rng(seed=42)
    image = rng.rayleigh(scale=10, size=(100, 100)) + 50
    
    bg_low = rng.rayleigh(scale=1, size=(50, 50))
    bg_high = rng.rayleigh(scale=20, size=(50, 50))
    
    result_low = subtract_noise(image, bg_low)
    result_high = subtract_noise(image, bg_high)
    
    # Results should be different
    assert not np.array_equal(result_low, result_high)
    # and in particular 
    np.testing.assert_array_less(result_high,result_low)


# Formula verification tests

def test_sigma_r_calculation():
    """
    Given: random rayleigh generator istantiated with a known sigma_r parameter
    When: sigma_r = std(bg_area) / 0.655 is manually computed
    Then: the result is close to the original sigma_r """
    rng = np.random.default_rng(seed=42)
    sigma_r = 10
    bg_area = rng.rayleigh(scale=sigma_r, size=(30, 30))
    
    expected_sigma_r = np.std(bg_area) / 0.655
    
    assert expected_sigma_r >= 0
    # Verify that the result is consistent with the Rayleigh statics
    assert np.isclose(expected_sigma_r,sigma_r,rtol=0.1) 

def test_sigma_r_estimation():
    """
    Given: random rayleigh generator istantiated with a known sigma_r parameter
    When: _estimate_rayleigh_noise is called
    Then: the result is close to the true sigma_r
    """
    rng = np.random.default_rng(seed=42)
    sigma_r = 10
    bg_area = rng.rayleigh(scale=sigma_r, size=(30, 30))
    
    estimated_sigma_r = mrnl._estimate_rayleigh_noise(bg_area)
    
    assert estimated_sigma_r >= 0
    # Verify that the result is similar to the true sigma_r
    assert np.isclose(estimated_sigma_r,sigma_r,rtol=0.1) 

def test_A_squared_formula_manual_calculation():
    """Test A_squared formula with manual calculation
    Given: The same uniform image and background affected by Rayleigh noise
    When: The subtract_noise function is called locally
    Then: The mean of resulting image should be close to the the mean of 
        expected image performing global manual calculation"""
    
    rng = np.random.default_rng(seed=42)
    image = rng.rayleigh(scale=10, size=(50, 50)) + 50
    bg_area = rng.rayleigh(scale=10, size=(30, 30))
    
    # Manual calculation global
    m_ave = np.full(np.shape(image), np.mean(image))
    m_sd = np.full(np.shape(image), np.std(image))
    m_sd_bg = np.full(np.shape(image), np.std(bg_area))
    sigma_r = np.divide(m_sd_bg, 0.655)
    
    expected_A_squared = np.square(m_ave) + np.square(m_sd) - 2*np.square(sigma_r)
    expected_A_squared[expected_A_squared < 0] = 0
    expected_A = np.sqrt(expected_A_squared)
    
    # f_size is the pixel-size of windowing used to perform same reduction locally
    result = subtract_noise(image, bg_area, f_size=10)
    
    assert np.isclose(np.mean(expected_A),np.mean(result),rtol=0.01)


def test_image_and_bg_identical():
    """Test when image and background are identical
    Given: Image and background are identical (same data)
    When: The subtract_noise function is called 
    Then: The output energy (mean squared intensity) should be reduced"""
    rng = np.random.default_rng(seed=42)
    data = rng.rayleigh(scale=50, size=(100, 100))
    
    with pytest.warns(RuntimeWarning, match="Obtained at least one negative"):     
        result = subtract_noise(data, data, f_size=10)
        
    energy_in = np.mean(data ** 2)
    energy_out = np.mean(result ** 2)

    assert energy_out < energy_in
    
    # One should expect result near zero but with the local version of the 
    # function is not possible even with filter size larger than image
    # assert np.isclose(np.mean(result),0,atol=0.5)


# Reproducibility tests

def test_same_inputs_give_same_outputs():
    """Test that identical inputs give identical outputs"""
    rng = np.random.default_rng(seed=42)
    image = rng.rayleigh(scale=50, size=(100, 100))
    bg_area = rng.rayleigh(scale=10, size=(50, 50))
    
    result1 = subtract_noise(image, bg_area)
    result2 = subtract_noise(image, bg_area)
    
    np.testing.assert_array_equal(result1, result2)


def test_same_seed_gives_reproducible_results():
    """Test that using the same seed for random noise generators gives reproducible results"""
    rng1 = np.random.default_rng(seed=123)
    image1 = rng1.rayleigh(scale=50, size=(100, 100))
    bg_area1 = rng1.rayleigh(scale=10, size=(50, 50))
    result1 = subtract_noise(image1, bg_area1)
    
    rng2 = np.random.default_rng(seed=123)
    image2 = rng2.rayleigh(scale=50, size=(100, 100))
    bg_area2 = rng2.rayleigh(scale=10, size=(50, 50))
    result2 = subtract_noise(image2, bg_area2)
    
    np.testing.assert_array_equal(result1, result2)

# Masked Array

def test_subtract_noise_masked_requires_maskedarray():
    """
    Given: a non-masked input image
    When: subtract_noise_masked is called
    Then: it raises TypeError
    """
    rng = np.random.default_rng(42)
    image = rng.rayleigh(scale=50, size=(100, 100))
    bg_area = rng.rayleigh(scale=10, size=(50, 50))
    
    with pytest.raises(TypeError, match="MaskedArray"):
        subtract_noise_masked(image, bg_area, f_size=5)


def test_masked_wrapper_preserves_mask():
    """
    Given: a masked input image with a defined mask
    When: subtract_noise_masked is applied
    Then: the output is a masked array and the mask is preserved exactly
    """
    rng = np.random.default_rng(42)
    img = rng.rayleigh(scale=10, size=(64, 64)) + 50
    rect = np.zeros_like(img, dtype=bool)
    rect[10:40, :] = True  # a rectangle is masked

    img_ma = np.ma.array(img, mask=rect)
    bg = rng.rayleigh(scale=10, size=(20, 20))

    output = subtract_noise_masked(img_ma, bg, f_size=5)

    assert np.ma.isMaskedArray(output)
    np.testing.assert_array_equal(output.mask, rect)
    

def test_masked_wrapper_masked_pixels_ignored_in_checks():
    """
    Given: an image containing negative values only in masked pixels
    When: subtract_noise_masked is applied
    Then: no ValueError is raised and masked pixels remain masked
    """
    rng = np.random.default_rng(0)
    img = rng.rayleigh(scale=10, size=(64, 64)) + 50
    img[0, 0] = -1.0  # negative value but then is masked

    mask = np.zeros_like(img, dtype=bool)
    mask[0, 0] = True # mask hides the negative pixel
    img_ma = np.ma.array(img, mask=mask)

    bg = rng.rayleigh(scale=10, size=(20, 20)) #it is not masked

    out = subtract_noise_masked(img_ma, bg, fill_value=50, f_size=3)

    # no ValueError is raised by checks
    # and pixel (0,0) is still masked
    assert out.mask[0, 0] is np.True_


def test_masked_bg_empty_raises():
    """
    Given: a background array where all pixels are masked
    When: subtract_noise_masked is applied
    Then: a ValueError is raised indicating no valid background pixels
    """
    img = np.ma.array(np.ones((16, 16)), mask=False) # no mask
    bg = np.ma.array(np.ones((10, 10)), mask=True)  # all masked

    with pytest.raises(ValueError, match=r"Background has no valid"):
        subtract_noise_masked(img, bg)
        
        
def test_masked_wrapper_equivalent_when_no_mask():
    """
    Given: an image with a mask entirely set to False
    When: subtract_noise and subtract_noise_masked are applied
    Then: the numerical results are identical
    """
    rng = np.random.default_rng(42)
    img = rng.rayleigh(scale=10, size=(64, 64)) + 50
    bg = rng.rayleigh(scale=10, size=(20, 20))

    img_ma = np.ma.array(img, mask=np.zeros_like(img, dtype=bool))

    out_plain = subtract_noise(img, bg, f_size=5)
    out_masked = subtract_noise_masked(img_ma, bg, f_size=5)

    np.testing.assert_allclose(out_masked.data, out_plain, rtol=0, atol=0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])