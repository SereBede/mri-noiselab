import pytest
import numpy as np
from mri_noiselab import subtract_noise 


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
    
    with pytest.warns(RuntimeWarning):
        subtract_noise(image, bg_area)


def test_bg_area_different_shapes():
    """Test with bg_area of different sizes
    Given: An image and background areas of various different sizes
    When: The subtract_noise function is called with each background size
    Then: The result should always have the same shape as the input image"""
    rng = np.random.default_rng(seed=42)
    image = rng.rayleigh(scale=10, size=(100, 100)) + 50
    
    bg_shapes = [(20, 20), (50, 50), (30, 40), (10, 100), (3, 3, 3)]
    
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
    
    with pytest.raises(ValueError):
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
    
    result = subtract_noise(image, bg_area)
    
    assert result.shape == image.shape


# Positivity and no infinity requirement tests

def test_positivity_requirement():
    """Test that A2[A2 < 0] = 0 works correctly before square root is performed
    Given: An image with low intensity and background with high noise (A2 might be negative)
    When: The subtract_noise function is called
    Then: The result should not contain any negative or NaN values"""
    # Create a case where A2 might be negative
    rng = np.random.default_rng(seed=42)
    image = rng.rayleigh(scale=10, size=(100, 100)) - 10 # low signal compared to
    bg_area = rng.rayleigh(scale=10, size=(50, 50)) # background noise
    
    result = subtract_noise(image, bg_area)
    
    # no square root of negative number performed
    assert not np.any(np.isnan(result))    
    assert np.all(result >= 0)
    

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


# Edge cases with uniform or zeros images or background

def test_zero_background():
    """Test with background containing only zeros, no noise is present
    Given: An image with noise and a background area containing only zeros
    When: The subtract_noise function is called
    Then: ValueError is raised"""
    rng = np.random.default_rng(seed=42)
    image = rng.rayleigh(scale=50, size=(100, 100)) + 50
    bg_area = np.zeros((50, 50))
    
    # Since background is uniformly 0, std(bg_area) = 0, so there is no noise  
    with pytest.raises(ValueError):
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
    with pytest.raises(ValueError):
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
    with pytest.warns(RuntimeWarning):
        result = subtract_noise(image, bg_area)
            
    assert np.isclose((np.mean(result)**2),(img_level**2 - 2*(sigma_r**2))) 


def test_uniform_background():
    """Test with uniform bakground (all pixels equal, meaning bias is present)
    Given: A uniform constant noiseless but biased background 
    When: The subtract_noise function is called
    Then: ValueError is raised since noise is estimated to be 0"""
    rng = np.random.default_rng(seed=42)
    bg_level= 30
    image = rng.rayleigh(scale= 50, size=(100, 100))
    bg_area = np.full((50, 50), bg_level)
    
    with pytest.raises(ValueError):
        subtract_noise(image, bg_area)
        
        
def test_zero_mean_image():
    """Test with a non uniform image with 0 mean
    Given: A non-uniform with zero mean image and a noisy background
    When: The subtract_noise function is called
    Then: A runtime warning is raised since mean=0 and the square(mean result) 
        should be close to square(m_ave) - 2*square(sigma_r) """
    rng = np.random.default_rng(seed=42)
    img_level = 50
    image = np.tile([[-1,0,1],[0,1,-1],[1,-1,0]],(9,9))*img_level
    bg_area = rng.rayleigh(scale=10, size=(50, 50))
    
    img_sd =np.std(image)
    sigma_r = np.std(bg_area)/0.655
        
    # With uniform image, std(image) = 0
    with pytest.warns(RuntimeWarning):
        result = subtract_noise(image, bg_area)
            
    assert np.isclose((np.mean(result)**2),(img_sd**2 - 2*(sigma_r**2)))       
        

# Noise level tests
   
def test_very_high_background():
    """Test with very high background standard deviation
    Given: An image with low noise and a background with very high standard deviation
    When: The subtract_noise function is called
    Then: The positivity constraint should handle negative A2 values correctly"""
    rng = np.random.default_rng(seed=42)
    image = rng.rayleigh(scale=10, size=(50, 50)) + 50
    bg_area = rng.rayleigh(scale=100, size=(30, 30))  # very high sigma respect image
    
    result = subtract_noise(image, bg_area)
    
    # With high bg std, A2 might become negative, but must be handled correctly
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
    """Test that sigma_r = std(bg_area) / 0.655 from random rayleigh generator"""
    rng = np.random.default_rng(seed=42)
    sigma_r = 10
    bg_area = rng.rayleigh(scale=sigma_r, size=(30, 30))
    
    expected_sigma_r = np.std(bg_area) / 0.655
    
    assert expected_sigma_r >= 0
    # Verify that the result is consistent with the Rayleigh statics
    assert np.isclose(expected_sigma_r,sigma_r,rtol=0.1) 


def test_A2_formula_manual_calculation():
    """Test A2 formula with manual calculation"""
    rng = np.random.default_rng(seed=42)
    image = rng.rayleigh(scale=10, size=(50, 50)) + 50
    bg_area = rng.rayleigh(scale=10, size=(30, 30))
    
    # Manual calculation
    m_ave = np.full(np.shape(image), np.mean(image))
    m_sd = np.full(np.shape(image), np.std(image))
    m_sd_bg = np.full(np.shape(image), np.std(bg_area))
    sigma_r = np.divide(m_sd_bg, 0.655)
    
    expected_A2 = np.square(m_ave) + np.square(m_sd) - 2*np.square(sigma_r)
    expected_A2[expected_A2 < 0] = 0
    expected_A = np.sqrt(expected_A2)
    
    result = subtract_noise(image, bg_area)
    
    np.testing.assert_array_almost_equal(result, expected_A)


def test_image_and_bg_identical():
    """Test when image and background are identical
    Given: Image and background are identical (same data)
    When: The subtract_noise function is called
    Then: The result should be zero """
    rng = np.random.default_rng(seed=42)
    data = rng.rayleigh(scale=50, size=(100, 100))
    
    result = subtract_noise(data, data)
    
    np.testing.assert_allclose(result,np.zeros_like(data))


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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])