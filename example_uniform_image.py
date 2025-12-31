# -*- coding: utf-8 -*-
"""
In this example a uniform image affected by Rayleigh noise is simulated and 
then cleaning is performed by using subtract_noise function from mri_noiselab module. 
In the end images are shown to see differences.

"""

import mri_noiselab as mrnl
import numpy as np
import matplotlib.pyplot as plt

def img_stats (image):
    '''
    Creates a dictionary with average, standard deviation and signal to noise 
    ratio of a uniform region

    Parameters
    ----------
    image : 2D numpy array as image.

    Returns
    -------
    stats : python dictionary where 
        'ave': mean of the sample image,
        'sd': standard deviation of the sample image,
        'snr': Signal to noise ratio computed as ratio of signal and noise powers

    '''
    ave = np.mean(image)
    sd = np.std(image)
    
    P_signal = np.mean(np.square(image))
    P_noise = np.var(image)
    
    snr = P_signal / P_noise if P_noise != 0 else 0
    
    stats = {
        'ave': ave,
        'sd': sd,
        'snr': snr
    }
    
    return stats

# Create a uniform signal image affected by Rayleigh noise
# change those values to vary noise contribution
true_signal = 50
noise_level = 20
nonc = np.sqrt(true_signal**2 + noise_level**2)

# Generate a uniform signal image
true_img = np.full((100,100),true_signal,dtype=float) 

# Generate the background noise image
rng = np.random.default_rng(seed=42)
bg_noise = rng.rayleigh(scale=noise_level/1.253, size=np.shape(true_img))

bg_stats = img_stats(bg_noise)

# Background Noise is summed in quadrature to pure signal
noisy_img = np.sqrt(np.square(true_img) + np.square(bg_noise))

noisy_stats = img_stats(noisy_img)

# Then given bg_noise clean it in noisy_img

cleaned_img = mrnl.subtract_noise(noisy_img, bg_noise)
cleaned_stats = img_stats(cleaned_img)

# After cleaning the SNR should improve
print(f'True Signal = {true_signal:.2f}, Noise Level = {noise_level:.2f} \n',
     f'Mean intensity with Noise = {noisy_stats['ave']:.2f}, SNR = {noisy_stats['snr']:.2f},\n',
     f'Mean intensity after noise reduction = {cleaned_stats['ave']:.2f}, SNR = {cleaned_stats['snr']:.2f}')

# Plot the images to see the differences
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

images = [true_img, noisy_img, cleaned_img]
titles = ["Image without noise", "Image with noise", "After noise reduction"]

vmin = 0
vmax = max(img.max() for img in images) * 1.2

# Loop all images
for ax, img, title in zip(axes, images, titles):
    im = ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis('off')

# Shared colorbar
fig.colorbar(im, ax=(axes[0],axes[1],axes[2]), orientation='vertical', 
             fraction=0.046, pad=0.04, label='Intensity')

plt.show()