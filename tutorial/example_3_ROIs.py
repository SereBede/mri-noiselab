# -*- coding: utf-8 -*-
"""
Example: clean one ROI at a time using masked arrays.

- Build a 3-level phantom + Rayleigh noise
- Define 3 disjoint region masks (outer ring, middle ring, core) and background
- Use subtract_noise_masked to compute correction, returning only the ROI
- Merge the three cleaned ROIs into a final cleaned image
"""

import numpy as np
import matplotlib.pyplot as plt
from mri_noiselab import subtract_noise_masked


def img_stats(image):
    """Stats over valid pixels; supports masked arrays."""
    if np.ma.isMaskedArray(image):
        data = image.compressed()
    else:
        data = np.ravel(image)

    ave = float(np.mean(data))
    sd = float(np.std(data))
    p_signal = float(np.mean(np.square(data)))
    p_noise = float(np.var(data))
    snr = p_signal / p_noise if p_noise != 0 else 0.0
    return {"ave": ave, "sd": sd, "snr": snr}


# -----------------------------
# 1) Simulate 3-level signal + Rayleigh noise
# -----------------------------

# Define nested slices of concentric squares setting true signal
nested_slices = {
    "Background" : ( (slice(None), slice(None)) ,  0.0),
    "Region 1": ( (slice(20, 100), slice(20, 100)),30.0), 
    "Region 2": ( (slice(35, 85),  slice(35, 85)), 50.0),
    "Region 3": ( (slice(50, 70),  slice(50, 70)), 100.0),
}

noise_level = 40
filter_size = 5

true_img = np.zeros((120, 120))
for name, ((rr, cc), true_sig) in nested_slices.items():
    true_img[rr, cc] = true_sig

rng = np.random.default_rng(seed=42)
bg_noise = rng.rayleigh(scale=noise_level/1.253, size=true_img.shape)
noisy_img = np.sqrt(np.square(true_img) + np.square(bg_noise))


# -----------------------------
# 2) Define DISJOINT region masks (one per signal level)
# -----------------------------
# Base ROIs (nested) create a dictionary 
roi = {}
for name, ((rr, cc), true_sig) in nested_slices.items():
    m = np.zeros_like(noisy_img, dtype=bool)
    m[rr, cc] = True
    roi[name] = m

# Make them disjoint:
# region3 = core = roi["Region 3"]
# region2 = middle ring (roi2 minus roi3)
roi["Region 2"] &= (~roi["Region 3"])
# region1 = outer ring (roi1 minus roi2 or 3)
roi["Region 1"] &= ~(roi["Region 2"] | roi["Region 3"])
# Background = outside any other ROI
roi["Background"] = ~(roi["Region 1"] | roi["Region 2"] | roi["Region 3"])

# Plot the resulting ROIs
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
order = [
    ("Background", (0, 0), "Background ROI"),
    ("Region 1",   (0, 1), "ROI Region 1 (outer ring)"),
    ("Region 2",   (1, 0), "ROI Region 2 (middle ring)"),
    ("Region 3",   (1, 1), "ROI Region 3 (core)"),
]

for name, (i, j), title in order:
    ax[i, j].imshow(roi[name], cmap="gray")
    ax[i, j].set_title(title)
    ax[i, j].axis("off")

plt.tight_layout()
plt.show()

# -----------------------------
# 3) Clean one region at a time (using masked image input)
# -----------------------------

# appling masks, do not forget to invert them to obtain ROIs
cleaned = noisy_img.copy()
bg_ma = np.ma.array(noisy_img, mask=~ roi["Background"])

for name, seg in roi.items(): # seg as "segmentation" == roi
    region_ma = np.ma.array(cleaned, mask=~seg) # (True mask => ignored pixel)
    fill = np.mean(region_ma) 
            
    region_ma = subtract_noise_masked(
        region_ma,
        bg_ma,       # noise estimate from background
        fill_value = fill,
        f_size=filter_size,
    )
    cleaned[seg] = region_ma.data[seg]

# -----------------------------
# 4) Stats: before vs after per region
# -----------------------------

statistics = {"ave": "Region MEAN", "sd": "Region STD", "snr": "Region SNR"}
for k, label in statistics.items():
    print(f"{label} | Before | After")
    for name, seg in roi.items():
        print(f"{name:10s} | {img_stats(noisy_img[seg])[k]:.2f} | {img_stats(cleaned[seg])[k]:.2f}")
    print()


# -----------------------------
# 5) Plot: true, noisy, cleaned-by-region
# -----------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

images = [true_img, noisy_img, cleaned]
titles = ["True image (no noise)", "Noisy image", f"Cleaned (ROI-by-ROI), f_size={filter_size}"]

vmin = 0
vmax = max(im.max() for im in images) * 1.2

for ax, im, title in zip(axes, images, titles):
    m = ax.imshow(im, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")

fig.colorbar(m, ax=axes, orientation="vertical", fraction=0.046, pad=0.04, label="Intensity")
plt.show()
