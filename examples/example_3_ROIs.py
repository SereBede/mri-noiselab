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
import mri_noiselab as mrnl


def img_stats(image):
    """Stats over valid pixels; supports masked arrays."""
    if np.ma.isMaskedArray(image):
        data = image.compressed()
    else:
        data = np.ravel(image)

    ave = float(np.mean(data))
    sd = float(np.std(data))
    p_signal = float(np.mean(data * data))
    p_noise = float(np.var(data))
    snr = p_signal / p_noise if p_noise != 0 else 0.0
    return {"ave": ave, "sd": sd, "snr": snr}


# -----------------------------
# 1) Simulate 3-level signal + Rayleigh noise
# -----------------------------
true_signal_1 = 30
true_signal_2 = 50
true_signal_3 = 100
noise_level = 40
filter_size = 5

true_img = np.zeros((120, 120))
true_img[20:100, 20:100] = true_signal_1
true_img[35:85, 35:85] = true_signal_2
true_img[50:70, 50:70] = true_signal_3

rng = np.random.default_rng(seed=42)
bg_noise = rng.rayleigh(scale=noise_level/1.253, size=true_img.shape)
noisy_img = np.sqrt(np.square(true_img) + np.square(bg_noise))


# -----------------------------
# 2) Define DISJOINT region masks (one per signal level)
# -----------------------------
# Base ROIs (nested)
roi1 = np.zeros_like(noisy_img, dtype=bool)
roi2 = np.zeros_like(noisy_img, dtype=bool)
roi3 = np.zeros_like(noisy_img, dtype=bool)

roi1[20:100, 20:100] = True
roi2[35:85, 35:85] = True
roi3[50:70, 50:70] = True

# Make them disjoint:
# region3 = core = roi3
# region2 = middle ring (roi2 minus roi3)
roi2 = roi2 & (~roi3)
# region1 = outer ring (roi1 minus roi2 or 3)
roi1 = roi1 & (~(roi2 | roi3))

region1 = np.ma.array(noisy_img, mask=~roi1)
region2 = np.ma.array(noisy_img, mask=~roi2)
region3 = np.ma.array(noisy_img, mask=~roi3)


# -----------------------------
# 3) Build a masked background for noise estimation
# -----------------------------
# Background = outside any other ROI
roi_bg = ~(roi1 | roi2 | roi3) 
bg_ma = np.ma.array(noisy_img, mask=~roi_bg)  # mask everything that's NOT background


# -----------------------------
# 4) Clean one region at a time (using masked image input)
# -----------------------------

cleaned_r1 = mrnl.subtract_noise_masked(
                region1, bg_ma, fill_value=np.mean(region1), f_size=filter_size)
cleaned_r2 = mrnl.subtract_noise_masked(
                region2, bg_ma, fill_value=np.mean(region2), f_size=filter_size)
cleaned_r3 = mrnl.subtract_noise_masked(
                region3, bg_ma, fill_value=np.mean(region3), f_size=filter_size)
cleaned_bg = mrnl.subtract_noise_masked(
                bg_ma, bg_ma, fill_value=0.0, f_size=filter_size)

# -----------------------------
# 5) Merge cleaned regions into a final image
# -----------------------------
cleaned_full = noisy_img.copy()

# Replace only pixels that are NOT masked in each region result
cleaned_full[~cleaned_r1.mask] = cleaned_r1.data[~cleaned_r1.mask]
cleaned_full[~cleaned_r2.mask] = cleaned_r2.data[~cleaned_r2.mask]
cleaned_full[~cleaned_r3.mask] = cleaned_r3.data[~cleaned_r3.mask]
cleaned_full[~cleaned_bg.mask] = cleaned_bg.data[~cleaned_bg.mask]

# -----------------------------
# 6) Stats: before vs after per region
# -----------------------------
noisy_r1_stats = img_stats(region1)
noisy_r2_stats = img_stats(region2)
noisy_r3_stats = img_stats(region3)

bg_stats = img_stats(bg_ma)  # masked background stats

clean_r1_stats = img_stats(cleaned_full[roi1])
clean_r2_stats = img_stats(cleaned_full[roi2])
clean_r3_stats = img_stats(cleaned_full[roi3])
clean_bg_stats = img_stats(cleaned_full[roi_bg])


statistics = {"ave": "Region mean", "sd": "Region std", "snr": "Signal-to-noise ratio"}
for k, label in statistics.items():
    print(
        f"{label} | Before | After\n"
        f"Background | {bg_stats[k]:.2f} | {clean_bg_stats[k]:.2f}\n"
        f"Region 1    | {noisy_r1_stats[k]:.2f} | {clean_r1_stats[k]:.2f}\n"
        f"Region 2    | {noisy_r2_stats[k]:.2f} | {clean_r2_stats[k]:.2f}\n"
        f"Region 3    | {noisy_r3_stats[k]:.2f} | {clean_r3_stats[k]:.2f}\n"
    )


# -----------------------------
# 7) Plot: true, noisy, cleaned-by-region
# -----------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

images = [true_img, noisy_img, cleaned_full]
titles = ["True image (no noise)", "Noisy image", f"Cleaned (ROI-by-ROI), f_size={filter_size}"]

vmin = 0
vmax = max(im.max() for im in images) * 1.2

for ax, im, title in zip(axes, images, titles):
    m = ax.imshow(im, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")

fig.colorbar(m, ax=axes, orientation="vertical", fraction=0.046, pad=0.04, label="Intensity")
plt.show()
