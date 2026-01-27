# -*- coding: utf-8 -*-
"""
HOW TO:
Load a DICOM image, apply a background-based noise subtraction,
export the corrected image as a new DICOM file,
and visually compare original vs corrected images.

The dicom file used in this short guide is provided in the 
repository, inside the 'data' folder.
"""

# --- Import required libraries ---
import pydicom                     # DICOM I/O
import numpy as np                  # Numerical operations
import matplotlib.pyplot as plt     # Visualization
from mri_noiselab import subtract_noise  # Custom noise reduction function


# ============================================================
# 1. Load DICOM image
# ============================================================

# Read the DICOM file, containig metadata and pixel data
# (to open other dicom file replace with the actual file path)
ds = pydicom.dcmread("..\data\example.dcm")

# Convert pixel data to float for numerical processing
# (important to avoid integer overflow during calculations)
img = ds.pixel_array.astype(np.float32)

print(f"Loaded image shape: {img.shape}, datatype: {img.dtype}")


# ============================================================
# 2. Define background region for noise estimation
# ============================================================
bg_sample = (slice(0, 100), slice(80, 430))
# Select a background area (no anatomical signal expected)
# This region is used to estimate noise statistics
background = img[bg_sample]
# (N.B. pay attention to not include the zeros padding!)

# ============================================================
# 3. Apply noise subtraction
# ============================================================

# Subtract estimated noise from the image
# f_size controls the filter/kernel pixel size used in the algorithm
filter_s = 3
img_corr = subtract_noise(img, background, f_size=filter_s)


# ============================================================
# 4. Convert corrected image back to DICOM-compatible format
# ============================================================

# Clip values to the valid DICOM intensity range, no change in contrast
# (defined in metadata by the number of stored bits)
img_corr = np.clip(img_corr, 0, 2**ds.BitsStored - 1)

# Convert back to the original integer datatype
img_corr = img_corr.astype(ds.pixel_array.dtype)


# ============================================================
# 5. Update DICOM dataset and save corrected image
# ============================================================

# Replace pixel data
ds.PixelData = img_corr.tobytes()

# Update image dimensions (safety check)
ds.Rows, ds.Columns = img_corr.shape

# Generate a new SOP Instance UID
# (mandatory when modifying pixel data)
ds.SOPInstanceUID = pydicom.uid.generate_uid()

# Save corrected DICOM file
ds.save_as("../data/corrected_image.dcm")


# ============================================================
# 6. Visual comparison: original vs corrected image
# ============================================================

# Create side-by-side plots
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

images = [img, img_corr]
titles = [f"Original image (with noise) bg_mean = {np.mean(background):.2f} ",
          f"After noise reduction (f_size={filter_s}) bg_mean={np.mean(img_corr[bg_sample]):.2f} "]

# Shared intensity scale for fair comparison
vmin = 0
vmax = max(im.max() for im in images) * 1.2

# Display images
for ax, image, title in zip(axes, images, titles):
    im = ax.imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")

# Add a shared colorbar
fig.colorbar(
    im,
    ax=axes,
    orientation="vertical",
    fraction=0.046,
    pad=0.04,
    label="Pixel intensity"
)
plt.show()