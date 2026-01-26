# DICOM workflow

What the script does:
- Load a DICOM image with `pydicom.dcmread(...)`
- Extract the pixel array and convert it to float for processing
- Select a **background ROI** (noise-only) for Rayleigh noise estimation
- Apply `mri_noiselab.subtract_noise(...)` using a local window (`f_size`)
- Clip the corrected values back to the valid DICOM intensity range
- Save a **new DICOM** with updated metadata (including a new `SOPInstanceUID`)
- Optionally plot original vs corrected images

## Requirements

- `pydicom`
- `numpy`, `matplotlib`
- your package (`mri_noiselab`)

If you haven't yet install them:

```bash
pip install pydicom,
```

## Background ROI advice

Choose a region with **no anatomical signal** (pure background noise), and avoid:
- any anatomy / signal spillover
- zero-padding / masked regions
- coil artifacts or structured background

A bad background ROI biases the Rayleigh σ estimate and can degrade the correction.

## Source code

```{literalinclude} ../../../howto/howto_clean_dicom.py
:language: python
:linenos:
```
