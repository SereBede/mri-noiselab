# DICOM workflow

This how-to shows an end-to-end DICOM pipeline using **pydicom** and `mri_noiselab.subtract_noise()`:

- Load a DICOM (`pydicom.dcmread`)
- Select a **background ROI** (noise-only, avoid zero padding)
- Apply noise subtraction with a local window (`f_size`)
- Convert back to a DICOM-compatible integer format
- Update metadata (new `SOPInstanceUID`) and save a new DICOM
- Visually compare original vs corrected images

The example DICOM used here is provided in the repository: 
[example.dcm](https://github.com/SereBede/mri-noiselab/tree/main/howto/howto_clean_dicom.py).

[Download code here](https://github.com/SereBede/mri-noiselab/tree/main/data/example.py)

## Load DICOM and convert pixel data to float

```{literalinclude} ../../../howto/howto_clean_dicom.py
:language: python
:linenos:
:lines: 11-32

```

## Define background ROI and apply noise subtraction

```{literalinclude} ../../../howto/howto_clean_dicom.py
:language: python
:linenos:
:lines: 34-50

```

### Background ROI advice

Choose a region with **no anatomical signal** (pure background noise), and avoid:
- any anatomy / signal spillover
- zero-padding / masked regions
- coil artifacts or structured background

A bad background ROI biases the Rayleigh σ estimate and can degrade the correction.

## Convert corrected image back to DICOM format and save

```{literalinclude} ../../../howto/howto_clean_dicom.py
:language: python
:linenos:
:lines: 53-80
```
Key points:

- np.clip(..., 0, 2**BitsStored - 1) ensures valid intensity range as defined in dicom metadata

- a new SOPInstanceUID is mandatory when pixel data changes to ensure that the processed image is treated as a distinct DICOM object and avoid data conflicts in viewers.


## Visual comparison 
```{literalinclude} ../../../howto/howto_clean_dicom.py
:language: python
:linenos:
:lines: 83-113
```
Background noise intensity is reduced.

![dicom result](../_static/images/dicom-clean.png)

## See also

- [PyDicom Documentation](https://pydicom.github.io/pydicom/stable/index.html)
- [Dicom Standard Browser](https://dicom.innolitics.com/ciods)

