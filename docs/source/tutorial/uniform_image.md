# Uniform signal region

This example simulates a **uniform magnitude image** affected by background **Rayleigh noise** and then applies
`mri_noiselab.subtract_noise()` to reduce the noise.

Core steps in the script:
1. Build a constant “true” image (`true_signal`)
2. Sample Rayleigh noise for the background (`rng.rayleigh(...)`)
3. Combine signal and noise in quadrature (`sqrt(signal^2 + noise^2)`)
4. Apply `mrnl.subtract_noise(noisy_img, bg_noise)`
5. Compare images and print simple statistics

[Download code here](https://github.com/SereBede/mri-noiselab/tree/main/tutorial/example_uniform_image.py)

Requires: numpy, matplotlib, mri-noiselab

## 1) Import modules and define useful function

```{literalinclude} ..\..\..\tutorial\example_uniform_image.py
:language: python
:linenos:
:lines: 9-44
```

## 2) Generate uniform image with noise and clean it

```{literalinclude} ..\..\..\tutorial\example_uniform_image.py
:language: python
:linenos:
:lines: 46-69
```

## 3) Show Results

```{literalinclude} ..\..\..\tutorial\example_uniform_image.py
:language: python
:linenos:
:lines: 71-95
```

After noise reduction the mean intensity approaches the true signal
and signal to noise ratio is significally improved.

```
True Signal = 50.00, Noise Level = 20.00 
Mean intensity with Noise = 54.64, SNR = 159.84,
Mean intensity after noise reduction = 49.97, SNR = 10298.18

```

![Uniform region noise reduction](../_static/images/uniform-clean.png)