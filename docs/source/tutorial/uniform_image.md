# uniform syntetic image

## Uniform image (synthetic) — Rayleigh noise + correction

This example simulates a **uniform magnitude image** affected by background **Rayleigh noise** and then applies
`mri_noiselab.subtract_noise()` to reduce the noise.

Core steps in the script:
1. Build a constant “true” image (`true_signal`)
2. Sample Rayleigh noise for the background (`rng.rayleigh(...)`)
3. Combine signal and noise in quadrature (`sqrt(signal^2 + noise^2)`)
4. Apply `mrnl.subtract_noise(noisy_img, bg_noise)`
5. Compare images and print simple statistics


```{literalinclude} ..\..\..\tutorial\example_uniform_image.py
:language: python
:linenos:
```