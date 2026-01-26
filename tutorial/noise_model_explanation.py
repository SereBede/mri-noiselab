# -*- coding: utf-8 -*-
"""
Explantion — MRI magnitude noise model:
- Background: Rayleigh (no signal)
- Signal present: Rician (signal + complex Gaussian noise -> magnitude)
- Quadrature sum simulation: M = sqrt(A^2 + N^2) 
- Histograms + PDF overlays
- Goodness-of-fit tests (KS) for Rayleigh and Rician
- Link back to the correction used in mri_noiselab.subtract_noise()

Reference: R. Mark Henkelman, "Measument of signal intensities in the presence of 
noise in MR images", 1985.
https://en.wikipedia.org/wiki/Rayleigh_distribution
https://en.wikipedia.org/wiki/Rice_distribution

Requires: numpy, scipy, matplotlib, mri_noiselab

See also:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rayleigh.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rice.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstest.html

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import mri_noiselab as mrnl


def simulate_complex_gaussian_noise(shape, sigma, rng):
    """Generate complex Gaussian noise: n_r, n_i ~ N(0, sigma^2)"""
    nr = rng.normal(loc=0.0, scale=sigma, size=shape)
    ni = rng.normal(loc=0.0, scale=sigma, size=shape)
    return nr, ni


def rayleigh_background(sigma, n, rng):
    """
    Rayleigh magnitude for background (A=0):
        M = sqrt(n_r^2 + n_i^2)  -> Rayleigh(sigma)
    """
    nr, ni = simulate_complex_gaussian_noise((n,), sigma, rng)
    return np.sqrt(nr**2 + ni**2)


def magnitude_from_signal_and_noise(A, sigma, rng):
    """
    Rician magnitude given true signal A (considered Real):
        M = sqrt( (A + n_r)^2 + (n_i)^2 )
    where n_r, n_i ~ N(0, sigma^2) are gaussian distributed
    """
    nr, ni = simulate_complex_gaussian_noise(np.shape(A), sigma, rng)
    return np.sqrt((A + nr) ** 2 + (ni) ** 2)


def plot_hist_with_pdf(ax, data, dist, params, title, bins=80):
    """Plot magnitude probability distribution"""
    ax.hist(data, bins=bins, density=True)
    x = np.linspace(0, np.max(data) * 1.05, 500)
    ax.plot(x, dist.pdf(x, *params))
    ax.set_title(title)
    ax.set_xlabel("Magnitude")
    ax.set_ylabel("Density")


def snr_measure(x, selection):
    """Signal and noise power ratio"""
    data = x[selection]
    p_signal = float(np.mean(data**2))
    p_noise = float(np.var(data))
    return p_signal / p_noise if p_noise > 0 else np.inf


# initialize random generator
rng = np.random.default_rng(42)


# -----------------------------
# 1) Background: Rayleigh distributed
# -----------------------------
sigma_true = 20.0
bg = rayleigh_background(sigma_true, n=200_000, rng=rng) #n backgruond samples

# Fit Rayleigh with loc fixed to 0 (magnitude is non-negative)
# scipy.stats.rayleigh paramization: (loc, scale) where scale ~ sigma
ray_params = stats.rayleigh.fit(bg, floc=0)  # -> (loc, scale)

# Kolmogorov–Smirnov test against fitted Rayleigh

D_ray, p_ray = stats.kstest(bg, "rayleigh", args=ray_params)

# Mean/std ratio check
ratio = float(np.mean(bg) / np.std(bg))
expected_ratio = 1.91  #
print("=== Background Area(Rayleigh) ===")
print(f"True sigma: {sigma_true:.3f}")
print(f"Fitted sigma (scale): {ray_params[1]:.3f}")
print(f"mean: {np.mean(bg):.3f}, std:{ np.std(bg):.3f}")
print(f"mean/std ratio: {ratio:.3f} (expected ~ {expected_ratio})")
print(f"KS test: D={D_ray:.5f}, p_value={p_ray:.5g}\n")

fig, ax = plt.subplots(figsize=(7, 5))

plot_hist_with_pdf(
    ax, bg, stats.rayleigh, ray_params,
    title="Background magnitude distribution ~ Rayleigh"
)

plt.tight_layout()
plt.show()

# -----------------------------
# 2) Signal present: Rician distributed
# -----------------------------
A_true = 60.0
A = np.full((200_000,), A_true, dtype=np.float32) #200_000 signal samples
sig = magnitude_from_signal_and_noise(A, sigma_true, rng=rng)

# Fit Rice (Rician). scipy.stats.rice has shape parameter 'b' plus loc, scale.
# We again fix loc=0 because magnitude is non-negative and typically loc=0.
rice_params = stats.rice.fit(sig, floc=0)  # -> (b, loc, scale)

D_rice, p_rice = stats.kstest(sig, "rice", args=rice_params)

print("=== Signal Area(Rician) ===")
print(f"True A: {A_true:.3f}, true sigma: {sigma_true:.3f}")
print(f"Fitted rice params: b={rice_params[0]:.3f}, scale={rice_params[2]:.3f}")
print(f"KS test: D={D_rice:.5f}, p_value={p_rice:.5g}\n")

fig, ax = plt.subplots(figsize=(7, 5))

plot_hist_with_pdf(
    ax, sig, stats.rice, rice_params,
    title="Signal magnitude distribution ~ Rician (Rice)"
)

plt.tight_layout()
plt.show()

# -----------------------------
# 3) Apply cleaning formula with a small phantom demo
# -----------------------------
# We make use of the magnitude correction (Henkelman 1985):
# A^2 = m_ave^2 + m_var - 2*sigma_r^2
# where sigma_r is estimated from background (Rayleigh).
#
# Here we make a 2D phantom, add Rician noise in-signal, Rayleigh in background,
# then run mrnl.subtract_noise(image, bg_area).

phantom = np.zeros((128, 128), dtype=np.float32)
phantom[32:96, 32:96] = 60.0  # constant signal region

noisy = magnitude_from_signal_and_noise(phantom, sigma_true, rng=rng)

# choose a background ROI with (ideally) no signal
bg_area = noisy[0:24, 0:24]

cleaned = mrnl.subtract_noise(noisy, bg_area, f_size=10)

roi = np.zeros_like(phantom, dtype=bool)
roi[32:96, 32:96] = True

print("=== Phantom demo + mrnl.subtract_noise) ===")
print(f"SNR measure (noisy, ROI):   {snr_measure(noisy, roi):.3f}")
print(f"SNR measure (cleaned, ROI): {snr_measure(cleaned, roi):.3f}")

fig2, ax2 = plt.subplots(1, 3, figsize=(15, 5))
ax2[0].imshow(phantom, cmap="gray"); ax2[0].set_title("True A (phantom)"); ax2[0].axis("off")
ax2[1].imshow(noisy, cmap="gray"); ax2[1].set_title("Noisy magnitude"); ax2[1].axis("off")
ax2[2].imshow(cleaned, cmap="gray"); ax2[2].set_title("After subtract_noise"); ax2[2].axis("off")
plt.tight_layout()
plt.show()



