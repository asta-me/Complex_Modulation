#%%
# =============================================================================
# BOLDUC — EXACT COMPLEX MODULATION WITH A PHASE-ONLY HOLOGRAM
# Physical-coordinate implementation (Python)
# Author: Marco Astarita
#
# Based on:
#   P. Bolduc et al., "Exact solution to simultaneous intensity and phase
#   encryption with a single phase-only hologram", Opt. Lett. 38, 3546 (2013).
#
# Description:
#   Exact complex-field encoding using a phase-only hologram via the Bolduc method.
#   The entire pipeline is implemented in PHYSICAL COORDINATES (meters),
#   consistently with the original MATLAB-style formulation.
#
#   NOTE ON GLOBAL PHASE:
#   The method inherently introduces a global phase shift of π (a factor of -1)
#   in the reconstructed field. This stems explicitly from Eq. (3) in the paper:
#       T1 = -sinc(πM - π) * exp(...)
#   The inverse sinc is computed in the domain [-π, 0], where the sinc function
#   yields positive values (Negative/Negative). Consequently, the leading minus
#   sign in Eq. (3) persists, resulting in:
#       E_reconstructed = -E_target (i.e., E_target * exp(iπ))
#   This global phase is physically irrelevant for intensity measurements but
#   is observable in exact phase reconstruction.
#
# Assumptions:
#   - Target amplitude a_target is normalized in [0, 1]
#   - Target phase phi_target is in radians
#   - All spatial coordinates are CENTERED
#
# Pipeline:
#   1) Build centered physical grid (x, y)
#   2) Generate LG(p,l) target field (amplitude + phase)
#   3) Bolduc exact encoding + physical blaze grating -> phase-only pattern psi
#   4) FFT of phase-only hologram
#   5) Circular filtering of +1 diffraction order
#   6) Shift +1 order to DC and reconstruct complex field
#   7) Visualization of amplitude, phase, and phase error
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt


#%%
# =============================================================================
# 0) PARAMETERS
# =============================================================================

# SLM grid size
n = 1080

# Physical pixel pitch [m]
pix_pitch = 8e-6

# Grating period (pixels)
lambda_pix = 10

# Physical grating period [m]
lambda_phys = pix_pitch * lambda_pix

# Circular mask radius in Fourier plane [pixels]
r_pix = 10

# Inverse-sinc LUT size
inv_sinc_lut_size = 100_000

# LG target parameters
lg_p = 0
lg_l = 2
w_out = 1.5e-3  # beam waist [m]

# Phase error visualization threshold
amp_thresh = 0.05

# Overlay transparency
overlay_alpha = 0.35


#%%
# =============================================================================
# 1) UTILITY FUNCTIONS
# =============================================================================

def fft2c(x):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))

def ifft2c(x):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x)))

def wrap_2pi(x):
    return np.mod(x, 2*np.pi)

def normalize_max(x):
    m = np.max(x)
    return x if m == 0 else x / m

def make_circular_mask(n, radius, center_yx):
    yy, xx = np.indices((n, n))
    cy, cx = center_yx
    return (xx - cx)**2 + (yy - cy)**2 <= radius**2


# --- inverse sinc -------------------------------------------------------------

def sinc_unorm(x):
    out = np.ones_like(x, dtype=np.float64)
    m = np.abs(x) > 1e-20
    out[m] = np.sin(x[m]) / x[m]
    return out

def inverse_sinc_unorm(a, lut_size=20000):
    """
    Inverts y = sin(x)/x on the monotonic branch x in [-pi, 0].
    Assumes a in [0,1].
    """
    x_lut = np.linspace(-np.pi, 0.0, int(lut_size))
    y_lut = sinc_unorm(x_lut)
    return np.interp(a.ravel(), y_lut, x_lut).reshape(a.shape)


#%%
# =============================================================================
# 2) PHYSICAL COORDINATE GRID (CENTERED)
# =============================================================================

x = np.linspace(-n/2, n/2, n, endpoint=False) * pix_pitch
y = np.linspace(-n/2, n/2, n, endpoint=False) * pix_pitch
x, y = np.meshgrid(x, y)


#%%
# =============================================================================
# 3) LAGUERRE–GAUSSIAN TARGET FIELD
# =============================================================================

def laguerre_poly(p, a, x):
    if p == 0:
        return np.ones_like(x)
    if p == 1:
        return 1 + a - x

    l0 = np.ones_like(x)
    l1 = 1 + a - x
    for k in range(2, p + 1):
        lk = ((2*k - 1 + a - x)*l1 - (k - 1 + a)*l0) / k
        l0, l1 = l1, lk
    return l1

def generate_lg_field(p, l, w0, x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    a = abs(l)
    rho2 = 2 * r**2 / w0**2
    L = laguerre_poly(p, a, rho2)

    amp = (np.sqrt(2)*r/w0)**a * L * np.exp(-r**2/w0**2)
    phase = l * theta
    return amp * np.exp(1j*phase)

# Target field
l_target = generate_lg_field(lg_p, lg_l, w_out, x, y)
a_target = normalize_max(np.abs(l_target))
phi_target = np.angle(l_target)
e_target = a_target * np.exp(1j*phi_target)


#%%
# =============================================================================
# 4) TARGET VISUALIZATION
# =============================================================================

plt.figure(); plt.title("Target amplitude")
plt.imshow(a_target, cmap="gray"); plt.colorbar(); plt.show()

plt.figure(); plt.title("Target phase")
plt.imshow(np.angle(np.exp(1j*phi_target)), cmap="twilight")
plt.colorbar(); plt.show()


#%%
# =============================================================================
# 5) BOLDUC EXACT ENCODING + PHYSICAL BLAZE
# =============================================================================

blaze = 2*np.pi * x / lambda_phys

x_inv = inverse_sinc_unorm(a_target, inv_sinc_lut_size)
m = 1 + x_inv/np.pi
f = phi_target - np.pi*m


psi = wrap_2pi(m * wrap_2pi(f + blaze))
t = np.exp(1j*psi)


#%%
# =============================================================================
# 5b) SLM PHASE PATTERN (psi) — grayscale view
# =============================================================================

psi_01 = psi / (2*np.pi)  # map [0,2pi) -> [0,1)

plt.figure(figsize=(6,5))
plt.title("SLM phase pattern psi (mapped to [0,1])")
plt.imshow(psi_01, cmap="gray", vmin=0, vmax=1)
plt.colorbar(label="phase / 2π")
plt.tight_layout()
plt.show()


#%%
# =============================================================================
# 6) FOURIER TRANSFORM OF PHASE-ONLY HOLOGRAM
# =============================================================================

u = fft2c(t)
u_log = np.log1p(np.abs(u))

k_shift = int(np.round(n / lambda_pix))
center = (n//2, n//2)
plus1 = (n//2, n//2 + k_shift)


plt.figure(figsize=(7,6))
plt.title("FFT of phase-only hologram")
plt.imshow(u_log, cmap="gray")
plt.scatter([center[1], plus1[1]], [center[0], plus1[0]],
            facecolors='none', edgecolors='r', s=80)
plt.colorbar()
plt.tight_layout()
plt.show()


#%%
# =============================================================================
# 7) FILTER +1 ORDER
# =============================================================================

mask = make_circular_mask(n, r_pix, plus1)

plt.figure(); plt.title("Fourier mask (+1 order)")
plt.imshow(mask, cmap="gray"); plt.colorbar(); plt.tight_layout(); plt.show()

plt.figure(figsize=(7,6))
plt.title("Mask overlay on FFT")
plt.imshow(u_log, cmap="gray")
plt.imshow(mask, alpha=overlay_alpha, cmap="Reds")
plt.tight_layout()
plt.show()


#%%
# =============================================================================
# 8) SHIFT TO DC AND RECONSTRUCT FIELD
# =============================================================================

u_filt = u * mask
u_dc = np.roll(u_filt, -k_shift, axis=1)

e_rec = ifft2c(u_dc)
a_rec = normalize_max(np.abs(e_rec))
phi_rec = np.angle(e_rec)


#%%
# =============================================================================
# 9) RESULTS
# =============================================================================

plt.figure(); plt.title("Reconstructed amplitude")
plt.imshow(a_rec, cmap="gray"); plt.colorbar(); plt.tight_layout(); plt.show()

plt.figure(); plt.title("Reconstructed phase")
plt.imshow(phi_rec, cmap="twilight"); plt.colorbar(); plt.tight_layout(); plt.show()

phase_err = np.angle(np.exp(1j*(phi_rec - phi_target)))
phase_err[a_target < amp_thresh] = 0

plt.figure(); plt.title("Phase error (masked)")
plt.imshow(phase_err, cmap="twilight"); plt.colorbar(); plt.tight_layout(); plt.show()
