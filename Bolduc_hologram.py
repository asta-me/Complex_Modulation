
# =============================================================================
# BOLDUC_e_holo.py — Arbitrary Far-Field (BMP) to Phase-Only Hologram
#
# Author: Marco Astarita
#
# This script implements the Bolduc exact complex modulation method, but
# IN CONTRAST to Bolduc.py, the target far-field intensity is loaded from an
# arbitrary BMP image (e.g., logo, text, or experimental pattern), not generated
# analytically (e.g., Laguerre-Gaussian modes).
#
# Key difference with Bolduc.py:
#   - Bolduc.py: Target field is generated analytically (LG mode, etc.)
#   - Bolduc_e_holo.py: Target is a user-supplied BMP image (any shape)
#
# Workflow:
#   1) Load BMP as desired far-field INTENSITY I_des(xf, yf)
#   2) Build complex far-field spectrum U_des = sqrt(I_des) * exp(i*Phi_far)
#      (Phi_far can be set arbitrarily or to zero)
#   3) IFFT -> target complex SLM field E_target (amplitude + phase)
#   4) Bolduc exact encoding + physical blaze -> phase-only pattern psi
#   5) FFT( exp(i*psi) ), circularly filter +1 order, shift to DC
#   6) IFFT -> reconstructed SLM field E_rec
#   7) FAR-FIELD check: compare |FFT(E_rec)|^2 vs desired intensity
#
# Practical use:
#   - Enables encoding of arbitrary intensity patterns (from images) in the far field
#   - Useful for experimental demonstrations, logo projection, or custom shapes
#
# Notes:
#   - All coordinates and transforms are in physical units (meters)
#   - Image resizing is handled by PIL for robustness
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


#%%
# =============================================================================
# 0) PARAMETERS
# =============================================================================

# Grid size (should match your SLM working resolution or padded/cropped version)
n = 1080

# Physical pixel pitch [m]
pix_pitch = 8e-6

# Grating period (pixels)
lambda_pix = 5
lambda_phys = pix_pitch * lambda_pix

# Fourier mask radius around +1 order [pixels]
r_pix = 75

# Inverse-sinc LUT size
inv_sinc_lut_size = 100_000

# Phase error threshold (for SLM-field comparison)
amp_thresh = 0.05

# Overlay transparency
overlay_alpha = 0.35

# --- BMP far-field target ---
bmp_path = "baboon_pad.bmp"
bmp_path = "logo_M.bmp"

# Target size control (kept but not a focus right now)
m = np.int64(0.75 * 250)   # set your desired target size here (pixels)
resize_mode = "pad"        # "pad" or "crop" (kept for future)
interp = "bilinear"        # kept for future
keep_aspect = True         # kept for future

# How to build FAR-FIELD complex target U_des:
# "zero"      -> phase = 0
# "random"    -> random phase in [0,2pi)
# "quadratic" -> defocus-like quadratic phase
phase_mode = "quadratic"

# Quadratic phase strength (only if phase_mode="quadratic")
quad_strength = 2e3   # unitless here (pixel-frequency coords)


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
    m_ = np.max(x)
    return x if m_ == 0 else x / m_

def make_circular_mask(n, radius, center_yx):
    yy, xx = np.indices((n, n))
    cy, cx = center_yx
    return (xx - cx)**2 + (yy - cy)**2 <= radius**2

# --- inverse sinc ------------------------------------------------------------

def sinc_unorm(x):
    out = np.ones_like(x, dtype=np.float64)
    m_ = np.abs(x) > 1e-20
    out[m_] = np.sin(x[m_]) / x[m_]
    return out

def inverse_sinc_unorm(a, lut_size=20000):
    """
    Invert y = sin(x)/x on the monotonic branch x in [-pi, 0].
    Assumes a in [0,1].
    """
    x_lut = np.linspace(-np.pi, 0.0, int(lut_size))
    y_lut = sinc_unorm(x_lut)
    return np.interp(a.ravel(), y_lut, x_lut).reshape(a.shape)

# --- image utils -------------------------------------------------------------

def load_bmp_gray01(path):
    """Load image (BMP/PNG/...) as grayscale float64 in [0,1]."""
    im = Image.open(path).convert("L")  # grayscale
    arr = np.asarray(im, dtype=np.float64) / 255.0
    return np.clip(arr, 0.0, 1.0)

def center_crop_or_pad(img, n):
    """Center-crop or zero-pad a 2D array to (n,n)."""
    h, w = img.shape
    out = np.zeros((n, n), dtype=np.float64)

    y0 = max(0, (h - n)//2)
    x0 = max(0, (w - n)//2)
    y1 = y0 + min(h, n)
    x1 = x0 + min(w, n)
    cropped = img[y0:y1, x0:x1]

    yy0 = (n - cropped.shape[0])//2
    xx0 = (n - cropped.shape[1])//2
    out[yy0:yy0+cropped.shape[0], xx0:xx0+cropped.shape[1]] = cropped
    return out

def place_center(img_small, n, fill=0.0):
    """Place a smaller (or equal) image at the center of an (n,n) canvas."""
    out = np.full((n, n), fill, dtype=np.float64)
    h, w = img_small.shape
    if h > n or w > n:
        raise ValueError("img_small is larger than canvas. Reduce m or crop.")
    y0 = (n - h)//2
    x0 = (n - w)//2
    out[y0:y0+h, x0:x0+w] = img_small
    return out


#%%
# =============================================================================
# 2) PHYSICAL COORDINATE GRID (CENTERED) — for blaze only
# =============================================================================

x = np.linspace(-n/2, n/2, n, endpoint=False) * pix_pitch
y = np.linspace(-n/2, n/2, n, endpoint=False) * pix_pitch
X, Y = np.meshgrid(x, y)


#%%
# =============================================================================
# 3) LOAD BMP AS FAR-FIELD INTENSITY TARGET
# =============================================================================

img0 = load_bmp_gray01(bmp_path)

# Simple, robust choice for now:
# - bring image to n×n (center-crop/pad)
# - optionally shrink it to m×m and place at center (kept because you already use it)

img0_nn = center_crop_or_pad(img0, n)

# (not focus) quick "shrink-to-m and center" using PIL
if m is not None:
    im_pil = Image.fromarray((img0_nn * 255).astype(np.uint8))
    im_small = im_pil.resize((int(m), int(m)), resample=Image.BILINEAR)
    img_m = np.asarray(im_small, dtype=np.float64) / 255.0
    img = place_center(img_m, n, fill=0.0)
else:
    img = img0_nn

plt.figure(figsize=(5,4))
plt.title("Far-field desired intensity (from BMP)")
plt.imshow(img, cmap="gray")
plt.colorbar()
plt.tight_layout()
plt.show()

# Far-field amplitude from intensity image
amp_far = normalize_max(np.sqrt(img))

plt.figure(figsize=(5,4))
plt.title("Far-field amplitude = sqrt(I)")
plt.imshow(amp_far, cmap="gray")
plt.colorbar()
plt.tight_layout()
plt.show()


#%%
# =============================================================================
# 4) BUILD FAR-FIELD PHASE AND COMPLEX SPECTRUM U_des
# =============================================================================

yy, xx = np.indices((n, n))
kx = (xx - (n/2)) / n   # normalized frequency coords
ky = (yy - (n/2)) / n
rho2 = kx**2 + ky**2

if phase_mode == "zero":
    phi_far = np.zeros((n, n))
elif phase_mode == "random":
    phi_far = 2*np.pi * np.random.rand(n, n)
elif phase_mode == "quadratic":
    phi_far = quad_strength * rho2
else:
    raise ValueError("phase_mode must be 'zero', 'random', or 'quadratic'")

u_des = amp_far * np.exp(1j * phi_far)

plt.figure(figsize=(5,4))
plt.title(f"Far-field phase ({phase_mode})")
plt.imshow(np.angle(np.exp(1j*phi_far)), cmap="twilight")
plt.colorbar()
plt.tight_layout()
plt.show()


#%%
# =============================================================================
# 5) IFFT -> TARGET COMPLEX SLM FIELD E_target(x,y)
# =============================================================================

e_target = ifft2c(u_des)

a_target = normalize_max(np.abs(e_target))    # must be in [0,1]
phi_target = np.angle(e_target)

plt.figure(figsize=(5,4))
plt.title("Target SLM-field amplitude a_target (normalized)")
plt.imshow(a_target, cmap="gray")
plt.colorbar()
plt.tight_layout()
plt.show()

plt.figure(figsize=(5,4))
plt.title("Target SLM-field phase phi_target")
plt.imshow(phi_target, cmap="twilight")
plt.colorbar()
plt.tight_layout()
plt.show()


#%%
# =============================================================================
# 6) BOLDUC EXACT ENCODING + PHYSICAL BLAZE -> psi (phase-only SLM pattern)
# =============================================================================

blaze = 2*np.pi * X / lambda_phys

x_inv = inverse_sinc_unorm(a_target, inv_sinc_lut_size)
m_map = 1 + x_inv/np.pi
f_map = phi_target - np.pi*m_map

psi = wrap_2pi(m_map * wrap_2pi(f_map + blaze))
t = np.exp(1j*psi)

# SLM phase pattern in grayscale (0..1)
psi_01 = psi / (2*np.pi)

plt.figure(figsize=(6,5))
plt.title("SLM phase pattern psi (0..1 grayscale)")
plt.imshow(psi_01, cmap="gray", vmin=0, vmax=1)
plt.colorbar(label="psi / 2π")
plt.tight_layout()
plt.show()


#%%
# =============================================================================
# 7) FFT OF PHASE-ONLY HOLOGRAM + FILTER +1 ORDER
# =============================================================================

u = fft2c(t)
u_log = np.log1p(np.abs(u))

k_shift = int(np.round(n / lambda_pix))
center = (n//2, n//2)
plus1 = (n//2, n//2 + k_shift)

plt.figure(figsize=(7,6))
plt.title("FFT of phase-only hologram (log magnitude)")
plt.imshow(u_log, cmap="gray")
plt.scatter([center[1], plus1[1]], [center[0], plus1[0]],
            facecolors='none', edgecolors='r', s=80)
plt.colorbar()
plt.tight_layout()
plt.show()

mask = make_circular_mask(n, r_pix, plus1)

plt.figure(figsize=(7,6))
plt.title("Mask overlay on FFT")
plt.imshow(u_log, cmap="gray")
plt.imshow(mask, alpha=overlay_alpha, cmap="Reds")
plt.tight_layout()
plt.show()

u_filt = u * mask


#%%
# =============================================================================
# 8) SHIFT +1 ORDER TO DC -> RECONSTRUCT SLM FIELD
# =============================================================================

u_dc = np.roll(u_filt, -k_shift, axis=1)

plt.figure(figsize=(7,6))
plt.title("Fourier space after shift (+1 -> DC), log magnitude")
plt.imshow(np.log1p(np.abs(u_dc)), cmap="gray")
plt.colorbar()
plt.tight_layout()
plt.show()

e_rec = ifft2c(u_dc)

a_rec = normalize_max(np.abs(e_rec))
phi_rec = np.angle(e_rec)

plt.figure(figsize=(5,4))
plt.title("Reconstructed SLM-field amplitude")
plt.imshow(a_rec, cmap="gray")
plt.colorbar()
plt.tight_layout()
plt.show()

plt.figure(figsize=(5,4))
plt.title("Reconstructed SLM-field phase")
plt.imshow(phi_rec, cmap="twilight")
plt.colorbar()
plt.tight_layout()
plt.show()


#%%
# =============================================================================
# 9) PHASE ERROR ON SLM FIELD (OPTIONAL DIAGNOSTIC)
# =============================================================================

phase_err = np.angle(np.exp(1j*(phi_rec - phi_target)))
phase_err[a_target < amp_thresh] = 0

plt.figure(figsize=(5,4))
plt.title("SLM-field phase error (masked)")
plt.imshow(phase_err, cmap="twilight")
plt.colorbar()
plt.tight_layout()
plt.show()


#%%
# =============================================================================
# 10) FAR-FIELD CHECK: DOES FFT(reconstructed) MATCH THE BMP TARGET?
# =============================================================================

u_rec = fft2c(e_rec)
i_rec = normalize_max(np.abs(u_rec)**2)

plt.figure(figsize=(5,4))
plt.title("Far-field intensity from reconstructed field (normalized)")
plt.imshow(i_rec, cmap="gray")
plt.colorbar()
plt.tight_layout()
plt.show()

plt.figure(figsize=(5,4))
plt.title("Desired far-field intensity (BMP, normalized)")
plt.imshow(normalize_max(img), cmap="gray")
plt.colorbar()
plt.tight_layout()
plt.show()
