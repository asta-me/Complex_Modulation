# =============================================================================
# Holographic Reconstruction Simulation (Jesacher et al., 2008)
# =============================================================================
# Description:
# Numerical simulation of the "Near-perfect hologram reconstruction with a 
# spatial light modulator" paper. The script computes phase masks P1 and P2
# to modulate both amplitude and phase using a Gerchberg-Saxton approach.
# It includes zero-padding for anti-aliasing and exact physical scaling for 3D.
# =============================================================================

#%% CELL 1: Imports and utility functions
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def load_and_pad_image(image_path, target_shape=(512, 512), shift_left=False):
    """
    Loads an image, resizes it to half-width, normalizes it, 
    and places it on a black canvas (left or right).
    """
    img = Image.open(image_path).convert('L')
    img = img.resize((target_shape[1]//2, target_shape[0]))
    img_arr = np.array(img, dtype=float)
    img_arr /= np.max(img_arr)
    
    canvas = np.zeros(target_shape)
    if shift_left:
        canvas[:, :target_shape[1]//2] = img_arr
    else:
        canvas[:, target_shape[1]//2:] = img_arr
    return canvas

def physical_defocus_phase(shape, delta_z_meters, f_FT_meters=0.3, wl_meters=1064e-9, pitch_meters=8e-6):
    """
    Generates a quadratic phase to physically shift the reconstructed image.
    delta_z_meters: Actual physical shift of the image plane in meters.
    f_FT_meters: Focal length of the physical Fourier-transforming lens.
    wl_meters: Wavelength (default 1064 nm as in the paper).
    pitch_meters: SLM pixel pitch (default 8 um as in the paper).
    """
    N_y, N_x = shape
    # Create a centered grid of physical spatial coordinates (in meters) on the SLM
    y = (np.arange(N_y) - N_y // 2) * pitch_meters
    x = (np.arange(N_x) - N_x // 2) * pitch_meters
    X, Y = np.meshgrid(x, y)
    
    R2 = X**2 + Y**2
    
    # Phase formula for a physical axial shift delta_z in a Fourier setup
    # Phi = (pi * delta_z * r^2) / (lambda * f_FT^2)
    phase = (np.pi * delta_z_meters * R2) / (wl_meters * (f_FT_meters**2))
    return phase

def pad_fourier_field(field, pad_factor=2):
    """
    Adds zero-padding to the field in the Fourier domain.
    Prevents aliasing artifacts when calculating intensity (which has double bandwidth).
    """
    h, w = field.shape
    new_h, new_w = int(h * pad_factor), int(w * pad_factor)
    padded = np.zeros((new_h, new_w), dtype=complex)
    
    # Insert the original field (with centered frequencies) into the new padded matrix
    start_h, start_w = (new_h - h) // 2, (new_w - w) // 2
    padded[start_h:start_h+h, start_w:start_w+w] = field
    return padded

print("Cell 1 executed: Libraries and physical functions loaded.")

#%% CELL 2: 2D Simulation (Standard)
# Configuration
iterations_2d = 15
img_2d_path = 'logo_M.bmp' # Use your logo here

# 1. Target preparation
img_2d = Image.open(img_2d_path).convert('L').resize((512, 512))
target_amplitude = np.array(img_2d, dtype=float)
target_amplitude /= np.max(target_amplitude)
a_target = target_amplitude * np.exp(1j * 0)

# 2. Target in the Fourier plane
A_target = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(a_target)))
target_A_amp = np.abs(A_target)
Phi = np.angle(A_target)

# 3. Gerchberg-Saxton for P1
P1 = np.random.rand(*target_amplitude.shape) * 2 * np.pi
print(f"Running {iterations_2d} GS iterations for P1 (2D)...")
for i in range(iterations_2d):
    E1 = np.exp(1j * P1)
    E_F = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(E1)))
    E_F_mod = target_A_amp * np.exp(1j * np.angle(E_F))
    E1_new = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(E_F_mod)))
    P1 = np.angle(E1_new)

# 4. P2 Calculation
E_F_sim = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(np.exp(1j * P1))))
Theta = np.angle(E_F_sim)
P2 = np.mod(Phi - Theta, 2 * np.pi)

# 5. Final reconstruction with ZERO-PADDING
E_after_P2 = np.abs(E_F_sim) * np.exp(1j * (Theta + P2))

# Apply padding before inverse-transforming to ensure adequate spatial grid
E_after_P2_padded = pad_fourier_field(E_after_P2, pad_factor=2)
a_sim_padded = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(E_after_P2_padded)))
rec_intensity_2d = np.abs(a_sim_padded)**2

print("Cell 2 executed: 2D Hologram calculated (anti-aliased).")

#%% CELL 3: 2D Visualization
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].imshow(target_amplitude, cmap='gray'); axes[0].set_title("Target |a(x,y)|")
axes[1].imshow(np.mod(P1, 2*np.pi), cmap='gray'); axes[1].set_title("P1 (Non Padded)")
axes[2].imshow(P2, cmap='gray'); axes[2].set_title("P2 (Non Padded)")
axes[3].imshow(rec_intensity_2d, cmap='gray'); axes[3].set_title("2D Reconstruction (Padded 2x)")
for ax in axes: ax.axis('off')
plt.tight_layout()
plt.show()

#%% CELL 4: 3D Simulation Setup with exact physical shift
img_left = load_and_pad_image('logo_m.bmp', shift_left=True)
img_right = load_and_pad_image('baboon_pad.bmp', shift_left=False)

# Let's shift the right image by exactly 1 meter physically (like the eagle in the paper)
physical_shift_meters = 1 
# Assuming the final Fourier lens has a focal length of 300 mm (0.3 m)
fourier_lens_focal = 0.3 

lens_phase = physical_defocus_phase(
    img_left.shape, 
    delta_z_meters=physical_shift_meters, 
    f_FT_meters=fourier_lens_focal
)

A_left = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img_left)))
A_right = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img_right)))

# Apply the exact physical defocus phase to the right image
A_target_3d = A_left + A_right * np.exp(1j * lens_phase)
target_A_amp_3d = np.abs(A_target_3d)
Phi_3d = np.angle(A_target_3d)

print(f"Cell 4 executed: 3D Target prepared. Right image shifted by {physical_shift_meters} m.")

#%% CELL 5: P1 and P2 Calculation for 3D
iterations_3d = 20
P1_3d = np.random.rand(*img_left.shape) * 2 * np.pi

print(f"Running {iterations_3d} GS iterations for P1 (3D)...")
for i in range(iterations_3d):
    E1 = np.exp(1j * P1_3d)
    E_F = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(E1)))
    E_F_mod = target_A_amp_3d * np.exp(1j * np.angle(E_F))
    E1_new = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(E_F_mod)))
    P1_3d = np.angle(E1_new)

E_F_sim_3d = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(np.exp(1j * P1_3d))))
Theta_3d = np.angle(E_F_sim_3d)
P2_3d = np.mod(Phi_3d - Theta_3d, 2 * np.pi)
E_after_P2_3d = np.abs(E_F_sim_3d) * np.exp(1j * (Theta_3d + P2_3d))

print("Cell 5 executed: 3D Hologram calculated.")

#%% CELL 6: 3D Reconstruction and Visualization with ZERO-PADDING

# Focus 1: Original plane (Z = 0)
E_after_P2_3d_padded = pad_fourier_field(E_after_P2_3d, pad_factor=2)
a_sim_focus1_padded = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(E_after_P2_3d_padded)))
rec_focus1_padded = np.abs(a_sim_focus1_padded)**2

# Focus 2: Compensate for the physical shift delta_z
E_shifted = E_after_P2_3d * np.exp(-1j * lens_phase)
E_shifted_padded = pad_fourier_field(E_shifted, pad_factor=2)
a_sim_focus2_padded = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(E_shifted_padded)))
rec_focus2_padded = np.abs(a_sim_focus2_padded)**2

fig, axes = plt.subplots(1, 2, figsize=(15, 7))
axes[0].imshow(rec_focus1_padded, cmap='gray')
axes[0].set_title("Focus Plane 1 (Z = 0 m)")
axes[0].axis('off')

axes[1].imshow(rec_focus2_padded, cmap='gray')
axes[1].set_title(f"Focus Plane 2 (Z = {physical_shift_meters} m)")
axes[1].axis('off')

plt.tight_layout()
plt.show()