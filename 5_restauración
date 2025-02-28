import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def adaptive_noise_reduction(image, window_size=5, k=2.0):
    """
    Implements an adaptive local noise reduction filter.
    Uses local statistics (mean and variance) to adapt filtering strength.
    
    Parameters:
    - image: Input grayscale image
    - window_size: Size of the local window
    - k: Adaptive parameter controlling filter strength
    
    Returns:
    - Filtered image
    """
    # Create output image
    output = np.zeros_like(image, dtype=np.float64)
    
    # Pad the image to handle border pixels
    padded = np.pad(image, (window_size//2, window_size//2), mode='reflect')
    
    # Get global image statistics
    global_mean = np.mean(image)
    global_var = np.var(image)
    
    # If global variance is zero, return the image as is
    if global_var == 0:
        return image
        
    # Process each pixel
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Extract local window
            i_pad, j_pad = i + window_size//2, j + window_size//2
            window = padded[i_pad-window_size//2:i_pad+window_size//2+1, 
                           j_pad-window_size//2:j_pad+window_size//2+1]
            
            # Calculate local statistics
            local_mean = np.mean(window)
            local_var = np.var(window)
            
            # Apply adaptive filter
            if local_var > 0:
                # Filter response is stronger in flat areas (low variance)
                # and weaker in detailed areas (high variance)
                factor = local_var / (local_var + k * global_var)
                output[i, j] = local_mean + factor * (image[i, j] - local_mean)
            else:
                output[i, j] = local_mean
    
    return np.clip(output, 0, 255).astype(np.uint8)

def adaptive_median_filter(image, max_window_size=7):
    """
    Implements an adaptive median filter.
    Window size adapts based on local noise characteristics.
    
    Parameters:
    - image: Input grayscale image
    - max_window_size: Maximum window size for median filtering
    
    Returns:
    - Filtered image
    """
    # Create output image
    output = np.zeros_like(image)
    
    # Pad the image to handle border pixels
    padded = np.pad(image, (max_window_size//2, max_window_size//2), mode='reflect')
    
    # Process each pixel
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Get the center pixel coordinates in the padded image
            i_pad, j_pad = i + max_window_size//2, j + max_window_size//2
            
            # Start with a small window size
            window_size = 3
            
            # Level A: Try to find a non-impulse window
            while window_size <= max_window_size:
                # Extract window
                half_window = window_size // 2
                window = padded[i_pad-half_window:i_pad+half_window+1, 
                               j_pad-half_window:j_pad+half_window+1]
                
                # Calculate window statistics
                z_min = np.min(window)
                z_max = np.max(window)
                z_med = np.median(window)
                z_xy = padded[i_pad, j_pad]  # Center pixel value
                
                # Level A: Check if median is between min and max
                if z_min < z_med < z_max:
                    # Level B: Check if pixel is an impulse
                    if z_min < z_xy < z_max:
                        # Not an impulse, keep original
                        output[i, j] = z_xy
                    else:
                        # Impulse, replace with median
                        output[i, j] = z_med
                    break
                else:
                    # Increase window size and try again
                    window_size += 2
                    
                    # If we've reached max window size, use median
                    if window_size > max_window_size:
                        output[i, j] = z_med
    
    return output

def generate_motion_blur_operator(shape, a=0.1, b=0.1, T=1.0):
    """
    Generates a motion blur degradation operator H(u,v) in the frequency domain.
    
    Parameters:
    - shape: Shape of the image (M, N)
    - a, b: Motion blur parameters controlling direction
    - T: Motion blur parameter controlling strength
    
    Returns:
    - H: Motion blur operator in frequency domain
    """
    M, N = shape
    H = np.zeros((M, N), dtype=np.complex128)
    
    # Generate centered frequency coordinates
    u = np.arange(-M//2, M//2) if M % 2 == 0 else np.arange(-(M-1)//2, (M-1)//2 + 1)
    v = np.arange(-N//2, N//2) if N % 2 == 0 else np.arange(-(N-1)//2, (N-1)//2 + 1)
    
    # Create frequency coordinate grids
    u_grid, v_grid = np.meshgrid(u, v, indexing='ij')
    
    # Calculate frequency term
    freq_term = np.pi * (u_grid * a + v_grid * b)
    
    # Handle division by zero
    non_zero_mask = np.abs(freq_term) > 1e-10
    H[non_zero_mask] = T * np.sin(freq_term[non_zero_mask]) / freq_term[non_zero_mask] * np.exp(-1j * freq_term[non_zero_mask])
    H[~non_zero_mask] = T  # At (0,0) frequency, the value is T
    
    return H

def wiener_filter_restoration(degraded_noisy, H, K):
    """
    Restore an image using Wiener filtering.
    
    Parameters:
    - degraded_noisy: Degraded and noisy image
    - H: Degradation operator in frequency domain
    - K: Wiener filter adjustment parameter
    
    Returns:
    - Restored image
    """
    # Compute FFT of degraded image
    G = np.fft.fftshift(np.fft.fft2(degraded_noisy))
    
    # Apply Wiener filter
    H_conj = np.conjugate(H)
    H_abs_sq = np.abs(H) ** 2
    F = H_conj / (H_abs_sq + K) * G
    
    # Compute inverse FFT to get restored image
    f_restored = np.real(np.fft.ifft2(np.fft.ifftshift(F)))
    
    # Normalize and clip to valid range
    f_restored = np.clip(f_restored, 0, 1)
    
    return f_restored

def degrade_image(image, H, noise_variance=0.001):
    """
    Degrade an image using a motion blur operator and add Gaussian noise.
    
    Parameters:
    - image: Input image (normalized)
    - H: Degradation operator in frequency domain
    - noise_variance: Variance of Gaussian noise
    
    Returns:
    - Degraded and noisy image
    """
    # Compute FFT of the image
    F = np.fft.fftshift(np.fft.fft2(image))
    
    # Apply degradation operator
    G = F * H
    
    # Compute inverse FFT to get degraded image
    g = np.real(np.fft.ifft2(np.fft.ifftshift(G)))
    
    # Add Gaussian noise
    noise = np.random.normal(0, np.sqrt(noise_variance), image.shape)
    g_noisy = g + noise
    
    # Clip to valid range
    g_noisy = np.clip(g_noisy, 0, 1)
    
    return g_noisy

def plot_results(images, titles, figsize=(15, 10)):
    """
    Plot multiple images side by side for comparison.
    
    Parameters:
    - images: List of images to plot
    - titles: List of titles for each image
    - figsize: Figure size (width, height)
    """
    plt.figure(figsize=figsize)
    
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i+1)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    # Read the input image
    image = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found. Please provide a valid path.")
    
    # Normalize the image
    image_norm = image.astype(np.float32) / 255.0
    
    # Part A: Implement adaptive filters
    print("Applying adaptive noise reduction...")
    adaptive_filtered = adaptive_noise_reduction(image, window_size=5, k=2.0)
    cv2.imwrite('Lena-FiltroAdaptableReduccionRuido.png', adaptive_filtered)
    
    print("Applying adaptive median filter...")
    median_filtered = adaptive_median_filter(image, max_window_size=7)
    cv2.imwrite('Lena-FiltroAdaptableMediana.png', median_filtered)
    
    # Part B: Wiener filter restoration
    print("Generating motion blur operator...")
    H = generate_motion_blur_operator(image.shape, a=0.1, b=0.1, T=1.0)
    
    print("Degrading the image...")
    degraded_noisy = degrade_image(image_norm, H, noise_variance=0.001)
    
    # Apply Wiener filter with different K values
    K_values = [0.001, 0.01, 0.05, 0.1]
    restored_images = []
    
    for K in K_values:
        print(f"Restoring with K={K}...")
        restored = wiener_filter_restoration(degraded_noisy, H, K)
        restored_images.append(restored)
        
        # Save the restored image
        cv2.imwrite(f'Lena-Restored-K{K}.png', (restored * 255).astype(np.uint8))
    
    # Calculate quality metrics
    print("\nQuality Metrics:")
    for i, K in enumerate(K_values):
        curr_psnr = psnr(image_norm, restored_images[i])
        curr_ssim = ssim(image_norm, restored_images[i])
        print(f"K={K}: PSNR={curr_psnr:.2f}dB, SSIM={curr_ssim:.4f}")
    
    # Display results
    plot_results(
        [image_norm, degraded_noisy] + restored_images,
        ['Original', 'Degraded & Noisy'] + [f'Restored (K={K})' for K in K_values]
    )

if __name__ == "__main__":
    main()