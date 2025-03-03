import numpy as np
import cv2
import matplotlib.pyplot as plt
from time import time
import os
from scipy import ndimage
import pandas as pd

def frequency_filter(image, params):
    """
    Apply frequency domain filtering to remove periodic noise.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image (grayscale)
    params : dict
        Dictionary containing filter parameters:
        - 'centers': List of (x, y) tuples for notch filter centers
        - 'radius': Radius of the notch filters
        - 'type': Type of filter ('notch', 'bandreject', etc.)
        - 'order': Filter order (for Butterworth filters)
    
    Returns:
    --------
    numpy.ndarray
        Filtered image
    """
    # Convert to grayscale if needed
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Convert image to float for processing
    img_float = np.float32(image)
    
    # Compute 2D FFT
    dft = np.fft.fft2(img_float)
    dft_shift = np.fft.fftshift(dft)
    
    # Generate the filter mask
    mask = create_notch_mask(dft_shift.shape, params)
    
    # Apply the filter
    filtered_dft = dft_shift * mask
    
    # Inverse FFT to get back to spatial domain
    img_back_shift = np.fft.ifftshift(filtered_dft)
    img_back = np.fft.ifft2(img_back_shift)
    
    # Get the real part and normalize
    img_filtered = np.abs(img_back)
    
    # Normalize to [0, 255] range
    img_filtered = cv2.normalize(img_filtered, None, 0, 255, cv2.NORM_MINMAX)
    
    return img_filtered.astype(np.uint8)

def create_notch_mask(shape, params):
    """
    Create a notch filter mask in the frequency domain.
    
    Parameters:
    -----------
    shape : tuple
        Shape of the frequency domain image (height, width)
    params : dict
        Dictionary containing filter parameters:
        - 'centers': List of (x, y) tuples for notch filter centers
        - 'radius': Radius of the notch filters
        - 'type': Type of filter ('notch', 'bandreject', etc.)
        - 'order': Filter order (for Butterworth filters)
    
    Returns:
    --------
    numpy.ndarray
        Filter mask (same size as the image)
    """
    rows, cols = shape
    mask = np.ones((rows, cols), np.float32)
    
    # Get parameters
    centers = params.get('centers', [])
    radius = params.get('radius', 10)
    filter_type = params.get('type', 'notch')
    
    # Create coordinate matrices
    y, x = np.ogrid[:rows, :cols]
    y_center, x_center = rows // 2, cols // 2
    
    # Create notch filters for each center
    for center_x, center_y in centers:
        # Convert center coordinates relative to the image center
        rel_x = center_x - x_center
        rel_y = center_y - y_center
        
        # Calculate distances from the notch center and its symmetric counterpart
        d1 = np.sqrt((x - x_center - rel_x)**2 + (y - y_center - rel_y)**2)
        d2 = np.sqrt((x - x_center + rel_x)**2 + (y - y_center + rel_y)**2)
        
        # Apply notch filter
        if filter_type == 'notch':
            # Butterworth notch filter
            n = params.get('order', 2)  # Filter order
            mask = mask * (1 / (1 + (radius / d1)**(2*n))) * (1 / (1 + (radius / d2)**(2*n)))
        else:
            # Simple binary notch
            mask[d1 <= radius] = 0
            mask[d2 <= radius] = 0
    
    return mask

def visualize_spectrum(dft_shift, title="Magnitude Spectrum (Log Scale)"):
    """
    Visualize the Fourier spectrum of an image.
    
    Parameters:
    -----------
    dft_shift : numpy.ndarray
        Shifted Fourier spectrum
    title : str
        Plot title
    """
    # Compute the magnitude spectrum (log scale for better visualization)
    magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title(title)
    plt.colorbar()
    plt.show()

def process_image(image_path, params, visualize=False):
    """
    Process a single image with frequency domain filtering.
    
    Parameters:
    -----------
    image_path : str
        Path to the input image
    params : dict
        Dictionary containing filter parameters
    visualize : bool
        Whether to visualize the Fourier spectrum and results
    
    Returns:
    --------
    tuple
        (Original image, Filtered image, Execution time, Filter mask)
    """
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Measure execution time
    start_time = time()
    
    # Compute FFT for visualization
    dft = np.fft.fft2(np.float32(img))
    dft_shift = np.fft.fftshift(dft)
    
    # Generate filter mask
    mask = create_notch_mask(dft_shift.shape, params)
    
    # Apply frequency domain filtering
    img_filtered = frequency_filter(img, params)
    
    # Record execution time
    execution_time = time() - start_time
    
    # Visualize results if requested
    if visualize:
        # Visualize the Fourier spectrum of the original image
        visualize_spectrum(dft_shift, "Original Magnitude Spectrum (Log Scale)")
        
        # Visualize the filter mask
        plt.figure(figsize=(10, 5))
        plt.imshow(mask, cmap='gray')
        plt.title('Filter Mask')
        plt.colorbar()
        plt.show()
        
        # Visualize the filtered spectrum
        filtered_spectrum = dft_shift * mask
        visualize_spectrum(filtered_spectrum, "Filtered Magnitude Spectrum (Log Scale)")
        
        # Show original and filtered images
        plt.figure(figsize=(12, 6))
        plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Original Image')
        plt.subplot(122), plt.imshow(img_filtered, cmap='gray'), plt.title('Filtered Image')
        plt.tight_layout()
        plt.show()
    
    return img, img_filtered, execution_time, mask

def batch_filter(image_paths, params_list, num_runs=5):
    """
    Process multiple images with multiple runs for statistical analysis.
    
    Parameters:
    -----------
    image_paths : list
        List of paths to input images
    params_list : list
        List of parameter dictionaries, one for each image
    num_runs : int
        Number of execution runs for each image
    
    Returns:
    --------
    dict
        Dictionary containing statistics for each image
    """
    results = {}
    
    for i, (image_path, params) in enumerate(zip(image_paths, params_list)):
        image_name = os.path.basename(image_path)
        print(f"Processing image {i+1}/{len(image_paths)}: {image_name}")
        
        # Store results for this image
        image_results = {
            'execution_times': [],
            'params': params
        }
        
        # Run multiple times for consistency
        for run in range(num_runs):
            print(f"  Run {run+1}/{num_runs}...")
            _, _, exec_time, _ = process_image(image_path, params, visualize=False)
            image_results['execution_times'].append(exec_time)
        
        # Calculate statistics
        times = np.array(image_results['execution_times'])
        image_results['mean_time'] = np.mean(times)
        image_results['median_time'] = np.median(times)
        image_results['std_time'] = np.std(times)
        image_results['min_time'] = np.min(times)
        image_results['max_time'] = np.max(times)
        
        # Store results for this image
        results[image_name] = image_results
    
    return results

def generate_report(results, output_file=None):
    """
    Generate a statistical report from the batch filtering results.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing statistics for each image
    output_file : str, optional
        Path to save the report, if None, only print to console
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the statistics
    """
    print("\n===== FREQUENCY FILTERING STATISTICAL REPORT =====\n")
    
    # Create DataFrame for the report
    report_data = []
    for image_name, stats in results.items():
        report_data.append({
            'Image': image_name,
            'Mean Time (s)': stats['mean_time'],
            'Median Time (s)': stats['median_time'],
            'Std Dev (s)': stats['std_time'],
            'Min Time (s)': stats['min_time'],
            'Max Time (s)': stats['max_time']
        })
    
    df = pd.DataFrame(report_data)
    
    # Print report
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n==================================================\n")
    
    # Save to file if requested
    if output_file is not None:
        df.to_csv(output_file, index=False, float_format="%.4f")
        print(f"Report saved to: {output_file}")
    
    return df

# --- Challenge Problem: Adaptive Noise Frequency Detection ---

def detect_noise_peaks(dft_shift, threshold_factor=0.9, min_distance=30):
    """
    Automatically detect peaks in the Fourier spectrum that might correspond to periodic noise.
    
    Parameters:
    -----------
    dft_shift : numpy.ndarray
        Shifted Fourier spectrum
    threshold_factor : float
        Factor to determine the threshold (relative to the maximum value)
    min_distance : int
        Minimum distance between peaks
    
    Returns:
    --------
    list
        List of (x, y) tuples representing detected noise peaks
    """
    # Compute magnitude spectrum
    magnitude_spectrum = np.abs(dft_shift)
    
    # Get the center coordinates
    rows, cols = magnitude_spectrum.shape
    center_y, center_x = rows // 2, cols // 2
    
    # Exclude the center region (DC component)
    center_mask = np.ones_like(magnitude_spectrum, dtype=bool)
    y, x = np.ogrid[:rows, :cols]
    center_mask[((y - center_y)**2 + (x - center_x)**2) < 100] = False
    
    # Apply mask to exclude the center
    masked_spectrum = magnitude_spectrum.copy()
    masked_spectrum[~center_mask] = 0
    
    # Find local maxima
    neighborhood_size = min_distance // 2
    local_max = ndimage.maximum_filter(masked_spectrum, size=neighborhood_size) == masked_spectrum
    
    # Get coordinates of peaks
    threshold = threshold_factor * np.max(masked_spectrum)
    peak_indices = np.where((local_max) & (masked_spectrum > threshold))
    
    # Convert to (x, y) coordinates
    peaks = [(peak_indices[1][i], peak_indices[0][i]) for i in range(len(peak_indices[0]))]
    
    # Filter out peaks too close to the image borders
    border_margin = 10
    peaks = [p for p in peaks if 
             border_margin <= p[0] < cols - border_margin and 
             border_margin <= p[1] < rows - border_margin]
    
    # Keep only a reasonable number of peaks (focus on the strongest ones)
    if len(peaks) > 20:
        peak_values = [magnitude_spectrum[y, x] for x, y in peaks]
        sorted_indices = np.argsort(peak_values)[::-1]  # Sort in descending order
        peaks = [peaks[i] for i in sorted_indices[:20]]
    
    # Find symmetric peaks (noise usually creates symmetric patterns in the FFT)
    symmetric_peaks = []
    for x, y in peaks:
        # Calculate symmetric position
        sym_x = 2 * center_x - x
        sym_y = 2 * center_y - y
        symmetric_peaks.append((sym_x, sym_y))
    
    # Combine and remove duplicates
    all_peaks = list(set(peaks + symmetric_peaks))
    
    # Sort peaks by distance from center
    all_peaks.sort(key=lambda p: (p[0] - center_x)**2 + (p[1] - center_y)**2)
    
    return all_peaks

def adaptive_frequency_filter(image, auto_params=None):
    """
    Apply adaptive frequency domain filtering to automatically remove periodic noise.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image (grayscale)
    auto_params : dict, optional
        Parameters for automatic noise detection:
        - 'threshold_factor': Factor for peak detection (default: 0.9)
        - 'min_distance': Minimum distance between peaks (default: 30)
        - 'radius': Radius for notch filters (default: 10)
        - 'order': Filter order for Butterworth filter (default: 2)
    
    Returns:
    --------
    tuple
        (Filtered image, Detected noise centers, Filter mask)
    """
    # Default parameters
    if auto_params is None:
        auto_params = {
            'threshold_factor': 0.9,
            'min_distance': 30,
            'radius': 10,
            'order': 2
        }
    
    # Convert to grayscale if needed
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Convert image to float for processing
    img_float = np.float32(image)
    
    # Compute 2D FFT
    dft = np.fft.fft2(img_float)
    dft_shift = np.fft.fftshift(dft)
    
    # Detect noise peaks
    detected_peaks = detect_noise_peaks(
        dft_shift, 
        threshold_factor=auto_params.get('threshold_factor', 0.9),
        min_distance=auto_params.get('min_distance', 30)
    )
    
    # Create filter parameters with detected peaks
    params = {
        'centers': detected_peaks,
        'radius': auto_params.get('radius', 10),
        'type': 'notch',
        'order': auto_params.get('order', 2)
    }
    
    # Generate the filter mask
    mask = create_notch_mask(dft_shift.shape, params)
    
    # Apply the filter
    filtered_dft = dft_shift * mask
    
    # Inverse FFT to get back to spatial domain
    img_back_shift = np.fft.ifftshift(filtered_dft)
    img_back = np.fft.ifft2(img_back_shift)
    
    # Get the real part and normalize
    img_filtered = np.abs(img_back)
    
    # Normalize to [0, 255] range
    img_filtered = cv2.normalize(img_filtered, None, 0, 255, cv2.NORM_MINMAX)
    
    return img_filtered.astype(np.uint8), detected_peaks, mask

def visualize_adaptive_results(image, filtered_image, dft_shift, detected_peaks, mask):
    """
    Visualize the results of adaptive filtering.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Original image
    filtered_image : numpy.ndarray
        Filtered image
    dft_shift : numpy.ndarray
        Shifted Fourier spectrum of the original image
    detected_peaks : list
        List of (x, y) tuples representing detected noise peaks
    mask : numpy.ndarray
        Filter mask used for filtering
    """
    # Compute magnitude spectrum
    magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1)
    
    # Create figure with subplots
    plt.figure(figsize=(15, 10))
    
    # Plot original image
    plt.subplot(231)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # Plot filtered image
    plt.subplot(232)
    plt.imshow(filtered_image, cmap='gray')
    plt.title('Filtered Image')
    plt.axis('off')
    
    # Plot magnitude spectrum
    plt.subplot(233)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum (Log Scale)')
    plt.colorbar()
    
    # Plot magnitude spectrum with detected peaks
    plt.subplot(234)
    plt.imshow(magnitude_spectrum, cmap='gray')
    rows, cols = magnitude_spectrum.shape
    center_y, center_x = rows // 2, cols // 2
    
    # Mark the detected peaks
    for x, y in detected_peaks:
        plt.plot(x, y, 'ro', markersize=5)
    
    plt.title(f'Detected Noise Peaks ({len(detected_peaks)} peaks)')
    
    # Plot the filter mask
    plt.subplot(235)
    plt.imshow(mask, cmap='gray')
    plt.title('Filter Mask')
    plt.colorbar()
    
    # Plot the filtered spectrum
    filtered_spectrum = magnitude_spectrum * mask
    plt.subplot(236)
    plt.imshow(filtered_spectrum, cmap='gray')
    plt.title('Filtered Spectrum')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()

def process_image_adaptive(image_path, auto_params=None, visualize=True):
    """
    Process a single image with adaptive frequency domain filtering.
    
    Parameters:
    -----------
    image_path : str
        Path to the input image
    auto_params : dict, optional
        Parameters for automatic noise detection
    visualize : bool
        Whether to visualize the results
    
    Returns:
    --------
    tuple
        (Original image, Filtered image, Execution time, Detected peaks)
    """
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Measure execution time
    start_time = time()
    
    # Compute FFT for visualization
    dft = np.fft.fft2(np.float32(img))
    dft_shift = np.fft.fftshift(dft)
    
    # Apply adaptive frequency domain filtering
    img_filtered, detected_peaks, mask = adaptive_frequency_filter(img, auto_params)
    
    # Record execution time
    execution_time = time() - start_time
    
    # Visualize results if requested
    if visualize:
        visualize_adaptive_results(img, img_filtered, dft_shift, detected_peaks, mask)
    
    return img, img_filtered, execution_time, detected_peaks

def compare_methods(image_path, manual_params, auto_params=None):
    """
    Compare manual and adaptive filtering methods on the same image.
    
    Parameters:
    -----------
    image_path : str
        Path to the input image
    manual_params : dict
        Parameters for manual filtering
    auto_params : dict, optional
        Parameters for automatic noise detection
    
    Returns:
    --------
    dict
        Dictionary containing comparison results
    """
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Manual filtering
    start_time_manual = time()
    img_manual, _, _, mask_manual = process_image(image_path, manual_params, visualize=False)
    exec_time_manual = time() - start_time_manual
    
    # Adaptive filtering
    start_time_adaptive = time()
    img_adaptive, detected_peaks, mask_adaptive = adaptive_frequency_filter(img, auto_params)
    exec_time_adaptive = time() - start_time_adaptive
    
    # Compute MSE between original and filtered images
    mse_manual = np.mean((img - img_manual) ** 2)
    mse_adaptive = np.mean((img - img_adaptive) ** 2)
    
    # Compute MSE between filtered images
    mse_between = np.mean((img_manual - img_adaptive) ** 2)
    
    # Visualize comparison
    plt.figure(figsize=(15, 10))
    
    plt.subplot(231)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(232)
    plt.imshow(img_manual, cmap='gray')
    plt.title(f'Manual Filtering\nTime: {exec_time_manual:.4f}s')
    plt.axis('off')
    
    plt.subplot(233)
    plt.imshow(img_adaptive, cmap='gray')
    plt.title(f'Adaptive Filtering\nTime: {exec_time_adaptive:.4f}s')
    plt.axis('off')
    
    plt.subplot(234)
    plt.imshow(mask_manual, cmap='gray')
    plt.title('Manual Filter Mask')
    plt.colorbar()
    
    plt.subplot(235)
    plt.imshow(mask_adaptive, cmap='gray')
    plt.title(f'Adaptive Filter Mask\n({len(detected_peaks)} peaks)')
    plt.colorbar()
    
    plt.subplot(236)
    diff = np.abs(img_manual - img_adaptive)
    plt.imshow(diff, cmap='hot')
    plt.title(f'Difference Between Filtered Images\nMSE: {mse_between:.2f}')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()
    
    # Return comparison results
    return {
        'exec_time_manual': exec_time_manual,
        'exec_time_adaptive': exec_time_adaptive,
        'mse_manual': mse_manual,
        'mse_adaptive': mse_adaptive,
        'mse_between': mse_between,
        'num_detected_peaks': len(detected_peaks)
    }

def main():
    """
    Main function to demonstrate frequency domain filtering.
    
    Change image paths and parameters as needed for your specific images.
    """
    # Example paths - replace with your actual image paths
    image_paths = [
        "noisy_image1.jpg",
        "noisy_image2.jpg",
        "noisy_image3.jpg",
        "noisy_image4.jpg",
        "noisy_image5.jpg"
    ]
    
    # Example parameters for each image - you'll need to tune these
    # based on your specific images
    params_list = [
        {
            'centers': [(100, 100), (356, 100)],  # Example coordinates
            'radius': 15,
            'type': 'notch',
            'order': 2
        },
        {
            'centers': [(120, 110), (336, 110)],
            'radius': 12,
            'type': 'notch',
            'order': 2
        },
        {
            'centers': [(130, 120), (326, 120)],
            'radius': 10,
            'type': 'notch',
            'order': 2
        },
        {
            'centers': [(140, 130), (316, 130)],
            'radius': 14,
            'type': 'notch',
            'order': 2
        },
        {
            'centers': [(150, 140), (306, 140)],
            'radius': 16,
            'type': 'notch',
            'order': 2
        }
    ]
    
    # 1. Process a single image (manual parameters)
    print("1. Processing a single image with manual parameters...")
    image_path = image_paths[0]  # Use the first image
    params = params_list[0]      # Use the first set of parameters
    
    _, _, _, _ = process_image(image_path, params, visualize=True)
    
    # 2. Run batch processing (multiple executions for statistics)
    print("\n2. Running batch processing for statistical analysis...")
    num_runs = 5  # Number of runs for statistics
    results = batch_filter(image_paths, params_list, num_runs)
    generate_report(results, "filtering_stats.csv")
    
    # 3. Process a single image with adaptive filtering
    print("\n3. Processing with adaptive filtering...")
    auto_params = {
        'threshold_factor': 0.85,
        'min_distance': 30,
        'radius': 15,
        'order': 2
    }
    
    _, _, _, _ = process_image_adaptive(image_paths[0], auto_params)
    
    # 4. Compare manual and adaptive methods
    print("\n4. Comparing manual and adaptive methods...")
    compare_results = compare_methods(image_paths[0], params_list[0], auto_params)
    
    print("\nComparison Results:")
    print(f"Manual filtering time: {compare_results['exec_time_manual']:.4f} seconds")
    print(f"Adaptive filtering time: {compare_results['exec_time_adaptive']:.4f} seconds")
    print(f"MSE (manual): {compare_results['mse_manual']:.2f}")
    print(f"MSE (adaptive): {compare_results['mse_adaptive']:.2f}")
    print(f"Number of detected peaks: {compare_results['num_detected_peaks']}")

if __name__ == "__main__":
    main()