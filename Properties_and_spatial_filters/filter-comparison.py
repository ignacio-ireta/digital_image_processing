import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import os
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def load_image(file_path: str) -> np.ndarray:
    """Load an image and convert to grayscale if needed."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Image file not found: {file_path}")
    
    img = cv2.imread(file_path)
    if img is None:
        raise ValueError(f"Could not read image: {file_path}")
    
    # Convert to grayscale if image has 3 channels
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def analyze_image_statistics(img: np.ndarray) -> Dict:
    """Calculate various statistics for an image."""
    return {
        'min': np.min(img),
        'max': np.max(img),
        'mean': np.mean(img),
        'std': np.std(img),
        'median': np.median(img),
        'histogram': np.histogram(img, bins=256, range=(0, 256))[0]
    }

def normalize_image(img: np.ndarray) -> np.ndarray:
    """Normalize image contrast to use full range [0, 255]."""
    min_val, max_val = np.min(img), np.max(img)
    
    # Avoid division by zero
    if min_val == max_val:
        return np.zeros_like(img)
    
    # Apply normalization formula
    normalized = ((img - min_val) * 255 / (max_val - min_val)).astype(np.uint8)
    return normalized

def apply_filters(img: np.ndarray, kernel_sizes: List[int]) -> Dict:
    """Apply various filters with different kernel sizes to an image."""
    results = {}
    
    for k in kernel_sizes:
        # Mean filter
        mean_filtered = cv2.blur(img, (k, k))
        results[f'mean_{k}'] = mean_filtered
        
        # Median filter
        median_filtered = cv2.medianBlur(img, k)
        results[f'median_{k}'] = median_filtered
        
        # Max filter (dilate)
        max_filtered = cv2.dilate(img, np.ones((k, k), np.uint8))
        results[f'max_{k}'] = max_filtered
        
        # Min filter (erode)
        min_filtered = cv2.erode(img, np.ones((k, k), np.uint8))
        results[f'min_{k}'] = min_filtered
    
    return results

def compare_images(original: np.ndarray, processed: np.ndarray) -> Dict:
    """Compare original and processed images using various metrics."""
    # Ensure both images are of the same type
    original = original.astype(np.uint8)
    processed = processed.astype(np.uint8)
    
    # Calculate SSIM (higher is better, max is 1)
    ssim_value = ssim(original, processed)
    
    # Calculate PSNR (higher is better)
    psnr_value = psnr(original, processed)
    
    # Calculate MSE (lower is better)
    mse = np.mean((original - processed) ** 2)
    
    # Calculate absolute difference
    abs_diff = np.abs(original - processed)
    mean_abs_diff = np.mean(abs_diff)
    
    return {
        'ssim': ssim_value,
        'psnr': psnr_value,
        'mse': mse,
        'mean_abs_diff': mean_abs_diff,
        'abs_diff_image': abs_diff
    }

def plot_histograms(images_dict: Dict[str, np.ndarray], figsize=(12, 8)):
    """Plot histograms for multiple images."""
    fig, axes = plt.subplots(len(images_dict), 1, figsize=figsize)
    
    if len(images_dict) == 1:
        axes = [axes]
    
    for (name, img), ax in zip(images_dict.items(), axes):
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        ax.plot(hist)
        ax.set_xlim([0, 256])
        ax.set_title(f'Histogram for {name}')
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Frequency')
    
    plt.tight_layout()
    return fig

def analyze_filters_effect(image_path: str, kernel_sizes: List[int] = [3, 5, 7]):
    """Analyze and visualize the effects of different filters on an image."""
    # Load and normalize image
    original = load_image(image_path)
    normalized = normalize_image(original)
    
    # Get filename for output
    filename = os.path.splitext(os.path.basename(image_path))[0]
    
    # Get image statistics
    original_stats = analyze_image_statistics(original)
    normalized_stats = analyze_image_statistics(normalized)
    
    print(f"\nAnalysis for {filename}:")
    print(f"  Original image - Min: {original_stats['min']}, Max: {original_stats['max']}, Mean: {original_stats['mean']:.2f}")
    print(f"  Normalized image - Min: {normalized_stats['min']}, Max: {normalized_stats['max']}, Mean: {normalized_stats['mean']:.2f}")
    
    # Apply filters with different kernel sizes
    filtered_images = apply_filters(normalized, kernel_sizes)
    
    # Compare results with normalized image
    comparisons = {}
    
    for name, filtered in filtered_images.items():
        comparisons[name] = compare_images(normalized, filtered)
        
        # Print comparison metrics
        filter_type, k = name.split('_')
        print(f"  {filter_type.capitalize()} filter (k={k}):")
        print(f"    SSIM: {comparisons[name]['ssim']:.4f}")
        print(f"    PSNR: {comparisons[name]['psnr']:.2f} dB")
        print(f"    Mean Absolute Difference: {comparisons[name]['mean_abs_diff']:.2f}")
    
    # Create visualization of original, normalized, and filtered images
    plt.figure(figsize=(15, 10))
    
    # Original and normalized images
    plt.subplot(2, 4, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(2, 4, 2)
    plt.imshow(normalized, cmap='gray')
    plt.title('Normalized')
    plt.axis('off')
    
    # Display one filtered image per filter type (using middle kernel size)
    middle_k = kernel_sizes[len(kernel_sizes)//2]
    
    filter_types = ['mean', 'median', 'max', 'min']
    for i, filter_type in enumerate(filter_types):
        plt.subplot(2, 4, i+3)
        img = filtered_images[f'{filter_type}_{middle_k}']
        plt.imshow(img, cmap='gray')
        plt.title(f'{filter_type.capitalize()} (k={middle_k})')
        plt.axis('off')
    
    # Display difference images for each filter type
    for i, filter_type in enumerate(filter_types):
        plt.subplot(2, 4, i+5)
        diff_img = comparisons[f'{filter_type}_{middle_k}']['abs_diff_image']
        plt.imshow(diff_img, cmap='hot')
        plt.title(f'{filter_type.capitalize()} Difference')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{filename}_filter_effects.png')
    
    # Plot histograms
    hist_images = {
        'Original': original,
        'Normalized': normalized
    }
    
    # Add one example of each filter type
    for filter_type in filter_types:
        hist_images[f'{filter_type.capitalize()} (k={middle_k})'] = filtered_images[f'{filter_type}_{middle_k}']
    
    fig_hist = plot_histograms(hist_images, figsize=(10, 12))
    fig_hist.savefig(f'{filename}_histograms.png')
    
    # Compare kernel sizes for one filter type (e.g., median)
    plt.figure(figsize=(15, 5))
    plt.suptitle(f'Effect of Kernel Size on Median Filter for {filename}', fontsize=16)
    
    for i, k in enumerate(kernel_sizes):
        plt.subplot(1, len(kernel_sizes), i+1)
        img = filtered_images[f'median_{k}']
        plt.imshow(img, cmap='gray')
        plt.title(f'k={k}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(f'{filename}_kernel_size_comparison.png')
    
    print(f"Analysis complete for {filename}. Output images saved.")
    return {
        'original': original,
        'normalized': normalized,
        'filtered': filtered_images,
        'comparisons': comparisons,
        'stats': {
            'original': original_stats,
            'normalized': normalized_stats
        }
    }

def analyze_all_images(image_paths: List[str], kernel_sizes: List[int] = [3, 5, 7]):
    """Analyze all images and generate a summary comparison."""
    results = {}
    
    for path in image_paths:
        try:
            filename = os.path.splitext(os.path.basename(path))[0]
            results[filename] = analyze_filters_effect(path, kernel_sizes)
        except Exception as e:
            print(f"Error processing {path}: {str(e)}")
    
    # Create a summary comparison across images
    plt.figure(figsize=(15, 12))
    plt.suptitle('Filter Effects Comparison Across Images', fontsize=16)
    
    row = 0
    for i, (filename, result) in enumerate(results.items()):
        # Show original and normalized for each image
        plt.subplot(len(results), 4, row*4 + 1)
        plt.imshow(result['normalized'], cmap='gray')
        plt.title(f'{filename} (Normalized)')
        plt.axis('off')
        
        # Pick a representative filter for each image
        middle_k = kernel_sizes[len(kernel_sizes)//2]
        filter_types = ['median', 'mean', 'max']
        
        for j, filter_type in enumerate(filter_types):
            plt.subplot(len(results), 4, row*4 + j + 2)
            plt.imshow(result['filtered'][f'{filter_type}_{middle_k}'], cmap='gray')
            plt.title(f'{filter_type.capitalize()} (k={middle_k})')
            plt.axis('off')
        
        row += 1
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig('all_images_comparison.png')
    
    # Print overall summary
    print("\nOverall Summary:")
    print("================")
    print(f"Analyzed {len(results)} images with kernel sizes {kernel_sizes}")
    
    # Summarize which filter worked best for each image (based on SSIM)
    print("\nBest filter per image (based on SSIM with normalized image):")
    for filename, result in results.items():
        best_filter = ""
        best_ssim = -1
        
        for name, comp in result['comparisons'].items():
            if comp['ssim'] > best_ssim:
                best_ssim = comp['ssim']
                best_filter = name
        
        filter_type, k = best_filter.split('_')
        print(f"  {filename}: {filter_type.capitalize()} filter with kernel size {k} (SSIM: {best_ssim:.4f})")
    
    return results

if __name__ == "__main__":
    # List of image paths
    image_paths = ["chest_xray.jpg", "cat.jpg", "forest.jpg", "papyrus.png"]
    
    # Kernel sizes to test
    kernel_sizes = [3, 5, 7]
    
    # Analyze all images
    results = analyze_all_images(image_paths, kernel_sizes)
    
    print("\nAnalysis complete. All output files have been saved.")