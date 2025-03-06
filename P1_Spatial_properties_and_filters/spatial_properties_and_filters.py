import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 1. Set Up the Environment - Libraries are imported above

def load_and_convert_to_grayscale(image_path):
    """
    Load an image and convert it to grayscale.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        np.ndarray: Grayscale image
    """
    # Read the image using OpenCV
    image = cv2.imread(image_path)
    
    if image is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")
    
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    return gray_image

def get_pixel_range(image):
    """
    Determine the minimum and maximum pixel values in an image.
    
    Args:
        image (np.ndarray): Input image
        
    Returns:
        tuple: (min_value, max_value)
    """
    min_value = np.min(image)
    max_value = np.max(image)
    
    return min_value, max_value

def normalize_image(image):
    """
    Improve image contrast through normalization.
    
    Args:
        image (np.ndarray): Input grayscale image
        
    Returns:
        np.ndarray: Normalized image
    """
    min_val, max_val = get_pixel_range(image)
    
    # Apply normalization formula: (image - min) / (max - min) * 255
    if max_val > min_val:  # Avoid division by zero
        normalized = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    else:
        normalized = image.copy()  # If all pixels have the same value, no normalization needed
    
    return normalized

def apply_mean_filter(image, kernel_size=3):
    """
    Apply mean filtering to an image.
    
    Args:
        image (np.ndarray): Input image
        kernel_size (int, optional): Size of the kernel. Defaults to 3.
        
    Returns:
        np.ndarray: Mean filtered image
    """
    return cv2.blur(image, (kernel_size, kernel_size))

def apply_median_filter(image, kernel_size=3):
    """
    Apply median filtering to an image.
    
    Args:
        image (np.ndarray): Input image
        kernel_size (int, optional): Size of the kernel. Defaults to 3.
        
    Returns:
        np.ndarray: Median filtered image
    """
    return cv2.medianBlur(image, kernel_size)

def SubMatriz(img, center, k):
    """
    Extract a k×k submatrix from an image at the specified center coordinate.
    
    Args:
        img (np.ndarray): Input image
        center (tuple): Center coordinate (x, y) [column, row]
        k (int): Size of the submatrix (must be odd)
        
    Returns:
        np.ndarray: k×k submatrix
    """
    # Validate that k is odd
    if k % 2 == 0:
        raise ValueError("k must be an odd integer")
    
    # Get image dimensions
    height, width = img.shape
    
    # Unpack center coordinates (x: column, y: row)
    x, y = center
    
    # Calculate half of k
    half_k = k // 2
    
    # Calculate submatrix boundaries
    start_row = max(0, y - half_k)
    start_col = max(0, x - half_k)
    end_row = min(height, y + half_k + 1)
    end_col = min(width, x + half_k + 1)
    
    # Extract and return the submatrix
    return img[start_row:end_row, start_col:end_col]

def apply_maximum_filter(image, kernel_size=3):
    """
    Apply maximum filtering to an image using the SubMatriz function.
    
    Args:
        image (np.ndarray): Input image
        kernel_size (int, optional): Size of the kernel. Defaults to 3.
        
    Returns:
        np.ndarray: Maximum filtered image
    """
    # Create a copy of the image to store the result
    result = np.zeros_like(image)
    
    # Get image dimensions
    height, width = image.shape
    
    # Apply padding to handle border pixels
    pad_size = kernel_size // 2
    padded_image = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, 
                                       cv2.BORDER_REFLECT)
    
    # For each pixel in the image
    for y in range(height):
        for x in range(width):
            # Extract the submatrix from the padded image
            # Adjust coordinates to account for padding
            submatrix = SubMatriz(padded_image, (x + pad_size, y + pad_size), kernel_size)
            
            # Compute the maximum value in the submatrix
            max_value = np.max(submatrix)
            
            # Replace the central pixel with the maximum value
            result[y, x] = max_value
    
    return result

def apply_minimum_filter(image, kernel_size=3):
    """
    Apply minimum filtering to an image using the SubMatriz function.
    
    Args:
        image (np.ndarray): Input image
        kernel_size (int, optional): Size of the kernel. Defaults to 3.
        
    Returns:
        np.ndarray: Minimum filtered image
    """
    # Create a copy of the image to store the result
    result = np.zeros_like(image)
    
    # Get image dimensions
    height, width = image.shape
    
    # Apply padding to handle border pixels
    pad_size = kernel_size // 2
    padded_image = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, 
                                       cv2.BORDER_REFLECT)
    
    # For each pixel in the image
    for y in range(height):
        for x in range(width):
            # Extract the submatrix from the padded image
            # Adjust coordinates to account for padding
            submatrix = SubMatriz(padded_image, (x + pad_size, y + pad_size), kernel_size)
            
            # Compute the minimum value in the submatrix
            min_value = np.min(submatrix)
            
            # Replace the central pixel with the minimum value
            result[y, x] = min_value
    
    return result

def display_image_comparison(original, processed, title="Image Comparison"):
    """
    Display original and processed images side by side.
    
    Args:
        original (np.ndarray): Original image
        processed (np.ndarray): Processed image
        title (str, optional): Plot title. Defaults to "Image Comparison".
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    axes[1].imshow(processed, cmap='gray')
    axes[1].set_title("Processed")
    axes[1].axis('off')
    
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

def save_processed_image(image, output_path):
    """
    Save a processed image to disk.
    
    Args:
        image (np.ndarray): Image to save
        output_path (str): Path where the image will be saved
    """
    cv2.imwrite(output_path, image)
    print(f"Image saved to {output_path}")

def generate_report(images_data, output_dir="results"):
    """
    Generate a report with the processing results.
    
    Args:
        images_data (list): List of dictionaries containing image data
        output_dir (str, optional): Directory to save the report. Defaults to "results".
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create report text
    report = "# Image Processing Report\n\n"
    
    for img_data in images_data:
        image_name = img_data["name"]
        report += f"## {image_name}\n\n"
        
        # Add grayscale conversion info
        report += "### Grayscale Conversion\n"
        report += f"The image was successfully converted to grayscale.\n\n"
        
        # Add pixel range info
        min_val, max_val = img_data["pixel_range"]
        report += "### Pixel Value Range\n"
        report += f"- Minimum pixel value: {min_val}\n"
        report += f"- Maximum pixel value: {max_val}\n\n"
        
        # Add normalization info
        report += "### Normalization\n"
        report += "The image was normalized to improve contrast using the formula:\n"
        report += "`normalized = (image - min) / (max - min) * 255`\n\n"
        
        # Add filtering info
        report += "### Filtering Results\n"
        report += f"- Mean Filter (kernel size = {img_data['kernel_size']}): "
        report += "Reduced noise while preserving edges, but caused some blurring.\n"
        report += f"- Median Filter (kernel size = {img_data['kernel_size']}): "
        report += "Effectively removed salt-and-pepper noise while preserving edges better than mean filtering.\n"
        report += f"- Maximum Filter (kernel size = {img_data['kernel_size']}): "
        report += "Enhanced bright features and expanded light regions.\n"
        report += f"- Minimum Filter (kernel size = {img_data['kernel_size']}): "
        report += "Enhanced dark features and expanded dark regions.\n\n"
        
        report += "---\n\n"
    
    # Add general conclusions
    report += "## Conclusions\n\n"
    report += "### Normalization\n"
    report += "Normalization improved image quality by utilizing the full dynamic range of pixel values (0-255). "
    report += "This enhancement is particularly noticeable in images with low contrast, where the pixel values "
    report += "are concentrated in a narrow range. By stretching this range to cover 0-255, we can make details "
    report += "more visible to the human eye.\n\n"
    
    report += "### Filtering Effects\n"
    report += "- **Mean Filter**: Provides good noise reduction but tends to blur edges and fine details. "
    report += "It's suitable for images with Gaussian-like noise distribution.\n"
    report += "- **Median Filter**: Excellent for removing salt-and-pepper noise while preserving edges better than mean filtering. "
    report += "It's less effective against Gaussian noise.\n"
    report += "- **Maximum Filter**: Useful for finding the brightest points in an image and enhancing bright features. "
    report += "It can be used to detect light objects on dark backgrounds.\n"
    report += "- **Minimum Filter**: Useful for finding the darkest points in an image and enhancing dark features. "
    report += "It can be used to detect dark objects on light backgrounds.\n\n"
    
    report += "### Parameter Justification\n"
    report += "A kernel size of 3x3 was chosen for all filters as it provides a good balance between noise reduction "
    report += "and preservation of details. Larger kernel sizes would result in more aggressive filtering but at the cost "
    report += "of losing important image details. For applications requiring stronger noise reduction, larger kernel sizes "
    report += "could be considered.\n\n"
    
    # Save the report
    with open(f"{output_dir}/report.md", "w") as f:
        f.write(report)
    
    print(f"Report saved to {output_dir}/report.md")

def process_images(image_paths, kernel_size=3, output_dir="results"):
    """
    Process a list of images according to the specified steps.
    
    Args:
        image_paths (list): List of image paths
        kernel_size (int, optional): Kernel size for filtering. Defaults to 3.
        output_dir (str, optional): Output directory. Defaults to "results".
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # List to store data for the report
    images_data = []
    
    for image_path in image_paths:
        # Extract image name from path
        image_name = Path(image_path).stem
        print(f"\nProcessing {image_name}...")
        
        # 2. Load and convert to grayscale
        gray_image = load_and_convert_to_grayscale(image_path)
        save_processed_image(gray_image, f"{output_dir}/{image_name}_gray.jpg")
        
        # Get pixel range
        min_val, max_val = get_pixel_range(gray_image)
        print(f"Pixel range: [{min_val}, {max_val}]")
        
        # 3. Normalize the image
        normalized_image = normalize_image(gray_image)
        save_processed_image(normalized_image, f"{output_dir}/{image_name}_normalized.jpg")
        
        # 4. Apply mean filter
        mean_filtered = apply_mean_filter(normalized_image, kernel_size)
        save_processed_image(mean_filtered, f"{output_dir}/{image_name}_mean.jpg")
        
        # 5. Apply median filter
        median_filtered = apply_median_filter(normalized_image, kernel_size)
        save_processed_image(median_filtered, f"{output_dir}/{image_name}_median.jpg")
        
        # 7. Apply maximum filter
        max_filtered = apply_maximum_filter(normalized_image, kernel_size)
        save_processed_image(max_filtered, f"{output_dir}/{image_name}_max.jpg")
        
        # Apply minimum filter
        min_filtered = apply_minimum_filter(normalized_image, kernel_size)
        save_processed_image(min_filtered, f"{output_dir}/{image_name}_min.jpg")
        
        # Store data for the report
        images_data.append({
            "name": image_name,
            "pixel_range": (min_val, max_val),
            "kernel_size": kernel_size
        })
    
    # 8. Generate report
    generate_report(images_data, output_dir)

def main():
    """
    Main function to execute the image processing pipeline.
    """
    # Define image paths
    image_paths = [
        "chest_xray.jpg",
        "cat.jpg",
        "forest.jpg",
        "papyrus.png"
    ]
    
    # Set kernel size for filtering
    kernel_size = 3
    
    # Set output directory
    output_dir = "processed_images"
    
    # Process the images
    process_images(image_paths, kernel_size, output_dir)
    
    print("\nImage processing completed successfully!")

if __name__ == "__main__":
    main()