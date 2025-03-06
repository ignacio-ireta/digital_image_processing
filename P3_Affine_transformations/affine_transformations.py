#!/usr/bin/env python3

import numpy as np
import cv2
import os
from pathlib import Path
from dotenv import load_dotenv

def manual_translation(image, translation_matrix):
    """
    Applies a translation to an image without using cv2.warpAffine().
    
    :param image: Input image (numpy array)
    :param translation_matrix: 2x1 translation matrix [tx, ty]
    :return: Translated image
    """
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Create output image with same dimensions and data type
    translated_image = np.zeros_like(image)
    
    # Extract translation values
    tx, ty = int(translation_matrix[0]), int(translation_matrix[1])
    
    # Source region (where to copy from in the original image)
    src_x_start = max(0, -tx)
    src_y_start = max(0, -ty)
    src_x_end = min(width, width - tx) if tx > 0 else width
    src_y_end = min(height, height - ty) if ty > 0 else height
    
    # Destination region (where to paste in the new image)
    dst_x_start = max(0, tx)
    dst_y_start = max(0, ty)
    dst_x_end = min(width, width + tx) if tx < 0 else width
    dst_y_end = min(height, height + ty) if ty < 0 else height
    
    # Ensure the regions have the same size
    width_to_copy = min(src_x_end - src_x_start, dst_x_end - dst_x_start)
    height_to_copy = min(src_y_end - src_y_start, dst_y_end - dst_y_start)
    
    if width_to_copy > 0 and height_to_copy > 0:
        # Copy the region from source to destination
        translated_image[dst_y_start:dst_y_start + height_to_copy, 
                         dst_x_start:dst_x_start + width_to_copy] = \
            image[src_y_start:src_y_start + height_to_copy, 
                  src_x_start:src_x_start + width_to_copy]
    
    return translated_image

def ensure_directory_exists(directory):
    """Create directory if it doesn't exist."""
    Path(directory).mkdir(parents=True, exist_ok=True)

def main():
    # Load environment variables
    load_dotenv()
    
    # Get paths from environment variables
    input_image = os.getenv('INPUT_IMAGE_PATH', 'images/imagen_prueba.jpg')
    output_dir = os.getenv('OUTPUT_DIR', 'output')
    
    # Ensure paths are relative to the script location
    script_dir = Path(__file__).parent
    input_image_path = script_dir / input_image
    output_dir_path = script_dir / output_dir
    
    # Create output directory if it doesn't exist
    ensure_directory_exists(output_dir_path)
    
    # Load image
    image = cv2.imread(str(input_image_path))
    
    if image is None:
        print(f"Error: Could not load the image '{input_image_path}'")
        return
    
    translations = [
        np.array([0, 50]),   # +50 units vertical
        np.array([0, -50]),  # -50 units vertical
        np.array([50, 0]),   # +50 units horizontal
        np.array([-50, 0])   # -50 units horizontal
    ]
    
    for i, trans in enumerate(translations):
        translated_image = manual_translation(image, trans)
        output_path = output_dir_path / f'imagen_trasladada_{i+1}.jpg'
        cv2.imwrite(str(output_path), translated_image)
        print(f"Imagen trasladada {i+1} guardada en {output_path}")

if __name__ == "__main__":
    main()