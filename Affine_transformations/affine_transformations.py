#!/usr/bin/env python3

import numpy as np
import cv2

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
    
    # Define the source and destination regions for the translation
    # For positive tx: move image to the right, for negative tx: move image to the left
    # For positive ty: move image down, for negative ty: move image up
    
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

def main():
    # Load image
    image = cv2.imread('imagen_prueba.jpg')
    
    if image is None:
        print("Error: Could not load the image 'imagen_prueba.jpg'")
        return
    
    translations = [
        np.array([0, 50]),   # +50 units vertical
        np.array([0, -50]),  # -50 units vertical
        np.array([50, 0]),   # +50 units horizontal
        np.array([-50, 0])   # -50 units horizontal
    ]
    
    for i, trans in enumerate(translations):
        translated_image = manual_translation(image, trans)
        cv2.imwrite(f'imagen_trasladada_{i+1}.jpg', translated_image)
        print(f"Imagen trasladada {i+1} guardada")

if __name__ == "__main__":
    main()