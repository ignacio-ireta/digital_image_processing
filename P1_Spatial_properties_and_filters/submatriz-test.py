import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

def SubMatriz(img: np.ndarray, center: Tuple[int, int], k: int) -> np.ndarray:
    """
    Extract a kxk submatrix from an image centered at the given coordinates.
    
    Args:
        img: Input image
        center: (x, y) coordinates for center of submatrix (x=column, y=row)
        k: Size of submatrix (must be odd)
        
    Returns:
        kxk submatrix
    """
    if k % 2 == 0:
        raise ValueError("k must be an odd integer")
    
    x, y = center
    half_k = k // 2
    
    # Calculate submatrix boundaries
    x_start = x - half_k
    x_end = x + half_k + 1
    y_start = y - half_k
    y_end = y + half_k + 1
    
    # Handle boundaries - use padding with zeros
    height, width = img.shape
    padded_img = np.pad(img, pad_width=half_k, mode='constant', constant_values=0)
    
    # Adjust coordinates for padded image
    x_padded = x + half_k
    y_padded = y + half_k
    
    # Extract submatrix from padded image
    submatrix = padded_img[y_padded-half_k:y_padded+half_k+1, x_padded-half_k:x_padded+half_k+1]
    
    return submatrix

def test_SubMatriz():
    """
    Test the SubMatriz function with various test cases.
    """
    # Create a test image with unique values
    test_img = np.arange(100).reshape(10, 10)
    print("Test image (10x10):")
    print(test_img)
    
    # Test case 1: Center of the image with k=3
    print("\nTest case 1: Center of the image (5,5) with k=3")
    center1 = (5, 5)
    k1 = 3
    sub1 = SubMatriz(test_img, center1, k1)
    print(f"SubMatriz result ({k1}x{k1}):")
    print(sub1)
    
    # Test case 2: Near the edge with k=5
    print("\nTest case 2: Near the edge (1,1) with k=5")
    center2 = (1, 1)
    k2 = 5
    sub2 = SubMatriz(test_img, center2, k2)
    print(f"SubMatriz result ({k2}x{k2}):")
    print(sub2)
    
    # Test case 3: At the corner with k=3
    print("\nTest case 3: At the corner (0,0) with k=3")
    center3 = (0, 0)
    k3 = 3
    sub3 = SubMatriz(test_img, center3, k3)
    print(f"SubMatriz result ({k3}x{k3}):")
    print(sub3)
    
    # Test case 4: Outside the image with k=3
    print("\nTest case 4: Outside the image (-1,-1) with k=3")
    center4 = (-1, -1)
    k4 = 3
    sub4 = SubMatriz(test_img, center4, k4)
    print(f"SubMatriz result ({k4}x{k4}):")
    print(sub4)
    
    # Visual verification - create a visualization of the padding and extraction
    plt.figure(figsize=(12, 10))
    
    # Display the original image
    plt.subplot(2, 2, 1)
    plt.imshow(test_img, cmap='viridis')
    plt.title('Original Image')
    plt.colorbar()
    
    # Display the test cases with markings
    for i, (center, k, title) in enumerate([
        (center1, k1, 'Center (5,5), k=3'),
        (center2, k2, 'Near Edge (1,1), k=5'),
        (center3, k3, 'Corner (0,0), k=3')
    ]):
        plt.subplot(2, 2, i+2)
        plt.imshow(test_img, cmap='viridis')
        plt.title(title)
        
        half_k = k // 2
        x, y = center
        
        # Draw the box representing the submatrix
        plt.gca().add_patch(plt.Rectangle(
            (x - half_k - 0.5, y - half_k - 0.5),
            k, k, 
            edgecolor='red', 
            facecolor='none', 
            linewidth=2
        ))
        
        # Mark the center
        plt.plot(x, y, 'rx', markersize=10)
        
    plt.tight_layout()
    plt.savefig('submatriz_test_visualization.png')
    
    # Return True if all tests passed
    return True

if __name__ == "__main__":
    test_SubMatriz()
    print("\nAll tests completed. Check the visualization saved as 'submatriz_test_visualization.png'")