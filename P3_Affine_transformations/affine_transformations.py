import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import math

def load_grayscale_image(image_path):
    """Load an image and convert it to grayscale."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    img = Image.open(image_path).convert('L')
    return np.array(img)

def bilinear_interpolation(image, x, y):
    """
    Perform bilinear interpolation at point (x, y) in the image.
    
    Args:
        image: Input grayscale image as numpy array
        x, y: Coordinates in the input image (can be floating point)
        
    Returns:
        Interpolated pixel value
    """
    height, width = image.shape
    
    # Check if the point is outside the image bounds
    if x < 0 or y < 0 or x >= width - 1 or y >= height - 1:
        return 0  # Return black for points outside the image
    
    # Get the four surrounding pixel coordinates
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = x0 + 1, y0 + 1
    
    # Ensure we don't go out of bounds
    x1 = min(x1, width - 1)
    y1 = min(y1, height - 1)
    
    # Get the four surrounding pixel values
    f00 = image[y0, x0]
    f01 = image[y0, x1]
    f10 = image[y1, x0]
    f11 = image[y1, x1]
    
    # Calculate interpolation weights
    wx = x - x0
    wy = y - y0
    
    # Perform bilinear interpolation
    top = f00 * (1 - wx) + f01 * wx
    bottom = f10 * (1 - wx) + f11 * wx
    return int(top * (1 - wy) + bottom * wy)

def calculate_output_dimensions(image_shape, matrix):
    """
    Calculate the output dimensions needed to contain the entire transformed image.
    
    Args:
        image_shape: Shape of the input image (height, width)
        matrix: Affine transformation matrix
        
    Returns:
        Tuple of (new_height, new_width, min_x, min_y)
    """
    height, width = image_shape
    
    # Get the coordinates of the four corners of the image
    corners = np.array([
        [0, 0, 1],
        [width - 1, 0, 1],
        [0, height - 1, 1],
        [width - 1, height - 1, 1]
    ])
    
    # Apply the transformation to each corner
    transformed_corners = np.dot(corners, matrix.T)
    
    # Calculate the minimum and maximum coordinates
    min_x = np.floor(np.min(transformed_corners[:, 0])).astype(int)
    max_x = np.ceil(np.max(transformed_corners[:, 0])).astype(int)
    min_y = np.floor(np.min(transformed_corners[:, 1])).astype(int)
    max_y = np.ceil(np.max(transformed_corners[:, 1])).astype(int)
    
    # Calculate the new dimensions
    new_width = max_x - min_x + 1
    new_height = max_y - min_y + 1
    
    return new_height, new_width, min_x, min_y

def affine_transform(image, matrix, output_size=None):
    """
    Apply an affine transformation to the input image.
    
    Args:
        image: Input grayscale image as numpy array
        matrix: 3x3 affine transformation matrix
        output_size: Optional tuple (height, width) for the output image
        
    Returns:
        Transformed image as numpy array
    """
    if not isinstance(matrix, np.ndarray) or matrix.shape != (3, 3):
        raise ValueError("Transformation matrix must be a 3x3 numpy array")
    
    height, width = image.shape
    
    # Determine output size
    if output_size is None:
        # Calculate output dimensions to contain the entire transformed image
        new_height, new_width, min_x, min_y = calculate_output_dimensions(image.shape, matrix)
        
        # Adjust the transformation matrix to account for the shift
        adjustment_matrix = np.array([
            [1, 0, -min_x],
            [0, 1, -min_y],
            [0, 0, 1]
        ])
        matrix = np.dot(adjustment_matrix, matrix)
    else:
        new_height, new_width = output_size
    
    # Create output image
    output = np.zeros((new_height, new_width), dtype=np.uint8)
    
    # Compute the inverse of the transformation matrix
    try:
        inv_matrix = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        raise ValueError("Transformation matrix is not invertible")
    
    # Loop through each pixel in the output image
    for j in range(new_height):
        for i in range(new_width):
            # Apply inverse transformation to get the source coordinates
            source = np.dot(inv_matrix, np.array([i, j, 1]))
            x, y = source[0], source[1]
            
            # Apply bilinear interpolation to get the pixel value
            output[j, i] = bilinear_interpolation(image, x, y)
    
    return output

def create_translation_matrix(tx, ty):
    """Create a translation matrix for affine transformation."""
    return np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ])

def create_scaling_matrix(sx, sy, center=None):
    """
    Create a scaling matrix for affine transformation.
    
    Args:
        sx, sy: Scale factors for x and y directions
        center: Optional (x, y) point to scale around, defaults to (0, 0)
        
    Returns:
        3x3 scaling matrix
    """
    if center is None:
        return np.array([
            [sx, 0, 0],
            [0, sy, 0],
            [0, 0, 1]
        ])
    else:
        cx, cy = center
        return np.array([
            [sx, 0, cx * (1 - sx)],
            [0, sy, cy * (1 - sy)],
            [0, 0, 1]
        ])

def create_rotation_matrix(angle_degrees, center=None):
    """
    Create a rotation matrix for affine transformation.
    
    Args:
        angle_degrees: Rotation angle in degrees (counterclockwise)
        center: Optional (x, y) point to rotate around, defaults to (0, 0)
        
    Returns:
        3x3 rotation matrix
    """
    angle_radians = np.radians(angle_degrees)
    cos_theta = np.cos(angle_radians)
    sin_theta = np.sin(angle_radians)
    
    if center is None:
        return np.array([
            [cos_theta, -sin_theta, 0],
            [sin_theta, cos_theta, 0],
            [0, 0, 1]
        ])
    else:
        cx, cy = center
        return np.array([
            [cos_theta, -sin_theta, cx * (1 - cos_theta) + cy * sin_theta],
            [sin_theta, cos_theta, cy * (1 - cos_theta) - cx * sin_theta],
            [0, 0, 1]
        ])

def display_images(original, transformed, title="Affine Transformation"):
    """Display original and transformed images side by side."""
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(transformed, cmap='gray')
    plt.title('Transformed Image')
    plt.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def save_image(image, output_path):
    """Save the image to the specified path."""
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    Image.fromarray(image).save(output_path)
    print(f"Transformed image saved to {output_path}")

def main():
    # Example usage
    current_dir = os.path.dirname(os.path.abspath('__file__'))
    input_dir = os.path.join(current_dir, r"P3_Affine_transformations\Input")
    output_dir = os.path.join(current_dir, r"P3_Affine_transformations\Output")
    input_image = os.listdir(input_dir)[0]
    input_path = os.path.join(input_dir, input_image)
    output_path = output_dir + "/transformed_image.jpg"  # Replace with your output path
    
    try:
        # Load the grayscale image
        image = load_grayscale_image(input_path)
        
        # Get image center for centered transformations
        height, width = image.shape
        center = (width // 2, height // 2)
        
        # Example transformations
        # Uncomment the one you want to use
        
        # Translation: Move the image 500 pixels right and 300 pixels down
        # matrix = create_translation_matrix(500, 300)
        # title = "Translation (500 right, 300 down)"
        
        # Scaling: Scale the image to twice its size around its center
        # matrix = create_scaling_matrix(2.0, 2.0, center=center)
        # title = "Scaling (2x) around center"
        
        # Rotation: Rotate the image by 45 degrees around its center
        matrix = create_rotation_matrix(45, center=center)
        title = "Rotation (45 degrees) around center"
        
        # Apply the affine transformation
        transformed_image = affine_transform(image, matrix)
        
        # Display the images
        display_images(image, transformed_image, title)
        
        # Save the transformed image
        save_image(transformed_image, output_path)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()