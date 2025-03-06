# Digital Image Processing

A comprehensive collection of digital image processing implementations and examples in Python.

## Overview

This repository contains various image processing algorithms and techniques implemented as part of a digital image processing course. The project includes implementations ranging from basic image operations to advanced processing techniques.

## Project Structure

- `00_OpenCV_and_Pillow/`: Basic image processing operations using OpenCV and Pillow
- `01_Foundations/`: Fundamental image processing concepts and algorithms
- `P1_Spatial_properties_and_filters/`: Spatial domain filtering and properties
- `P2_Border_detection/`: Edge detection algorithms
- `P3_Affine_transformations/`: Image transformation techniques
- `P4_Frequency_domain_filtering/`: Frequency domain analysis and filtering
- `P5_Image_restoration/`: Image restoration techniques
- `P6A_Segmentation/`: Image segmentation algorithms
- `P6B_Feature_extraction_and_descriptors/`: Feature extraction and description methods

## Requirements

### Core Dependencies
- Python 3.12 or higher
- NumPy: Array operations and mathematical computations
- OpenCV (cv2): Image processing operations
- Matplotlib: Data visualization and image display
- Pillow (PIL): Image processing and file format support

### Machine Learning Dependencies
- scikit-learn: Machine learning algorithms and tools
  - KNeighborsClassifier
  - SVC (Support Vector Classification)
  - StandardScaler
  - Pipeline
  - GridSearchCV

### GUI Dependencies
- tkinter: GUI development for desktop applications

### Additional Libraries
- datetime: Date and time operations
- os: Operating system interface
- time: Time-related functions
- random: Random number generation
- typing: Type hints and annotations

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/digital_image_processing.git
```

2. Install required dependencies:
```bash
pip install opencv-python numpy matplotlib pillow scikit-learn
```

## Key Features

- Image format conversion and basic operations
- SVD-based image compression
- Spatial domain filtering
- Border detection algorithms
- Affine transformations
- Frequency domain analysis
- Image restoration
- Segmentation techniques
- Feature extraction and classification
- GUI tools for image processing

## Usage Examples

### Grayscale Conversion
```python
from PIL import Image

# Load and convert image to grayscale
image = Image.open('your_image.jpg')
grayscale = image.convert('L')
grayscale.save('grayscale_image.png')
```

### SVD Image Compression
```python
import cv2
import numpy as np

# Load image and perform SVD
image = cv2.imread('your_image.jpg', 0)
U, S, Vt = np.linalg.svd(image, full_matrices=False)
```

### Image Translation
```python
import numpy as np
import cv2

# Create translation matrix
translation_matrix = np.array([50, 0])  # Move 50 pixels horizontally
```

## Author

José Ignacio Esparza Ireta

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.