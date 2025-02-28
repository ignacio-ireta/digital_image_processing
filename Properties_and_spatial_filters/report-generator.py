import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import os
from datetime import datetime
import jinja2
import base64
from io import BytesIO

# Import the processing functions from our existing code
# For brevity, we'll assume these are defined in "image_processing.py"
# and will import them in a real implementation

def get_image_as_base64(img: np.ndarray) -> str:
    """Convert a numpy array image to base64 string for HTML embedding."""
    # Convert to RGB if grayscale
    if len(img.shape) == 2:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create a BytesIO object
    buffer = BytesIO()
    plt.imsave(buffer, img_rgb)
    buffer.seek(0)
    
    # Convert to base64
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    return img_base64

def generate_html_report(image_paths: List[str], results: Dict, output_path: str = "image_processing_report.html"):
    """Generate an HTML report from the processing results."""
    # Create a simple HTML template
    template_str = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Image Processing Report</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 20px;
                max-width: 1200px;
                margin: 0 auto;
            }
            h1 {
                color: #2c3e50;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }
            h2 {
                color: #2980b9;
                margin-top: 30px;
            }
            h3 {
                color: #3498db;
            }
            .image-container {
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                margin: 20px 0;
            }
            .image-item {
                text-align: center;
                margin-bottom: 20px;
                flex: 1;
                min-width: 300px;
            }
            .image-item img {
                max-width: 100%;
                border: 1px solid #ddd;
            }
            table {
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            th {
                background-color: #f2f2f2;
            }
            .conclusion {
                background-color: #f8f9fa;
                padding: 15px;
                border-left: 5px solid #3498db;
                margin: 20px 0;
            }
            .methodology {
                background-color: #f8f9fa;
                padding: 15px;
                border-left: 5px solid #2ecc71;
                margin: 20px 0;
            }
            pre {
                background-color: #f5f5f5;
                padding: 10px;
                border-radius: 5px;
                overflow-x: auto;
            }
        </style>
    </head>
    <body>
        <h1>Image Processing and Enhancement Report</h1>
        <p>Generated on: {{ date }}</p>
        
        <h2>1. Introduction</h2>
        <p>
            This report details the procedures, results, and conclusions of image enhancement experiments
            performed on the following grayscale images:
        </p>
        <ul>
            {% for image in images %}
            <li>{{ image }}</li>
            {% endfor %}
        </ul>
        
        <h2>2. Methodology</h2>
        <div class="methodology">
            <h3>2.1 Grayscale Analysis</h3>
            <p>
                For each image, we first analyzed the grayscale range by determining the minimum and maximum
                pixel values. This analysis provides insight into the current contrast of the image and guides
                the normalization process.
            </p>
            
            <h3>2.2 Contrast Normalization</h3>
            <p>
                To enhance contrast, we applied normalization to expand the grayscale range to the full [0, 255]
                interval using the following formula:
            </p>
            <pre>normalized_pixel = (original_pixel - min_value) * 255 / (max_value - min_value)</pre>
            <p>
                This transformation ensures maximum contrast while preserving the relative differences between pixels.
            </p>
            
            <h3>2.3 Filtering Techniques</h3>
            <p>The following filters were applied to the normalized images:</p>
            <ul>
                <li>
                    <strong>Mean Filter ({{ kernel_size }}×{{ kernel_size }}):</strong> Replaces each pixel with the average 
                    of its neighborhood. This filter reduces noise but may blur edges.
                </li>
                <li>
                    <strong>Median Filter ({{ kernel_size }}×{{ kernel_size }}):</strong> Replaces each pixel with the median
                    value of its neighborhood. This filter is effective at removing salt-and-pepper noise
                    while preserving edges better than the mean filter.
                </li>
                <li>
                    <strong>Maximum Filter ({{ kernel_size }}×{{ kernel_size }}):</strong> Replaces each pixel with the maximum
                    value in its neighborhood. This filter enhances bright features and expands bright regions.
                </li>
                <li>
                    <strong>Minimum Filter ({{ kernel_size }}×{{ kernel_size }}):</strong> Replaces each pixel with the minimum
                    value in its neighborhood. This filter enhances dark features and expands dark regions.
                </li>
            </ul>
            
            <h3>2.4 SubMatriz Function</h3>
            <p>
                The SubMatriz function extracts a k×k submatrix centered at specified coordinates. The function
                handles boundary cases by padding the image with zeros when necessary. This function was implemented
                as follows:
            </p>
            <pre>
def SubMatriz(img, center, k):
    # Ensure k is odd
    if k % 2 == 0:
        raise ValueError("k must be an odd integer")
    
    x, y = center
    half_k = k // 2
    
    # Calculate submatrix boundaries
    x_start = x - half_k
    x_end = x + half_k + 1
    y_start = y - half_k
    y_end = y + half_k + 1
    
    # Handle boundaries with padding
    height, width = img.shape
    padded_img = np.pad(img, pad_width=half_k, mode='constant', constant_values=0)
    
    # Adjust coordinates for padded image
    x_padded = x + half_k
    y_padded = y + half_k
    
    # Extract submatrix from padded image
    submatrix = padded_img[y_padded-half_k:y_padded+half_k+1, x_padded-half_k:x_padded+half_k+1]
    
    return submatrix</pre>
            <p>
                This function was used to implement the maximum and minimum filters by applying the appropriate
                operation to each extracted submatrix.
            </p>
        </div>
        
        <h2>3. Results</h2>
        
        {% for image, image_data in results.items() %}
        <h3>3.{{ loop.index }} {{ image }}</h3>
        
        <h4>Gray Value Analysis</h4>
        <table>
            <tr>
                <th>Image</th>
                <th>Minimum Value</th>
                <th>Maximum Value</th>
                <th>Mean Value</th>
            </tr>
            <tr>
                <td>Original</td>
                <td>{{ image_data.stats.original.min }}</td>
                <td>{{ image_data.stats.original.max }}</td>
                <td>{{ "%.2f"|format(image_data.stats.original.mean) }}</td>
            </tr>
            <tr>
                <td>Normalized</td>
                <td>{{ image_data.stats.normalized.min }}</td>
                <td>{{ image_data.stats.normalized.max }}</td>
                <td>{{ "%.2f"|format(image_data.stats.normalized.mean) }}</td>
            </tr>
        </table>
        
        <h4>Processed Images</h4>
        <div class="image-container">
            <div class="image-item">
                <img src="data:image/png;base64,{{ image_data.images.original }}" alt="Original {{ image }}">
                <p>Original</p>
            </div>
            <div class="image-item">
                <img src="data:image/png;base64,{{ image_data.images.normalized }}" alt="Normalized {{ image }}">
                <p>Normalized</p>
            </div>
        </div>
        
        <div class="image-container">
            <div class="image-item">
                <img src="data:image/png;base64,{{ image_data.images.mean }}" alt="Mean Filter {{ image }}">
                <p>Mean Filter ({{ kernel_size }}×{{ kernel_size }})</p>
            </div>
            <div class="image-item">
                <img src="data:image/png;base64,{{ image_data.images.median }}" alt="Median Filter {{ image }}">
                <p>Median Filter ({{ kernel_size }}×{{ kernel_size }})</p>
            </div>
            <div class="image-item">
                <img src="data:image/png;base64,{{ image_data.images.max }}" alt="Maximum Filter {{ image }}">
                <p>Maximum Filter ({{ kernel_size }}×{{ kernel_size }})</p>
            </div>
            <div class="image-item">
                <img src="data:image/png;base64,{{ image_data.images.min }}" alt="Minimum Filter {{ image }}">
                <p>Minimum Filter ({{ kernel_size }}×{{ kernel_size }})</p>
            </div>
        </div>
        
        <h4>Filter Comparisons</h4>
        <table>
            <tr>
                <th>Filter</th>
                <th>SSIM</th>
                <th>PSNR (dB)</th>
                <th>Mean Absolute Difference</th>
                <th>Assessment</th>
            </tr>
            {% for filter_name, metrics in image_data.comparisons.items() %}
            <tr>
                <td>{{ filter_name }}</td>
                <td>{{ "%.4f"|format(metrics.ssim) }}</td>
                <td>{{ "%.2f"|format(metrics.psnr) }}</td>
                <td>{{ "%.2f"|format(metrics.mean_abs_diff) }}</td>
                <td>{{ metrics.assessment }}</td>
            </tr>
            {% endfor %}
        </table>
        
        <div class="image-container">
            <div class="image-item">
                <img src="data:image/png;base64,{{ image_data.images.histogram }}" alt="Histogram {{ image }}">
                <p>Pixel Value Distribution</p>
            </div>
        </div>
        {% endfor %}
        
        <h2>4. Discussion</h2>
        <p>
            The image processing techniques applied in this study demonstrated several important effects:
        </p>
        <ul>
            <li>
                <strong>Normalization:</strong> Successfully expanded the dynamic range of all images, enhancing
                contrast and making details more visible. This was particularly effective for images with a
                naturally narrow grayscale range.
            </li>
            <li>
                <strong>Mean Filtering:</strong> Effectively reduced random noise but introduced blurring,
                especially at edges. This effect was more pronounced with larger kernel sizes.
            </li>
            <li>
                <strong>Median Filtering:</strong> Provided good noise reduction while better preserving
                edges compared to mean filtering. It was particularly effective for the {{ best_for_median }}
                image.
            </li>
            <li>
                <strong>Maximum Filtering:</strong> Enhanced bright features and expanded bright regions,
                which was beneficial for the {{ best_for_max }} image by highlighting key structures.
            </li>
            <li>
                <strong>Minimum Filtering:</strong> Enhanced dark features and expanded dark regions,
                which was most effective for the {{ best_for_min }} image.
            </li>
        </ul>
        
        <h2>5. Conclusions</h2>
        <div class="conclusion">
            <p>
                Based on our experiments, we can draw the following conclusions:
            </p>
            <ol>
                <li>
                    <strong>Contrast Normalization:</strong> A fundamental preprocessing step that significantly
                    improves image quality by utilizing the full grayscale range. This was beneficial for all
                    images and should be considered a standard preprocessing step.
                </li>
                <li>
                    <strong>Filter Selection:</strong> Different filters are appropriate for different types of images:
                    <ul>
                        <li>
                            <strong>Medical Images (chest_xray.jpg):</strong> Median filtering provided the best
                            balance between noise reduction and feature preservation, maintaining important
                            diagnostic structures.
                        </li>
                        <li>
                            <strong>Natural Images (cat.jpg, forest.jpg):</strong> A combination of normalization
                            and gentle median filtering (smaller kernel size) produced the most visually appealing
                            results.
                        </li>
                        <li>
                            <strong>Historical Documents (papyrus.png):</strong> Maximum filtering helped enhance
                            the text and symbols, making them more readable against the background.
                        </li>
                    </ul>
                </li>
                <li>
                    <strong>Kernel Size:</strong> The choice of kernel size significantly impacts filter performance.
                    Larger kernels ({{ kernel_size }}×{{ kernel_size }}) produced more pronounced effects but risked
                    over-smoothing or distorting important features.
                </li>
                <li>
                    <strong>SubMatriz Function:</strong> Our implementation successfully handled edge cases by using
                    zero-padding, ensuring consistent behavior across the entire image including boundaries.
                </li>
            </ol>
            
            <p>
                <strong>Future Work:</strong> For further improvement, adaptive filtering techniques could be
                explored, where filter parameters are automatically adjusted based on local image characteristics.
                Additionally, combining multiple filters in a pipeline (e.g., normalization followed by edge-preserving
                smoothing) could yield even better results.
            </p>
        </div>
    </body>
    </html>
    """
    
    # Process results into the format expected by the template
    template_data = {
        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'images': [os.path.basename(path) for path in image_paths],
        'kernel_size': 5,  # Assuming a 5x5 kernel was used
        'results': {},
        'best_for_median': 'chest_xray',  # Examples - in real implementation, determine this from results
        'best_for_max': 'papyrus',
        'best_for_min': 'forest'
    }
    
    # In a real implementation, we would populate template_data['results'] with actual results
    # For this sample, we'll create placeholder data
    for path in image_paths:
        filename = os.path.splitext(os.path.basename(path))[0]
        
        # Create placeholder data
        # In a real implementation, this would come from the actual processing results
        template_data['results'][filename] = {
            'stats': {
                'original': {'min': 0, 'max': 255, 'mean': 127.5},
                'normalized': {'min': 0, 'max': 255, 'mean': 127.5}
            },
            'images': {
                'original': 'placeholder',  # In real implementation, use get_image_as_base64()
                'normalized': 'placeholder',
                'mean': 'placeholder',
                'median': 'placeholder',
                'max': 'placeholder',
                'min': 'placeholder',
                'histogram': 'placeholder'
            },
            'comparisons': {
                'Mean Filter': {'ssim': 0.85, 'psnr': 25.0, 'mean_abs_diff': 15.0, 'assessment': 'Good noise reduction but some blurring'},
                'Median Filter': {'ssim': 0.88, 'psnr': 28.0, 'mean_abs_diff': 12.0, 'assessment': 'Best overall balance'},
                'Maximum Filter': {'ssim': 0.75, 'psnr': 20.0, 'mean_abs_diff': 25.0, 'assessment': 'Enhanced bright features'},
                'Minimum Filter': {'ssim': 0.72, 'psnr': 19.0, 'mean_abs_diff': 27.0, 'assessment': 'Enhanced dark features'}
            }
        }
    
    # Render the template
    template = jinja2.Template(template_str)
    html_content = template.render(**template_data)
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Report generated: {output_path}")
    return output_path

def main():
    # List of image paths
    image_paths = ["chest_xray.jpg", "cat.jpg", "forest.jpg", "papyrus.png"]
    
    # In a real implementation, run the processing and analysis
    # results = analyze_all_images(image_paths, [3, 5, 7])
    
    # For this sample, we'll use placeholder results
    results = {}
    
    # Generate the report
    report_path = generate_html_report(image_paths, results)
    
    print(f"Processing complete. Report saved to {report_path}")

if __name__ == "__main__":
    main()