import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

def detect_human_thermal(image_path, threshold=None, visualize=True, output_path=None):
    """
    Detect humans in thermal images and visualize them with bounding boxes.
    
    Args:
        image_path (str): Path to the thermal image
        threshold (float, optional): Threshold for anomaly detection. If None, use adaptive method.
        visualize (bool): Whether to visualize the result
        output_path (str, optional): Path to save the visualization
        
    Returns:
        dict: Dictionary containing detection results:
            - 'human_count': Number of humans detected
            - 'bboxes': List of bounding boxes [(x1, y1, x2, y2), ...]
            - 'visualization_path': Path to saved visualization if output_path provided
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Convert BGR to RGB for visualization
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to grayscale for processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # For thermal images, humans typically appear as bright spots
    # Use adaptive thresholding or the provided threshold
    if threshold is None:
        # Adaptive method - use statistical properties of the image
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        threshold = mean_val + 2 * std_val  # 2 standard deviations above mean
    
    # Threshold the image to get bright regions
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # Remove small noise with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Find contours for the bright regions
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by size to identify potential humans
    human_contours = []
    min_area = 50  # Minimum area for a human signature
    max_area = 10000  # Maximum area (may need adjustment based on image size)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            human_contours.append(contour)
    
    # Extract bounding boxes
    bboxes = []
    for contour in human_contours:
        x, y, w, h = cv2.boundingRect(contour)
        bboxes.append((x, y, x+w, y+h))
    
    # Visualization
    if visualize:
        plt.figure(figsize=(10, 8))
        plt.imshow(rgb_image)
        
        # Draw bounding boxes
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                             fill=False, edgecolor='red', linewidth=2))
        
        plt.axis('off')
        plt.title(f"Detected Humans: {len(bboxes)}")
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0.0)
            plt.close()
        else:
            plt.show()
    
    return {
        'human_count': len(bboxes),
        'bboxes': bboxes,
        'visualization_path': output_path if output_path else None
    }

def process_image_with_detection(image_path, anomaly_score, optimal_threshold, output_dir=None):
    """
    Process an image with anomaly detection and human detection.
    
    Args:
        image_path (str): Path to the image
        anomaly_score (float): Anomaly score from the model
        optimal_threshold (float): Optimal threshold for anomaly detection
        output_dir (str, optional): Directory to save the output
        
    Returns:
        dict: Dictionary containing process results
    """
    is_anomaly = anomaly_score >= optimal_threshold
    
    result = {
        'image_path': image_path,
        'anomaly_score': float(anomaly_score),
        'is_anomaly': bool(is_anomaly),
        'human_count': 0,
        'visualization_path': None
    }
    
    # Only perform human detection if it's flagged as anomaly
    if is_anomaly:
        # Create output directory if not exists
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.basename(image_path)
            output_path = os.path.join(output_dir, f"detected_{filename}")
        else:
            output_path = None
            
        # Detect humans
        detection_result = detect_human_thermal(
            image_path=image_path,
            threshold=None,  # Use adaptive threshold
            visualize=True,
            output_path=output_path
        )
        
        result.update({
            'human_count': detection_result['human_count'],
            'bboxes': detection_result['bboxes'],
            'visualization_path': detection_result['visualization_path']
        })
    
    return result

def batch_process_with_threshold(image_paths, anomaly_scores, optimal_threshold, output_dir=None):
    """
    Process a batch of images with the optimal threshold.
    
    Args:
        image_paths (list): List of image paths
        anomaly_scores (list): List of anomaly scores
        optimal_threshold (float): Optimal threshold for anomaly detection
        output_dir (str, optional): Directory to save the outputs
        
    Returns:
        list: List of processing results
    """
    results = []
    
    for i, (image_path, score) in enumerate(zip(image_paths, anomaly_scores)):
        print(f"Processing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
        
        result = process_image_with_detection(
            image_path=image_path,
            anomaly_score=score,
            optimal_threshold=optimal_threshold,
            output_dir=output_dir
        )
        
        results.append(result)
        
    # Summary
    anomaly_count = sum(1 for r in results if r['is_anomaly'])
    human_count = sum(r.get('human_count', 0) for r in results)
    
    print(f"\nProcessing Summary:")
    print(f"Total images: {len(results)}")
    print(f"Anomalies detected: {anomaly_count}")
    print(f"Total humans detected: {human_count}")
    
    return results

if __name__ == "__main__":
    # Example usage
    test_image = "path/to/thermal/image.jpg"
    result = detect_human_thermal(test_image, visualize=True, output_path="detected.jpg")
    print(f"Detected {result['human_count']} humans") 