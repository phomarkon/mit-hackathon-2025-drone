import sys
import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Import the AnomalyDetector class from the anomaly_detector.py file
from anomaly_detector import AnomalyDetector

def test_single_image(image_path):
    """Test anomaly detection on a single image"""
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} does not exist")
        return
    
    print(f"Testing anomaly detection on {image_path}")
    detector = AnomalyDetector()
    
    try:
        result = detector.predict(image_path)
        
        # Display results
        print("\nResults:")
        print(f"Is anomaly: {result['is_anomaly']}")
        print(f"Confidence: {result['confidence']:.2f}%")
        print(f"Number of detected regions: {len(result['bounding_boxes'])}")
        
        # Display the image with bounding boxes
        img = result['heatmap_img']
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        
        status = "ANOMALY" if result['is_anomaly'] else "NORMAL"
        plt.title(f"{os.path.basename(image_path)} - {status} ({result['confidence']:.2f}%)")
        
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_model.py <path_to_image>")
        sys.exit(1)
    
    test_single_image(sys.argv[1]) 