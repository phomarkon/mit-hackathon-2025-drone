#!/usr/bin/env python
"""
Script to test human detection on a single thermal image.
"""
import os
import argparse
import matplotlib.pyplot as plt
from src.human_detection import detect_human_thermal

def parse_args():
    parser = argparse.ArgumentParser(description='Test human detection on a thermal image')
    parser.add_argument('--image', type=str, default='data/test/abnormal/DJI_M300_H20t_0003.jpg',
                        help='Path to the thermal image to analyze')
    parser.add_argument('--output', type=str, default='detected_humans.jpg',
                        help='Path to save the visualization')
    parser.add_argument('--sensitivity', type=float, default=2.0,
                        help='Sensitivity for adaptive thresholding (standard deviations above mean)')
    parser.add_argument('--min_area', type=int, default=50,
                        help='Minimum area for a human signature')
    parser.add_argument('--max_area', type=int, default=10000,
                        help='Maximum area for a human signature')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Check if the image exists
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        print(f"Working directory: {os.getcwd()}")
        print(f"Searching for the image in test/abnormal directory...")
        
        # Try to find the image in the default test/abnormal directory
        for root, dirs, files in os.walk('data'):
            for file in files:
                if file == os.path.basename(args.image):
                    full_path = os.path.join(root, file)
                    print(f"Found image at: {full_path}")
                    args.image = full_path
                    break
        
        if not os.path.exists(args.image):
            print(f"Still couldn't find the image. Please provide the correct path.")
            return
    
    print(f"Processing image: {args.image}")
    
    # Custom detect_human_thermal function with modified parameters
    def custom_detect_thermal(image_path, sensitivity=2.0, min_area=50, max_area=10000):
        import cv2
        import numpy as np
        
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Convert BGR to RGB for visualization
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # For thermal images, humans typically appear as bright spots
        # Use adaptive thresholding based on sensitivity parameter
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        threshold = mean_val + sensitivity * std_val
        
        print(f"Image statistics: Mean={mean_val:.2f}, StdDev={std_val:.2f}")
        print(f"Computed threshold: {threshold:.2f}")
        
        # Threshold the image to get bright regions
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        # Remove small noise with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Find contours for the bright regions
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size to identify potential humans
        human_contours = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                human_contours.append(contour)
        
        print(f"Found {len(contours)} total contours")
        print(f"After filtering by size ({min_area} < area < {max_area}): {len(human_contours)} human contours")
        
        # Extract bounding boxes
        bboxes = []
        for contour in human_contours:
            x, y, w, h = cv2.boundingRect(contour)
            bboxes.append((x, y, x+w, y+h))
        
        # Visualization
        plt.figure(figsize=(15, 10))
        
        # Original image
        plt.subplot(2, 2, 1)
        plt.imshow(rgb_image)
        plt.title("Original Image")
        plt.axis('off')
        
        # Binary threshold
        plt.subplot(2, 2, 2)
        plt.imshow(binary, cmap='gray')
        plt.title(f"Binary Threshold (value={threshold:.2f})")
        plt.axis('off')
        
        # Original with bounding boxes
        plt.subplot(2, 2, 3)
        plt.imshow(rgb_image)
        plt.title(f"Detected Humans: {len(bboxes)}")
        
        # Draw bounding boxes
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                            fill=False, edgecolor='red', linewidth=2))
        plt.axis('off')
        
        # Contours
        plt.subplot(2, 2, 4)
        contour_img = np.zeros_like(gray)
        cv2.drawContours(contour_img, human_contours, -1, (255, 255, 255), 2)
        plt.imshow(contour_img, cmap='gray')
        plt.title("Human Contours")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(args.output)
        print(f"Visualization saved to {args.output}")
        plt.show()
        
        return {
            'human_count': len(bboxes),
            'bboxes': bboxes
        }
    
    # Run custom detection
    try:
        result = custom_detect_thermal(
            args.image,
            sensitivity=args.sensitivity,
            min_area=args.min_area,
            max_area=args.max_area
        )
        print(f"\nDetected {result['human_count']} humans in the image.")
        
        # Also run the regular detection for comparison
        print("\nRunning regular detection for comparison:")
        standard_result = detect_human_thermal(
            image_path=args.image,
            threshold=None,
            visualize=True,
            output_path="standard_detection.jpg"
        )
        print(f"Standard detection found {standard_result['human_count']} humans.")
        
    except Exception as e:
        print(f"Error in detection: {e}")

if __name__ == "__main__":
    main() 