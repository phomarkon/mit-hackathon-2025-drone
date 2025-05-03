import os
import sys
import argparse
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
import matplotlib.pyplot as plt
from tqdm import tqdm

class AnomalyDetector:
    def __init__(self):
        # Load a pre-trained ResNet model and modify for anomaly detection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model = self.load_model()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.threshold = 0.5  # Threshold for anomaly detection

    def load_model(self):
        # For the MVP, we'll use a pre-trained ResNet18 model
        # In a real implementation, you would fine-tune this on the thermal drone dataset
        model = resnet18(pretrained=True)
        # Modify the model for anomaly detection (simplified for MVP)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        model = model.to(self.device)
        model.eval()
        return model

    def predict(self, image_path):
        # Open image and convert to RGB if grayscale
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Transform and predict
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            score = self.model(image_tensor).item()
        
        is_anomaly = score > self.threshold
        confidence = score if is_anomaly else 1 - score
        
        # Simple heatmap for visualization (in a real implementation, this would be more sophisticated)
        # For MVP, we'll just highlight areas with high thermal signatures
        img_np = np.array(image)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Simple thresholding to find bright areas (potential thermal signatures)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw bounding boxes around potential anomalies
        result_img = img_np.copy()
        if is_anomaly:
            for contour in contours:
                if cv2.contourArea(contour) > 100:  # Filter small contours
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        return {
            'is_anomaly': is_anomaly,
            'confidence': confidence * 100,  # Convert to percentage
            'heatmap_img': result_img,
            'bounding_boxes': [(cv2.boundingRect(c), cv2.contourArea(c)) for c in contours if cv2.contourArea(c) > 100]
        }

def process_directory(input_dir, output_dir=None, save_images=False):
    """Process all images in a directory and output results"""
    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} does not exist")
        return
    
    if save_images and output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    detector = AnomalyDetector()
    results = []
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
    image_files = [f for f in os.listdir(input_dir) 
                  if os.path.isfile(os.path.join(input_dir, f)) and 
                  os.path.splitext(f)[1].lower() in image_extensions]
    
    print(f"Found {len(image_files)} images in {input_dir}")
    
    for filename in tqdm(image_files):
        file_path = os.path.join(input_dir, filename)
        try:
            result = detector.predict(file_path)
            result['filename'] = filename
            results.append(result)
            
            if save_images and output_dir:
                # Convert from RGB to BGR for OpenCV
                output_img = cv2.cvtColor(result['heatmap_img'], cv2.COLOR_RGB2BGR)
                
                # Add text with prediction results
                status = "ANOMALY" if result['is_anomaly'] else "NORMAL"
                confidence = f"{result['confidence']:.1f}%"
                cv2.putText(output_img, f"{status} - {confidence}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if result['is_anomaly'] else (0, 255, 0), 2)
                
                output_path = os.path.join(output_dir, f"processed_{filename}")
                cv2.imwrite(output_path, output_img)
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    # Sort results by confidence (prioritize high confidence anomalies)
    results.sort(key=lambda x: x['confidence'] if x['is_anomaly'] else 0, reverse=True)
    
    # Print summary
    anomalies = [r for r in results if r['is_anomaly']]
    print("\n===== RESULTS SUMMARY =====")
    print(f"Total images processed: {len(results)}")
    print(f"Anomalies detected: {len(anomalies)} ({len(anomalies)/len(results)*100:.1f}%)")
    
    if anomalies:
        print("\nTop 5 anomalies (highest confidence):")
        for i, result in enumerate(anomalies[:5]):
            print(f"{i+1}. {result['filename']} - Confidence: {result['confidence']:.1f}%")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Thermal drone footage anomaly detector")
    parser.add_argument("--input", "-i", required=True, help="Input directory containing images")
    parser.add_argument("--output", "-o", help="Output directory for processed images")
    parser.add_argument("--save", "-s", action="store_true", help="Save processed images with annotations")
    args = parser.parse_args()
    
    process_directory(args.input, args.output, args.save)

if __name__ == "__main__":
    main() 