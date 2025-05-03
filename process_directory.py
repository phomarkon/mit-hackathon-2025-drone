import os
import sys
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights
from PIL import Image
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from pathlib import Path
import yaml

def load_config(config_path):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config from {config_path}: {e}")
        return None

class AnomalyDetector:
    def __init__(self, backbone="resnet18"):
        # Load a pre-trained ResNet model and modify for anomaly detection
        self.backbone = backbone
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model = self.load_model()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load human detection model (HOG descriptor)
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        self.threshold = 0.5  # Threshold for anomaly detection

    def load_model(self):
        """Load trained model based on backbone name"""
        print(f"Loading {self.backbone} model...")
        
        try:
            # Define model path based on backbone
            model_path = f"outputs/patchcore_{self.backbone}/patchcore_{self.backbone}.pth"
            
            # Select the appropriate model
            if self.backbone == "resnet18":
                model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            elif self.backbone == "resnet50":
                model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            else:
                raise ValueError(f"Unsupported backbone: {self.backbone}")
            
            # Modify for anomaly detection
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Linear(num_ftrs, 512),
                nn.ReLU(),
                nn.Linear(512, 1),
                nn.Sigmoid()
            )
            
            # Only attempt to load weights if model file exists
            if os.path.exists(model_path):
                # Handle PyTorch 2.6 security changes for torch.load
                load_success = False
                error_messages = []
                
                # Attempt 1: Try with weights_only=False (less secure but handles numpy arrays)
                try:
                    print(f"Loading model weights from {model_path} (attempt 1: weights_only=False)")
                    model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False))
                    print(f"Successfully loaded model: {model_path}")
                    load_success = True
                except Exception as e1:
                    error_messages.append(f"Attempt 1 failed: {str(e1)}")
                    
                    # Attempt 2: Add safe globals and try with weights_only=True (more secure)
                    try:
                        print(f"Attempt 1 failed, trying with safe_globals (attempt 2)")
                        import numpy
                        import torch.serialization
                        torch.serialization.add_safe_globals(['numpy._core.multiarray._reconstruct'])
                        model.load_state_dict(torch.load(model_path, map_location=self.device))
                        print(f"Successfully loaded model using safe_globals")
                        load_success = True
                    except Exception as e2:
                        error_messages.append(f"Attempt 2 failed: {str(e2)}")
                        
                        # Attempt 3: Last resort - try pickle load with custom handler
                        try:
                            print(f"Attempt 2 failed, trying with direct pickle loading (attempt 3)")
                            import pickle
                            with open(model_path, 'rb') as f:
                                state_dict = pickle.load(f, encoding='latin1')
                            model.load_state_dict(state_dict)
                            print(f"Successfully loaded model using pickle")
                            load_success = True
                        except Exception as e3:
                            error_messages.append(f"Attempt 3 failed: {str(e3)}")
                
                # If all loading attempts failed, show warning but continue with default model
                if not load_success:
                    print(f"⚠️ Could not load pre-trained model. Using default ImageNet weights instead.")
                    print(f"Make sure the model file exists and is in the correct format.")
                    print("The application will still work, but results may be less accurate.")
                    print("\nError details:")
                    for msg in error_messages:
                        print(f"  - {msg}")
            else:
                print(f"⚠️ Model file not found: {model_path}")
                print("Using default ImageNet weights instead. Results may be less accurate.")
            
            model = model.to(self.device)
            model.eval()
            print(f"Model ready for inference")
            return model
        except Exception as e:
            print(f"Critical error loading model: {e}")
            raise

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
        
        # Create heatmap for visualization
        img_np = np.array(image)
        
        # Apply CLAHE for better thermal contrast enhancement
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Create heatmap overlay
        heatmap = cv2.applyColorMap(enhanced, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Detect humans if it's an anomaly
        detected_humans = []
        result_img = img_np.copy()
        
        if is_anomaly:
            # Resize for human detection if needed
            detection_img = img_np
            if min(img_np.shape[0], img_np.shape[1]) < 200:
                # Scale up small images for better detection
                detection_img = cv2.resize(img_np, None, fx=2, fy=2)
            
            # Detect humans
            humans, weights = self.hog.detectMultiScale(
                detection_img, 
                winStride=(8, 8),
                padding=(16, 16), 
                scale=1.05,
                useMeanshiftGrouping=False
            )
            
            if len(humans) > 0:
                # Store the detections
                detected_humans = [
                    {
                        'box': (x, y, w, h), 
                        'confidence': float(weights[i])
                    } 
                    for i, (x, y, w, h) in enumerate(humans)
                ]
                
                # Draw bounding boxes
                for (x, y, w, h) in humans:
                    # If detection was done on a resized image, adjust coordinates
                    if detection_img.shape != img_np.shape:
                        ratio = img_np.shape[0] / detection_img.shape[0]
                        x, y, w, h = int(x * ratio), int(y * ratio), int(w * ratio), int(h * ratio)
                    
                    # Draw rectangle with red color
                    cv2.rectangle(result_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Create alpha blend of original and heatmap
        alpha = 0.3 if is_anomaly else 0.1  # More visible for anomalies
        result_img_with_heatmap = cv2.addWeighted(result_img, 1 - alpha, heatmap, alpha, 0)
        
        return {
            'is_anomaly': is_anomaly,
            'confidence': confidence * 100,  # Convert to percentage
            'heatmap_img': result_img_with_heatmap,
            'raw_score': score,
            'humans_detected': len(detected_humans),
            'human_boxes': detected_humans
        }


def get_priority_label(confidence, is_anomaly, humans_detected=0):
    """Get priority label based on confidence, anomaly status and human detection"""
    if not is_anomaly:
        return "Normal"
    elif humans_detected > 0 or confidence >= 80:
        return "High"
    elif confidence >= 60:
        return "Medium"
    else:
        return "Low"


def process_directory(input_dir, output_dir=None, save_images=False, create_summary=True, backbone="resnet18"):
    """Process all images in a directory and output results"""
    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} does not exist")
        return
    
    # Create output directory if needed
    if save_images and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Results will be saved to {output_dir}")
    
    # Try to load matching config file
    config_path = f"configs/patchcore_{backbone}.yaml"
    if os.path.exists(config_path):
        config = load_config(config_path)
        if config:
            print(f"Configuration loaded from: {config_path}")
    
    # Initialize detector with selected backbone
    print(f"Initializing detector with {backbone} backbone...")
    detector = AnomalyDetector(backbone=backbone)
    results = []
    
    # Get list of image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
    image_files = [f for f in os.listdir(input_dir) 
                  if os.path.isfile(os.path.join(input_dir, f)) and 
                  os.path.splitext(f)[1].lower() in image_extensions]
    
    print(f"Found {len(image_files)} images in {input_dir}")
    
    # Process each image
    for filename in tqdm(image_files, desc="Processing images"):
        file_path = os.path.join(input_dir, filename)
        try:
            result = detector.predict(file_path)
            result['filename'] = filename
            result['priority'] = get_priority_label(
                result['confidence'], 
                result['is_anomaly'],
                result['humans_detected']
            )
            results.append(result)
            
            if save_images and output_dir:
                # Convert from RGB to BGR for OpenCV
                output_img = cv2.cvtColor(result['heatmap_img'], cv2.COLOR_RGB2BGR)
                
                # Add text with prediction results
                status = "ANOMALY" if result['is_anomaly'] else "NORMAL"
                confidence = f"{result['confidence']:.1f}%"
                priority = result['priority']
                humans = result['humans_detected']
                
                # Set color based on priority
                if priority == "High":
                    color = (0, 0, 255)  # Red (BGR)
                elif priority == "Medium":
                    color = (0, 165, 255)  # Orange
                elif priority == "Low":
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (0, 255, 0)  # Green
                
                # Create a background rectangle for text
                cv2.rectangle(output_img, (10, 10), (350, 110), (0, 0, 0), -1)
                
                # Add text
                cv2.putText(output_img, f"{status} - {confidence}", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(output_img, f"Priority: {priority}", (20, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(output_img, f"Humans detected: {humans}", (20, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                # Save the image
                output_path = os.path.join(output_dir, f"analyzed_{filename}")
                cv2.imwrite(output_path, output_img)
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    # Sort results by confidence (prioritize high confidence anomalies)
    results.sort(key=lambda x: x['confidence'] if x['is_anomaly'] else 0, reverse=True)
    
    # Create summary report
    if create_summary and results:
        create_summary_report(results, output_dir if output_dir else input_dir, backbone)
    
    # Print summary
    anomalies = [r for r in results if r['is_anomaly']]
    human_detections = sum(r['humans_detected'] for r in results)
    
    print("\n===== RESULTS SUMMARY =====")
    print(f"Model: {backbone}")
    print(f"Total images processed: {len(results)}")
    print(f"Anomalies detected: {len(anomalies)} ({len(anomalies)/len(results)*100:.1f}%)")
    print(f"Human detections: {human_detections}")
    
    if anomalies:
        print("\nTop 5 anomalies (highest confidence):")
        for i, result in enumerate(anomalies[:5]):
            humans_info = f" - Humans: {result['humans_detected']}" if result['humans_detected'] > 0 else ""
            print(f"{i+1}. {result['filename']} - Confidence: {result['confidence']:.1f}% - Priority: {result['priority']}{humans_info}")
    
    return results


def create_summary_report(results, output_dir, backbone="resnet18"):
    """Create summary report files (CSV, HTML, and JSON)"""
    # Prepare report prefix
    prefix = f"anomaly_results_{backbone}"
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame([
        {
            'Filename': r['filename'],
            'Status': 'ANOMALY' if r['is_anomaly'] else 'NORMAL',
            'Confidence': f"{r['confidence']:.1f}%",
            'Priority': r['priority'],
            'Humans': r.get('humans_detected', 0),
            'Raw Score': f"{r['raw_score']:.4f}"
        }
        for r in results
    ])
    
    # Save as CSV
    csv_path = os.path.join(output_dir, f"{prefix}.csv")
    df.to_csv(csv_path, index=False)
    print(f"CSV report saved to {csv_path}")
    
    # Save as HTML report
    html_path = os.path.join(output_dir, f"{prefix}.html")
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Thermal Drone Anomaly Detection Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #333; }
            table { border-collapse: collapse; width: 100%; margin-top: 20px; }
            th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background-color: #f2f2f2; }
            tr:hover { background-color: #f5f5f5; }
            .anomaly-high { color: #ff0000; font-weight: bold; }
            .anomaly-medium { color: #ff6600; font-weight: bold; }
            .anomaly-low { color: #ffcc00; font-weight: bold; }
            .normal { color: #00cc00; }
            .summary { margin: 20px 0; padding: 10px; background-color: #f9f9f9; border-radius: 5px; }
            .humans { background-color: #ffeeee; }
        </style>
    </head>
    <body>
        <h1>Thermal Drone Anomaly Detection Report</h1>
        <p><b>Model:</b> {backbone}</p>
        
        <div class="summary">
            <h2>Summary</h2>
            <p>Total images: {total}</p>
            <p>Anomalies detected: {anomalies} ({percentage:.1f}%)</p>
            <p>Human detections: {humans}</p>
        </div>
        
        <h2>Detailed Results</h2>
        <p>Results are sorted by priority (highest confidence anomalies first)</p>
        <table>
            <tr>
                <th>Filename</th>
                <th>Status</th>
                <th>Confidence</th>
                <th>Priority</th>
                <th>Humans</th>
            </tr>
            {table_rows}
        </table>
    </body>
    </html>
    """
    
    # Create table rows with priority-based styling
    table_rows = ""
    for _, row in df.iterrows():
        status_class = ""
        if row['Status'] == 'ANOMALY':
            if row['Priority'] == 'High':
                status_class = "anomaly-high"
            elif row['Priority'] == 'Medium':
                status_class = "anomaly-medium"
            elif row['Priority'] == 'Low':
                status_class = "anomaly-low"
        else:
            status_class = "normal"
        
        human_class = " humans" if row['Humans'] > 0 else ""
        
        table_rows += f"""
        <tr>
            <td>{row['Filename']}</td>
            <td class="{status_class}">{row['Status']}</td>
            <td>{row['Confidence']}</td>
            <td class="{status_class}">{row['Priority']}</td>
            <td class="{status_class}{human_class}">{row['Humans']}</td>
        </tr>
        """
    
    # Calculate summary statistics
    total = len(results)
    anomalies = sum(1 for r in results if r['is_anomaly'])
    percentage = (anomalies / total * 100) if total > 0 else 0
    humans = sum(r.get('humans_detected', 0) for r in results)
    
    # Generate complete HTML
    html_content = html_content.format(
        backbone=backbone,
        total=total,
        anomalies=anomalies,
        percentage=percentage,
        humans=humans,
        table_rows=table_rows
    )
    
    # Write HTML file
    with open(html_path, 'w') as f:
        f.write(html_content)
    print(f"HTML report saved to {html_path}")
    
    # Save as JSON
    json_path = os.path.join(output_dir, f"{prefix}.json")
    
    # Create a serializable version of the results
    json_results = []
    for r in results:
        # Create a copy without non-serializable objects
        json_result = {
            'filename': r['filename'],
            'is_anomaly': r['is_anomaly'],
            'confidence': r['confidence'],
            'priority': r['priority'],
            'humans_detected': r.get('humans_detected', 0),
            'raw_score': r['raw_score']
        }
        json_results.append(json_result)
    
    with open(json_path, 'w') as f:
        json.dump({
            'model': backbone,
            'summary': {
                'total_images': total,
                'anomalies_detected': anomalies,
                'anomaly_percentage': percentage,
                'humans_detected': humans
            },
            'results': json_results
        }, f, indent=2)
    
    print(f"JSON report saved to {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Thermal drone footage anomaly detector")
    parser.add_argument("--input", "-i", required=True, help="Input directory containing images")
    parser.add_argument("--output", "-o", help="Output directory for processed images and reports")
    parser.add_argument("--save", "-s", action="store_true", help="Save processed images with annotations")
    parser.add_argument("--no-summary", action="store_true", help="Don't create summary reports")
    parser.add_argument("--model", "-m", choices=["resnet18", "resnet50"], default="resnet18", 
                       help="Model backbone to use (default: resnet18)")
    args = parser.parse_args()
    
    process_directory(
        args.input, 
        args.output, 
        save_images=args.save, 
        create_summary=not args.no_summary,
        backbone=args.model
    )

if __name__ == "__main__":
    main() 

    