#!/usr/bin/env python
import os
import sys
import json
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.utils import load_config
from src.models.patchcore import PatchCore
from src.data_loader import get_transforms
from src.human_detection import detect_human_thermal

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze a single thermal image using a trained model')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to the thermal image to analyze')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the model config file')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to the trained model. If not provided, will look for the model in the output directory')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Manual threshold for anomaly detection. If not provided, will use the optimal threshold from training')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Path to save the visualization. If not provided, will display the result')
    parser.add_argument('--threshold_file', type=str, default=None,
                        help='Path to the optimal threshold file. If not provided, will look for it in the output directory')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    return parser.parse_args()

def load_threshold(threshold_file, default=0.5):
    """Load the optimal threshold from a JSON file."""
    if not os.path.exists(threshold_file):
        print(f"Warning: Threshold file not found: {threshold_file}. Using default: {default}")
        return default
    
    try:
        with open(threshold_file, 'r') as f:
            threshold_data = json.load(f)
        threshold = threshold_data.get('optimal_threshold', default)
        print(f"Loaded optimal threshold: {threshold:.4f} from {threshold_file}")
        return threshold
    except Exception as e:
        print(f"Error loading threshold from {threshold_file}: {e}. Using default: {default}")
        return default

def analyze_image(image_path, model, transforms, threshold, output_path=None, verbose=False):
    """Analyze a single image and visualize the result with human detection if anomalous."""
    # Load and transform the image
    try:
        from PIL import Image
        image = Image.open(image_path).convert('RGB')
        image_tensor = transforms(image)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None
    
    # Get anomaly score
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(model.device)
        anomaly_map, anomaly_score = model.predict(image_tensor)
    
    # Determine if it's an anomaly
    is_anomaly = anomaly_score >= threshold
    
    if verbose:
        print(f"Image: {image_path}")
        print(f"Anomaly score: {anomaly_score:.4f}")
        print(f"Threshold: {threshold:.4f}")
        print(f"Is anomaly: {is_anomaly}")
    
    # If it's an anomaly, detect humans
    human_count = 0
    bboxes = []
    
    if is_anomaly:
        if verbose:
            print("Detecting humans...")
        try:
            detection_result = detect_human_thermal(
                image_path=image_path,
                threshold=None,  # Use adaptive threshold
                visualize=False  # We'll visualize later
            )
            human_count = detection_result['human_count']
            bboxes = detection_result['bboxes']
            
            if verbose:
                print(f"Detected {human_count} humans")
        except Exception as e:
            print(f"Error in human detection: {e}")
    
    # Visualize the result
    plt.figure(figsize=(12, 8))
    
    # Create subplot for original image with detections
    plt.subplot(1, 2, 1)
    image_np = np.array(image)
    plt.imshow(image_np)
    plt.title("Original Image")
    plt.axis('off')
    
    # Draw bounding boxes for humans
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                        fill=False, edgecolor='red', linewidth=2))
    
    # Create subplot for heatmap if available
    plt.subplot(1, 2, 2)
    if anomaly_map is not None:
        anomaly_map = anomaly_map.squeeze().cpu().numpy()
        plt.imshow(anomaly_map, cmap='jet')
        plt.title("Anomaly Heatmap")
    else:
        plt.imshow(image_np, alpha=0.5)
        plt.title("No heatmap available")
    plt.axis('off')
    
    # Add overall title
    plt.suptitle(f"Anomaly Score: {anomaly_score:.4f} | Threshold: {threshold:.4f} | " + 
                 f"{'ANOMALY' if is_anomaly else 'NORMAL'} | " +
                 f"Humans: {human_count}", fontsize=14)
    
    plt.tight_layout()
    
    # Save or display the result
    if output_path:
        plt.savefig(output_path)
        if verbose:
            print(f"Result saved to {output_path}")
        plt.close()
    else:
        plt.show()
    
    return {
        'image_path': image_path,
        'anomaly_score': float(anomaly_score),
        'threshold': float(threshold),
        'is_anomaly': bool(is_anomaly),
        'human_count': human_count,
        'bboxes': bboxes
    }

def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Check if the image exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image not found: {args.image_path}")
        sys.exit(1)
    
    # Set device
    device_cfg = config['training'].get('device', 'cpu')
    if device_cfg == 'cuda' and not torch.cuda.is_available():
        print("CUDA requested but not available, using CPU instead.")
        device = 'cpu'
    else:
        device = device_cfg
    if args.verbose:
        print(f"Using device: {device}")
    
    # Get model path
    if args.model_path:
        model_path = args.model_path
    else:
        backbone_name = config['model']['backbone']
        if isinstance(backbone_name, list):
            # Ensemble - select the first model for simplicity
            backbone_name = backbone_name[0]
            print(f"Model is an ensemble. Using the first backbone: {backbone_name}")
        model_filename = config['model'].get('model_filename', f"patchcore_{backbone_name}.pth")
        model_path = os.path.join(config['output_path'], model_filename)
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found: {model_path}")
        sys.exit(1)
    
    # Get threshold
    if args.threshold is not None:
        threshold = args.threshold
    else:
        # Load threshold from file
        if args.threshold_file:
            threshold_file = args.threshold_file
        else:
            backbone_name = config['model']['backbone']
            if isinstance(backbone_name, list):
                backbone_name = 'ensemble'
            threshold_file = os.path.join(config['output_path'], f'optimal_threshold_{backbone_name}.json')
        threshold = load_threshold(threshold_file)
    
    # Initialize model
    model = PatchCore(
        backbone_name=config['model']['backbone'],
        layers=config['model']['layers'],
        coreset_sampling_ratio=config['model'].get('coreset_sampling_ratio', 0.1),
        device=device
    )
    
    if args.verbose:
        print(f"Loading model from {model_path}")
    model.load(model_path)
    
    # Get transforms
    transform = get_transforms(config, train=False)
    
    # Analyze the image
    result = analyze_image(
        image_path=args.image_path,
        model=model,
        transforms=transform,
        threshold=threshold,
        output_path=args.output_path,
        verbose=args.verbose
    )
    
    # Print results if verbose
    if args.verbose and result:
        print("\nAnalysis result:")
        for key, value in result.items():
            if key != 'bboxes':  # Skip printing all bounding boxes
                print(f"  {key}: {value}")

if __name__ == "__main__":
    main() 