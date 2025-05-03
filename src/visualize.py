import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm
import argparse

from src.utils import load_config
from src.models.patchcore import PatchCore
from src.data_loader import get_transforms
from src.human_detection import detect_human_thermal  # Import our new human detection module

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize anomaly detection results')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--scores_path', type=str, default=None,
                        help='Path to saved anomaly scores file')
    parser.add_argument('--labels_path', type=str, default=None,
                        help='Path to saved ground truth labels file')
    parser.add_argument('--paths_file', type=str, default=None,
                        help='Path to saved image paths file')
    parser.add_argument('--topk', type=int, default=10,
                        help='Number of top anomalies to visualize')
    parser.add_argument('--output_dir', type=str, default='outputs/visualizations',
                        help='Output directory for visualizations')
    parser.add_argument('--detect_humans', action='store_true',
                        help='Enable human detection in thermal images')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Optimal threshold for anomaly detection')
    return parser.parse_args()

def visualize_anomalies(anomaly_scores, gt_labels, image_paths, config, topk=10, output_dir='outputs/visualizations', return_images=False, detect_humans=True, threshold=None):
    """Visualize top anomalies or all if topk >= len(anomaly_scores)."""
    os.makedirs(output_dir, exist_ok=True)
    n_images = len(anomaly_scores)
    
    # Try to load optimal threshold from file if not provided and human detection is enabled
    if detect_humans and threshold is None:
        # Check if optimal threshold file exists
        model_name = config['model'].get('backbone', 'model')
        if isinstance(model_name, list):
            model_name = 'ensemble'
        threshold_file = os.path.join(config['output_path'], f'optimal_threshold_{model_name}.json')
        if os.path.exists(threshold_file):
            try:
                import json
                with open(threshold_file, 'r') as f:
                    threshold_data = json.load(f)
                threshold = threshold_data.get('optimal_threshold')
                print(f"Loaded optimal threshold: {threshold:.4f} from {threshold_file}")
            except Exception as e:
                print(f"Error loading threshold from {threshold_file}: {e}")
    
    if topk >= n_images:
        indices = np.argsort(anomaly_scores)[::-1]
    else:
        indices = np.argsort(anomaly_scores)[::-1][:topk]
    
    # Create summary figure (up to 10 for display)
    n_summary = min(10, len(indices))
    plt.figure(figsize=(15, 10))
    individual_images = []
    
    for i, idx in enumerate(indices):
        image_path = image_paths[idx]
        score = anomaly_scores[idx]
        label = gt_labels[idx]
        label_str = "Anomaly" if label == 1 else "Normal"
        
        try:
            image = Image.open(image_path).convert('RGB')
            image_np = np.array(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            continue
        
        # Detect humans in thermal images if the image is an anomaly (or predicted as one)
        is_anomaly = (label == 1) or (threshold is not None and score >= threshold)
        human_count = 0
        bboxes = []
        
        if detect_humans and is_anomaly:
            try:
                # Detect humans in the thermal image
                detection_result = detect_human_thermal(
                    image_path=image_path,
                    threshold=None,  # Use adaptive threshold
                    visualize=False  # Don't visualize yet
                )
                human_count = detection_result['human_count']
                bboxes = detection_result['bboxes']
                print(f"Detected {human_count} humans in {os.path.basename(image_path)}")
            except Exception as e:
                print(f"Error in human detection for {image_path}: {e}")
        
        # Create figure for this individual image
        plt.figure(figsize=(8, 8))
        plt.imshow(image_np)
        
        # Draw bounding boxes for detected humans
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                            fill=False, edgecolor='red', linewidth=2))
        
        title = f"Anomaly Score: {score:.4f}\nGround Truth: {label_str}"
        if human_count > 0:
            title += f"\nHumans detected: {human_count}"
        title += f"\nImage: {os.path.basename(image_path)}"
        
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        
        # Save individual image
        individual_img_path = os.path.join(output_dir, f"anomaly_{i+1}_score_{score:.4f}_label_{label}.png")
        plt.savefig(individual_img_path)
        plt.close()
        
        if return_images:
            individual_images.append((individual_img_path, score, label))
        
        # Add to summary plot if within limit
        if i < n_summary:
            plt.figure(1)  # Switch back to the summary figure
            plt.subplot(2, 5, i + 1)
            plt.imshow(image_np)
            
            # Draw bounding boxes in the summary plot too
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                                fill=False, edgecolor='red', linewidth=2))
            
            summary_title = f"Score: {score:.4f}\nLabel: {label_str}"
            if human_count > 0:
                summary_title += f"\nHumans: {human_count}"
            plt.title(summary_title)
            plt.axis('off')
    
    # Save summary figure
    plt.figure(1)
    plt.tight_layout()
    summary_path = os.path.join(output_dir, "top_anomalies_summary.png")
    plt.savefig(summary_path)
    plt.close()
    print(f"Visualizations saved to {output_dir}")
    
    if return_images:
        return summary_path, individual_images
    return None

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    
    # Load results
    output_path = config['output_path']
    
    scores_path = args.scores_path or os.path.join(output_path, 'anomaly_scores.npy')
    labels_path = args.labels_path or os.path.join(output_path, 'ground_truth.npy')
    paths_file = args.paths_file or os.path.join(output_path, 'image_paths.txt')
    
    if not os.path.exists(scores_path) or not os.path.exists(labels_path) or not os.path.exists(paths_file):
        print("Results files not found. Run main.py with --test first.")
        exit(1)
    
    anomaly_scores = np.load(scores_path)
    gt_labels = np.load(labels_path)
    with open(paths_file, 'r') as f:
        image_paths = [line.strip() for line in f.readlines()]
    
    # Visualize top anomalies
    visualize_anomalies(
        anomaly_scores=anomaly_scores,
        gt_labels=gt_labels,
        image_paths=image_paths,
        config=config,
        topk=args.topk,
        output_dir=args.output_dir,
        detect_humans=args.detect_humans,
        threshold=args.threshold
    ) 