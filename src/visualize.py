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
    return parser.parse_args()

def visualize_anomalies(anomaly_scores, gt_labels, image_paths, config, topk=10, output_dir='outputs/visualizations', return_images=False):
    """Visualize top anomalies."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get indices of top anomalies
    top_indices = np.argsort(anomaly_scores)[::-1][:topk]
    
    # Create summary figure
    plt.figure(figsize=(15, 10))
    
    # For returning image info if needed (for wandb)
    individual_images = []
    
    for i, idx in enumerate(top_indices):
        # Get image and its score
        image_path = image_paths[idx]
        score = anomaly_scores[idx]
        label = gt_labels[idx]
        label_str = "Anomaly" if label == 1 else "Normal"
        
        # Load image using PIL
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            continue
        
        # Add to plot
        plt.subplot(2, 5, i + 1)
        plt.imshow(image)
        plt.title(f"Score: {score:.4f}\nLabel: {label_str}")
        plt.axis('off')
        
        # Also save individual image
        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.title(f"Anomaly Score: {score:.4f}\nGround Truth: {label_str}\nImage: {os.path.basename(image_path)}")
        plt.axis('off')
        plt.tight_layout()
        
        # Save individual image
        individual_img_path = os.path.join(output_dir, f"anomaly_{i+1}_score_{score:.4f}_label_{label}.png")
        plt.savefig(individual_img_path)
        plt.close()
        
        # Add to list for returning if needed
        if return_images:
            individual_images.append((individual_img_path, score, label))
    
    # Save summary figure
    plt.tight_layout()
    summary_path = os.path.join(output_dir, "top_anomalies_summary.png")
    plt.savefig(summary_path)
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")
    
    # Return info for wandb if requested
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
        output_dir=args.output_dir
    ) 