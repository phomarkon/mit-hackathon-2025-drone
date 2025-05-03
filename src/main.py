import os
import torch
import numpy as np
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, confusion_matrix
import wandb
import yaml
import time
from datetime import datetime

from src.utils import load_config
from src.data_loader import get_dataloaders
from src.models.patchcore import PatchCore
from src.visualize import visualize_anomalies

def parse_args():
    parser = argparse.ArgumentParser(description='Anomaly Detection Pipeline')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the model-specific config file (e.g., configs/patchcore.yaml)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Override output directory specified in config')
    return parser.parse_args()

def init_wandb(config, args):
    """Initialize wandb if enabled in config."""
    if config['wandb'].get('use_wandb', False):
        run_name = f"{config['experiment']['run_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Determine output directory
        output_dir = args.output_dir if args.output_dir else config['output_path']
        wandb_dir = os.path.join(output_dir, "wandb")
        os.makedirs(wandb_dir, exist_ok=True)

        # Initialize wandb
        try:
            wandb.init(
                project=config['wandb']['project'],
                entity=config['wandb']['entity'],
                name=run_name,
                config=config, # Log the entire config
                tags=config['wandb'].get('tags', []),
                dir=wandb_dir # Set wandb directory
            )
            print(f"Wandb initialized. Run name: {run_name}")
            # Log configuration details (redundant if config is passed above, but explicit)
            # wandb.config.update(config)
            return True
        except Exception as e:
            print(f"Error initializing wandb: {e}. Running without wandb.")
            return False
    else:
        print("Wandb logging disabled in config.")
        return False

def train(model, train_loader, config, output_dir, using_wandb=False):
    """Build memory bank using normal data."""
    print("--- Training (Building Memory Bank) ---")
    start_time = time.time()
    
    model.build_memory_bank(train_loader)
    
    train_time = time.time() - start_time
    print(f"Memory bank built in {train_time:.2f} seconds")
    
    # Save model
    model_filename = config['model'].get('model_filename', f"{config['model']['name']}_model.pth")
    model_path = os.path.join(output_dir, model_filename)
    model.save(model_path)
    print(f"Memory bank saved to {model_path}")
    
    # Log to wandb
    if using_wandb:
        wandb.log({
            "train/time": train_time,
            "train/memory_bank_size": len(model.memory_bank),
        })
        # Save model as artifact
        model_artifact = wandb.Artifact(f"{config['experiment']['run_name']}-model", type="model")
        model_artifact.add_file(model_path)
        wandb.log_artifact(model_artifact)

def test(model, test_loader, config, output_dir, using_wandb=False):
    """Test the model on normal and abnormal data."""
    print("--- Testing Model ---")
    start_time = time.time()
    
    anomaly_scores, gt_labels, image_paths = model.compute_anomaly_scores(test_loader)
    
    # Evaluate
    auroc, fpr, tpr, thresholds = model.evaluate(anomaly_scores, gt_labels)
    precision, recall, _ = precision_recall_curve(gt_labels, anomaly_scores)
    
    test_time = time.time() - start_time
    
    # Print results
    print(f"Test AUROC: {auroc:.4f}")
    print(f"Testing completed in {test_time:.2f} seconds")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    scores_path = os.path.join(output_dir, 'anomaly_scores.npy')
    labels_path = os.path.join(output_dir, 'ground_truth.npy')
    paths_file = os.path.join(output_dir, 'image_paths.txt')

    np.save(scores_path, anomaly_scores)
    np.save(labels_path, gt_labels)
    with open(paths_file, 'w') as f:
        for path in image_paths:
            f.write(f"{path}\n")
    
    # Create ROC curve and PR curve plots
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr)
    plt.title(f'ROC Curve (AUROC = {auroc:.4f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True)
    
    # Plot PR curve
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision)
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    
    plt.tight_layout()
    plots_path = os.path.join(output_dir, 'evaluation_curves.png')
    plt.savefig(plots_path)
    plt.close()
    print(f"Evaluation results saved to {output_dir}")
    
    # Compute confusion matrix at threshold 0.5
    threshold = 0.5
    y_pred = (anomaly_scores >= threshold).astype(int)
    cm = confusion_matrix(gt_labels, y_pred)
    fig_cm, ax = plt.subplots()
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title('Confusion Matrix (threshold=0.5)')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='black')
    fig_cm.colorbar(im)
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    fig_cm.savefig(cm_path)
    plt.close(fig_cm)
    
    # Log metrics to wandb
    if using_wandb:
        wandb.log({
            "test/auroc": auroc,
            "test/time": test_time,
            "test/evaluation_curves": wandb.Image(plots_path),
            "test/confusion_matrix": wandb.Image(cm_path),
            "test/confusion_matrix_raw": cm.tolist(),
        })
        
        # Log histogram of anomaly scores for normal vs abnormal
        normal_scores = anomaly_scores[gt_labels == 0]
        abnormal_scores = anomaly_scores[gt_labels == 1]
        
        wandb.log({
            "test/normal_score_histogram": wandb.Histogram(normal_scores),
            "test/abnormal_score_histogram": wandb.Histogram(abnormal_scores)
        })

        # Log results as artifact
        results_artifact = wandb.Artifact(f"{config['experiment']['run_name']}-results", type="results")
        results_artifact.add_file(scores_path)
        results_artifact.add_file(labels_path)
        results_artifact.add_file(paths_file)
        results_artifact.add_file(plots_path)
        results_artifact.add_file(cm_path)
        wandb.log_artifact(results_artifact)

    return anomaly_scores, gt_labels, image_paths

def run_visualization(config, output_dir, anomaly_scores=None, gt_labels=None, image_paths=None, using_wandb=False):
    """Run visualization if enabled."""
    print("--- Visualizing Results ---")
    
    # If no results are provided, load them from files
    if anomaly_scores is None or gt_labels is None or image_paths is None:
        scores_path = os.path.join(output_dir, 'anomaly_scores.npy')
        labels_path = os.path.join(output_dir, 'ground_truth.npy')
        paths_file = os.path.join(output_dir, 'image_paths.txt')
        
        if not os.path.exists(scores_path) or not os.path.exists(labels_path) or not os.path.exists(paths_file):
            print(f"Results files not found in {output_dir}. Skipping visualization.")
            return
        
        try:
            anomaly_scores = np.load(scores_path)
            gt_labels = np.load(labels_path)
            with open(paths_file, 'r') as f:
                image_paths = [line.strip() for line in f.readlines()]
        except Exception as e:
            print(f"Error loading results files: {e}. Skipping visualization.")
            return

    # Output directory for visualizations
    vis_output_dir = os.path.join(output_dir, 'visualizations')
    
    # Visualize top anomalies
    vis_results = visualize_anomalies(
        anomaly_scores=anomaly_scores,
        gt_labels=gt_labels,
        image_paths=image_paths,
        config=config,
        topk=len(anomaly_scores),  # Show all images
        output_dir=vis_output_dir,
        return_images=True  # Always return all images for wandb logging
    )
    
    # Log visualizations to wandb
    if using_wandb and vis_results:
        summary_path, individual_images = vis_results
        wandb.log({
            "visualizations/top_anomalies_summary": wandb.Image(summary_path),
        })
        vis_images_artifact = wandb.Artifact(f"{config['experiment']['run_name']}-visualizations", type="visualization")
        for i, (img_path, score, label) in enumerate(individual_images):
            caption = f"Score: {score:.4f}, Label: {'Anomaly' if label else 'Normal'}"
            wandb.log({
                f"visualizations/anomaly_{i+1}": wandb.Image(img_path, caption=caption)
            })
            vis_images_artifact.add_file(img_path, name=f"anomaly_{i+1}.png")
        vis_images_artifact.add_file(summary_path, name="top_anomalies_summary.png")
        wandb.log_artifact(vis_images_artifact)

def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Determine output directory
    output_dir = args.output_dir if args.output_dir else config['output_path']
    os.makedirs(output_dir, exist_ok=True)
    print(f"Using output directory: {output_dir}")

    # Initialize wandb (always attempt based on config)
    using_wandb = init_wandb(config, args)
    
    # Determine device
    device_cfg = config['training'].get('device', 'cpu')
    if device_cfg == 'cuda' and not torch.cuda.is_available():
        print("CUDA requested but not available, using CPU instead.")
        device = 'cpu'
    else:
        device = device_cfg
    print(f"Using device: {device}")
    
    # Get data loaders
    try:
        train_loader, test_loader = get_dataloaders(config)
        print(f"Train dataset size: {len(train_loader.dataset)}")
        print(f"Test dataset size: {len(test_loader.dataset)}")
    except RuntimeError as e:
        print(f"Error creating dataloaders: {e}")
        if using_wandb:
            wandb.finish(exit_code=1)
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during dataloader creation: {e}")
        if using_wandb:
            wandb.finish(exit_code=1)
        exit(1)

    # Initialize model
    # --- Model Selection Logic (Future Enhancement) ---
    # Currently hardcoded for PatchCore, but could be dynamic based on config['model']['name']
    if config['model']['name'].lower() == 'patchcore':
        model = PatchCore(
            backbone_name=config['model']['backbone'],
            layers=config['model']['layers'],
            coreset_sampling_ratio=config['model'].get('coreset_sampling_ratio', 0.1),
            device=device
        )
    else:
        print(f"Error: Model '{config['model']['name']}' not implemented.")
        if using_wandb: wandb.finish(exit_code=1)
        exit(1)
    # -------------------------------------------------

    # Determine actions from config
    run_train = config['experiment'].get('train', False)
    run_test = config['experiment'].get('test', False)
    run_visualize = config['experiment'].get('visualize', False)
    
    # Model path
    model_filename = config['model'].get('model_filename', f"{config['model']['name']}_model.pth")
    model_path = os.path.join(output_dir, model_filename)
    
    # Results storage
    anomaly_scores = None
    gt_labels = None
    image_paths = None
    
    # Training phase
    if run_train:
        try:
            train(model, train_loader, config, output_dir, using_wandb)
        except Exception as e:
            print(f"An error occurred during training: {e}")
            if using_wandb: wandb.finish(exit_code=1)
            exit(1)
    else:
        # Try to load existing model if not training
        if os.path.exists(model_path):
            try:
                model.load(model_path)
            except Exception as e:
                 print(f"Error loading model from {model_path}: {e}. Cannot proceed without training or a valid model.")
                 if using_wandb: wandb.finish(exit_code=1)
                 exit(1)
        else:
            print(f"No model found at {model_path} and training is disabled. Cannot proceed.")
            if using_wandb: wandb.finish(exit_code=1)
            exit(1)
    
    # Testing phase
    if run_test:
        try:
            anomaly_scores, gt_labels, image_paths = test(model, test_loader, config, output_dir, using_wandb)
        except Exception as e:
            print(f"An error occurred during testing: {e}")
            if using_wandb: wandb.finish(exit_code=1)
            # Allow visualization to proceed if scores exist
            if not os.path.exists(os.path.join(output_dir, 'anomaly_scores.npy')):
                 exit(1)

    # Visualization phase
    if run_visualize:
        try:
            run_visualization(config, output_dir, anomaly_scores, gt_labels, image_paths, using_wandb)
        except Exception as e:
             print(f"An error occurred during visualization: {e}")
             # Don't terminate the whole run for visualization error

    # Finish wandb run
    if using_wandb:
        wandb.finish()
    
    # Data leakage check: ensure no overlap between train and test image paths
    train_image_set = set(train_loader.dataset.image_paths)
    test_image_set = set(test_loader.dataset.image_paths)
    overlap = train_image_set & test_image_set
    if overlap:
        print(f"WARNING: Data leakage detected! {len(overlap)} overlapping images between train and test.")
        if using_wandb:
            wandb.alert(title="Data Leakage Detected", text=f"{len(overlap)} overlapping images between train and test.")
    else:
        print("No data leakage detected between train and test sets.")
    
    print("--- Pipeline Finished ---")

if __name__ == "__main__":
    main() 