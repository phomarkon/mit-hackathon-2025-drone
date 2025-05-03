import os
import torch
import numpy as np
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, confusion_matrix, f1_score
import wandb
import yaml
import time
from datetime import datetime

from src.utils import load_config
from src.data_loader import get_dataloaders
from src.models.patchcore import PatchCore
from src.visualize import visualize_anomalies

torch.classes.__path__ = [] # Fix for Streamlit/PyTorch watcher conflict

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

def test(model, test_loader, config, output_dir, using_wandb=False, backbone_name=None):
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
    
    # Find best threshold using F1 score
    best_f1 = 0
    best_thresh = 0.5
    thresholds = np.linspace(np.min(anomaly_scores), np.max(anomaly_scores), 100)
    for t in thresholds:
        y_pred = (anomaly_scores >= t).astype(int)
        f1 = f1_score(gt_labels, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    # Compute confusion matrix at best threshold
    y_pred_best = (anomaly_scores >= best_thresh).astype(int)
    cm_best = confusion_matrix(gt_labels, y_pred_best)
    fig_cm_best, ax = plt.subplots()
    im = ax.imshow(cm_best, cmap='Blues')
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title(f'Confusion Matrix (Best F1 Threshold={best_thresh:.4f})')
    for i in range(cm_best.shape[0]):
        for j in range(cm_best.shape[1]):
            ax.text(j, i, cm_best[i, j], ha='center', va='center', color='black')
    fig_cm_best.colorbar(im)
    cm_best_path = os.path.join(output_dir, f'confusion_matrix_best_{backbone_name or "model"}.png')
    fig_cm_best.savefig(cm_best_path)
    plt.close(fig_cm_best)
    if using_wandb:
        wandb.log({
            f"test/confusion_matrix_best_{backbone_name or 'model'}": wandb.Image(cm_best_path),
            f"test/confusion_matrix_best_raw_{backbone_name or 'model'}": cm_best.tolist(),
            f"test/best_threshold_{backbone_name or 'model'}": best_thresh,
            f"test/best_f1_{backbone_name or 'model'}": best_f1,
        })
    return anomaly_scores, gt_labels, image_paths, best_thresh

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
    # Determine actions from config
    run_train = config['experiment'].get('train', False)
    run_test = config['experiment'].get('test', False)
    run_visualize = config['experiment'].get('visualize', False)
    # Model factory for different backbones
    def model_factory(backbone_name):
        return PatchCore(
            backbone_name=backbone_name,
            layers=config['model']['layers'],
            coreset_sampling_ratio=config['model'].get('coreset_sampling_ratio', 0.1),
            device=device
        )
    # Support for running a single backbone/config or all (ensemble)
    # PATCH: Only run the backbone specified in the config, unless explicitly requesting ensemble
    backbone_value = config['model']['backbone']
    if isinstance(backbone_value, str):
        backbone_list = [(backbone_value, config['model'].get('model_filename', f"patchcore_{backbone_value}.pth"))]
    elif isinstance(backbone_value, list):
        backbone_list = [(b, f"patchcore_{b}.pth") for b in backbone_value]
    else:
        raise ValueError("model.backbone must be a string (single backbone) or list (ensemble)")
    all_scores = []
    all_labels = None
    all_paths = None
    all_thresholds = []
    for backbone_name, model_filename in backbone_list:
        print(f"\n--- Running experiment for backbone: {backbone_name} ---")
        model = model_factory(backbone_name)
        model_path = os.path.join(output_dir, model_filename)
        if run_train:
            train(model, train_loader, config, output_dir, using_wandb)
        else:
            if os.path.exists(model_path):
                model.load(model_path)
            else:
                print(f"No model found at {model_path} and training is disabled. Cannot proceed.")
                if using_wandb: wandb.finish(exit_code=1)
                exit(1)
        if run_test:
            scores, labels, paths, best_thresh = test(model, test_loader, config, output_dir, using_wandb, backbone_name=backbone_name)
            all_scores.append(scores)
            all_thresholds.append(best_thresh)
            if all_labels is None:
                all_labels = labels
            if all_paths is None:
                all_paths = paths
    # Ensemble: average the anomaly scores from all models
    if len(all_scores) > 1:
        ensemble_scores = np.mean(np.stack(all_scores, axis=0), axis=0)
        # Find best threshold for ensemble
        best_f1_ens = 0
        best_thresh_ens = 0.5
        thresholds = np.linspace(np.min(ensemble_scores), np.max(ensemble_scores), 100)
        for t in thresholds:
            y_pred = (ensemble_scores >= t).astype(int)
            f1 = f1_score(all_labels, y_pred)
            if f1 > best_f1_ens:
                best_f1_ens = f1
                best_thresh_ens = t
        y_pred_ens = (ensemble_scores >= best_thresh_ens).astype(int)
        cm_ens = confusion_matrix(all_labels, y_pred_ens)
        fig_cm_ens, ax = plt.subplots()
        im = ax.imshow(cm_ens, cmap='Blues')
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        ax.set_title(f'Confusion Matrix (Ensemble Best F1 Threshold={best_thresh_ens:.4f})')
        for i in range(cm_ens.shape[0]):
            for j in range(cm_ens.shape[1]):
                ax.text(j, i, cm_ens[i, j], ha='center', va='center', color='black')
        fig_cm_ens.colorbar(im)
        cm_ens_path = os.path.join(output_dir, 'confusion_matrix_ensemble.png')
        fig_cm_ens.savefig(cm_ens_path)
        plt.close(fig_cm_ens)
        if using_wandb:
            wandb.log({
                "test/confusion_matrix_ensemble": wandb.Image(cm_ens_path),
                "test/confusion_matrix_ensemble_raw": cm_ens.tolist(),
                "test/best_threshold_ensemble": best_thresh_ens,
                "test/best_f1_ensemble": best_f1_ens,
            })
    # Visualization phase
    if run_visualize:
        try:
            run_visualization(config, output_dir, all_scores[-1], all_labels, all_paths, using_wandb)
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