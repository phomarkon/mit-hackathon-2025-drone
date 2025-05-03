# Anomaly Detection with PatchCore

This project implements an unsupervised anomaly detection system based on PatchCore for detecting anomalies in drone images. The implementation is designed for flexibility, reproducibility, and extensibility, supporting multiple CNN backbones and ensemble evaluation. It is fully integrated with Weights & Biases (wandb) for experiment tracking and visualization.

## Project Structure

```
├── configs/            # Configuration files (per-backbone and ensemble)
├── data/               # Dataset directory
│   ├── train/          # Normal training images
│   └── test/           # Test images
│       ├── normal/     # Normal test images
│       └── abnormal/   # Abnormal test images
├── outputs/            # Output directory for model and results
├── src/                # Source code
│   ├── data_loader.py  # Data loading and preprocessing
│   ├── main.py         # Main script for training, testing, and visualization
│   ├── models/         # Model implementations
│   │   ├── feature_extractor.py # Feature extraction backbone
│   │   └── patchcore.py        # PatchCore implementation
│   ├── utils.py        # Utility functions
│   └── visualize.py    # Visualization script
└── requirements.txt    # Dependencies
```

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Configuration

All settings are controlled via YAML config files in `configs/`. You can use:
- `patchcore_resnet18.yaml` for ResNet18
- `patchcore_resnet50.yaml` for ResNet50
- `patchcore_wideresnet50_2.yaml` for WideResNet50_2
- Or create an ensemble config with `model.backbone: ["resnet18", "resnet50", "wide_resnet50_2"]`

Example (ResNet50):
```yaml
data:
  train_path: "data/train"
  test_normal_path: "data/test/normal"
  test_abnormal_path: "data/test/abnormal"
  image_size: 224
model:
  name: "patchcore"
  backbone: "resnet50"
  layers: ["layer2", "layer3"]
  coreset_sampling_ratio: 0.1
  model_filename: "patchcore_resnet50.pth"
training:
  batch_size: 8
  device: "cpu"
  num_workers: 2
evaluation:
  batch_size: 4
output_path: "outputs/patchcore_resnet50"
experiment:
  run_name: "patchcore_resnet50"
  train: true
  test: true
  visualize: true
  vis_topk: 10
wandb:
  use_wandb: true
  project: "drone-anomaly-detection"
  entity: null
  tags: ["patchcore", "resnet50"]
```

## Usage

### Run a Single Backbone

```sh
python -m src.main --config configs/patchcore_resnet50.yaml --output_dir outputs/patchcore_resnet50
```

### Run All Backbones and Ensemble

Create a config with:
```yaml
model:
  backbone: ["resnet18", "resnet50", "wide_resnet50_2"]
```
Then run:
```sh
python -m src.main --config configs/patchcore.yaml --output_dir outputs/patchcore_ensemble
```

### Manual Control

You can also manually control which steps to run (if you add CLI flags):
```sh
python -m src.main --train --config configs/patchcore_resnet50.yaml
python -m src.main --test --config configs/patchcore_resnet50.yaml
python -m src.main --visualize --config configs/patchcore_resnet50.yaml
```

## Technical Implementation Details

### PatchCore Pipeline

1. **Feature Extraction**
   - Uses a pre-trained CNN backbone (ResNet18, ResNet50, or WideResNet50_2).
   - Extracts patch embeddings from specified layers (default: `layer2`, `layer3`).
   - Features are frozen (no fine-tuning).

2. **Memory Bank & Coreset Selection**
   - All patch embeddings from normal training images are collected.
   - A greedy coreset selection algorithm reduces the memory bank size (default: 10% of all patches).
   - The memory bank is saved for fast inference.

3. **Anomaly Scoring**
   - For each test image, patch embeddings are extracted.
   - Each patch is scored by its minimum L2 distance to the memory bank.
   - The image-level anomaly score is the maximum patch score in the image.

4. **Threshold Selection & Evaluation**
   - The ROC and Precision-Recall curves are computed.
   - The best threshold is automatically selected by maximizing the F1 score on the test set.
   - The confusion matrix at this optimal threshold is computed and logged.
   - AUROC, best F1, and the threshold are reported.

5. **Visualization & Logging**
   - All test images are visualized with their anomaly scores and ground truth labels.
   - Top anomalies are summarized in a grid.
   - All results, metrics, and images are logged to Weights & Biases (wandb).
   - Data leakage checks are performed and warnings are logged if any overlap is found between train and test sets.

6. **Ensemble Support**
   - If multiple backbones are specified, each is trained/tested independently.
   - Their anomaly scores are ensembled (by averaging) and the ensemble is evaluated with the same threshold selection and logging pipeline.

### Extensibility

- **Add a new backbone:**
  - Add the backbone to `feature_extractor.py` and create a config file.
  - Specify the backbone name and layers in the config.
- **Change coreset ratio or layers:**
  - Edit the config file for the desired experiment.
- **Add new metrics or logging:**
  - Extend `main.py` or `patchcore.py` as needed; all metrics are logged to wandb.

### Troubleshooting & Performance Tips

- **Coreset selection is slow:**
  - Try a smaller coreset ratio (e.g., 0.05) or use fewer training images for quick experiments.
- **Running on CPU:**
  - For large models or datasets, use a machine with a GPU and set `device: "cuda"` in your config.
- **Data leakage warning:**
  - The pipeline checks for overlap between train and test images and logs a warning if found.
- **wandb issues:**
  - If you have wandb login or sync issues, try `wandb login` or run with `use_wandb: false` in your config.

## License

This project is part of the MIT Hackathon 2025 Drone Challenge.