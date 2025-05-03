# Anomaly Detection with PatchCore

This project implements an unsupervised anomaly detection system based on PatchCore for detecting anomalies in drone images. The implementation follows the strategy outlined in the hackathon requirements, focusing on domain adaptation through transfer learning.

## Project Structure

```
├── configs/            # Configuration files
├── data/               # Dataset directory
│   ├── train/          # Normal training images
│   └── test/           # Test images
│       ├── normal/     # Normal test images
│       └── abnormal/   # Abnormal test images
├── outputs/            # Output directory for model and results
├── src/                # Source code
│   ├── data_loader.py  # Data loading and preprocessing
│   ├── main.py         # Main script for training and testing
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

## Usage

### Config-Driven Workflow

The easiest way to run the entire pipeline is using the configuration file:

```bash
python -m src.main --config configs/config.yaml
```

This will automatically run training, testing, and visualization according to the settings in the config file.

### Manual Control

You can also manually control which steps to run:

```bash
# Just training
python -m src.main --train

# Just testing
python -m src.main --test

# Just visualization
python -m src.main --visualize

# Combined operations
python -m src.main --train --test
```

### Disabling Weights & Biases

If you want to run without Weights & Biases logging:

```bash
python -m src.main --no_wandb
```

## Configuration

The model, dataset, and experiment settings can be configured in `configs/config.yaml`:

```yaml
# Data settings
data:
  train_path: "data/train"
  test_normal_path: "data/test/normal"
  test_abnormal_path: "data/test/abnormal"
  image_size: 224

# Model settings
model:
  name: "patchcore"
  backbone: "resnet18"
  layers: ["layer2", "layer3"]
  coreset_sampling_ratio: 0.1

# Experiment settings
experiment:
  run_name: "patchcore_baseline"
  train: true      # Whether to run training
  test: true       # Whether to run testing
  visualize: true  # Whether to run visualization

# Weights & Biases logging
wandb:
  use_wandb: true
  project: "drone-anomaly-detection"
  entity: null
  tags: ["patchcore", "baseline"]
```

## Implementation Details

### PatchCore Algorithm

1. **Feature Extraction**: Extracts patch embeddings from normal training images using a pre-trained CNN backbone (ResNet18 by default).
2. **Memory Bank**: Stores embeddings of normal patches, with optional coreset selection to reduce size.
3. **Anomaly Detection**: Computes distance from test image patches to memory bank; high distance indicates anomaly.

### Performance Metrics

The system evaluates detection performance using standard metrics:
- AUROC (Area Under Receiver Operating Characteristic)
- Precision-Recall curve

### Visualization

The visualization script creates images showing the top detected anomalies with their scores.

### Experiment Tracking with Weights & Biases

The implementation integrates with Weights & Biases (wandb) for experiment tracking:
- Logs metrics (AUROC, training time, etc.)
- Visualizes ROC curves and precision-recall curves
- Tracks anomaly score distributions
- Displays top anomalies with their scores

## Model Components

- **Feature Extractor**: Extracts features from pre-trained CNN backbone (transfer learning)
- **Memory Bank**: Efficiently stores normal patch embeddings
- **Coreset Selection**: Optional greedy sampling algorithm to reduce memory requirements
- **Nearest-Neighbor Search**: Computes distances from query patches to memory bank

## License

This project is part of the MIT Hackathon 2025 Drone Challenge.