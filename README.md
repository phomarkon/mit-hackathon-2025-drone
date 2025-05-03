# Thermal Drone Footage Anomaly Detector

An application for anomaly detection in thermal drone footage for search and rescue operations. This MVP prototype is designed to help identify missing persons in thermal drone footage by detecting and prioritizing anomalies in thermal imagery.

## Features

- **Interactive Dashboard**: User-friendly Streamlit interface for analyzing thermal images
- **Batch Processing**: Support for analyzing directories of thermal drone images
- **Anomaly Detection**: Identification of potential human presence using deep learning
- **Human Detection**: Automatic identification and bounding box display of potential humans in thermal images
- **Prioritization System**: Results sorted by confidence score to focus on most likely detections first
- **Visual Analysis**: Enhanced thermal heatmaps to highlight areas of interest
- **Detailed Reporting**: CSV, HTML, and JSON summary reports
- **Pre-trained Models**: Support for ResNet18 (faster) and ResNet50 (more accurate) pre-trained models

## Setup Instructions

### Prerequisites

- Python 3.8+ installed
- Git (for cloning the repository)

### Installation

1. Clone the repository or download the source code:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     source venv/bin/activate
     ```

4. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running the Application

### Recommended Method: Cross-Platform Python Script

The easiest way to run the application is using our cross-platform Python script:

```
python run_dashboard.py
```

This script will:
- Check if dependencies are installed
- Automatically install any missing packages
- Launch the Streamlit dashboard
- Work on any operating system (Windows, macOS, or Linux)

### Testing Script

We provide a comprehensive testing script that can be used to:
- Validate that model files are available
- Run the Streamlit dashboard
- Process a directory of images with both models

```
# Launch the dashboard interface
./test.sh --dashboard

# Process a directory of images with both models
./test.sh --process --input /path/to/images --output /path/to/results --save

# Show help
./test.sh --help
```

### Alternative Methods

#### Option 1: Streamlit Command

Run the Streamlit dashboard directly:

```
streamlit run dashboard.py
```

#### Option 2: Shell Scripts

- On macOS/Linux:
  ```
  ./run.sh
  ```

- On Windows:
  ```
  run.bat
  ```

#### Option 3: Command-Line Batch Processing

For batch processing of image directories:

```
python process_directory.py --input /path/to/images --output /path/to/results --save
```

Arguments:
- `--input` or `-i`: Directory containing thermal images (required)
- `--output` or `-o`: Directory to save results (optional)
- `--save` or `-s`: Save processed images with annotations (optional)
- `--no-summary`: Skip creating summary reports (optional)
- `--model` or `-m`: Select model backbone ('resnet18' or 'resnet50', default: 'resnet18')

## Usage Guide

### Analyzing Images with the Dashboard

1. Launch the Streamlit dashboard using one of the methods above
2. Select the desired model (ResNet18 or ResNet50) from the sidebar
3. Upload one or more thermal drone images
4. View the prioritized results table, sorted by anomaly confidence
5. Select any image from the table to view detailed analysis
6. View human detection bounding boxes and details if humans are detected
7. Use the "Simulate Alert" button to test the alert system

### Understanding the Results

Results are categorized into priority levels:

- **High Priority (Red)**: Strong anomaly detection with 80%+ confidence or human detected
- **Medium Priority (Orange)**: Moderate anomaly detection with 60-80% confidence
- **Low Priority (Yellow)**: Possible anomaly detection with <60% confidence
- **Normal (Green)**: No anomaly detected

Each result includes:
- The image filename
- Anomaly status (Normal or Anomaly)
- Confidence score (0-100%)
- Priority level based on confidence and human detection
- Number of humans detected (if any)
- Thermal heatmap visualization with human bounding boxes

### Batch Processing Output

When using the command-line tool, the following outputs are generated:

- **Terminal summary**: Overview of anomalies and humans detected with top priorities
- **CSV report**: Tabular data of all processed images with results
- **HTML report**: Interactive web report with color-coded priorities
- **JSON data**: Machine-readable format for integration with other systems
- **Annotated images**: Original images with heatmap overlays, human bounding boxes, and analysis text

## Technical Details

- **Anomaly Detection**: Uses pre-trained PatchCore models based on ResNet-18/50 to detect anomalies in thermal images
- **Human Detection**: Uses HOG (Histogram of Oriented Gradients) descriptor with SVM for human detection
- **Preprocessing**: Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance thermal contrast
- **Visual Analysis**: Generates heat map overlays to highlight potential areas of interest
- **Confidence Scoring**: Provides reliability metrics for each detection
- **Transfer Learning**: Leverages PatchCore architecture with pre-trained models

## Model Paths

The application expects pre-trained models at the following locations:
- ResNet18 model: `outputs/patchcore_resnet18/patchcore_resnet18.pth`
- ResNet50 model: `outputs/patchcore_resnet50/patchcore_resnet50.pth`

The test script will validate these paths before running.

## Troubleshooting

- **PyTorch Version Issues**: This application uses PyTorch 2.1.0 and torchvision 0.16.0. If you encounter issues with model loading, ensure you have these versions installed.
- **PyTorch 2.6+ Security**: Starting with PyTorch 2.6, there are new security measures for loading models. Our code handles this by either loading with `weights_only=False` or by using `torch.serialization.add_safe_globals()`.
- **CUDA Availability**: The application will use GPU acceleration if available, but will fall back to CPU if not.
- **Memory Issues**: If you encounter memory errors, try processing fewer images at once.
- **Model Loading Errors**: If the model fails to load, check that the model files exist at the expected paths.
- **Human Detection Issues**: The human detection is based on HOG descriptors and may not be as accurate on thermal images as on visible light images.

## Limitations (MVP Version)

- The prototype uses a pre-trained model that would benefit from fine-tuning on thermal drone imagery
- Thermal anomaly detection has inherent challenges with distinguishing human signatures from other heat sources
- HOG-based human detection may produce false positives on thermal images
- The thresholds for anomaly detection and priority levels may need adjustment based on specific deployment conditions

## Future Enhancements

- Fine-tuning the model on dedicated thermal drone datasets
- Improved human detection with thermal-specific models
- GPS/location data integration
- Alert system integration for rescue teams
- Real-time video stream processing
- Mobile application support