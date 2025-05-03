# Thermal Drone Anomaly Detector Updates

## Latest Updates

We've made several significant improvements to the Thermal Drone Anomaly Detector application:

### 1. Pre-trained Model Loading

- Added support for loading pre-trained PatchCore models
- Application now looks for models at:
  - `outputs/patchcore_resnet18/patchcore_resnet18.pth`
  - `outputs/patchcore_resnet50/patchcore_resnet50.pth`
- Fixed model loading code to correctly utilize these trained models
- Added compatibility with PyTorch 2.6+ by handling the new weights_only security feature

### 2. Human Detection with Bounding Boxes

- Implemented human detection using HOG (Histogram of Oriented Gradients) with SVM
- Added automatic bounding box visualization for detected humans
- Enhanced priority classification to elevate images with human detections to "High" priority
- Modified visualization to make human bounding boxes clearly visible
- Added human detection count to reports and UI

### 3. Enhanced Reporting

- Updated CSV, HTML, and JSON reports to include human detection information
- Added color-coding in HTML reports for human detections
- Added human detection counts to summary statistics in all reports
- Improved terminal output to highlight images with human detections

### 4. Testing Utilities

- Created a comprehensive test script (`run_test.py`) that can:
  - Validate model file availability
  - Launch the Streamlit dashboard
  - Process directories with both ResNet18 and ResNet50 models
- Added a convenience shell script (`test.sh`) for easier command-line usage
- Enhanced error handling and user feedback

### 5. UI Improvements

- Added human detection information to the Streamlit dashboard
- Updated the details panel to show human detection count and confidence
- Added an expandable section to view details about each detected human
- Enhanced visualization with human bounding boxes

### 6. Compatibility Fixes

- Fixed compatibility issues with PyTorch 2.6+ by implementing multi-stage model loading
- Added three fallback methods for loading models with varying security levels
- Created a model converter tool (`convert_model.py`) to transform models between formats
- Fixed Streamlit hot-reloading issues when working with PyTorch classes
- Enhanced error handling with detailed messages and graceful fallbacks
- Improved resilience by allowing the application to continue with default models

## How to Use the New Features

### Human Detection

The human detection feature is automatically active. When an anomaly is detected, the system now:

1. Attempts to identify human shapes in the thermal image
2. Draws bounding boxes around potential humans
3. Displays the count of humans in the UI and reports
4. Elevates the priority level if humans are detected

### Using the Test Scripts

```bash
# To run the dashboard with model validation:
./test.sh --dashboard

# To process a directory of images with both models:
./test.sh --process --input /path/to/images --output /path/to/results --save

# To see all options:
./test.sh --help
```

### Model Conversion Tool

If you encounter issues with PyTorch 2.6+ compatibility, you can use the model conversion tool:

```bash
# Convert all models in batch mode (recommended)
python convert_model.py --batch

# Convert a specific model with a specific format
python convert_model.py --input model_path.pth --mode torch16
```

Conversion modes:
- `torch16`: PyTorch 1.6 format (most compatible across versions)
- `safe`: PyTorch safe format with allowlisted classes
- `pickle`: Basic pickle format (less secure but most compatible)

## Technical Notes

- The human detection utilizes OpenCV's HOG descriptor with default people detection
- The detection is more accurate on larger images; small images are automatically resized for better detection
- Bounding boxes are drawn in blue on the heatmap overlay
- The detection threshold is set to balance between false positives and false negatives
- Due to the nature of thermal imagery, human detection may not be as accurate as in visible light images
- The model loading uses a multi-stage approach to handle PyTorch 2.6's new security measures for loading saved models

## Future Work

- Train a dedicated human detector specifically for thermal imagery
- Implement tracking of detected humans across multiple frames
- Add distance estimation for detected humans
- Improve the bounding box placement using more advanced algorithms 