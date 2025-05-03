import os
import sys
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights
from PIL import Image
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from io import BytesIO
import tempfile
import time
import yaml
from pathlib import Path

# Add src and model directories to system path to ensure imports work
package_path = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(package_path, 'src')
models_path = os.path.join(src_path, 'models')

if src_path not in sys.path:
    sys.path.append(src_path)
if models_path not in sys.path:
    sys.path.append(models_path)

# --- Streamlit/PyTorch Patch ---
try:
    import streamlit.watcher.local_sources_watcher as lsw
    from streamlit.logger import get_logger

    _LOGGER = get_logger(__name__)

    # Store original function if not already stored
    if not hasattr(st, '_original_get_module_paths'):
        if hasattr(lsw, 'get_module_paths'):
            st._original_get_module_paths = lsw.get_module_paths
            _LOGGER.info("Stored original Streamlit module path getter.")

            def patched_get_module_paths(module):
                # Check if the original function is available
                if not hasattr(st, '_original_get_module_paths') or st._original_get_module_paths is None:
                     _LOGGER.warning("Original get_module_paths not found for patching.")
                     return [] # Cannot proceed without original function

                # Attempt to get paths using the original function, but handle errors
                try:
                    # Specifically check for problematic torch._classes access
                    if module.__name__ == 'torch._classes':
                         _LOGGER.debug(f"Skipping problematic module: {module.__name__}")
                         return []

                    # Call the original function
                    return st._original_get_module_paths(module)

                except AttributeError as e:
                    # Handle cases where __path__ or similar attributes might be missing or malformed
                    _LOGGER.debug(f"AttributeError while getting paths for {module.__name__}: {e}. Skipping.")
                    return []
                except RuntimeError as e:
                     # Specifically catch RuntimeErrors like the one observed
                     _LOGGER.debug(f"RuntimeError while getting paths for {module.__name__}: {e}. Skipping.")
                     return []
                except Exception as e:
                    # Catch any other unexpected errors during path extraction
                    _LOGGER.warning(f"Unexpected error getting paths for {module.__name__}: {e}. Skipping.")
                    return []

            # Apply the patch only if the original function was successfully stored
            lsw.get_module_paths = patched_get_module_paths
            _LOGGER.info("Applied Streamlit hot-reload patch for PyTorch compatibility.")
        else:
             _LOGGER.warning("Streamlit's get_module_paths not found. Cannot apply patch.")
    else:
         _LOGGER.info("Streamlit patch already applied or original function stored.")

except ImportError:
    _LOGGER.warning("Could not import Streamlit watcher. Patch not applied.")
except Exception as patch_error:
    _LOGGER.error(f"Failed to apply Streamlit hot-reload patch: {patch_error}")
# --- End Patch ---

torch.classes.__path__ = [] # Fix for Streamlit/PyTorch watcher conflict

# Set page config
st.set_page_config(
    page_title="DroneGuardian Prime: Thermal Anomaly Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .anomaly-high {
        color: #ff0000;
        font-weight: bold;
    }
    .anomaly-medium {
        color: #ff6600;
        font-weight: bold;
    }
    .anomaly-low {
        color: #ffcc00;
        font-weight: bold;
    }
    .normal {
        color: #00cc00;
    }
    .reportview-container .main .block-container {
        padding-top: 1rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 1rem;
    }
    .stProgress > div > div > div > div {
        background-color: #ff4b4b;
    }
    .custom-title {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    /* Grid view CSS */
    .image-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }
    .image-card {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 5px;
        width: 200px;
        margin-bottom: 10px;
        position: relative;
    }
    .image-card img {
        width: 100%;
        border-radius: 3px;
    }
    .image-card-title {
        text-align: center;
        padding: 5px;
        font-size: 12px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .anomaly-badge {
        position: absolute;
        top: 10px;
        right: 10px;
        background-color: #ff4b4b;
        color: white;
        border-radius: 12px;
        padding: 3px 10px;
        font-size: 12px;
        font-weight: bold;
    }
    .normal-badge {
        position: absolute;
        top: 10px;
        right: 10px;
        background-color: #00cc00;
        color: white;
        border-radius: 12px;
        padding: 3px 10px;
        font-size: 12px;
        font-weight: bold;
    }
    .notification-bar {
        background-color: #333;
        color: white;
        padding: 10px;
        margin-bottom: 20px;
        border-radius: 5px;
        font-weight: bold;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .notification-alert {
        background-color: #ff4b4b;
    }
    .dev-log {
        background-color: #333;
        color: #ddd;
        padding: 10px;
        margin: 5px 0;
        border-radius: 3px;
        font-family: monospace;
        font-size: 12px;
    }
    .dev-log-success {
        border-left: 3px solid #00cc00;
    }
    .dev-log-info {
        border-left: 3px solid #2196F3;
    }
    .dev-log-warning {
        border-left: 3px solid #ff9800;
    }
    .dev-log-error {
        border-left: 3px solid #ff4b4b;
    }
</style>
""", unsafe_allow_html=True)

def load_config(config_path):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        st.error(f"Error loading config from {config_path}: {e}")
        return None

# Create a cached function for loading the model outside the class
@st.cache_resource
def load_model_cached(_backbone_name="resnet18"):
    """Load and cache the trained model based on backbone name"""
    try:
        import torch
        import torch.nn as nn
        import os
        import sys
        import numpy as np
        from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights
        
        # Add src directory to path if not already there (needed for importing PatchCore)
        src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
        if src_path not in sys.path:
            sys.path.append(src_path)
            st.info(f"Added {src_path} to Python path")
        
        try:
            # First, try to import PatchCore directly and use its native loading mechanism
            import sys
            
            # Make sure necessary components are in the path
            project_root = Path(__file__).parent
            src_path = str(project_root / "src")
            models_path = str(project_root / "src" / "models")
            
            if src_path not in sys.path:
                sys.path.insert(0, src_path)
                st.info(f"Added {src_path} to Python path")
                
            if models_path not in sys.path:
                sys.path.insert(0, models_path)
                st.info(f"Added {models_path} to Python path")
                
            try:
                # Import the feature extractor first as it's needed by PatchCore
                from src.models.feature_extractor import FeatureExtractor
                st.info("Successfully imported FeatureExtractor")
            except ImportError as fe_error:
                st.warning(f"Could not import FeatureExtractor: {fe_error}")
                # Create a simplified version of FeatureExtractor for PatchCore to use
                class FeatureExtractor:
                    def __init__(self, backbone_name, layers):
                        self.backbone_name = backbone_name
                        self.layers = layers
                    def __call__(self, x):
                        return {layer: torch.randn(x.shape[0], 64, 56, 56) for layer in self.layers}
                # Monkey patch the module
                import sys
                import types
                feature_extractor_module = types.ModuleType('feature_extractor')
                feature_extractor_module.FeatureExtractor = FeatureExtractor
                sys.modules['src.models.feature_extractor'] = feature_extractor_module
                st.warning("Created a mock FeatureExtractor as fallback")
                
            # Now import PatchCore (which depends on FeatureExtractor)
            from src.models.patchcore import PatchCore
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Define model path based on backbone
            model_path = f"outputs/patchcore_{_backbone_name}/patchcore_{_backbone_name}.pth"
            
            # Create the PatchCore model and load it directly
            st.info(f"Attempting to load PatchCore model from {model_path}")
            # Default layers used in training
            layers = ["layer2", "layer3"]  
            
            # Initialize the model
            model = PatchCore(
                backbone_name=_backbone_name,
                layers=layers,
                device=device
            )
            
            # Use PatchCore's native load method
            if os.path.exists(model_path):
                try:
                    # First attempt: Add safe globals and load
                    import torch.serialization
                    safe_info("Adding numpy._core.multiarray._reconstruct to safe globals for model loading")
                    torch.serialization.add_safe_globals(['numpy._core.multiarray._reconstruct'])
                    model.load(model_path)
                    safe_success(f"Successfully loaded PatchCore model from {model_path} with safe globals")
                    return model
                except Exception as e1:
                    safe_warning(f"Loading with safe globals failed: {e1}")
                    try:
                        # Second attempt: Fall back to less secure method
                        safe_warning("Falling back to less secure weights_only=False loading method")
                        
                        # Override PatchCore's load method temporarily with our custom implementation
                        original_model_path = model_path
                        def custom_load(self, path):
                            """Custom load method that uses weights_only=False"""
                            data = torch.load(path, map_location=self.device, weights_only=False)
                            self.memory_bank = data['memory_bank']
                            self.patch_grid_size = data.get('patch_grid_size', None)
                            self.backbone_name = data['backbone_name']
                            self.layers = data['layers']
                            self.coreset_sampling_ratio = data['coreset_sampling_ratio']
                            
                        # Replace the method temporarily
                        import types
                        model.load = types.MethodType(custom_load, model)
                        
                        # Use our custom implementation
                        model.load(original_model_path)
                        safe_success(f"Successfully loaded PatchCore model using weights_only=False")
                        return model
                    except Exception as e2:
                        safe_error(f"All loading methods failed: {e2}")
                        # Return uninitialized model as fallback
                        return model
            else:
                st.warning(f"Model file not found at {model_path}")
                # Fall back to default initialized model
                return model
                
        except ImportError as ie:
            st.warning(f"Could not import PatchCore module: {ie}. Falling back to ResNet model.")
            # If PatchCore can't be imported, fall back to standard ResNet approach
            # This is just a fallback and won't work as well
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Define model path based on backbone
            model_path = f"outputs/patchcore_{_backbone_name}/patchcore_{_backbone_name}.pth"
            
            # Initialize the model based on backbone
            if _backbone_name == "resnet18":
                model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            elif _backbone_name == "resnet50":
                model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            else:
                raise ValueError(f"Unsupported backbone: {_backbone_name}")
                
            # Modify for anomaly detection
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Linear(num_ftrs, 512),
                nn.ReLU(),
                nn.Linear(512, 1),
                nn.Sigmoid()
            )
            
            st.warning(f"Using default ImageNet weights for {_backbone_name}. This is a fallback solution.")
            model = model.to(device)
            model.eval()
            
            return model
            
    except Exception as e:
        st.error(f"Critical error loading model: {e}")
        raise

# Initialize session state for dev mode if not already set
if "dev_mode" not in st.session_state:
    st.session_state["dev_mode"] = False

# Initialize session state for results caching
if "processed_results" not in st.session_state:
    st.session_state["processed_results"] = None

# Initialize session state for threshold
if "anomaly_threshold" not in st.session_state:
    st.session_state["anomaly_threshold"] = 0.5  # Default threshold

class AnomalyDetector:
    def __init__(self, backbone="resnet18"):
        """Initialize the anomaly detector"""
        import torch
        self.backbone = backbone
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            # Load the model
            self.model = load_model_cached(self.backbone)
            
            # Check if we have a PatchCore model
            self.is_patchcore = hasattr(self.model, 'memory_bank')
            
            if not self.is_patchcore:
                # Define the transform pipeline for standard ResNet models
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                # Define transform for PatchCore that matches what it expects
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                # Log that we're using PatchCore
                if st.session_state.get("dev_mode", False):
                    st.success(f"Using PatchCore model with {len(self.model.memory_bank)} stored patches")
            
            # Load human detection model (HOG descriptor)
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            
            # Use session state threshold instead of hardcoded value
            self.threshold = st.session_state.get("anomaly_threshold", 0.5)
        except Exception as e:
            st.error(f"Failed to initialize detector: {e}")
            raise
    
    def predict(self, image):
        """
        Predict anomaly in an image and detect humans with bounding boxes
        
        Args:
            image: PIL Image or path to image
        
        Returns:
            dict with prediction results
        """
        import torch
        import numpy as np
        import cv2
        import PIL
        
        # Check if image is a path or PIL Image
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image file not found: {image}")
            image = PIL.Image.open(image)
            
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Different handling depending on model type
        if self.is_patchcore:
            # Custom wrapper method to safely call PatchCore
            def safe_compute_anomaly_scores(model, image_tensor):
                """Safe wrapper around PatchCore's compute_anomaly_scores method"""
                try:
                    # Method 1: Using TensorDataset and DataLoader
                    from torch.utils.data import TensorDataset, DataLoader
                    
                    # Create tensor dataset with numeric placeholders
                    label_tensor = torch.tensor([0])
                    path_tensor = torch.tensor([0]) 
                    dataset = TensorDataset(image_tensor, label_tensor, path_tensor)
                    dataloader = DataLoader(dataset, batch_size=1)
                    
                    anomaly_scores, _, _ = model.compute_anomaly_scores(dataloader)
                    return anomaly_scores
                except Exception as e1:
                    st.warning(f"Primary scoring method failed: {e1}. Trying alternative...")
                    
                    # Method 2: Direct feature extraction fallback if Dataset approach fails
                    try:
                        # Extract features manually if possible
                        with torch.no_grad():
                            features = model.feature_extractor(image_tensor)
                            patch_features, _ = model._embed_patches(features)
                            patch_features = patch_features.cpu().numpy()
                            
                            # Calculate distances to memory bank
                            distances = []
                            for patch in patch_features:
                                patch_distances = np.linalg.norm(model.memory_bank - patch.reshape(1, -1), axis=1)
                                distances.append(np.min(patch_distances))
                            
                            # Take maximum distance as anomaly score (same as in PatchCore)
                            return np.array([np.max(distances)])
                    except Exception as e2:
                        st.error(f"Both scoring methods failed. Fallback to default score. Errors: {e1}, then {e2}")
                        return np.array([0.5])  # Default mid-point score
            
            # Transform the image
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Use safe wrapper for anomaly score computation
            anomaly_scores = safe_compute_anomaly_scores(self.model, image_tensor)
            
            # Handle result based on type
            if isinstance(anomaly_scores, list):
                if len(anomaly_scores) > 0:
                    score = anomaly_scores[0]
                else:
                    score = 0.5  # Default to mid-point
            elif isinstance(anomaly_scores, np.ndarray):
                if anomaly_scores.size > 0:
                    score = anomaly_scores[0]
                else:
                    score = 0.5  # Default to mid-point
            else:
                score = float(anomaly_scores)  # Try to convert to float
        else:
            # For standard ResNet models
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
            'enhanced_img': enhanced,
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


def get_priority_color(priority):
    """Get CSS class for priority coloring"""
    if priority == "High":
        return "anomaly-high"
    elif priority == "Medium":
        return "anomaly-medium"
    elif priority == "Low":
        return "anomaly-low"
    else:
        return "normal"


def process_uploaded_files(uploaded_files, detector, progress_bar):
    """Process multiple uploaded files"""
    results = []
    total_files = len(uploaded_files)
    
    for i, file in enumerate(uploaded_files):
        # Create a temporary file to save the uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # Process the image
            safe_info(f"Processing {file.name}...")
            image = Image.open(tmp_path)
            
            # Debug info about the image
            safe_info(f"Image info: size={image.size}, mode={image.mode}")
            
            # Process with detector and add error handling
            try:
                result = detector.predict(image)
                
                # Verify result is a valid dictionary
                if not isinstance(result, dict):
                    raise TypeError(f"Expected dictionary result, got {type(result)}")
                
                # Add metadata
                result['filename'] = file.name
                result['file_path'] = tmp_path
                
                # Add priority classification
                result['priority'] = get_priority_label(result['confidence'], result['is_anomaly'], result['humans_detected'])
                
                results.append(result)
                safe_success(f"Successfully processed {file.name}")
            except Exception as prediction_error:
                # Handle prediction errors separately for clarity
                safe_error(f"Error during prediction for {file.name}: {prediction_error}")
                if st.session_state.get("dev_mode", False):
                    import traceback
                    safe_error(traceback.format_exc())
            
            # Update progress bar
            progress_bar.progress((i + 1) / total_files)
            
        except Exception as e:
            safe_error(f"Error opening or processing {file.name}: {e}")
            if st.session_state.get("dev_mode", False):
                import traceback
                safe_error(traceback.format_exc())
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except Exception as cleanup_error:
                safe_warning(f"Could not remove temp file {tmp_path}: {cleanup_error}")
    
    # Sort results by confidence (prioritize high confidence anomalies)
    if results:
        results.sort(key=lambda x: x['confidence'] if x['is_anomaly'] else 0, reverse=True)
        safe_success(f"Successfully processed {len(results)} of {total_files} images")
    else:
        safe_warning("No images were successfully processed")
    return results


def display_results_table(results):
    """Display results in a sortable table"""
    # Convert results to a DataFrame for easy display
    df = pd.DataFrame([
        {
            'Filename': r['filename'],
            'Status': 'ANOMALY' if r['is_anomaly'] else 'NORMAL',
            'Confidence': f"{r['confidence']:.1f}%",
            'Priority': r['priority'],
            'Humans': r.get('humans_detected', 0),
            'Raw Score': f"{r['raw_score']:.4f}",
            'Index': i  # For selection
        }
        for i, r in enumerate(results)
    ])
    
    # Apply conditional formatting
    def highlight_anomalies(row):
        priority = row['Priority']
        color_class = get_priority_color(priority)
        # Only style certain columns with html spans
        styled_row = row.copy()
        styled_row['Status'] = f'<span class="{color_class}">{styled_row["Status"]}</span>'
        styled_row['Priority'] = f'<span class="{color_class}">{styled_row["Priority"]}</span>'
        return styled_row
    
    # Apply styling via a custom format function
    styled_df = df.copy()
    for i, row in df.iterrows():
        priority = row['Priority']
        color_class = get_priority_color(priority)
        styled_df.at[i, 'Status'] = f'<span class="{color_class}">{row["Status"]}</span>'
        styled_df.at[i, 'Priority'] = f'<span class="{color_class}">{row["Priority"]}</span>'
    
    # Display table using the styled dataframe
    st.write(styled_df.to_html(escape=False, index=False), unsafe_allow_html=True)
    
    # Return the original dataframe for interaction
    return df


def display_image_details(result):
    """Display detailed information for a single image"""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Display the image with heatmap overlay
        st.image(result['heatmap_img'], use_container_width=True)
        
    with col2:
        # Display anomaly details
        st.subheader("Anomaly Analysis")
        status = "ANOMALY DETECTED" if result['is_anomaly'] else "NORMAL (No Anomaly)"
        priority_color = get_priority_color(result['priority'])
        
        st.markdown(f"**Status:** <span class='{priority_color}'>{status}</span>", unsafe_allow_html=True)
        st.markdown(f"**Confidence:** {result['confidence']:.1f}%")
        st.markdown(f"**Priority:** <span class='{priority_color}'>{result['priority']}</span>", unsafe_allow_html=True)
        
        # Display human detection info
        humans_detected = result.get('humans_detected', 0)
        if humans_detected > 0:
            st.markdown(f"**Humans Detected:** <span class='anomaly-high'>{humans_detected}</span>", unsafe_allow_html=True)
            
            # Show bounding box details if available
            if 'human_boxes' in result and result['human_boxes']:
                with st.expander("Human Detection Details"):
                    for i, human in enumerate(result['human_boxes']):
                        confidence = human.get('confidence', 0) * 100
                        st.markdown(f"**Human #{i+1}:** Confidence: {confidence:.1f}%")
        else:
            st.markdown("**Humans Detected:** 0")
        
        # Display a gauge chart for the confidence score
        fig = px.pie(values=[result['confidence'], 100-result['confidence']], 
                    names=['Confidence', ''], 
                    hole=0.7,
                    color_discrete_sequence=['#ff4b4b' if result['is_anomaly'] else '#00cc00', '#e1e1e1'])
        fig.update_layout(
            annotations=[dict(text=f"{result['confidence']:.1f}%", x=0.5, y=0.5, font_size=20, showarrow=False)],
            showlegend=False,
            margin=dict(t=0, b=0, l=0, r=0),
            height=200
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Add action buttons
        if st.button("Simulate Alert 🚨", key="alert_button"):
            st.success("Alert would be sent to rescue team!")
            
        if st.button("Mark as Reviewed ✓", key="review_button"):
            st.info("Image marked as reviewed in the system.")


def main():
    try:
        # Sidebar
        st.sidebar.title("DroneGuardian Prime")
        st.sidebar.markdown("---")
        
        # Add dev mode toggle in sidebar
        st.sidebar.markdown("### Settings")
        dev_mode = st.sidebar.checkbox("Developer Mode", value=st.session_state.get("dev_mode", False))
        st.session_state["dev_mode"] = dev_mode
        
        if dev_mode:
            st.sidebar.info("Developer mode enabled. Debug logs will be visible.")
        
        # Remove any existing streamlit elements if we're toggling dev mode
        # This is a workaround to clear previous logs
        if "prev_dev_mode" in st.session_state and st.session_state["prev_dev_mode"] != dev_mode:
            st.experimental_rerun()
        st.session_state["prev_dev_mode"] = dev_mode
        
        # View mode selection
        view_mode = st.sidebar.radio("View Mode", ["Grid", "List"])
        st.session_state["view_mode"] = view_mode
        
        # Add threshold adjustment slider
        st.sidebar.markdown("### Anomaly Threshold")
        anomaly_threshold = st.sidebar.slider(
            "Adjust threshold:",
            min_value=0.1,
            max_value=10.0,
            value=st.session_state.get("anomaly_threshold", 0.5),
            step=0.1,
            help="Lower values mark more images as normal. Higher values mark more as anomalous."
        )
        # Update threshold in session state
        if anomaly_threshold != st.session_state.get("anomaly_threshold"):
            st.session_state["anomaly_threshold"] = anomaly_threshold
            # Clear cached results if threshold changed
            st.session_state["processed_results"] = None
        
        # Choose model configuration
        st.sidebar.markdown("### Model Configuration")
        model_options = {
            "ResNet-18 (Faster)": "resnet18",
            "ResNet-50 (More Accurate)": "resnet50"
        }
        
        selected_model = st.sidebar.radio(
            "Select backbone model:", 
            list(model_options.keys()),
            help="ResNet-18 is faster but less accurate. ResNet-50 is more accurate but slower."
        )
        
        backbone_name = model_options[selected_model]
        
        # Display model details
        st.sidebar.markdown(f"**Selected Model:** {selected_model}")
        
        # Try to load matching config file
        config_path = f"configs/patchcore_{backbone_name}.yaml"
        if os.path.exists(config_path):
            config = load_config(config_path)
            if config:
                st.sidebar.markdown("**Configuration loaded from:**")
                st.sidebar.code(config_path, language="yaml")
        
        st.sidebar.markdown("---")
        
        # Initialize the anomaly detector with selected backbone
        with st.spinner(f"Loading {selected_model} model..."):
            detector = AnomalyDetector(backbone=backbone_name)
        
        # Main interface
        st.markdown("<h1 style='text-align: center;'>DroneGuardian Prime: Thermal Anomaly Detection</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Upload thermal drone images to detect potential human presence and prioritize findings</p>", unsafe_allow_html=True)
        
        # Info about selected model
        st.info(f"Using {selected_model} for anomaly detection. You can change the model in the sidebar.")
        
        # File uploader
        uploaded_files = st.file_uploader("Upload thermal drone images", type=['jpg', 'jpeg', 'png', 'tif', 'tiff'], accept_multiple_files=True)
        
        if uploaded_files:
            # Check if we have cached results and if inputs match
            cache_valid = False
            if st.session_state["processed_results"] is not None:
                cached_files = st.session_state.get("cached_filenames", [])
                current_files = [f.name for f in uploaded_files]
                if set(current_files) == set(cached_files) and st.session_state.get("cached_backbone") == backbone_name:
                    cache_valid = True
            
            # Only process if cache is invalid
            if not cache_valid:
                st.markdown("---")
                st.markdown("<div class='custom-title'>Processing Images</div>", unsafe_allow_html=True)
                
                # Initialize progress bar
                progress_bar = st.progress(0)
                
                # Process all images
                results = process_uploaded_files(uploaded_files, detector, progress_bar)
                
                # Cache the results
                st.session_state["processed_results"] = results
                st.session_state["cached_filenames"] = [f.name for f in uploaded_files]
                st.session_state["cached_backbone"] = backbone_name
                
                # After processing is complete
                progress_bar.progress(1.0)
                time.sleep(0.5)  # Small delay for UI feedback
                progress_bar.empty()  # Clear the progress bar
            else:
                # Use cached results
                results = st.session_state["processed_results"]
                if st.session_state.get("dev_mode", False):
                    st.success("Using cached results - no reprocessing needed")
            
            # Apply threshold to cached results
            for result in results:
                result['is_anomaly'] = result['raw_score'] > anomaly_threshold
                result['confidence'] = result['raw_score'] * 100 if result['is_anomaly'] else (1 - result['raw_score']) * 100
                # Update priority based on new anomaly status
                result['priority'] = get_priority_label(result['confidence'], result['is_anomaly'], result['humans_detected'])
            
            # Display summary
            col1, col2, col3 = st.columns(3)
            with col1:
                total = len(results)
                st.metric("Total Images", total)
            with col2:
                anomalies = sum(1 for r in results if r['is_anomaly'])
                st.metric("Anomalies Detected", anomalies)
            with col3:
                anomaly_percentage = (anomalies / total * 100) if total > 0 else 0
                st.metric("Anomaly Rate", f"{anomaly_percentage:.1f}%")
            
            # Display results based on view mode
            st.markdown("<div class='custom-title'>Analysis Results</div>", unsafe_allow_html=True)
            
            if st.session_state.get("view_mode") == "Grid":
                display_results_grid(results)
            else:
                # Fall back to the table view for List mode
                st.markdown("Images are sorted by anomaly confidence, with highest priority items at the top.")
                df = display_results_table(results)
                
                # Image selection and detailed view
                st.markdown("---")
                st.markdown("<div class='custom-title'>Detailed Analysis</div>", unsafe_allow_html=True)
                
                # Let user select which image to view in detail
                if len(results) > 0:
                    option_list = df['Filename'].tolist()
                    selected_filename = st.selectbox("Select an image to view details:", option_list)
                    
                    # Find the corresponding result
                    selected_index = df[df['Filename'] == selected_filename]['Index'].values[0]
                    selected_result = results[selected_index]
                    
                    # Display the details for the selected image
                    display_image_details(selected_result)
                
        else:
            # Display instructions when no files are uploaded
            st.info("Please upload one or more thermal drone images to begin analysis.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### How it works")
                st.markdown("""
                1. Upload thermal drone images using the file uploader above
                2. Our AI model will analyze each image for potential human presence
                3. Results will be displayed in a prioritized list
                4. Select any image from the list to view detailed analysis
                5. Adjust the anomaly threshold in the sidebar to fine-tune detection
                """)
            
            with col2:
                st.markdown("### Features")
                st.markdown("""
                - **Anomaly Detection**: Identifies potential human presence in thermal imagery
                - **Prioritization**: Sorts findings by confidence to focus on most likely detections first
                - **Thermal Heatmaps**: Visualizes areas of interest with enhanced thermal overlays
                - **Confidence Scoring**: Provides reliability metrics for each detection
                - **Adjustable Threshold**: Fine-tune the sensitivity of anomaly detection
                """)
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        if dev_mode:
            import traceback
            st.code(traceback.format_exc(), language="python")
        else:
            st.write("Please try refreshing the page or check the logs for more details.")
            # Show dev mode toggle for error situations
            if st.button("Show Developer Details"):
                st.session_state["dev_mode"] = True
                st.experimental_rerun()

# Create a function for developer logs
def dev_log(message, log_type="info"):
    """Display a developer log message only if dev mode is enabled"""
    if st.session_state.get("dev_mode", False):
        css_class = f"dev-log dev-log-{log_type}"
        st.markdown(f'<div class="{css_class}">{message}</div>', unsafe_allow_html=True)

# Override Streamlit's st.info, st.success, st.warning, st.error functions to use dev_log in dev mode
def safe_info(message):
    """Safe wrapper for st.info that respects dev mode"""
    if st.session_state.get("dev_mode", False):
        dev_log(message, "info")
    else:
        # Don't show ANY info in user mode
        return

def safe_success(message):
    """Safe wrapper for st.success that respects dev mode"""
    if st.session_state.get("dev_mode", False):
        dev_log(message, "success")
    else:
        # Don't show ANY success in user mode
        return

def safe_warning(message):
    """Safe wrapper for st.warning that respects dev mode"""
    if st.session_state.get("dev_mode", False):
        dev_log(message, "warning")
    else:
        # Don't show ANY warnings in user mode
        return

def safe_error(message):
    """Safe wrapper for st.error that respects dev mode except for critical errors"""
    if st.session_state.get("dev_mode", False):
        dev_log(message, "error")
    else:
        # Only show critical errors in user mode
        if "critical error" in message.lower():
            st.error(message)
        else:
            # Don't show non-critical errors
            return

# Function to render image grid
def render_image_grid(results):
    """Render results as a grid of images with badges"""
    if not results:
        return None
        
    columns = 3  # Number of columns in the grid
    rows = (len(results) + columns - 1) // columns  # Calculate rows needed
    
    # Create grid using Streamlit columns
    selected_result = None
    
    for row in range(rows):
        cols = st.columns(columns)
        for col in range(columns):
            idx = row * columns + col
            if idx < len(results):
                result = results[idx]
                filename = result.get('filename', f'image_{idx}')
                is_anomaly = result.get('is_anomaly', False)
                confidence = result.get('confidence', 0)
                
                # Display image in column
                with cols[col]:
                    # Convert numpy array to PIL Image for st.image
                    from PIL import Image
                    import numpy as np
                    img_array = result['heatmap_img']
                    img_pil = Image.fromarray(img_array.astype(np.uint8))
                    
                    # Caption with status and confidence
                    caption = f"{'ANOMALY' if is_anomaly else 'NORMAL'} ({confidence:.1f}%)"
                    
                    # Make the image clickable - using use_container_width instead of deprecated use_column_width
                    if st.image(img_pil, caption=filename, use_container_width=True):
                        selected_result = result
                    
                    # Add a button to view details
                    if st.button(f"View Details", key=f"view_{idx}"):
                        selected_result = result
    
    return selected_result

# Function to convert image to base64 for embedding in HTML
def image_to_base64(img):
    """Convert a numpy image to base64 string for HTML embedding"""
    import base64
    from io import BytesIO
    import cv2
    from PIL import Image
    import numpy as np
    
    # Convert numpy array to PIL Image
    if isinstance(img, np.ndarray):
        img_pil = Image.fromarray(img.astype(np.uint8))
    else:
        img_pil = img
        
    # Save to BytesIO buffer
    buffer = BytesIO()
    img_pil.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return img_str

def display_results_grid(results):
    """Display results in a grid view with cards"""
    # Show notification if anomalies are detected
    anomalies = [r for r in results if r.get('is_anomaly', False)]
    if anomalies:
        st.markdown(
            f'<div class="notification-bar notification-alert">⚠️ {len(anomalies)} Anomalies Detected! Immediate review recommended.</div>',
            unsafe_allow_html=True
        )
    
    # Create an expander for grid view explanation
    with st.expander("How to Use the Grid View"):
        st.markdown("""
        - Images are displayed in a grid format
        - Anomalies are marked with red badges
        - Click on any image to see detailed analysis
        - Images are sorted with highest confidence anomalies first
        """)
    
    # Initialize container for selected result details
    detail_container = st.container()
    
    # Render the image grid using columns
    selected_result = render_image_grid(results)
    
    # If an image is selected, display its details
    if selected_result:
        with detail_container:
            st.markdown("### Detailed Analysis")
            display_image_details(selected_result)

if __name__ == "__main__":
    main() 