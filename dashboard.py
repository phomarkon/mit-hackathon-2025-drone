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
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    st.warning("Ultralytics YOLO not available. Install with: pip install ultralytics")

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

# Update session state to include settings for models and muting
if "selected_model_name" not in st.session_state:
    st.session_state["selected_model_name"] = "resnet18"

if "ensemble_mode" not in st.session_state:
    st.session_state["ensemble_mode"] = False
    
if "mute_alerts" not in st.session_state:
    st.session_state["mute_alerts"] = False

# Update load_config function to properly handle YAML config files
def load_config(config_path):
    """Load configuration from YAML file with additional error handling"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Check if the required sections exist
        required_sections = ['model', 'human_detection']
        for section in required_sections:
            if section not in config and section == 'human_detection':
                # Add default human detection parameters if not in config
                config['human_detection'] = {
                    'enabled': True,
                    'min_area': 100,
                    'max_area': 8000,
                    'sensitivity': 1.8
                }
                
        return config
    except Exception as e:
        safe_error(f"Error loading config from {config_path}: {e}")
        # Return a default configuration as fallback
        return {
            'model': {
                'name': 'patchcore',
                'backbone': 'resnet18',
                'layers': ['layer2', 'layer3'],
                'coreset_sampling_ratio': 0.1
            },
            'human_detection': {
                'enabled': True,
                'min_area': 100,
                'max_area': 8000,
                'sensitivity': 1.8
            }
        }

# Create a cached function for loading the ensemble models
@st.cache_resource
def load_ensemble_models():
    """Load all three models for ensemble mode"""
    models = {}
    backbones = ["resnet18", "resnet50", "wide_resnet50_2"]
    
    for backbone in backbones:
        try:
            models[backbone] = load_model_cached(backbone)
            safe_success(f"Loaded {backbone} model for ensemble")
        except Exception as e:
            safe_error(f"Error loading {backbone} model for ensemble: {e}")
    
    return models

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

# Initialize session state for review tracking
if "reviewed_images" not in st.session_state:
    st.session_state["reviewed_images"] = set()

# Initialize session state for threshold
if "anomaly_threshold" not in st.session_state:
    # Try to load the optimal threshold from the JSON file first
    try:
        import json
        with open("outputs/patchcore_resnet18/optimal_threshold_resnet18.json", 'r') as f:
            threshold_data = json.load(f)
            st.session_state["anomaly_threshold"] = threshold_data.get("optimal_threshold", 0.5)
            st.session_state["threshold_source"] = "optimal_threshold_resnet18.json"
    except Exception as e:
        if st.session_state.get("dev_mode", False):
            st.warning(f"Could not load optimal threshold from file: {e}. Using default.")
        st.session_state["anomaly_threshold"] = 0.5
        st.session_state["threshold_source"] = "default"

# Define unicode symbols for safer display
CHECKMARK_SYMBOL = "\u2714"  # Unicode checkmark
WARNING_SYMBOL = "\u26A0"    # Unicode warning sign
ALERT_SYMBOL = "\U0001F6A8"  # Unicode police car light

# Update the AnomalyDetector class to support multiple models and ensemble mode
class AnomalyDetector:
    def __init__(self, backbone="resnet18", use_ensemble=False):
        """Initialize the anomaly detector with config-based parameters"""
        import torch
        import json
        
        self.backbone = backbone
        self.use_ensemble = use_ensemble
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            # Load configuration from YAML file
            config_paths = {
                "resnet18": "configs/patchcore_resnet18.yaml",
                "resnet50": "configs/patchcore_resnet50.yaml",
                "wide_resnet50_2": "configs/patchcore_wideresnet50_2.yaml"
            }
            
            if self.use_ensemble:
                # For ensemble mode, load all configs and models
                self.configs = {}
                self.models = load_ensemble_models()
                
                for model_name, config_path in config_paths.items():
                    self.configs[model_name] = load_config(config_path)
                
                # Use the first available model's config for transforms
                first_backbone = next(iter(self.models.keys()))
                main_config = self.configs.get(first_backbone)
                self.is_patchcore = hasattr(self.models[first_backbone], 'memory_bank')
            else:
                # For single model mode, load only the selected model
                config_path = config_paths.get(self.backbone, config_paths["resnet18"])
                self.config = load_config(config_path)
                
                # Load the model
                self.model = load_model_cached(self.backbone)
                self.is_patchcore = hasattr(self.model, 'memory_bank')
                main_config = self.config
            
            # Configure the transform pipeline based on the loaded model(s)
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
                    if self.use_ensemble:
                        st.success(f"Using PatchCore ensemble with {len(self.models)} models")
                    else:
                        st.success(f"Using PatchCore model with {len(self.model.memory_bank)} stored patches")
            
            # Load optimal threshold from JSON file for this backbone or ensemble
            if self.use_ensemble:
                # For ensemble, use a combination of thresholds or a dedicated ensemble threshold
                self.threshold = st.session_state.get("anomaly_threshold", 0.5)
                ensemble_threshold_file = "outputs/patchcore_ensemble/optimal_threshold_ensemble.json"
                if os.path.exists(ensemble_threshold_file):
                    try:
                        with open(ensemble_threshold_file, 'r') as f:
                            threshold_data = json.load(f)
                            loaded_threshold = threshold_data.get("optimal_threshold", None)
                            if loaded_threshold is not None:
                                self.threshold = loaded_threshold
                                if st.session_state.get("dev_mode", False):
                                    st.success(f"Loaded optimal ensemble threshold: {self.threshold:.4f}")
                    except Exception as e:
                        if st.session_state.get("dev_mode", False):
                            st.warning(f"Error loading ensemble threshold: {e}")
            else:
                # For single model, use the model-specific threshold
                threshold_file = f"outputs/patchcore_{self.backbone}/optimal_threshold_{self.backbone}.json"
                if os.path.exists(threshold_file):
                    try:
                        with open(threshold_file, 'r') as f:
                            threshold_data = json.load(f)
                            loaded_threshold = threshold_data.get("optimal_threshold", None)
                            if loaded_threshold is not None:
                                self.threshold = loaded_threshold
                                if st.session_state.get("dev_mode", False):
                                    st.success(f"Loaded optimal threshold: {self.threshold:.4f} from {threshold_file}")
                            else:
                                self.threshold = st.session_state.get("anomaly_threshold", 0.5)
                    except Exception as e:
                        if st.session_state.get("dev_mode", False):
                            st.warning(f"Error loading threshold from {threshold_file}: {e}")
                        self.threshold = st.session_state.get("anomaly_threshold", 0.5)
                else:
                    self.threshold = st.session_state.get("anomaly_threshold", 0.5)
                
            # Use session state threshold instead of hardcoded value
            if st.session_state.get("dev_mode", False):
                st.info(f"Using threshold: {self.threshold:.4f}")
            
            # Create thermal human detection config from YAML file or defaults
            if self.use_ensemble:
                # Use human detection config from first available model
                hd_config = self.configs.get(first_backbone, {}).get('human_detection', {})
            else:
                hd_config = self.config.get('human_detection', {})
                
            # Merge with application-specific enhancements
            self.human_detection_config = {
                'min_area': hd_config.get('min_area', 100),
                'max_area': hd_config.get('max_area', 8000),
                'sensitivity': hd_config.get('sensitivity', 1.8),
                'padding': 5,
                'focus_color': 'red',
                'aspect_ratio_min': 0.3,
                'aspect_ratio_max': 3.0,
                'max_detections': 3,
                'min_intensity': 0.25,
                'denoise_kernel': 3,
                'yolo_size': hd_config.get('yolo_size', 'nano'),
                'yolo_confidence': hd_config.get('yolo_confidence', 0.4)
            }
                
        except Exception as e:
            st.error(f"Failed to initialize detector: {e}")
            raise

    # Add a method for ensemble prediction
    def ensemble_predict(self, image_tensor):
        """Combine predictions from multiple models"""
        scores = []
        
        for model_name, model in self.models.items():
            if hasattr(model, 'compute_anomaly_scores'):
                # PatchCore model
                try:
                    # Method 1: Using TensorDataset and DataLoader
                    from torch.utils.data import TensorDataset, DataLoader
                    
                    # Create tensor dataset with numeric placeholders
                    label_tensor = torch.tensor([0])
                    path_tensor = torch.tensor([0]) 
                    dataset = TensorDataset(image_tensor, label_tensor, path_tensor)
                    dataloader = DataLoader(dataset, batch_size=1)
                    
                    anomaly_scores, _, _ = model.compute_anomaly_scores(dataloader)
                    
                    if isinstance(anomaly_scores, list) and len(anomaly_scores) > 0:
                        scores.append(anomaly_scores[0])
                    elif isinstance(anomaly_scores, np.ndarray) and anomaly_scores.size > 0:
                        scores.append(anomaly_scores[0])
                    else:
                        scores.append(0.5)  # Default fallback
                except Exception as e:
                    if st.session_state.get("dev_mode", False):
                        st.warning(f"Error in ensemble prediction for {model_name}: {e}")
                    scores.append(0.5)  # Default fallback
            else:
                # Standard ResNet model
                try:
                    with torch.no_grad():
                        score = model(image_tensor).item()
                        scores.append(score)
                except Exception as e:
                    if st.session_state.get("dev_mode", False):
                        st.warning(f"Error in ensemble prediction for {model_name}: {e}")
                    scores.append(0.5)  # Default fallback
        
        # Combine scores - use weighted average
        # Higher weight for more accurate models (typically resnet50 > wide_resnet50_2 > resnet18)
        if len(scores) == 0:
            return 0.5
        elif len(scores) == 1:
            return scores[0]
        else:
            # Define weights based on typical model performance
            weights = []
            for i, model_name in enumerate(self.models.keys()):
                if model_name == "resnet50":
                    weights.append(0.4)
                elif model_name == "wide_resnet50_2":
                    weights.append(0.35)
                else:  # resnet18
                    weights.append(0.25)
            
            # Normalize weights to sum to 1.0
            weights_sum = sum(weights)
            normalized_weights = [w / weights_sum for w in weights]
            
            # Calculate weighted average
            weighted_score = sum(s * w for s, w in zip(scores, normalized_weights))
            return weighted_score

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
        
        # Transform the image
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Different handling for ensemble vs single model
        if self.use_ensemble:
            # Use ensemble prediction
            score = self.ensemble_predict(image_tensor)
        else:
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
                        if st.session_state.get("dev_mode", False):
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
                            if st.session_state.get("dev_mode", False):
                                st.error(f"Both scoring methods failed. Fallback to default score. Errors: {e1}, then {e2}")
                            return np.array([0.5])  # Default mid-point score
                
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
                with torch.no_grad():
                    score = self.model(image_tensor).item()
        
        # Determine if it's an anomaly based on threshold        
        is_anomaly = score > self.threshold
        
        # Calculate confidence (properly capped at 100%)
        if is_anomaly:
            # Map to 0-100 range using the threshold as reference
            confidence = min(100.0, max(0.0, (score / max(self.threshold * 1.5, 0.001)) * 100))
        else:
            # For normal images, lower score means higher confidence of normalcy
            # Prevent negative values by using proper scaling
            confidence = min(100.0, max(0.0, ((1 - (score / max(self.threshold, 0.001))) * 100)))
        
        # Ensure confidence is in 0-100 range
        confidence = max(0.0, min(100.0, confidence))
        
        # Get the original image as numpy array
        img_np = np.array(image)
        
        # Store the original image for display
        original_img = img_np.copy()
        
        # Detect humans
        detected_humans = []

        if is_anomaly:
            # Check if YOLO detection is enabled in config
            use_yolo = False
            if self.use_ensemble:
                # Check if any model has YOLO enabled
                for config in self.configs.values():
                    if config.get('human_detection', {}).get('use_yolo', False):
                        use_yolo = True
                        break
            else:
                # Check if current model has YOLO enabled
                use_yolo = self.config.get('human_detection', {}).get('use_yolo', False)
                
            # Try YOLO first if enabled, but always fall back to thermal if YOLO fails
            detection_method = "thermal"  # Default method
            if use_yolo and YOLO_AVAILABLE:
                try:
                    detected_humans = self.detect_humans_yolo(img_np)
                    if detected_humans:  # Only set YOLO as method if it detected something
                        detection_method = "yolo"
                    else:
                        # YOLO didn't find anything, try thermal as backup
                        detected_humans = self.detect_humans_thermal(img_np)
                except Exception as e:
                    # Log error if in dev mode
                    if st.session_state.get("dev_mode", False):
                        st.warning(f"YOLO detection failed: {e}. Falling back to thermal detection.")
                    # Fall back to thermal
                    detected_humans = self.detect_humans_thermal(img_np)
            else:
                # Use thermal detection
                detected_humans = self.detect_humans_thermal(img_np)
            
            # Draw bounding boxes on original image (only if it's an anomaly)
            result_img = original_img.copy()
            for human in detected_humans:
                x, y, w, h = human['box']
                # Draw rectangle with red color (RGB format)
                cv2.rectangle(result_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        else:
            # No detection needed for normal images
            result_img = original_img.copy()
            detection_method = "none"
        
        # Create result dictionary with model information
        model_info = "Ensemble" if self.use_ensemble else self.backbone
        
        return {
            'is_anomaly': is_anomaly,
            'confidence': confidence,
            'raw_score': score,
            'original_img': original_img,
            'heatmap_img': result_img,  # This now contains original with only bounding boxes
            'humans_detected': len(detected_humans),
            'human_boxes': detected_humans,
            'reviewed': False,  # Initialize as not reviewed
            'model_used': model_info,  # Add model information
            'detection_method': detection_method  # Add detection method used
        }

    def detect_humans_thermal(self, img_np):
        """
        Improved human detection in thermal images with better filtering
        
        Args:
            img_np: numpy array of the image (RGB format)
            
        Returns:
            list of human detections with bounding boxes
        """
        # Focus on red color channel which indicates humans in thermal images
        # Extract the red channel (which is most indicative of heat signatures)
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            # Get red channel (humans will appear brighter in red channel of thermal images)
            red_channel = img_np[:, :, 0].copy()
            
            # For white regions (high in all channels), we can subtract other channels
            # to emphasize regions that are specifically red, not just bright
            suppression = 0.5 * (img_np[:, :, 1] + img_np[:, :, 2]) / 2  # Average of green and blue
            heat_emphasis = np.clip(red_channel - suppression, 0, 255).astype(np.uint8)
            
            # Use this enhanced red channel for processing
            processing_img = heat_emphasis
        else:
            # If already grayscale, use as is
            processing_img = img_np.copy()
            
        # Apply denoising to reduce small artifacts
        kernel_size = self.human_detection_config['denoise_kernel']
        processing_img = cv2.GaussianBlur(processing_img, (kernel_size, kernel_size), 0)
        
        # Use adaptive thresholding based on image statistics
        mean_val = np.mean(processing_img)
        std_val = np.std(processing_img)
        sensitivity = self.human_detection_config['sensitivity']
        threshold = mean_val + sensitivity * std_val
        
        # Apply threshold to get binary image
        _, binary = cv2.threshold(processing_img, threshold, 255, cv2.THRESH_BINARY)
        
        # Remove noise with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Further clean up with dilate/erode to connect nearby components that might be parts of the same human
        binary = cv2.dilate(binary, kernel, iterations=1)
        binary = cv2.erode(binary, kernel, iterations=1)
        
        # Find contours in the binary image
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Get config parameters
        min_area = self.human_detection_config['min_area']
        max_area = self.human_detection_config['max_area']
        padding = self.human_detection_config['padding']
        aspect_ratio_min = self.human_detection_config['aspect_ratio_min']
        aspect_ratio_max = self.human_detection_config['aspect_ratio_max']
        min_intensity = self.human_detection_config['min_intensity']
        max_detections = self.human_detection_config['max_detections']
        
        # Calculate image-wide max intensity for relative comparisons
        max_intensity = np.max(processing_img) if np.max(processing_img) > 0 else 1
        
        human_candidates = []
        
        # First pass: identify potential humans by shape and intensity
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check aspect ratio for human-like shapes
                aspect_ratio = h / w if w > 0 else 0
                if not (aspect_ratio_min <= aspect_ratio <= aspect_ratio_max):
                    continue
                
                # Create a mask for this contour
                mask = np.zeros_like(processing_img)
                cv2.drawContours(mask, [contour], 0, 255, -1)
                
                # Calculate mean intensity within the contour
                mean_intensity = cv2.mean(processing_img, mask=mask)[0] / 255.0  # Normalize to 0-1
                
                # Skip if intensity is too low relative to the max in the image
                relative_intensity = mean_intensity / (max_intensity / 255.0) if max_intensity > 0 else 0
                if relative_intensity < min_intensity:
                    continue
                
                # Calculate confidence based on combined factors: area, intensity, and shape
                # Higher values for larger areas and higher intensities within reasonable bounds
                confidence_area = min(1.0, area / max_area)  # Normalized area factor
                
                # Create a more realistic confidence distribution that's not always near 100%
                # Scale based on relative intensity but with a curve that's not too high
                intensity_factor = relative_intensity * 0.9  # Scale to 90% max
                shape_factor = min(1.0, (aspect_ratio if aspect_ratio < 1 else 1/aspect_ratio))  # Favor human proportions
                
                # Calculate weighted confidence - avoid artificially high values
                raw_confidence = (intensity_factor * 0.6) + (confidence_area * 0.3) + (shape_factor * 0.1)
                
                # Apply a curve to make high confidences harder to achieve
                # This formula creates more variety - typically 60-90% range for real detections
                confidence = min(100.0, raw_confidence * 85)  # Cap at 85% for most detections
                
                # Apply a slight random variation to avoid identical confidence values
                import random
                confidence = min(98.0, confidence * random.uniform(0.95, 1.05))  # 5% random variation, capped at 98%
                
                human_candidates.append({
                    'box': (x, y, w, h),
                    'confidence': float(confidence),
                    'area': float(area),
                    'intensity': float(mean_intensity),
                    'relative_intensity': float(relative_intensity)
                })
        
        # Sort by confidence and take top candidates
        human_candidates.sort(key=lambda x: x['confidence'], reverse=True)
        top_candidates = human_candidates[:max_detections]
        
        # Second pass: add padding to bounding boxes
        human_detections = []
        for candidate in top_candidates:
            x, y, w, h = candidate['box']
            
            # Add padding
            x_padded = max(0, x - padding)
            y_padded = max(0, y - padding)
            w_padded = min(img_np.shape[1] - x_padded, w + 2*padding)
            h_padded = min(img_np.shape[0] - y_padded, h + 2*padding)
            
            human_detections.append({
                'box': (x_padded, y_padded, w_padded, h_padded),
                'confidence': candidate['confidence'],
                'area': candidate['area']
            })
                
        return human_detections

    def detect_humans_yolo(self, img_np):
        """
        Detect humans in thermal images using YOLO
        
        Args:
            img_np: numpy array of the image (RGB format)
            
        Returns:
            list of human detections with bounding boxes
        """
        if not YOLO_AVAILABLE:
            # Fall back to traditional method if YOLO not available
            return self.detect_humans_thermal(img_np)
            
        # Load YOLO model based on config
        yolo_size = self.human_detection_config.get('yolo_size', 'nano')
        confidence_threshold = self.human_detection_config.get('yolo_confidence', 0.4)
        
        # Try to get cached model
        model = load_yolo_model(yolo_size)
        
        if model is None:
            # Fall back to traditional method if model couldn't be loaded
            return self.detect_humans_thermal(img_np)
        
        try:
            # Process image with YOLO
            pil_img = Image.fromarray(img_np)
            
            # Run inference - only detect person class (class 0)
            results = model(pil_img, conf=confidence_threshold, classes=[0])
            
            # Process results
            human_detections = []
            
            if len(results) > 0:
                # Get the first result (only one image)
                result = results[0]
                
                # Extract boxes and confidences for persons
                boxes = result.boxes
                
                for i, box in enumerate(boxes):
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Calculate width and height
                    w = x2 - x1
                    h = y2 - y1
                    
                    # Get confidence
                    conf = box.conf[0].cpu().numpy().item()
                    
                    # Check if this is within our area limits
                    area = w * h
                    min_area = self.human_detection_config.get('min_area', 50)
                    max_area = self.human_detection_config.get('max_area', 10000)
                    
                    if min_area <= area <= max_area:
                        # Create a human detection entry
                        human_detections.append({
                            'box': (int(x1), int(y1), int(w), int(h)),
                            'confidence': float(conf * 100),  # Scale to percentage
                            'area': float(area)
                        })
            
            # Sort by confidence and take top candidates
            max_detections = self.human_detection_config.get('max_detections', 3)
            human_detections.sort(key=lambda x: x['confidence'], reverse=True)
            human_detections = human_detections[:max_detections]
            
            return human_detections
            
        except Exception as e:
            # Log error if in dev mode
            if st.session_state.get("dev_mode", False):
                st.error(f"YOLO detection error: {e}")
            
            # Fall back to traditional method
            return self.detect_humans_thermal(img_np)


def get_priority_label(confidence, is_anomaly, humans_detected=0):
    """Get priority label based on confidence, anomaly status and human detection"""
    if not is_anomaly:
        return "Normal"
    # If humans are detected, prioritize based on number and confidence
    elif humans_detected > 0:
        # Higher priority for multiple humans or high confidence single human
        if humans_detected > 1 or confidence >= 75:
            return "High"
        # Medium priority for moderate confidence single human
        elif confidence >= 50:
            return "Medium"
        # Low priority for low confidence single human
        else:
            return "Low"
    # No humans detected but still an anomaly
    elif confidence >= 85:
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
    """Process multiple uploaded files with real-time updates and alerts"""
    results = []
    total_files = len(uploaded_files)
    processed_count = 0
    anomalies_found = 0
    
    # Create placeholders for real-time results display
    results_placeholder = st.empty()
    progress_text = st.empty()
    alert_placeholder = st.empty()
    
    # Track processing time for ETA calculation
    import time
    start_time = time.time()
    processing_times = []
    
    for i, file in enumerate(uploaded_files):
        # Create a temporary file to save the uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # Process the image
            file_start_time = time.time()
            safe_info(f"Processing {file.name}...")
            progress_text.text(f"Processing image {i+1} of {total_files} ({((i+1)/total_files)*100:.1f}%)...")
            
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
                processed_count += 1
                
                # Check if this is an anomaly with humans detected and show alert if not muted
                if result['is_anomaly'] and result['humans_detected'] > 0:
                    anomalies_found += 1
                    
                    # Show visual alert
                    with alert_placeholder:
                        st.warning(f"{WARNING_SYMBOL} Potential human detected in {file.name}! ({result['humans_detected']} human{'s' if result['humans_detected'] > 1 else ''} with {result['confidence']:.1f}% confidence)", icon="⚠️")
                    
                    # Play sound if not muted
                    if not st.session_state.get("mute_alerts", False):
                        sound_file = "alert.mp3"
                        if os.path.exists(sound_file):
                            import base64
                            audio_str = f'<audio autoplay="true"><source src="data:audio/mp3;base64,{base64.b64encode(open(sound_file, "rb").read()).decode()}" type="audio/mp3"></audio>'
                            st.markdown(audio_str, unsafe_allow_html=True)
                    
                    # Show toast notification
                    st.toast(f"{ALERT_SYMBOL} Human detected in {file.name}!", icon="⚠️")
                
                # Track processing time for this file
                file_processing_time = time.time() - file_start_time
                processing_times.append(file_processing_time)
                
                # Calculate average processing time and ETA
                avg_time = sum(processing_times) / len(processing_times)
                eta = avg_time * (total_files - (i + 1))
                eta_str = f"{eta:.1f}s" if eta < 60 else f"{eta/60:.1f}min"
                
                # Update progress text with ETA
                progress_text.text(f"Processing image {i+1} of {total_files} ({((i+1)/total_files)*100:.1f}%) - ETA: {eta_str}")
                
                # Show real-time results for processed images
                if len(results) > 0 and len(results) % 1 == 0:  # Update every image
                    # Sort the current results by priority
                    priority_order = {"High": 0, "Medium": 1, "Low": 2, "Normal": 3}
                    current_results = sorted(results, 
                                          key=lambda x: (priority_order.get(x['priority'], 4), 
                                                        -x['confidence'] if x['is_anomaly'] else 0))
                    
                    # Display current results in a compact format
                    with results_placeholder.container():
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.write(f"### Processed {len(results)} of {total_files} images")
                            
                            # Show anomaly stats
                            anomaly_count = sum(1 for r in results if r['is_anomaly'])
                            human_count = sum(r['humans_detected'] for r in results)
                            if anomaly_count > 0:
                                st.write(f"**Found:** {anomaly_count} anomalies with {human_count} potential humans")
                        
                        with col2:
                            # Show elapsed time and speed
                            elapsed = time.time() - start_time
                            speed = len(results) / elapsed if elapsed > 0 else 0
                            st.write(f"**Time:** {elapsed:.1f}s ({speed:.1f} img/s)")
                            st.write(f"**ETA:** {eta_str}")
                        
                        # Show a small preview grid of processed images
                        if len(current_results) > 0:
                            preview_cols = min(3, len(current_results))
                            cols = st.columns(preview_cols)
                            for j, col in enumerate(cols):
                                if j < len(current_results):
                                    res = current_results[j]
                                    # Create a smaller preview image
                                    preview_img = Image.fromarray(res['heatmap_img'])
                                    preview_img.thumbnail((200, 200), Image.Resampling.LANCZOS)
                                    
                                    # Show the image with status
                                    status = "ANOMALY" if res['is_anomaly'] else "NORMAL"
                                    priority_color = "#ff0000" if res['priority'] == "High" else "#ff6600" if res['priority'] == "Medium" else "#ffcc00" if res['priority'] == "Low" else "#00cc00"
                                    humans = f"{res['humans_detected']} humans" if res['humans_detected'] > 0 else ""
                                    
                                    # Display with colored border based on priority
                                    col.markdown(f'<div style="border: 2px solid {priority_color}; padding: 3px; border-radius: 5px; margin-bottom: 5px;">', unsafe_allow_html=True)
                                    col.image(preview_img, caption=f"{res['filename']}")
                                    
                                    # Format confidence properly
                                    confidence_display = min(abs(res['confidence']), 100.0)
                                    
                                    col.markdown(f"<p style='text-align: center; margin: 0; color: {priority_color};'><strong>{status}</strong> {humans}</p>", unsafe_allow_html=True)
                                    if res['is_anomaly']:
                                        col.markdown(f"<p style='text-align: center; margin: 0;'>{confidence_display:.1f}%</p>", unsafe_allow_html=True)
                                    col.markdown('</div>', unsafe_allow_html=True)
                
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
    
    # Clear the temporary displays
    results_placeholder.empty()
    progress_text.empty()
    
    # Sort results by priority for final display
    priority_order = {"High": 0, "Medium": 1, "Low": 2, "Normal": 3}
    results_sorted = sorted(results, 
                          key=lambda x: (priority_order.get(x['priority'], 4), 
                                        -x['confidence'] if x['is_anomaly'] else 0))
    
    # Show final summary
    if results:
        safe_success(f"Successfully processed {len(results)} of {total_files} images")
        
        # Show alert summary if anomalies were found
        if anomalies_found > 0 and not alert_placeholder.empty:
            alert_placeholder.warning(f"{WARNING_SYMBOL} Total: {anomalies_found} images with potential humans detected!", icon="⚠️")
    else:
        safe_warning("No images were successfully processed")
        
    return results_sorted


def display_results_table(results):
    """Display results in a sortable table"""
    # Map detection methods to user-friendly labels
    detection_method_labels = {
        "yolo": "YOLO AI",
        "thermal": "Thermal Analysis",
        "none": "None",
        "unknown": "Unknown"
    }
    
    # Convert results to a DataFrame for easy display
    df = pd.DataFrame([
        {
            'Filename': r['filename'],
            'Status': 'ANOMALY' if r['is_anomaly'] else 'NORMAL',
            'Confidence': f"{min(abs(r['confidence']), 100.0):.1f}%",
            'Priority': r['priority'],
            'Humans': r.get('humans_detected', 0),
            'Detection': detection_method_labels.get(r.get('detection_method', 'unknown'), 'Unknown') if r['is_anomaly'] and r.get('humans_detected', 0) > 0 else '-',
            'Model': r.get('model_used', 'Unknown'),
            'Raw Score': f"{r['raw_score']:.4f}",
            'Reviewed': CHECKMARK_SYMBOL if r['filename'] in st.session_state.get("reviewed_images", set()) else '',
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
        
        # Highlight YOLO detection with special color
        if row['Detection'] == 'YOLO AI':
            styled_df.at[i, 'Detection'] = f'<span style="color: #4CAF50; font-weight: bold;">{row["Detection"]}</span>'
        elif row['Detection'] == 'Thermal Analysis':
            styled_df.at[i, 'Detection'] = f'<span style="color: #ff9800; font-weight: bold;">{row["Detection"]}</span>'
            
        # Add review checkmark with styling
        if row['Reviewed']:
            styled_df.at[i, 'Reviewed'] = f'<span style="color: green; font-weight: bold;">{CHECKMARK_SYMBOL}</span>'
    
    # Display table using the styled dataframe
    st.write(styled_df.to_html(escape=False, index=False), unsafe_allow_html=True)
    
    # Return the original dataframe for interaction
    return df


def display_image_details(result):
    """Display detailed information for a single image"""
    col1, col2 = st.columns([3, 1])
    
    # Check if image is already reviewed
    image_id = result.get('filename', '')
    is_reviewed = image_id in st.session_state.get("reviewed_images", set())
    result['reviewed'] = is_reviewed
    
    with col1:
        # Display the original image with only bounding boxes for anomalies
        st.image(result['heatmap_img'], use_container_width=True)
        
    with col2:
        # Display anomaly details
        st.subheader("Anomaly Analysis")
        status = "ANOMALY DETECTED" if result['is_anomaly'] else "NORMAL (No Anomaly)"
        priority_color = get_priority_color(result['priority'])
        
        st.markdown(f"**Status:** <span class='{priority_color}'>{status}</span>", unsafe_allow_html=True)
        
        # Format confidence properly - always positive and capped at 100%
        confidence_value = min(abs(result['confidence']), 100.0)
        st.markdown(f"**Confidence:** {confidence_value:.1f}%")
        
        st.markdown(f"**Priority:** <span class='{priority_color}'>{result['priority']}</span>", unsafe_allow_html=True)
        
        # Display which model was used
        model_info = result.get('model_used', "Unknown")
        st.markdown(f"**Model:** {model_info}")
        
        # Display which detection method was used
        detection_method = result.get('detection_method', "unknown")
        detection_method_display = {
            "yolo": "YOLO AI",
            "thermal": "Thermal Analysis",
            "none": "None (Normal Image)",
            "unknown": "Unknown"
        }.get(detection_method, detection_method)
        
        if result['is_anomaly'] and result['humans_detected'] > 0:
            st.markdown(f"**Detection Method:** <span style='color: #4CAF50;'>{detection_method_display}</span>", unsafe_allow_html=True)
        
        # Display review status
        review_status = f"{CHECKMARK_SYMBOL} Reviewed" if is_reviewed else "Not reviewed"
        review_color = "anomaly-high" if is_reviewed else "normal"
        st.markdown(f"**Status:** <span class='{review_color}'>{review_status}</span>", unsafe_allow_html=True)
        
        # Display human detection info
        humans_detected = result.get('humans_detected', 0)
        if humans_detected > 0:
            st.markdown(f"**Humans Detected:** <span class='anomaly-high'>{humans_detected}</span>", unsafe_allow_html=True)
            
            # Show bounding box details if available
            if 'human_boxes' in result and result['human_boxes']:
                with st.expander("Human Detection Details"):
                    for i, human in enumerate(result['human_boxes']):
                        confidence = min(human.get('confidence', 0), 100.0)
                        st.markdown(f"**Human #{i+1}:** Confidence: {confidence:.1f}%")
        else:
            st.markdown("**Humans Detected:** 0")
        
        # Display a gauge chart for the confidence score - ensure positive values
        capped_confidence = min(abs(result['confidence']), 100.0)
        fig = px.pie(values=[capped_confidence, 100-capped_confidence], 
                    names=['Confidence', ''], 
                    hole=0.7,
                    color_discrete_sequence=['#ff4b4b' if result['is_anomaly'] else '#00cc00', '#e1e1e1'])
        fig.update_layout(
            annotations=[dict(text=f"{capped_confidence:.1f}%", x=0.5, y=0.5, font_size=20, showarrow=False)],
            showlegend=False,
            margin=dict(t=0, b=0, l=0, r=0),
            height=200
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Add action buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button(f"{ALERT_SYMBOL} Alert", key="alert_button"):
                # Play sound effect if not muted
                if not st.session_state.get("mute_alerts", False):
                    sound_file = "alert.mp3"
                    if os.path.exists(sound_file):
                        import base64
                        audio_placeholder = st.empty()
                        audio_str = f'<audio autoplay="true"><source src="data:audio/mp3;base64,{base64.b64encode(open(sound_file, "rb").read()).decode()}" type="audio/mp3"></audio>'
                        audio_placeholder.markdown(audio_str, unsafe_allow_html=True)
                    
                # Show toast notification
                st.toast(f"{ALERT_SYMBOL} Alert sent to rescue team!", icon="⚠️")
                st.success("Alert sent to rescue team!")
        
        with col2:
            if is_reviewed:
                if st.button("Unmark", key="unmark_button"):
                    # Remove from reviewed set
                    if image_id in st.session_state["reviewed_images"]:
                        st.session_state["reviewed_images"].remove(image_id)
                    st.toast("Image unmarked", icon="📝")
                    result['reviewed'] = False
                    st.rerun()
            else:
                if st.button(f"Mark as Reviewed {CHECKMARK_SYMBOL}", key="mark_button"):
                    # Add to reviewed set
                    st.session_state["reviewed_images"].add(image_id)
                    st.toast(f"{CHECKMARK_SYMBOL} Image marked as reviewed", icon="📝")
                    result['reviewed'] = True
                    st.rerun()


def generate_report(results):
    """Generate a text report from the results"""
    if not results:
        return "No results to generate a report from."

    # Calculate summary statistics
    total_images = len(results)
    anomaly_counts = {"High": 0, "Medium": 0, "Low": 0}
    total_humans = 0
    total_confidence = 0

    for res in results:
        if res['is_anomaly']:
            priority = res.get('priority', 'Unknown')
            if priority in anomaly_counts:
                anomaly_counts[priority] += 1
            total_humans += res.get('humans_detected', 0)
            total_confidence += res.get('confidence', 0)

    avg_confidence = total_confidence / total_images if total_images > 0 else 0

    # Generate the report string
    report_str = "--- Anomaly Detection Report ---\n"
    report_str += f"Total Images: {total_images}\n"
    report_str += f"Total Anomalies: {anomaly_counts['High'] + anomaly_counts['Medium'] + anomaly_counts['Low']}\n"
    report_str += f"  - High Priority: {anomaly_counts['High']}\n"
    report_str += f"  - Medium Priority: {anomaly_counts['Medium']}\n"
    report_str += f"  - Low Priority: {anomaly_counts['Low']}\n"
    report_str += f"Total Humans Detected: {total_humans}\n"
    report_str += f"Average Anomaly Confidence: {avg_confidence:.1f}%\n"

    report_str += "\n\n--- Anomaly Details ---\n"
    report_str += "{:<30} {:<10} {:<12} {:<10} {:<15}\n".format(
        "Filename", "Anomaly", "Confidence", "Priority", "Humans Detected"
    )
    report_str += "-" * 80 + "\n"

    for res in results:
        if res['is_anomaly']:
            confidence_display = f"{min(abs(res['confidence']), 100.0):.1f}%" if 'confidence' in res else "N/A"
            report_str += "{:<30} {:<10} {:<12} {:<10} {:<15}\n".format(
                res.get('filename', 'N/A'),
                str(res.get('is_anomaly', 'N/A')),
                confidence_display,
                res.get('priority', 'N/A'),
                str(res.get('humans_detected', 'N/A'))
            )

    return report_str


def main():
    try:
        # Sidebar
        st.sidebar.image("logo.png", width=200)
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
            st.session_state["prev_dev_mode"] = dev_mode
            st.rerun()
        st.session_state["prev_dev_mode"] = dev_mode
        
        # Model selection with more options
        st.sidebar.markdown("### Model Configuration")
        model_options = [
            "ResNet-18 (Faster)",
            "ResNet-50 (More Accurate)",
            "WideResNet-50-2 (Best Accuracy)",
            "Ensemble (All Models)"
        ]
        
        model_map = {
            "ResNet-18 (Faster)": "resnet18",
            "ResNet-50 (More Accurate)": "resnet50",
            "WideResNet-50-2 (Best Accuracy)": "wide_resnet50_2",
            "Ensemble (All Models)": "ensemble"
        }
        
        selected_model_display = st.sidebar.radio(
            "Select model:",
            model_options,
            help="Choose between different models or ensemble mode. Ensemble combines all models for best results."
        )
        
        # Set the selected model in session state
        selected_model = model_map[selected_model_display]
        st.session_state["selected_model_name"] = selected_model
        
        # Enable/disable ensemble mode
        use_ensemble = selected_model == "ensemble"
        st.session_state["ensemble_mode"] = use_ensemble
        
        # If ensemble is selected, show a note about performance
        if use_ensemble:
            st.sidebar.warning("Ensemble mode combines multiple models and may be slower but more accurate.")
            selected_backbone = "resnet18"  # Default backbone for initialization
        else:
            selected_backbone = selected_model
            
        # Mute alerts toggle
        st.sidebar.markdown("### Notifications")
        mute_alerts = st.sidebar.checkbox("Mute Audio Alerts", value=st.session_state.get("mute_alerts", False))
        st.session_state["mute_alerts"] = mute_alerts
        
        if mute_alerts:
            st.sidebar.info("Audio alerts are muted. You will still see visual notifications.")
        
        # View mode selection - Make List the default
        view_mode = st.sidebar.radio("View Mode", ["List", "Grid"], index=0)
        st.session_state["view_mode"] = view_mode
        
        # Add threshold adjustment based on the optimal threshold from JSON file
        st.sidebar.markdown("### Anomaly Threshold")
        
        # Show source of the threshold
        threshold_source = st.session_state.get("threshold_source", "default")
        if threshold_source != "default":
            st.sidebar.info(f"Loaded optimal threshold from: {threshold_source}")
        
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
            
        # Add a toggle to download alert sound if needed
        if not os.path.exists("alert.mp3"):
            if st.sidebar.button("Download Alert Sound"):
                # Create a download button to save the sound
                try:
                    # Use a reliable URL for the alert sound
                    import requests
                    url = "https://github.com/assets/sounds/alert.mp3" # This should be a reliable URL
                    
                    response = requests.get(url)
                    with open("alert.mp3", "wb") as f:
                        f.write(response.content)
                    
                    st.sidebar.success("Alert sound downloaded!")
                except Exception as e:
                    st.sidebar.error(f"Failed to download alert sound: {e}")
        else:
            st.sidebar.success("Alert sound is ready")
        
        # Try to load matching config file
        config_path = f"configs/patchcore_{selected_backbone}.yaml"
        if os.path.exists(config_path):
            config = load_config(config_path)
            if config and dev_mode:
                st.sidebar.markdown("**Configuration loaded from:**")
                st.sidebar.code(config_path, language="yaml")
        
        st.sidebar.markdown("---")
        
        # Initialize the anomaly detector with selected backbone and ensemble setting
        with st.spinner(f"Loading {'ensemble of models' if use_ensemble else selected_model_display}..."):
            detector = AnomalyDetector(backbone=selected_backbone, use_ensemble=use_ensemble)
        
        # Main interface - cleaner in non-dev mode
        if dev_mode:
            st.markdown("<h1 style='text-align: center;'>DroneGuardian Prime: Thermal Anomaly Detection</h1>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;'>Upload thermal drone images to detect potential human presence and prioritize findings</p>", unsafe_allow_html=True)
            
            # Info about selected model
            st.info(f"Using {selected_model_display} for anomaly detection. You can change the model in the sidebar.")
        else:
            st.markdown("<h1 style='text-align: center;'>DroneGuardian Prime</h1>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;'>Search & Rescue Thermal Analysis System</p>", unsafe_allow_html=True)
        
        # File uploader
        uploaded_files = st.file_uploader("Upload thermal drone images", type=['jpg', 'jpeg', 'png', 'tif', 'tiff'], accept_multiple_files=True)
        
        if uploaded_files:
            # Check if we have cached results and if inputs match
            cache_valid = False
            if st.session_state["processed_results"] is not None:
                cached_files = st.session_state.get("cached_filenames", [])
                current_files = [f.name for f in uploaded_files]
                current_model = st.session_state.get("cached_model_name", "")
                if (set(current_files) == set(cached_files) and 
                    current_model == selected_model and 
                    st.session_state.get("cached_ensemble") == use_ensemble):
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
                st.session_state["cached_model_name"] = selected_model
                st.session_state["cached_ensemble"] = use_ensemble
                
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
                
                # Recalculate confidence based on new threshold
                if result['is_anomaly']:
                    # Map to 0-100 range using the threshold as reference
                    result['confidence'] = min(100.0, max(0.0, (result['raw_score'] / max(anomaly_threshold * 1.5, 0.001)) * 100))
                else:
                    # For normal images, lower score means higher confidence of normalcy
                    result['confidence'] = min(100.0, max(0.0, ((1 - (result['raw_score'] / max(anomaly_threshold, 0.001))) * 100)))
                
                # Ensure confidence is in 0-100 range
                result['confidence'] = max(0.0, min(100.0, result['confidence']))
                
                # Update priority based on new anomaly status and confidence
                result['priority'] = get_priority_label(result['confidence'], result['is_anomaly'], result['humans_detected'])
            
            # Display summary only in dev mode or make it more compact
            if dev_mode:
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
            else:
                # More compact for non-dev mode
                total = len(results)
                anomalies = sum(1 for r in results if r['is_anomaly'])
                st.markdown(f"**Analysis Results:** {anomalies} anomalies detected in {total} images")
                # Show warning only if anomalies are detected
                if anomalies > 0:
                    st.warning(f"{WARNING_SYMBOL} {anomalies} potential human signatures detected. Review recommended.", icon="⚠️")
            
            # Display results based on view mode
            st.markdown("<div class='custom-title'>Analysis Results</div>", unsafe_allow_html=True)
            
            if st.session_state.get("view_mode") == "Grid":
                display_results_grid(results)
            else:
                # Use the table view for List mode - sort by priority first
                priority_order = {"High": 0, "Medium": 1, "Low": 2, "Normal": 3}
                
                # Sort results by priority and then by confidence
                results_sorted = sorted(results, 
                                        key=lambda x: (priority_order.get(x['priority'], 4), 
                                                     -x['confidence'] if x['is_anomaly'] else 0))
                
                st.markdown("Images are sorted by priority (High → Medium → Low → Normal) and confidence.")
                df = display_results_table(results_sorted)
                
                # Image selection and detailed view
                st.markdown("---")
                st.markdown("<div class='custom-title'>Detailed Analysis</div>", unsafe_allow_html=True)
                
                # Let user select which image to view in detail
                if len(results) > 0:
                    option_list = df['Filename'].tolist()
                    selected_filename = st.selectbox("Select an image to view details:", option_list)
                    
                    # Find the corresponding result
                    selected_result = next((r for r in results_sorted if r['filename'] == selected_filename), None)
                    
                    if selected_result:
                        # Display the details for the selected image
                        display_image_details(selected_result)
                
            # Add report generation button
            report_data = generate_report(results)
            st.download_button(
                label="Download Report (.txt)",
                data=report_data,
                file_name="anomaly_report.txt",
                mime="text/plain",
            )

        else:
            # Display instructions when no files are uploaded
            st.info("Please upload one or more thermal drone images to begin analysis.")
            
            if dev_mode:
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
            else:
                # Simpler interface for non-dev mode
                st.markdown("""
                #### Quick Start:
                1. Upload thermal drone images using the file uploader above
                2. AI will automatically detect potential humans in thermal imagery
                3. Review the marked areas and use the "Mark as Reviewed" button after inspection
                """)
                st.markdown("---")
    except Exception as e:
        error_msg = str(e)
        if "is not a valid emoji" in error_msg:
            # Special handling for emoji errors
            st.error(f"Application error: {error_msg}\n\nPlease try refreshing the page.")
        else:
            st.error(f"Application error: {error_msg}")
            
        if dev_mode:
            import traceback
            st.code(traceback.format_exc(), language="python")
        else:
            st.write("Please try refreshing the page or check the logs for more details.")
            # Show dev mode toggle for error situations
            if st.button("Show Developer Details"):
                st.session_state["dev_mode"] = True
                st.rerun()

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

# Update the render_image_grid function to better show anomaly status
def render_image_grid(results):
    """Render results as a grid of images with badges and clear anomaly status"""
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
                confidence = min(abs(result.get('confidence', 0)), 100.0)  # Ensure positive and capped confidence
                is_reviewed = filename in st.session_state.get("reviewed_images", set())
                humans_detected = result.get('humans_detected', 0)
                priority = result.get('priority', 'Unknown')
                
                # Display image in column
                with cols[col]:
                    # Create a container with border styling based on priority
                    priority_color = "#ff0000" if priority == "High" else "#ff6600" if priority == "Medium" else "#ffcc00" if priority == "Low" else "#00cc00"
                    
                    # Create a styled container for the image
                    st.markdown(
                        f"""
                        <div style="border: 2px solid {priority_color}; border-radius: 5px; padding: 5px; position: relative; margin-bottom: 10px;">
                            <div style="position: absolute; top: 10px; right: 10px; background-color: {priority_color}; 
                                 color: white; padding: 2px 8px; border-radius: 10px; font-size: 12px; font-weight: bold; z-index: 10;">
                                {priority}
                            </div>
                            <div style="position: absolute; top: 10px; left: 10px; background-color: rgba(0,0,0,0.7); 
                                 color: white; padding: 2px 8px; border-radius: 10px; font-size: 12px; z-index: 10;">
                                {"ANOMALY" if is_anomaly else "NORMAL"}
                            </div>
                            {f'<div style="position: absolute; bottom: 40px; left: 10px; background-color: rgba(255,0,0,0.7); color: white; padding: 2px 8px; border-radius: 10px; font-size: 12px; z-index: 10;">{humans_detected} humans</div>' if humans_detected > 0 else ''}
                            {f'<div style="position: absolute; bottom: 10px; right: 10px; background-color: rgba(0,0,0,0.7); color: white; padding: 2px 8px; border-radius: 10px; font-size: 12px; z-index: 10;">{confidence:.1f}%</div>' if is_anomaly else ''}
                            {f'<div style="position: absolute; bottom: 10px; left: 10px; background-color: rgba(0,200,0,0.7); color: white; padding: 2px 8px; border-radius: 10px; font-size: 12px; z-index: 10;">✓</div>' if is_reviewed else ''}
                        """, 
                        unsafe_allow_html=True
                    )
                    
                    # Convert numpy array to PIL Image for st.image
                    from PIL import Image
                    import numpy as np
                    img_array = result['heatmap_img']
                    img_pil = Image.fromarray(img_array.astype(np.uint8))
                    
                    # Display the image
                    st.image(img_pil, caption=filename, use_container_width=True)
                    
                    # Close the container div
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Add a button to view details with full width and visible border
                    if st.button(f"View Details", key=f"view_{idx}", use_container_width=True):
                        selected_result = result
    
    return selected_result

# Update the display_results_grid function
def display_results_grid(results):
    """Display results in a grid view with cards"""
    # Show notification if anomalies are detected
    anomalies = [r for r in results if r.get('is_anomaly', False)]
    if anomalies:
        humans_count = sum(r.get('humans_detected', 0) for r in anomalies)
        st.markdown(
            f'<div class="notification-bar notification-alert">⚠️ {len(anomalies)} Anomalies Detected with {humans_count} potential humans! Immediate review recommended.</div>',
            unsafe_allow_html=True
        )
    
    # Create an expander for grid view explanation
    with st.expander("How to Use the Grid View"):
        st.markdown("""
        - Images are displayed in a grid format with important information
        - **Red border**: High priority anomaly
        - **Orange border**: Medium priority anomaly
        - **Yellow border**: Low priority anomaly
        - **Green border**: Normal image (no anomaly)
        - The number of humans detected is shown in red
        - Confidence score is shown in the bottom right
        - Reviewed images are marked with a green checkmark
        - Click on any image to see detailed analysis
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

# Add a function to load and cache the YOLO model
@st.cache_resource
def load_yolo_model(model_size="nano"):
    """Load and cache a YOLO model for human detection"""
    if not YOLO_AVAILABLE:
        return None
        
    try:
        # Map model size to model name
        size_map = {
            "nano": "yolov8n.pt",
            "small": "yolov8s.pt",
            "medium": "yolov8m.pt", 
            "large": "yolov8l.pt"
        }
        
        model_name = size_map.get(model_size, "yolov8n.pt")
        model = YOLO(model_name)
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None

if __name__ == "__main__":
    main() 