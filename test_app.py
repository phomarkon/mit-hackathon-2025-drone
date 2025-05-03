import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import sys

def test_model_loading():
    """Test if the model can be loaded correctly"""
    print("Testing model loading...")
    try:
        # Load the ResNet18 model with updated syntax
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Modify for anomaly detection
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        model = model.to(device)
        model.eval()
        
        print("✓ Model loaded successfully!")
        return True
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False

if __name__ == "__main__":
    success = test_model_loading()
    sys.exit(0 if success else 1) 