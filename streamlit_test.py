import streamlit as st
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# Set page config
st.set_page_config(
    page_title="Model Test",
    page_icon="✅"
)

st.title("Thermal Drone Anomaly Detection - Model Test")

# Create a cached function for loading the model
@st.cache_resource
def load_model_cached():
    st.write("Loading model (this will only happen once)...")
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
    model = model.to(device)
    model.eval()
    return model

# Try to load the model
try:
    model = load_model_cached()
    st.success("✅ Model loaded successfully!")
    st.write(f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    st.write("The application should now work correctly.")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.write("Please check the error message and troubleshoot accordingly.") 