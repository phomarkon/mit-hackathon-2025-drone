import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet50_Weights, Wide_ResNet50_2_Weights

class FeatureExtractor(nn.Module):
    """Feature extractor based on a pretrained CNN backbone."""
    
    def __init__(self, backbone_name='resnet18', layers=None):
        """
        Args:
            backbone_name (str): Name of the backbone CNN (e.g., 'resnet18', 'resnet50')
            layers (list): List of layer names to extract features from
        """
        super(FeatureExtractor, self).__init__()
        self.backbone_name = backbone_name
        self.layers = layers or ["layer2", "layer3"]  # Default to these layers if none specified
        
        # Get the pretrained model
        if backbone_name == 'resnet18':
            self.backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        elif backbone_name == 'resnet50':
            self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        elif backbone_name == 'wide_resnet50_2':
            self.backbone = models.wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        # Set the model to evaluation mode and freeze parameters
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Get the layers that we want to extract features from
        self.feature_layers = {}
        for name, module in self.backbone.named_children():
            if name in self.layers:
                self.feature_layers[name] = module
        
    def forward(self, x):
        """
        Forward pass through the feature extractor.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary of feature maps from the specified layers
        """
        features = {}
        
        # Pass through the backbone up to the final layer we're interested in
        for name, module in self.backbone.named_children():
            x = module(x)
            if name in self.layers:
                features[name] = x
                
            # If we've reached the last layer we care about, we can stop
            if name == self.layers[-1]:
                break
                
        return features 