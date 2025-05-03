import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from src.models.feature_extractor import FeatureExtractor
from sklearn.random_projection import SparseRandomProjection
from sklearn.metrics import roc_auc_score, roc_curve
import os

class PatchCore(nn.Module):
    """
    PatchCore anomaly detection model.
    """
    def __init__(self, backbone_name="resnet18", layers=None, coreset_sampling_ratio=0.1, device="cuda"):
        """
        Args:
            backbone_name (str): Name of the backbone CNN
            layers (list): Layers to extract features from
            coreset_sampling_ratio (float): Ratio of patches to keep in the memory bank
            device (str): Device to use ('cuda' or 'cpu')
        """
        super(PatchCore, self).__init__()
        self.device = device
        self.backbone_name = backbone_name
        self.layers = layers or ["layer2", "layer3"]
        self.coreset_sampling_ratio = coreset_sampling_ratio
        
        # Initialize the feature extractor
        self.feature_extractor = FeatureExtractor(backbone_name=backbone_name, layers=self.layers)
        self.feature_extractor.to(device)
        
        # Memory bank for normal patches
        self.memory_bank = None
        
    def _embed_patches(self, features):
        """
        Extracts, interpolates, and concatenates patch embeddings from feature maps.
        
        Args:
            features (dict): Dictionary of feature maps from different layers
            
        Returns:
            Tuple[torch.Tensor, tuple]: Concatenated patch embeddings and the size of the feature map grid (H, W)
        """
        # Process feature maps: interpolate and create patches
        embeddings_list = []
        target_h, target_w = -1, -1 # Determine target size from the first layer

        # Determine target spatial size (usually from the earliest layer)
        if self.layers:
            first_layer_features = features[self.layers[0]]
            target_h, target_w = first_layer_features.shape[2], first_layer_features.shape[3]
        
        for layer_name in self.layers:
            feature_map = features[layer_name]
            batch_size, num_channels, height, width = feature_map.shape
            
            # Interpolate feature map if necessary
            if height != target_h or width != target_w:
                feature_map = F.interpolate(feature_map, size=(target_h, target_w), mode='bilinear', align_corners=False)
            
            # Reshape to patches: [B, H*W, C]
            patches = feature_map.permute(0, 2, 3, 1).reshape(batch_size, target_h * target_w, num_channels)
            embeddings_list.append(patches)

        # Concatenate embeddings from different layers along the channel dimension
        # Result: [B, H*W, C1+C2+...]
        concatenated_embeddings = torch.cat(embeddings_list, dim=2)
        
        # Reshape to [B * H*W, C_total]
        batch_size, num_patches, total_channels = concatenated_embeddings.shape
        all_embeddings = concatenated_embeddings.reshape(batch_size * num_patches, total_channels)
        
        return all_embeddings, (target_h, target_w)
    
    def _greedy_coreset_selection(self, embeddings, sampling_ratio):
        """
        Greedy coreset selection algorithm to reduce memory bank size.
        
        Args:
            embeddings (np.ndarray): Embeddings to sample from, shape [N, D]
            sampling_ratio (float): Ratio of embeddings to keep
            
        Returns:
            np.ndarray: Indices of selected embeddings
        """
        if sampling_ratio >= 1.0:
            return np.arange(len(embeddings))  # Return all indices if ratio >= 1
            
        n_samples = int(len(embeddings) * sampling_ratio)
        
        # Make sure we select at least one sample
        n_samples = max(1, n_samples)
        
        # Initialize with a random example
        selected_indices = [np.random.choice(len(embeddings))]
        selected_samples = embeddings[selected_indices]
        
        # Compute distances to the selected samples for all points
        min_distances = np.linalg.norm(embeddings - selected_samples[0].reshape(1, -1), axis=1)
        
        # Greedily select the remaining samples
        with tqdm(total=n_samples-1, desc="Selecting coreset", leave=True) as pbar:
            for _ in range(n_samples - 1):
                # Select the point with the maximum minimum distance to already selected points
                new_idx = np.argmax(min_distances)
                selected_indices.append(new_idx)
                
                # Update selected samples
                if len(selected_samples.shape) == 1:  # If only one sample so far
                    selected_samples = embeddings[selected_indices]
                else:
                    selected_samples = np.vstack([selected_samples, embeddings[new_idx]])
                
                # Update min distances by comparing with the newly selected point
                new_distances = np.linalg.norm(embeddings - embeddings[new_idx].reshape(1, -1), axis=1)
                min_distances = np.minimum(min_distances, new_distances)
                
                pbar.update(1)
            
        return np.array(selected_indices)
    
    def build_memory_bank(self, dataloader):
        """
        Build memory bank from normal training samples.
        
        Args:
            dataloader: DataLoader for normal training images
        """
        self.feature_extractor.eval()
        patch_features_list = []
        self.patch_grid_size = None # Store the grid size H, W
        
        with torch.no_grad():
            with tqdm(dataloader, desc="Extracting features", leave=True) as pbar:
                for batch in pbar:
                    # Unpack and move to device
                    images, _, _ = batch  # Ignore labels and paths for training
                    images = images.to(self.device)
                    
                    # Extract features
                    features = self.feature_extractor(images)
                    
                    # Extract patch embeddings and grid size
                    batch_patch_features, grid_size = self._embed_patches(features)
                    patch_features_list.append(batch_patch_features.cpu())
                    
                    if self.patch_grid_size is None:
                        self.patch_grid_size = grid_size
                    elif self.patch_grid_size != grid_size:
                        # This should not happen if input image sizes are consistent
                        raise ValueError("Inconsistent feature map grid sizes detected.")

                    
                    # Show current progress details
                    pbar.set_postfix({"Patches": sum(feat.shape[0] for feat in patch_features_list)})
                
        # Concatenate all patch features
        if patch_features_list:
            patch_features = torch.cat(patch_features_list, dim=0).numpy()
            print(f"Total patches extracted: {patch_features.shape[0]}")
            
            # Apply coreset selection if ratio < 1.0
            if self.coreset_sampling_ratio < 1.0:
                print(f"Applying coreset selection with ratio {self.coreset_sampling_ratio}")
                print(f"Will select approximately {int(patch_features.shape[0] * self.coreset_sampling_ratio)} patches from {patch_features.shape[0]} total patches")
                selected_indices = self._greedy_coreset_selection(patch_features, self.coreset_sampling_ratio)
                self.memory_bank = patch_features[selected_indices]
            else:
                self.memory_bank = patch_features
                
            print(f"Memory bank built with {len(self.memory_bank)} patches")
        else:
            print("No patch features extracted. Check the feature extractor.")
            self.memory_bank = np.array([])
        
    def compute_anomaly_scores(self, dataloader):
        """
        Compute anomaly scores for test images.
        
        Args:
            dataloader: DataLoader for test images
            
        Returns:
            tuple: (anomaly_scores, labels, image_paths)
        """
        if self.memory_bank is None or len(self.memory_bank) == 0:
            raise RuntimeError("Memory bank is empty or not built. Call build_memory_bank first.")
        if self.patch_grid_size is None:
             raise RuntimeError("Patch grid size not set. Build memory bank first.")
            
        self.feature_extractor.eval()
        
        anomaly_scores = []
        gt_labels = []
        paths = []
        # Store patch scores for potential visualization later (optional)
        # patch_score_maps = [] 
        
        h, w = self.patch_grid_size
        num_patches_per_image = h * w

        with torch.no_grad():
            with tqdm(dataloader, desc="Computing anomaly scores", leave=True) as pbar:
                for batch in pbar:
                    # Unpack and move to device
                    images, labels, image_paths = batch
                    images = images.to(self.device)
                    batch_size = images.shape[0]
                    
                    # Extract features
                    features = self.feature_extractor(images)
                    
                    # Extract patch embeddings
                    # Note: grid_size is implicitly handled by _embed_patches output shape
                    batch_patch_features, _ = self._embed_patches(features)
                    query_patches = batch_patch_features.cpu().numpy()
                    
                    # Compute distances to memory bank for all patches in the batch
                    all_patch_scores = []
                    for query_patch in query_patches:
                        distances = np.linalg.norm(self.memory_bank - query_patch.reshape(1, -1), axis=1)
                        min_distance = np.min(distances)
                        all_patch_scores.append(min_distance)
                    
                    all_patch_scores = np.array(all_patch_scores)
                    
                    # Reshape scores to match [Batch, NumPatches]
                    if len(all_patch_scores) != batch_size * num_patches_per_image:
                         # Handle potential issue if batch sizes drop
                         current_num_patches = len(all_patch_scores)
                         expected_num_patches = batch_size * num_patches_per_image
                         if current_num_patches % num_patches_per_image == 0:
                              actual_batch_size = current_num_patches // num_patches_per_image
                              scores_per_image = all_patch_scores.reshape(actual_batch_size, num_patches_per_image)
                         else:
                              print(f"Warning: Mismatch in expected patch count ({expected_num_patches}) and actual ({current_num_patches}). Skipping score calculation for batch.")
                              continue # Or handle differently
                    else:
                         scores_per_image = all_patch_scores.reshape(batch_size, num_patches_per_image)

                    # Image anomaly score is the maximum patch anomaly score for each image
                    image_scores = np.max(scores_per_image, axis=1)
                    
                    # Store results for the batch
                    anomaly_scores.extend(image_scores.tolist())
                    gt_labels.extend(labels.cpu().numpy().tolist())
                    paths.extend(list(image_paths))
                    
                    # Update progress bar with count info
                    pbar.set_postfix({"Scored": len(anomaly_scores)})
                    
        return np.array(anomaly_scores), np.array(gt_labels), paths
    
    def save(self, path):
        """Save the model memory bank."""
        data = {
            'memory_bank': self.memory_bank,
            'patch_grid_size': self.patch_grid_size,
            'backbone_name': self.backbone_name,
            'layers': self.layers,
            'coreset_sampling_ratio': self.coreset_sampling_ratio
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(data, path)
        print(f"Model saved to {path}")
        
    def load(self, path):
        """Load the model memory bank."""
        data = torch.load(path)
        self.memory_bank = data['memory_bank']
        self.patch_grid_size = data.get('patch_grid_size', None) # Handle older saves
        self.backbone_name = data['backbone_name']
        self.layers = data['layers']
        self.coreset_sampling_ratio = data['coreset_sampling_ratio']
        print(f"Model loaded from {path}")
        
    def evaluate(self, anomaly_scores, gt_labels):
        """
        Evaluate the model using ROC AUC.
        
        Args:
            anomaly_scores (np.ndarray): Anomaly scores for test images
            gt_labels (np.ndarray): Ground truth labels (0=normal, 1=anomaly)
            
        Returns:
            float: AUROC (Area Under ROC Curve)
        """
        # Compute the AUROC
        auroc = roc_auc_score(gt_labels, anomaly_scores)
        
        # Compute ROC curve for plotting if needed
        fpr, tpr, thresholds = roc_curve(gt_labels, anomaly_scores)
        
        return auroc, fpr, tpr, thresholds 