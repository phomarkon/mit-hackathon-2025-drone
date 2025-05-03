import os
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class AnomalyDataset(Dataset):
    """Custom Dataset for loading anomaly detection data."""
    def __init__(self, root_dir, transform=None, is_train=True):
        """
        Args:
            root_dir (string): Directory with all the images (for train) or 
                             parent directory containing 'normal' and 'abnormal' 
                             subdirs (for test).
            transform (callable, optional): Optional transform to be applied 
                on a sample.
            is_train (bool): Flag indicating if this is the training dataset.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        self.image_paths = []
        self.labels = [] # 0 for normal, 1 for abnormal

        if self.is_train:
            # Training data only contains normal images
            self.image_paths = sorted(glob(os.path.join(self.root_dir, "*.jpg"))) # Assuming jpg format
            self.labels = [0] * len(self.image_paths)
        else:
            # Test data contains normal and abnormal images in subdirectories
            normal_paths = sorted(glob(os.path.join(self.root_dir, "normal", "*.jpg")))
            abnormal_paths = sorted(glob(os.path.join(self.root_dir, "abnormal", "*.jpg")))
            self.image_paths = normal_paths + abnormal_paths
            self.labels = [0] * len(normal_paths) + [1] * len(abnormal_paths)

        if not self.image_paths:
            raise RuntimeError(f"Found 0 images in {self.root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder or skip? For now, re-raise.
            raise e
            
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label, img_path # Return image path for potential debugging/analysis

def get_transforms(image_size):
    """Gets the transformations for the dataset."""
    # Use ImageNet mean and std dev for normalization with pre-trained models
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # Note: The strategy guide mentioned CLAHE for thermal data (SAR track).
    # This could be added here conditionally based on config if we pursue that track.
    # For now, standard transforms.
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

def get_dataloaders(config):
    """Creates training and testing dataloaders."""
    image_size = config['data']['image_size']
    train_batch_size = config['training']['batch_size']
    eval_batch_size = config['evaluation']['batch_size']
    num_workers = config['training']['num_workers'] # Use same for eval

    transform = get_transforms(image_size)

    train_dataset = AnomalyDataset(
        root_dir=config['data']['train_path'],
        transform=transform,
        is_train=True
    )
    # The test root path needs to be the parent directory containing 'normal' and 'abnormal'
    # Assuming the config paths point directly to the image folders, we need the parent of test paths
    test_root = os.path.dirname(config['data']['test_normal_path']) 
    test_dataset = AnomalyDataset(
        root_dir=test_root, # Pass the parent directory 'data/test'
        transform=transform,
        is_train=False
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False, # No need to shuffle test data
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader 