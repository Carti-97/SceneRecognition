import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import v2
from PIL import Image
import os
import glob

from config import MEAN, STD

class SceneDataset(Dataset):
    """
    Dataset class for scene recognition datasets (SUN397, MIT Indoor67)
    """
    def __init__(self, image_dir, train=True):
        self.image_dir = image_dir
        
        # Find all image files in the directory
        self.image_extensions = ['.jpg', '.jpeg', '.png']
        self.image_paths = []
        
        # Get all image files recursively from directory
        for root, _, files in os.walk(image_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in self.image_extensions):
                    self.image_paths.append(os.path.join(root, file))
        
        # Get class labels from directory structure
        # Assumes image_dir/class_name/image.jpg structure
        self.labels = [os.path.basename(os.path.dirname(path)) for path in self.image_paths]
        
        # Create label to index mapping
        self.classes = sorted(list(set(self.labels)))
        self.label_to_index = {label: idx for idx, label in enumerate(self.classes)}
        
        # Create appropriate transforms
        if train:
            self.transform = v2.Compose([
                v2.RandomHorizontalFlip(),
                v2.RandomResizedCrop(size=(224, 224)),
                v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                v2.RandomAdjustSharpness(sharpness_factor=2),
                v2.ColorJitter(brightness=0.2, contrast=0.2),
                v2.ToTensor(),
                v2.Normalize(mean=MEAN, std=STD)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.TenCrop(size=(224, 224)),
                transforms.Lambda(lambda crops: torch.stack([
                    transforms.Normalize(mean=MEAN, std=STD)(
                        transforms.ToTensor()(crop)
                    ) for crop in crops
                ]))
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load and convert image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a blank image in case of error
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transforms
        transformed_crops = self.transform(image)
        
        # Get label index
        label_index = self.label_to_index[label]
        
        # Return image and label
        return transformed_crops, label_index 