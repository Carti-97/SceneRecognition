import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import v2
from PIL import Image
import pandas as pd
import os

# ImageNet normalization values for pretrained models
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

class SceneDataset(Dataset):
    """
    Dataset class for scene recognition datasets (SUN397, MIT Indoor67)
    """
    def __init__(self, image_dir, tags_csv, train=True):
        self.image_dir = image_dir
        self.tags_df = pd.read_csv(tags_csv)
        
        # Extract image paths and tags from CSV file
        self.image_paths = self.tags_df['image_path'].tolist()
        self.tags = self.tags_df['tag'].tolist()
        
        # Handle relative or absolute paths in the CSV
        self.full_image_paths = []
        for path in self.image_paths:
            if os.path.isabs(path):
                self.full_image_paths.append(path)
            else:
                # Handle relative paths in the CSV
                # First check if path contains subdirectories
                if os.path.dirname(path):
                    # If path already contains subdirectories like 'train/class/image.jpg'
                    self.full_image_paths.append(os.path.join(os.path.dirname(image_dir), path))
                else:
                    # If path is just a filename
                    self.full_image_paths.append(os.path.join(image_dir, path))
        
        # Get class labels
        if 'class' in self.tags_df.columns:
            # If CSV has class column, use it directly
            self.labels = self.tags_df['class'].tolist()
        else:
            # Otherwise infer labels from image paths
            self.labels = [os.path.basename(os.path.dirname(path)) for path in self.full_image_paths]
        
        # Create label to index mapping
        self.label_to_index = {label: idx for idx, label in enumerate(sorted(set(self.labels)))}
        
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
        return len(self.full_image_paths)

    def __getitem__(self, idx):
        image_path = self.full_image_paths[idx]
        label = self.labels[idx]
        tag = self.tags[idx] if idx < len(self.tags) else ""
        
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