import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import v2
from PIL import Image
import pandas as pd
import os
from config import IMAGE_SIZE, MEAN, STD

class SceneDataset(Dataset):
    def __init__(self, image_dir, tags_csv, train=True):
        self.image_dir = image_dir
        self.tags_df = pd.read_csv(tags_csv)
        
        # Extract image paths and tags from CSV file
        self.image_paths = self.tags_df['image_path'].tolist()
        
        # Convert image paths to relative paths
        self.image_paths = [os.path.join(self.image_dir, os.path.basename(path)) 
                          for path in self.image_paths]
        
        self.tags = self.tags_df['tag'].tolist()
        
        # Labels are folder names from image paths
        self.labels = [os.path.basename(os.path.dirname(path)) for path in self.image_paths]
        self.label_to_index = {label: idx for idx, label in enumerate(sorted(set(self.labels)))}

        if train:
            self.transform = v2.Compose([
                v2.RandomHorizontalFlip(),
                v2.RandomResizedCrop(size=(IMAGE_SIZE, IMAGE_SIZE)),
                v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                v2.RandomAdjustSharpness(sharpness_factor=2),
                v2.ColorJitter(brightness=0.2, contrast=0.2),
                v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                v2.ToTensor(),
                v2.Normalize(mean=MEAN, std=STD)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.TenCrop(size=(IMAGE_SIZE, IMAGE_SIZE)),
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
        
        image = Image.open(image_path).convert('RGB')
        transformed_crops = self.transform(image)
        
        label_index = self.label_to_index[label]
        return transformed_crops, label_index 