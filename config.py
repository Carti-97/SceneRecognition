import torch
import os

# Set project root directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Basic settings
DEVICE = 'cuda:1'
NUM_CLASSES = 397
LEARNING_RATE = 0.0001
BATCH_SIZE = 16
EPOCHS = 200

# Data paths (relative paths)
DATA_DIR = os.path.join(ROOT_DIR, 'data')
TRAIN_IMAGES = os.path.join(DATA_DIR, 'SUN397_split', 'train')
TRAIN_TAGS = os.path.join(DATA_DIR, 'SUN397_split', 'train_tags.csv')
VAL_IMAGES = os.path.join(DATA_DIR, 'SUN397_split', 'val')
VAL_TAGS = os.path.join(DATA_DIR, 'SUN397_split', 'val_tags.csv')

# Model checkpoints (relative paths)
CHECKPOINT_DIR = os.path.join(ROOT_DIR, 'checkpoints')
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'RGB_ResNet50_SUN.pth.tar')

# Data transformation settings
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225] 