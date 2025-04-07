import torch
import os

# Root directory of the project
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Device configuration
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Training mode ('rgb' or 'dynamic')
TRAIN_MODE = 'rgb'  # Change to 'dynamic' for training the dynamic integration model

# Dataset selection ('sun397' or 'indoor67')
DATASET = 'sun397'  # Change to 'indoor67' for MIT Indoor 67 dataset

# Data paths
DATA_DIR = os.path.join(ROOT_DIR, 'data')

# Dataset-specific paths
SUN397_DIR = os.path.join(DATA_DIR, 'sun397')
SUN397_TRAIN_IMAGES = os.path.join(SUN397_DIR, 'train')
SUN397_TRAIN_TAGS = os.path.join(SUN397_DIR, 'train_tags.csv')
SUN397_VAL_IMAGES = os.path.join(SUN397_DIR, 'val')
SUN397_VAL_TAGS = os.path.join(SUN397_DIR, 'val_tags.csv')
SUN397_TEST_IMAGES = os.path.join(SUN397_DIR, 'test')
SUN397_TEST_TAGS = os.path.join(SUN397_DIR, 'test_tags.csv')
SUN397_NUM_CLASSES = 397

INDOOR67_DIR = os.path.join(DATA_DIR, 'indoor67')
INDOOR67_TRAIN_IMAGES = os.path.join(INDOOR67_DIR, 'train')
INDOOR67_TRAIN_TAGS = os.path.join(INDOOR67_DIR, 'train_tags.csv')
INDOOR67_VAL_IMAGES = os.path.join(INDOOR67_DIR, 'val')
INDOOR67_VAL_TAGS = os.path.join(INDOOR67_DIR, 'val_tags.csv')
INDOOR67_TEST_IMAGES = os.path.join(INDOOR67_DIR, 'test')
INDOOR67_TEST_TAGS = os.path.join(INDOOR67_DIR, 'test_tags.csv')
INDOOR67_NUM_CLASSES = 67

# Set active dataset paths based on DATASET variable
if DATASET == 'sun397':
    TRAIN_IMAGES = SUN397_TRAIN_IMAGES
    TRAIN_TAGS = SUN397_TRAIN_TAGS
    VAL_IMAGES = SUN397_VAL_IMAGES
    VAL_TAGS = SUN397_VAL_TAGS
    TEST_IMAGES = SUN397_TEST_IMAGES
    TEST_TAGS = SUN397_TEST_TAGS
    NUM_CLASSES = SUN397_NUM_CLASSES
    DATASET_NAME = 'SUN397'
elif DATASET == 'indoor67':
    TRAIN_IMAGES = INDOOR67_TRAIN_IMAGES
    TRAIN_TAGS = INDOOR67_TRAIN_TAGS
    VAL_IMAGES = INDOOR67_VAL_IMAGES
    VAL_TAGS = INDOOR67_VAL_TAGS
    TEST_IMAGES = INDOOR67_TEST_IMAGES
    TEST_TAGS = INDOOR67_TEST_TAGS
    NUM_CLASSES = INDOOR67_NUM_CLASSES
    DATASET_NAME = 'MIT Indoor67'
else:
    raise ValueError(f"Unsupported dataset: {DATASET}. Choose 'sun397' or 'indoor67'.")

# Checkpoint paths
CHECKPOINT_DIR = os.path.join(ROOT_DIR, 'checkpoints', DATASET)
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, f'rgb_model_{DATASET}.pth')
DYNAMIC_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, f'dynamic_model_{DATASET}.pth')

# Training hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
EPOCHS = 50  # Reduced for faster training

# Tag generation parameters
TAG_MODEL_TYPE = 'dummy'  # Options: 'dummy', 'ram', 'tag2text'
TAG_MODEL_PATH = ''  # Path to the tag generation model if needed

# Data transformation settings
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225] 