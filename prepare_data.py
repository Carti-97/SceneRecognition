import os
import shutil
import pandas as pd
from config import DATA_DIR, TRAIN_IMAGES, TRAIN_TAGS, VAL_IMAGES, VAL_TAGS

def prepare_dataset_structure():
    """
    Prepare dataset structure
    """
    # Create necessary directories
    os.makedirs(TRAIN_IMAGES, exist_ok=True)
    os.makedirs(VAL_IMAGES, exist_ok=True)
    
    print(f"Dataset directories have been created:")
    print(f"- Training images directory: {TRAIN_IMAGES}")
    print(f"- Validation images directory: {VAL_IMAGES}")
    
    # Create sample tag files
    train_tags_sample = pd.DataFrame({
        'image_path': ['sample_image_1.jpg', 'sample_image_2.jpg'],
        'tag': ['indoor', 'outdoor']
    })
    
    val_tags_sample = pd.DataFrame({
        'image_path': ['sample_image_3.jpg', 'sample_image_4.jpg'],
        'tag': ['indoor', 'outdoor']
    })
    
    # Save CSV files
    os.makedirs(os.path.dirname(TRAIN_TAGS), exist_ok=True)
    os.makedirs(os.path.dirname(VAL_TAGS), exist_ok=True)
    
    train_tags_sample.to_csv(TRAIN_TAGS, index=False)
    val_tags_sample.to_csv(VAL_TAGS, index=False)
    
    print("Sample tag files have been created:")
    print(f"- Training tags file: {TRAIN_TAGS}")
    print(f"- Validation tags file: {VAL_TAGS}")
    
    print("\nData preparation completed!")
    print("Place your data files as follows:")
    print("1. Copy training images to data/SUN397_split/train/ directory")
    print("2. Copy validation images to data/SUN397_split/val/ directory")
    print("3. Modify tag files according to your needs")

if __name__ == "__main__":
    prepare_dataset_structure() 