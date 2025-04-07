#!/usr/bin/env python3
"""
Data preparation script for scene recognition datasets.
This script downloads and prepares MIT Indoor67 and SUN397 datasets.
"""

import os
import sys
import argparse
import shutil
import urllib.request
import tarfile
import zipfile
from tqdm import tqdm
import random
import csv

# Add the project root to import from config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, ROOT_DIR

# URLs for downloading datasets
DATASET_URLS = {
    'sun397': 'http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz',
    'indoor67': 'http://web.mit.edu/torralba/www/indoor.zip'
}

def download_file(url, destination):
    """Download a file with progress bar"""
    if os.path.exists(destination):
        print(f"File already exists: {destination}")
        return
    
    print(f"Downloading {url} to {destination}")
    
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, destination, reporthook=t.update_to)

def extract_archive(archive_path, extract_dir):
    """Extract archive file (tar.gz, zip, etc.)"""
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)
    
    print(f"Extracting {archive_path} to {extract_dir}")
    
    if archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(path=extract_dir)
    elif archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")
    
    print(f"Extraction complete: {archive_path}")

def prepare_sun397(data_dir):
    """Prepare SUN397 dataset"""
    sun_dir = os.path.join(data_dir, 'SUN397')
    download_path = os.path.join(data_dir, 'SUN397.tar.gz')
    extract_dir = data_dir
    
    # Download dataset
    download_file(DATASET_URLS['sun397'], download_path)
    
    # Extract dataset
    if not os.path.exists(sun_dir):
        extract_archive(download_path, extract_dir)
    
    # Create train/val/test splits
    sun_train_dir = os.path.join(data_dir, 'sun397', 'train')
    sun_val_dir = os.path.join(data_dir, 'sun397', 'val')
    sun_test_dir = os.path.join(data_dir, 'sun397', 'test')
    
    # Create directories
    os.makedirs(sun_train_dir, exist_ok=True)
    os.makedirs(sun_val_dir, exist_ok=True)
    os.makedirs(sun_test_dir, exist_ok=True)
    
    # Get all class directories
    class_dirs = []
    for root, dirs, files in os.walk(sun_dir):
        for dir_name in dirs:
            if dir_name.startswith('a') or dir_name.startswith('b'):  # SUN397 class naming pattern
                class_path = os.path.join(root, dir_name)
                if os.path.isdir(class_path) and len(os.listdir(class_path)) > 0:
                    class_dirs.append((dir_name, class_path))
    
    print("Creating SUN397 dataset splits...")
    for class_name, class_path in tqdm(class_dirs, desc="Processing SUN397 classes"):
        # Create class directories in train/val/test
        os.makedirs(os.path.join(sun_train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(sun_val_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(sun_test_dir, class_name), exist_ok=True)
        
        # Get all images in class directory
        image_files = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(image_files)
        
        # Split images (70% train, 15% val, 15% test)
        n_train = int(len(image_files) * 0.7)
        n_val = int(len(image_files) * 0.15)
        
        train_images = image_files[:n_train]
        val_images = image_files[n_train:n_train+n_val]
        test_images = image_files[n_train+n_val:]
        
        # Process train images
        for img_file in train_images:
            src_path = os.path.join(class_path, img_file)
            dst_path = os.path.join(sun_train_dir, class_name, img_file)
            shutil.copy2(src_path, dst_path)
        
        # Process validation images
        for img_file in val_images:
            src_path = os.path.join(class_path, img_file)
            dst_path = os.path.join(sun_val_dir, class_name, img_file)
            shutil.copy2(src_path, dst_path)
        
        # Process test images
        for img_file in test_images:
            src_path = os.path.join(class_path, img_file)
            dst_path = os.path.join(sun_test_dir, class_name, img_file)
            shutil.copy2(src_path, dst_path)
    
    print(f"SUN397 dataset prepared in {os.path.join(data_dir, 'sun397')}")
    print(f"- Train images: {os.path.join(data_dir, 'sun397', 'train')}")
    print(f"- Validation images: {os.path.join(data_dir, 'sun397', 'val')}")
    print(f"- Test images: {os.path.join(data_dir, 'sun397', 'test')}")

def prepare_indoor67(data_dir):
    """Prepare MIT Indoor67 dataset"""
    indoor_dir = os.path.join(data_dir, 'indoor')
    download_path = os.path.join(data_dir, 'indoor.zip')
    extract_dir = data_dir
    
    # Download dataset
    download_file(DATASET_URLS['indoor67'], download_path)
    
    # Extract dataset
    if not os.path.exists(indoor_dir):
        extract_archive(download_path, extract_dir)
    
    # Create train/val/test splits
    indoor_train_dir = os.path.join(data_dir, 'indoor67', 'train')
    indoor_val_dir = os.path.join(data_dir, 'indoor67', 'val')
    indoor_test_dir = os.path.join(data_dir, 'indoor67', 'test')
    
    # Create directories
    os.makedirs(indoor_train_dir, exist_ok=True)
    os.makedirs(indoor_val_dir, exist_ok=True)
    os.makedirs(indoor_test_dir, exist_ok=True)
    
    # Get all class directories
    Images_dir = os.path.join(indoor_dir, 'Images')
    class_dirs = [(d, os.path.join(Images_dir, d)) for d in os.listdir(Images_dir) 
                  if os.path.isdir(os.path.join(Images_dir, d))]
    
    print("Creating MIT Indoor67 dataset splits...")
    for class_name, class_path in tqdm(class_dirs, desc="Processing Indoor67 classes"):
        # Create class directories in train/val/test
        os.makedirs(os.path.join(indoor_train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(indoor_val_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(indoor_test_dir, class_name), exist_ok=True)
        
        # Get all images in class directory
        image_files = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(image_files)
        
        # Split images (70% train, 15% val, 15% test)
        n_train = int(len(image_files) * 0.7)
        n_val = int(len(image_files) * 0.15)
        
        train_images = image_files[:n_train]
        val_images = image_files[n_train:n_train+n_val]
        test_images = image_files[n_train+n_val:]
        
        # Process train images
        for img_file in train_images:
            src_path = os.path.join(class_path, img_file)
            dst_path = os.path.join(indoor_train_dir, class_name, img_file)
            shutil.copy2(src_path, dst_path)
        
        # Process validation images
        for img_file in val_images:
            src_path = os.path.join(class_path, img_file)
            dst_path = os.path.join(indoor_val_dir, class_name, img_file)
            shutil.copy2(src_path, dst_path)
        
        # Process test images
        for img_file in test_images:
            src_path = os.path.join(class_path, img_file)
            dst_path = os.path.join(indoor_test_dir, class_name, img_file)
            shutil.copy2(src_path, dst_path)
    
    print(f"MIT Indoor67 dataset prepared in {os.path.join(data_dir, 'indoor67')}")
    print(f"- Train images: {os.path.join(data_dir, 'indoor67', 'train')}")
    print(f"- Validation images: {os.path.join(data_dir, 'indoor67', 'val')}")
    print(f"- Test images: {os.path.join(data_dir, 'indoor67', 'test')}")

def main():
    parser = argparse.ArgumentParser(description='Prepare scene recognition datasets')
    parser.add_argument('--dataset', type=str, default='all', choices=['sun397', 'indoor67', 'all'],
                        help='Dataset to prepare (sun397, indoor67, or all)')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR,
                        help='Directory to store datasets')
    
    args = parser.parse_args()
    
    # Create data directory
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Prepare datasets
    if args.dataset in ['sun397', 'all']:
        prepare_sun397(args.data_dir)
    
    if args.dataset in ['indoor67', 'all']:
        prepare_indoor67(args.data_dir)
    
    print("Dataset preparation complete!")

if __name__ == '__main__':
    main() 