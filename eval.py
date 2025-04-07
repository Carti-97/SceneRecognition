import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys
from tqdm import tqdm
import argparse

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset import SceneDataset
from model import RGBBranch, DynamicContextIntegration
from config import *

def evaluate_rgb_branch(model, val_loader, device):
    """Evaluate the RGB branch model"""
    model.eval()
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for inputs_image, labels in tqdm(val_loader, desc="Evaluating RGB Model", unit="batch"):
            bs, crops, c, h, w = inputs_image.size()
            inputs_image = inputs_image.view(-1, c, h, w).to(device)
            labels = labels.to(dtype=torch.long, device=device)

            outputs = model(inputs_image)
            outputs = outputs.view(bs, crops, -1)
            
            _, preds = torch.max(outputs, 2)
            predicted_val = torch.mode(preds, 1).values

            total_val += labels.size(0)
            correct_val += (predicted_val == labels).sum().item()

    return correct_val / total_val

def evaluate_dynamic_integration(model, val_loader, device):
    """Evaluate the dynamic integration model"""
    model.eval()
    correct_val = 0
    total_val = 0

    # Get RAM model for tag generation if available
    if TAG_MODEL_TYPE.lower() == 'ram':
        try:
            from ram.inference import inference_ram
            print("Using RAM model for tag generation")
            # TODO: Load the RAM model here
            use_ram = True
        except ImportError:
            print("RAM model not available, using dummy tags")
            use_ram = False
    else:
        use_ram = False

    # Tag generation function
    def generate_tags_for_images(images, batch_size=32):
        """
        Generate tags for images using RAM or a placeholder
        """
        if use_ram:
            # TODO: Implement actual RAM tag generation
            # This is a placeholder for now
            return ["indoor scene with furniture, living room" for _ in range(batch_size)]
        else:
            # Mock tags for each image in the batch
            return ["indoor scene with furniture, living room" for _ in range(batch_size)]

    with torch.no_grad():
        for inputs_image, labels in tqdm(val_loader, desc="Evaluating Dynamic Model", unit="batch"):
            bs, crops, c, h, w = inputs_image.size()
            inputs_image = inputs_image.view(-1, c, h, w).to(device)
            labels = labels.to(dtype=torch.long, device=device)

            # Generate tags for validation images
            tags = generate_tags_for_images(inputs_image, bs * crops)

            # Forward pass with both images and tags
            outputs = model(inputs_image, tags)
            outputs = outputs.view(bs, crops, -1)
            
            _, preds = torch.max(outputs, 2)
            predicted_val = torch.mode(preds, 1).values

            total_val += labels.size(0)
            correct_val += (predicted_val == labels).sum().item()

    return correct_val / total_val

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Evaluate scene recognition models')
    parser.add_argument('--model', type=str, default='rgb', choices=['rgb', 'dynamic'],
                        help='Model type to evaluate (rgb or dynamic)')
    args = parser.parse_args()

    # Create dataset and dataloader
    val_dataset = SceneDataset(VAL_IMAGES, train=False)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=4,
        pin_memory=True
    )

    if args.model == 'rgb':
        # Initialize RGB branch model
        model = RGBBranch(arch='ResNet-50', scene_classes=NUM_CLASSES)
        model.to(DEVICE)

        # Load checkpoint
        if os.path.exists(CHECKPOINT_PATH):
            checkpoint = torch.load(CHECKPOINT_PATH)
            model.load_state_dict(checkpoint['state_dict'])
            print(f"Loaded RGB branch checkpoint from epoch {checkpoint['epoch']} with validation accuracy: {checkpoint['best_val_accuracy']:.4f}")
        else:
            print(f"RGB branch checkpoint file not found at {CHECKPOINT_PATH}. Using model with random weights.")

        # Run evaluation
        accuracy = evaluate_rgb_branch(model, val_loader, DEVICE)
        print(f'RGB Branch Model Validation Accuracy: {accuracy:.4f}')

    elif args.model == 'dynamic':
        # Initialize dynamic integration model
        model = DynamicContextIntegration(backbone='ResNet-50', num_classes=NUM_CLASSES)
        model.to(DEVICE)

        # Load checkpoint
        if os.path.exists(DYNAMIC_CHECKPOINT_PATH):
            checkpoint = torch.load(DYNAMIC_CHECKPOINT_PATH)
            model.load_state_dict(checkpoint['state_dict'])
            print(f"Loaded dynamic model checkpoint from epoch {checkpoint['epoch']} with validation accuracy: {checkpoint['best_val_accuracy']:.4f}")
        else:
            print(f"Dynamic model checkpoint file not found at {DYNAMIC_CHECKPOINT_PATH}. Using model with random weights.")

        # Run evaluation
        accuracy = evaluate_dynamic_integration(model, val_loader, DEVICE)
        print(f'Dynamic Integration Model Validation Accuracy: {accuracy:.4f}')

if __name__ == '__main__':
    main() 