import torch
import os
from torch.utils.data import DataLoader
from utils.dataset import SceneDataset
from SASNET.RGBBranch import RGBBranch
from config import *

def evaluate(model, val_loader, device):
    model.eval()
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for inputs_image, labels in val_loader:
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

def main():
    # Initialize model
    model = RGBBranch(arch='ResNet-50', scene_classes=NUM_CLASSES)
    model.to(DEVICE)

    # Load checkpoint
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with validation accuracy: {checkpoint['best_val_accuracy']:.4f}")
    else:
        print(f"Checkpoint file not found at {CHECKPOINT_PATH}. Using model with random weights.")

    # Create dataset and dataloader
    val_dataset = SceneDataset(VAL_IMAGES, VAL_TAGS, train=False)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=4,
        pin_memory=True
    )

    # Run evaluation
    accuracy = evaluate(model, val_loader, DEVICE)
    print(f'Final Validation Accuracy: {accuracy:.4f}')

if __name__ == '__main__':
    main() 