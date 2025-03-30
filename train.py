import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from utils.dataset import SceneDataset
from utils.trainer import Trainer
from SASNET.RGBBranch import RGBBranch
from config import *

def main():
    # Create checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Initialize model
    model = RGBBranch(arch='ResNet-50', scene_classes=NUM_CLASSES)
    model.to(DEVICE)

    # Create dataset and dataloader
    train_dataset = SceneDataset(TRAIN_IMAGES, TRAIN_TAGS, train=True)
    val_dataset = SceneDataset(VAL_IMAGES, VAL_TAGS, train=False)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=4,
        pin_memory=True,
        shuffle=True
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=4,
        pin_memory=True
    )

    # Set loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Initialize trainer and start training
    trainer = Trainer(model, train_loader, val_loader, optimizer, criterion)
    best_model_state = trainer.train(EPOCHS)

    # Save best model
    torch.save({
        'epoch': trainer.best_epoch,
        'model_state_dict': best_model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_accuracy': trainer.best_val_accuracy,
    }, CHECKPOINT_PATH)

if __name__ == '__main__':
    main() 