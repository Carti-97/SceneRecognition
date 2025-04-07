import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import sys
from tqdm import tqdm

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset import SceneDataset
from model import RGBBranch, DynamicContextIntegration
from config import *

def train_rgb_branch():
    """Train the RGB branch model"""
    # Create checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Initialize model
    model = RGBBranch(arch='ResNet-50', scene_classes=NUM_CLASSES)
    model.to(DEVICE)

    # Create dataset and dataloader
    train_dataset = SceneDataset(TRAIN_IMAGES, train=True)
    val_dataset = SceneDataset(VAL_IMAGES, train=False)

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

    # Set optimizer - only train the classifier parameters
    # Filter parameters that require gradients
    parameters_to_train = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(parameters_to_train, lr=LEARNING_RATE)

    # Training loop
    best_val_accuracy = 0.0
    best_epoch = 0
    best_model_state = None

    for epoch in range(EPOCHS):
        model.train()
        correct_train = 0
        total_train = 0
        train_loss = 0.0
        
        for inputs_image, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}', unit='batch'):
            inputs_image = inputs_image.to(DEVICE)
            labels = labels.to(device=DEVICE)

            # Forward pass
            outputs = model(inputs_image)
            loss = model.loss(outputs, labels)  # Use the model's loss function

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted_train = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted_train == labels).sum().item()

            train_loss += loss.item()

        # Validation phase
        model.eval()
        correct_val = 0
        total_val = 0
        val_loss = 0.0

        with torch.no_grad():
            for inputs_image, labels in val_loader:
                bs, crops, c, h, w = inputs_image.size()
                inputs_image = inputs_image.view(-1, c, h, w).to(DEVICE)
                labels = labels.to(dtype=torch.long, device=DEVICE)

                outputs = model(inputs_image)
                outputs = outputs.view(bs, crops, -1)
                
                _, preds = torch.max(outputs, 2)
                predicted_val = torch.mode(preds, 1).values

                total_val += labels.size(0)
                correct_val += (predicted_val == labels).sum().item()

        val_accuracy = correct_val / total_val
        print(f'Epoch {epoch+1}, Validation Accuracy: {val_accuracy:.4f}')

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1
            best_model_state = model.state_dict().copy()

    print(f'Best Validation Accuracy: {best_val_accuracy:.4f}, Best Epoch: {best_epoch}')

    # Save best model
    torch.save({
        'epoch': best_epoch,
        'state_dict': best_model_state,
        'best_val_accuracy': best_val_accuracy,
    }, CHECKPOINT_PATH)

def train_dynamic_integration():
    """Train the dynamic integration model that combines RGB features with text embeddings"""
    # Create checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Initialize model
    model = DynamicContextIntegration(backbone='ResNet-50', num_classes=NUM_CLASSES)
    model.to(DEVICE)

    # If available, load pretrained RGB branch weights
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH)
        # Load only RGB branch weights from checkpoint
        rgb_branch_dict = {k: v for k, v in checkpoint['state_dict'].items() if 'rgb_branch' in k}
        if rgb_branch_dict:
            model.rgb_branch.load_state_dict(rgb_branch_dict)
            print(f"Loaded RGB branch weights from checkpoint epoch {checkpoint['epoch']}")
    
    # Create dataset and dataloader
    train_dataset = SceneDataset(TRAIN_IMAGES, train=True)
    val_dataset = SceneDataset(VAL_IMAGES, train=False)

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

    # Set optimizer - only train the gate and classifier parameters
    # Freeze RGB branch
    for param in model.rgb_branch.parameters():
        param.requires_grad = False
    
    # Set trainable parameters
    # Updated to include only gate_layer and classifier parameters
    parameters_to_train = list(model.gate_layer.parameters()) + list(model.classifier.parameters())
    optimizer = torch.optim.AdamW(parameters_to_train, lr=LEARNING_RATE)

    # Training loop
    best_val_accuracy = 0.0
    best_epoch = 0
    best_model_state = None

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

    # Tag generation function (mock implementation - replace with actual tag generator)
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

    for epoch in range(EPOCHS):
        model.train()
        correct_train = 0
        total_train = 0
        train_loss = 0.0
        
        for inputs_image, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}', unit='batch'):
            current_batch_size = inputs_image.size(0)
            inputs_image = inputs_image.to(DEVICE)
            labels = labels.to(device=DEVICE)

            # Generate tags for images (replace with actual tag generation)
            tags = generate_tags_for_images(inputs_image, current_batch_size)

            # Forward pass with both images and tags
            outputs = model(inputs_image, tags)
            loss = model.loss(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted_train = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted_train == labels).sum().item()

            train_loss += loss.item()

        # Validation phase
        model.eval()
        correct_val = 0
        total_val = 0
        val_loss = 0.0

        with torch.no_grad():
            for inputs_image, labels in val_loader:
                bs, crops, c, h, w = inputs_image.size()
                inputs_image = inputs_image.view(-1, c, h, w).to(DEVICE)
                labels = labels.to(dtype=torch.long, device=DEVICE)

                # Generate tags for validation images
                tags = generate_tags_for_images(inputs_image, bs * crops)

                # Forward pass with both images and tags
                outputs = model(inputs_image, tags)
                outputs = outputs.view(bs, crops, -1)
                
                _, preds = torch.max(outputs, 2)
                predicted_val = torch.mode(preds, 1).values

                total_val += labels.size(0)
                correct_val += (predicted_val == labels).sum().item()

        val_accuracy = correct_val / total_val
        print(f'Epoch {epoch+1}, Validation Accuracy: {val_accuracy:.4f}')

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1
            best_model_state = model.state_dict().copy()

    print(f'Best Validation Accuracy: {best_val_accuracy:.4f}, Best Epoch: {best_epoch}')

    # Save best model
    torch.save({
        'epoch': best_epoch,
        'state_dict': best_model_state,
        'best_val_accuracy': best_val_accuracy,
    }, DYNAMIC_CHECKPOINT_PATH)

def main():
    # Choose training mode
    if TRAIN_MODE == 'rgb':
        train_rgb_branch()
    elif TRAIN_MODE == 'dynamic':
        train_dynamic_integration()
    else:
        print(f"Unknown training mode: {TRAIN_MODE}. Choose 'rgb' or 'dynamic'.")

if __name__ == '__main__':
    main() 