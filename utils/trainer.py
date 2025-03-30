import torch
import torch.nn as nn
from tqdm import tqdm
from config import DEVICE

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, criterion):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = DEVICE
        self.best_val_accuracy = 0.0
        self.best_epoch = 0
        self.best_model_state = None

    def train_epoch(self):
        self.model.train()
        correct_train = 0
        total_train = 0
        train_loss = 0.0
        
        for inputs_image, labels in tqdm(self.train_loader, desc='Training', unit='batch'):
            inputs_image = inputs_image.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            outputs = self.model(inputs_image)
            loss = self.criterion(outputs, labels)

            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            _, predicted_train = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted_train == labels).sum().item()
            train_loss += loss.item()

        return correct_train / total_train, train_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        correct_val = 0
        total_val = 0
        val_loss = 0.0

        with torch.no_grad():
            for inputs_image, labels in self.val_loader:
                bs, crops, c, h, w = inputs_image.size()
                inputs_image = inputs_image.view(-1, c, h, w).to(self.device)
                labels = labels.to(dtype=torch.long, device=self.device)

                outputs = self.model(inputs_image)
                outputs = outputs.view(bs, crops, -1)
                
                _, preds = torch.max(outputs, 2)
                predicted_val = torch.mode(preds, 1).values

                total_val += labels.size(0)
                correct_val += (predicted_val == labels).sum().item()
                val_loss += self.criterion(outputs.view(bs * crops, -1), labels.repeat_interleave(crops)).item()

        return correct_val / total_val, val_loss / len(self.val_loader)

    def train(self, epochs):
        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')
            
            # Training phase
            train_acc, train_loss = self.train_epoch()
            
            # Validation phase
            val_acc, val_loss = self.validate()
            
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                self.best_epoch = epoch + 1
                self.best_model_state = self.model.state_dict().copy()

        print(f'\nBest Validation Accuracy: {self.best_val_accuracy:.4f} at epoch {self.best_epoch}')
        return self.best_model_state 