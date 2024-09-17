import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from datetime import datetime
import argparse
import math
from torch.optim.lr_scheduler import LambdaLR

# 假设这些是您的自定义模块
from dataset import get_train_val_dataloaders
from model import ClassificationNet1, ClassificationNet2, ClassificationNet3, ClassificationNet4

class WarmupCosineSchedule(LambdaLR):
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))

def train_model(train_loader, val_loader, num_classes, num_epochs, learning_rate, model_choice, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if model_choice == 1:
        model = ClassificationNet1(num_classes).to(device)
        model_name = 'ClassificationNet1'
    elif model_choice == 2:
        model = ClassificationNet2(num_classes).to(device)
        model_name = 'ClassificationNet2'
    elif model_choice == 3:
        model = ClassificationNet3(num_classes=num_classes).to(device)
        model_name = 'ClassificationNet3'
    elif model_choice == 4:
        model = ClassificationNet4(num_classes=num_classes).to(device)
        model_name = 'ClassificationNet4'
    else:
        raise ValueError("Invalid model choice. Choose either 1, 2, 3, or 4.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)
    
    # Warmup and cosine schedule
    warmup_steps = int(0.1 * num_epochs * len(train_loader))  # 10% of total steps
    total_steps = num_epochs * len(train_loader)
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps, total_steps)

    scaler = GradScaler()

    # Create checkpoints and logs folders
    checkpoint_dir = f'checkpoints/{args.data_type}_data/{model_name}'
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(f'logs/{args.data_type}', exist_ok=True)

    # Create a log file with a timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    log_file_path = f'logs/{args.data_type}/training_log_{timestamp}.txt'
    best_val_loss = float('inf')
    
    # Gradient accumulation steps
    gradient_accumulation_steps = 2  # Adjust as needed
    
    with open(log_file_path, 'w') as log_file:
        for epoch in range(num_epochs):
            print(f"Training Epoch {epoch+1}/{num_epochs}")
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            train_loader_tqdm = tqdm(train_loader, desc="Training")
            optimizer.zero_grad()
            
            for i, (inputs, labels) in enumerate(train_loader_tqdm):
                inputs, labels = inputs.to(device), labels.to(device)
                
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss = loss / gradient_accumulation_steps
                
                scaler.scale(loss).backward()
                
                if (i + 1) % gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()

                running_loss += loss.item() * gradient_accumulation_steps
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                train_loader_tqdm.set_postfix(loss=running_loss / (train_loader_tqdm.n + 1),
                                              acc=100. * correct / total,
                                              lr=optimizer.param_groups[0]['lr'])

            avg_train_loss = running_loss / len(train_loader)
            train_accuracy = 100. * correct / total
            print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")
            log_file.write(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.2f}%\n")

            print(f"Validating Epoch {epoch+1}/{num_epochs}")
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            val_loader_tqdm = tqdm(val_loader, desc="Validation")
            with torch.no_grad():
                for inputs, labels in val_loader_tqdm:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

                    val_loader_tqdm.set_postfix(loss=val_loss / (val_loader_tqdm.n + 1),
                                                acc=100. * correct / total)

            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = 100 * correct / total
            print(f'Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%')
            log_file.write(f'Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%\n')
            log_file.flush()

            # Save the model for each epoch
            torch.save(model.state_dict(), f'{checkpoint_dir}/model_epoch_{epoch+1}.pth')

            # Save the best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), f'{checkpoint_dir}/best_model.pth')
                print(f"New best model saved with validation loss: {best_val_loss:.4f}")
                log_file.write(f"New best model saved with validation loss: {best_val_loss:.4f}\n")

    print("Training completed.")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a classification model.')
    parser.add_argument('--model', type=int, choices=[1, 2, 3, 4], required=True,
                        help='Choose which model to train: 1 for ClassificationNet1, 2 for ClassificationNet2, 3 for ClassificationNet3, or 4 for ClassificationNet4')
    parser.add_argument('--data_type', type=str, choices=["generated", "original"], default="generated",
                        help='Choose which imagenet data to be used')
    args = parser.parse_args()

    train_csv = 'train_embeddings.csv'
    val_csv = 'validation_embeddings.csv'
    data_folder = '/data/ty45972/taming-transformers/codebook_explanation_classification/datasets/VQGAN_16384_generated_new'
    batch_size = 128  # Reduced batch size to accommodate gradient accumulation
    num_classes = 1000  # Adjust according to the actual number of classes
    num_epochs = 100  # Increased number of epochs
    learning_rate = 3e-4  # Adjusted learning rate

    train_loader, val_loader = get_train_val_dataloaders(train_csv, val_csv, data_folder, batch_size)
    train_model(train_loader, val_loader, num_classes, num_epochs, learning_rate, args.model, args)