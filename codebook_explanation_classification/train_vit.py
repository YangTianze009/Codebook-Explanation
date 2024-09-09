import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import get_train_val_dataloaders
from model import ClassificationNet3
from tqdm import tqdm
import argparse
from datetime import datetime

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    train_loader_tqdm = tqdm(train_loader, desc="Training")
    for inputs, labels in train_loader_tqdm:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        train_loader_tqdm.set_postfix(
            loss=running_loss / (train_loader_tqdm.n + 1),
            accuracy=100. * correct / total
        )
    
    return running_loss / len(train_loader), 100. * correct / total

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        val_loader_tqdm = tqdm(val_loader, desc="Validation")
        for inputs, labels in val_loader_tqdm:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            val_loader_tqdm.set_postfix(
                loss=running_loss / (val_loader_tqdm.n + 1),
                accuracy=100. * correct / total
            )
    
    return running_loss / len(val_loader), 100. * correct / total

def train_model(train_loader, val_loader, num_classes, num_epochs, learning_rate, device, model_name, data_type):
    model = ClassificationNet3(num_classes=num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Create checkpoints and logs folders
    checkpoint_dir = f'checkpoints/{data_type}_data/{model_name}'
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(f'logs/{data_type}', exist_ok=True)
    
    # Create a log file with a timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    log_file_path = f'logs/{data_type}/training_log_{timestamp}.txt'
    
    best_val_loss = float('inf')
    
    with open(log_file_path, 'w') as log_file:
        for epoch in range(num_epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            
            scheduler.step()
            
            # Log to file
            log_message = f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, LR: {scheduler.get_last_lr()[0]:.6f}"
            print(log_message)
            log_file.write(log_message + "\n")
            log_file.flush()
            
            # Save the model for each epoch
            torch.save(model.state_dict(), f'{checkpoint_dir}/model_epoch_{epoch+1}.pth')
            
            # Save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f'{checkpoint_dir}/best_model.pth')
                print(f"New best model saved with validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ClassificationNet3 model.')
    parser.add_argument('--data_type', type=str, choices=["generated", "original"], default="generated",
                        help='Choose which imagenet data to be used')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_csv = 'train_embeddings.csv'
    val_csv = 'validation_embeddings.csv'
    data_folder = '/data/ty45972/taming-transformers/codebook_explanation_classification/datasets/VQGAN_16384_generated_new'
    num_classes = 1000  # Adjust according to the actual number of classes

    train_loader, val_loader = get_train_val_dataloaders(train_csv, val_csv, data_folder, args.batch_size)

    train_model(train_loader, val_loader, num_classes, args.num_epochs, args.learning_rate, device, "ClassificationNet3", args.data_type)