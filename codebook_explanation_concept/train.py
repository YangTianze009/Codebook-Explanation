import os
import torch
import torch.optim as optim
from torch.nn import BCEWithLogitsLoss
from torch.optim.lr_scheduler import StepLR
from dataset import get_train_val_dataloaders
from model import ConceptModel1, ConceptModel2
from tqdm import tqdm
from datetime import datetime
import argparse
import numpy as np

def train_model(model, train_loader, val_loader, num_epochs, learning_rate, step_size, gamma):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model_name = type(model).__name__

    criterion = BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    # Create checkpoints and logs folders
    checkpoint_dir = f'checkpoints/{model_name}'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # Create a log file with a timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    log_file_path = f'logs/training_log_{timestamp}.txt'
    best_val_loss = float('inf')
    
    with open(log_file_path, 'w') as log_file:
        for epoch in range(num_epochs):
            print(f"Training Epoch {epoch+1}/{num_epochs}")
            model.train()
            running_loss = 0.0
            train_loader_tqdm = tqdm(train_loader, desc="Training")
            for inputs, labels in train_loader_tqdm:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"NaN or Inf detected in loss at epoch {epoch}, batch {train_loader_tqdm.n}")
                    continue
                
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                train_loader_tqdm.set_postfix(loss=running_loss / (train_loader_tqdm.n + 1))

            avg_train_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}")
            log_file.write(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}\n")

            print(f"Validating Epoch {epoch+1}/{num_epochs}")
            model.eval()
            val_loss = 0.0
            val_rmse = 0.0
            total_samples = 0
            val_loader_tqdm = tqdm(val_loader, desc="Validation")
            with torch.no_grad():
                for inputs, labels in val_loader_tqdm:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    # Compute RMSE
                    rmse_batch = torch.sqrt(torch.mean((outputs - labels) ** 2))
                    val_rmse += rmse_batch.item() * labels.size(0)
                    total_samples += labels.size(0)

                    val_loader_tqdm.set_postfix(loss=val_loss / (val_loader_tqdm.n + 1))

            avg_val_loss = val_loss / len(val_loader)
            avg_val_rmse = val_rmse / total_samples
            print(f'Validation Loss: {avg_val_loss:.4f}, RMSE: {avg_val_rmse:.4f}')
            log_file.write(f'Validation Loss: {avg_val_loss:.4f}, RMSE: {avg_val_rmse:.4f}\n')
            log_file.flush()

            # Save the model for each epoch
            torch.save(model.state_dict(), f'{checkpoint_dir}/model.pth')

            # Save the best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), f'{checkpoint_dir}/best_model.pth')

            # Step the scheduler
            scheduler.step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a concept classification model.')
    parser.add_argument('--model', type=int, choices=[1, 2], required=True, help='Choose the model: 1 for ConceptModel1, 2 for ConceptModel2')
    args = parser.parse_args()
    num_concepts = 23
    if args.model == 1:
        model = ConceptModel1(num_concepts)  # 根据需要调整 num_concepts 的值
    elif args.model == 2:
        model = ConceptModel2(num_concepts)  # 根据需要调整 num_concepts 的值

    train_csv = 'train_embeddings.csv'
    val_csv = 'val_embeddings.csv'
    data_folder = '/data/ty45972/taming-transformers/codebook_explanation_concept/datasets/CUB/image_embedding'
    batch_size = 256
      # Adjust according to the actual number of concepts
    num_epochs = 200
    learning_rate = 1e-4
    step_size = 50  # Decay learning rate every step_size epochs
    gamma = 0.1  # Multiply learning rate by gamma at each step

    train_loader, val_loader = get_train_val_dataloaders(train_csv, val_csv, data_folder, batch_size)
    train_model(model, train_loader, val_loader, num_epochs, learning_rate, step_size, gamma)
