import torch
import numpy as np
import csv
from tqdm import tqdm
from dataset import get_test_dataloader, get_train_val_dataloaders
from model import ClassificationNet1, ClassificationNet2, ClassificationNet3
import argparse
import os
import random

class AverageSaliency:
    def __init__(self, model, output_index=0):
        self.model = model
        self.output_index = output_index

    def get_grad(self, input_image):
        raise NotImplementedError("Subclasses should implement this!")

    def get_average_grad(self, input_image, stdev_spread=0.2, nsamples=50):
        stdev = stdev_spread * (torch.max(input_image) - torch.min(input_image))

        total_gradients = torch.zeros_like(input_image, dtype=torch.float64)
        for _ in range(nsamples):
            noise = torch.normal(mean=0, std=stdev, size=input_image.shape).to(input_image.device)
            noisy_image = input_image + noise

            gradients = self.get_grad(noisy_image)
            total_gradients += gradients

        return total_gradients / nsamples

class SingleSaliency(AverageSaliency):
    def __init__(self, model, output_index=0):
        super(SingleSaliency, self).__init__(model, output_index)

    def get_grad(self, input_image):
        input_image.requires_grad = True
        output = self.model(input_image)
        self.model.zero_grad()
        output[0, self.output_index].backward()
        return input_image.grad

def compute_contribution(test_loader, model_path, device, results_dir, selected_labels):
    num_classes = 1000  # Adjust according to your actual number of classes
    if args.model == 1:
        model = ClassificationNet1(num_classes)
    elif args.model == 2:
        model = ClassificationNet2(num_classes)
    elif args.model == 3:
        model = ClassificationNet3(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    os.makedirs(results_dir, exist_ok=True)

    label_files = {}

    for idx, (inputs, labels) in enumerate(tqdm(test_loader, desc="Computing Saliency Maps")):
        inputs, labels, filenames = inputs.to(device), labels.to(device), test_loader.dataset.data[idx * test_loader.batch_size: (idx + 1) * test_loader.batch_size]

        for i in range(inputs.size(0)):
            input_embedding = inputs[i].unsqueeze(0)
            label = labels[i].item()
            filename = filenames[i]

            # 如果label不属于选取的100个类别，跳过该样本
            if label not in selected_labels:
                continue

            output = model(input_embedding)
            predicted_label = int(torch.argmax(output, dim=1).cpu())

            if predicted_label == label:
                single_saliency = SingleSaliency(model, output_index=label)
                grad = single_saliency.get_average_grad(input_embedding).squeeze(0)
                grad = grad.cpu().numpy()
                avg_grad_flat = grad.mean(axis=0).flatten()  # Flatten and average over the 256-dim embeddings
                max_grad_flat = grad.max(axis=0).flatten()

                # Check if the CSV for this label already exists
                csv_path = os.path.join(results_dir, f"label_{label}.csv")
                if not os.path.exists(csv_path):
                    label_files[label] = open(csv_path, 'a', newline='')
                    csvwriter = csv.writer(label_files[label])
                    csvwriter.writerow(['filename', 'label', 'contribution'])  # Only write header if new file
                else:
                    label_files[label] = open(csv_path, 'a', newline='')
                    csvwriter = csv.writer(label_files[label])
                
                # Write results to the CSV after processing each image
                if args.method == "max":
                    csvwriter.writerow([filename, label, max_grad_flat.tolist()])
                elif args.method == "ave":
                    csvwriter.writerow([filename, label, avg_grad_flat.tolist()])

                # Flush the file to ensure data is written to disk
                label_files[label].flush()

    # Close all open files
    for file in label_files.values():
        file.close()

    print(f'Token contributions saved to individual label CSVs in {results_dir}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute saliency maps for a classification model.')
    parser.add_argument('--method', type=str, default="max", help='Method for contribution calculation: "max" or "ave"')
    parser.add_argument('--gpu', type=int, default=0, help='Specify which GPU to use for computation')
    parser.add_argument('--model', type=int, choices=[1, 2, 3], required=True,
                        help='Choose which model to test: 1 for ClassificationNet1, 2 for ClassificationNet2, or 3 for ClassificationNet3')
    parser.add_argument('--data', type=str, choices=["generated", "original"], default="generated",
                        help='Choose which model to test: 1 for ClassificationNet1, 2 for ClassificationNet2, or 3 for ClassificationNet3')
    args = parser.parse_args()

    # 随机从0到999中选取100个label
    selected_labels = random.sample(range(1000), 100)
    # selected_labels = list(range(1, 101))
    print(f"Selected labels: {selected_labels}")

    test_csv = 'train_embeddings.csv'
    val_csv = 'validation_embeddings.csv'
    if args.data == "generated":
        data_folder = '/data2/ty45972_data2/taming-transformers/codebook_explanation_classification/datasets/VQGAN_16384_generated_new'
    elif args.data == "original":
        data_folder = '/data2/ty45972_data2/taming-transformers/codebook_explanation_classification/datasets/VQGAN_16384_original'
    batch_size = 512
    model_path = f"/data2/ty45972_data2/taming-transformers/codebook_explanation_classification/checkpoints/{args.data}_data/ClassificationNet{args.model}/best_model.pth"
    results_dir = f"results/Explanation/{args.data}_data/label/Net{args.model}/label_activation_results"
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    test_loader, val_loader = get_train_val_dataloaders(test_csv, val_csv, data_folder, batch_size, shuffle=False)
    compute_contribution(test_loader, model_path, device, results_dir, selected_labels)
