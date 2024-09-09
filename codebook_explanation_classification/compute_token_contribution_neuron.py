import torch
import numpy as np
import csv
from tqdm import tqdm
from dataset import get_test_dataloader
from model import ClassificationNet1, ClassificationNet2, ClassificationNet3
import argparse
import os

def get_layer_outputs(model, inputs, target_layer_names):
    outputs = {}

    def hook(module, input, output, layer_name):
        outputs[layer_name] = output

    handles = []
    for name, layer in model.named_modules():
        if name in target_layer_names:
            handle = layer.register_forward_hook(lambda module, input, output, layer_name=name: hook(module, input, output, layer_name))
            handles.append(handle)

    # Forward pass
    model(inputs)

    # Remove hooks
    for handle in handles:
        handle.remove()

    return outputs

def map_feature_to_token(feature_map, feature_map_size, token_map_size):
    scale_factor = token_map_size // feature_map_size
    token_contributions = np.zeros((token_map_size, token_map_size), dtype=np.float32)

    for i in range(feature_map_size):
        for j in range(feature_map_size):
            token_contributions[i * scale_factor: (i + 1) * scale_factor, j * scale_factor: (j + 1) * scale_factor] = feature_map[i, j]

    return token_contributions

def process_conv_layer(layer_name, outputs, filename, label, results_dir, token_map_size=16):
    layer_output = outputs[layer_name]
    C, H, W = layer_output.shape[1], layer_output.shape[2], layer_output.shape[3]

    layer_dir = os.path.join(results_dir, layer_name)
    os.makedirs(layer_dir, exist_ok=True)

    for c in range(C):
        csv_filename = os.path.join(layer_dir, f"channel_{c}.csv")

        token_contributions = map_feature_to_token(layer_output[0, c, :, :].detach().cpu().numpy(), H, token_map_size)
    
        write_contributions_to_csv(token_contributions, csv_filename, filename, label)

def write_contributions_to_csv(contributions, csv_filename, filename, label):
    file_exists = os.path.isfile(csv_filename)

    with open(csv_filename, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        if not file_exists:
            csvwriter.writerow(['filename', 'label', 'contribution'])
        print(f"filename is {filename}")
        csvwriter.writerow([filename, label, contributions.flatten().tolist()])

def compute_contribution(test_loader, model_path, device, results_dir):
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

    target_layer_names = [
    name for name, layer in model.named_modules() 
    if isinstance(layer, torch.nn.Conv2d) and not name.startswith('conv1_')
]

    total_images = len(test_loader.dataset)
    current_image_idx = 0

    for idx, (inputs, labels) in enumerate(tqdm(test_loader, desc="Computing Token Contributions", total=len(test_loader))):
        inputs, labels, filenames = inputs.to(device), labels.to(device), test_loader.dataset.data[idx * test_loader.batch_size: (idx + 1) * test_loader.batch_size]

        for i in range(inputs.size(0)):
            input_embedding = inputs[i].unsqueeze(0)
            output = model(input_embedding)
            predicted_label = int(torch.argmax(output, dim=1).cpu())

            label = labels[i].item()
            filename = filenames[i]
            if predicted_label == label:
                outputs = get_layer_outputs(model, input_embedding, target_layer_names)
                for layer_name in target_layer_names:
                    process_conv_layer(layer_name, outputs, filename, label, results_dir)

            current_image_idx += 1
            print(f"Processed {current_image_idx}/{total_images} images.")

    print(f'Token contributions saved in directory: {results_dir}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute token contributions for each layer.')
    parser.add_argument('--gpu', type=int, default=0, help='Specify which GPU to use for computation')
    parser.add_argument('--model', type=int, choices=[1, 2, 3], default=1,
                        help='Choose which model to test: 1 for ClassificationNet1, 2 for ClassificationNet2, or 3 for ClassificationNet3')
    args = parser.parse_args()

    test_csv = 'test_embeddings.csv'
    data_folder = '/data2/ty45972_data2/taming-transformers/codebook_explanation_classification/datasets/VQGAN_16384_generated_new'
    batch_size = 16
    model_path = f"/data2/ty45972_data2/taming-transformers/codebook_explanation_classification/checkpoints/generated_data/ClassificationNet{args.model}/best_model.pth"
    results_dir = f"results/Explanation/neuron/Net{args.model}/neuron_activation_results"
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    test_loader = get_test_dataloader(test_csv, data_folder, batch_size)
    compute_contribution(test_loader, model_path, device, results_dir)
