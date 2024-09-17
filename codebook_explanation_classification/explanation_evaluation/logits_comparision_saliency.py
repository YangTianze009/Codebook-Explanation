import torch
import torchvision
from torchvision import transforms
from PIL import Image
import os
import pickle
import csv
import argparse
from tqdm import tqdm

def load_top_tokens(csv_path, top_n, token_number):
    target_token_list = []
    with open(csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        in_top_n_section = False
        current_row = 0

        for row in reader:
            if f"Top {top_n} Tokens" in row:
                in_top_n_section = True
                next(reader)
                current_row = 0
                continue

            if in_top_n_section:
                if "Top" in row[0] or current_row == token_number:
                    break
                token = int(row[0])
                target_token_list.append(token)
                current_row += 1

    return target_token_list

def load_top_tokens_baseline(input_csv, top_n):
    tokens = []
    with open(input_csv, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            token = int(row['Token'])
            count = int(row['Count'])
            tokens.append((token, count))
    
    tokens.sort(key=lambda x: x[1], reverse=True)
    top_n_tokens = [token for token, _ in tokens[:top_n]]
    
    return top_n_tokens

def preprocess_image(image, preprocess):
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch

def calculate_logits_diff(model, image, target_label, token_list, target_token_list, grid_size, device, preprocess):
    # Original image prediction
    input_batch = preprocess_image(image, preprocess).to(device)
    with torch.no_grad():
        original_output = model(input_batch)
    original_logits = original_output[0, target_label].item()
    # predicted_class = original_output.argmax(dim=1).item()

    
    # Mask token areas and predict logits
    masked_image = image.copy()
    image_size = masked_image.size[0]
    patch_size = image_size // grid_size
    pixels = masked_image.load()
    
    for token_position in range(len(token_list)):
        if token_list[token_position] in target_token_list:
            row = token_position // grid_size
            col = token_position % grid_size
            for i in range(patch_size):
                for j in range(patch_size):
                    pixels[col * patch_size + i, row * patch_size + j] = (0, 0, 0)
    
    masked_batch = preprocess_image(masked_image, preprocess).to(device)
    with torch.no_grad():
        masked_output = model(masked_batch)
    masked_logits = masked_output[0, target_label].item()

    return original_logits - masked_logits

def main(args):
    # Set up the model
    if args.model == 'vit_b_16':
        model = torchvision.models.vit_b_16(pretrained=True)
    elif args.model == 'vit_b_32':
        model = torchvision.models.vit_b_32(pretrained=True)
    elif args.model == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
    elif args.model == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
    else:
        raise ValueError("Invalid model name")

    model.eval()

    # Set up device
    if args.gpu is not None:
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    print(f"Using device: {device}")

    # Define image preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load token dictionary
    with open('/data2/ty45972_data2/taming-transformers/codebook_explanation_classification/datasets/VQGAN_16384_generated_new/test_token_indices.pkl', 'rb') as f:
        token_dict = pickle.load(f)

    # Define paths
    image_base_path = '/data2/ty45972_data2/taming-transformers/datasets/imagenet_VQGAN_generated/'
    test_csv = "/data2/ty45972_data2/taming-transformers/codebook_explanation_classification/datasets/VQGAN_16384_generated_new/test_embeddings.csv"

    # Read test CSV once and organize data
    label_to_files = {i: [] for i in range(1000)}  # Dictionary to store files for each label
    with open(test_csv, 'r', encoding='utf-8') as test_csvfile:
        reader = csv.reader(test_csvfile)
        next(reader)  # Skip header
        for row in reader:
            filename = row[0]
            label = int(row[1])
            label_to_files[label].append(filename)

    # Prepare output CSV
    output_csv = f"logits_diff_{args.model}_top{args.top_n}_token{args.token_num}.csv"
    evaluation_result_path = "evaluation_results/"
    os.makedirs(evaluation_result_path, exist_ok=True)
    with open(evaluation_result_path + output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Label', 'Avg_Diff_Target', 'Avg_Diff_Baseline'])

        total_diff_target_all = 0
        total_diff_baseline_all = 0
        total_count_all = 0

        # Main loop for processing all labels
        for target_label in tqdm(range(1000), desc="Processing labels"):
            csv_path = f"/data2/ty45972_data2/taming-transformers/codebook_explanation_classification/results/Explanation/generated_data/label/Net1/label_activation_statistics/label_{target_label}.csv"
            baseline_path = f"/data2/ty45972_data2/taming-transformers/codebook_explanation_classification/results/Explanation/baseline_statistics/label_{target_label}.csv"

            # Load tokens
            target_token_list = load_top_tokens(csv_path, args.top_n, args.token_num)
            target_token_list_baseline = load_top_tokens_baseline(baseline_path, args.token_num)

            # Get file list for the current label
            npy_file_list = label_to_files[target_label]

            total_diff_target = 0
            total_diff_baseline = 0
            count = 0

            for npy_file in npy_file_list:
                subfolder, image_name = npy_file.split('_')
                image_name = image_name.replace('.npy', '.png')
                image_path = os.path.join(image_base_path, subfolder, image_name)
                
                if not os.path.exists(image_path):
                    continue
                
                image = Image.open(image_path)
                token_list = token_dict.get(npy_file)
                
                if token_list is None:
                    continue
                
                diff_target = calculate_logits_diff(model, image, target_label, token_list, target_token_list, 16, device, preprocess)
                diff_baseline = calculate_logits_diff(model, image, target_label, token_list, target_token_list_baseline, 16, device, preprocess)
                
                total_diff_target += diff_target
                total_diff_baseline += diff_baseline
                count += 1

            if count > 0:
                avg_diff_target = total_diff_target / count
                avg_diff_baseline = total_diff_baseline / count
                
                # Update overall statistics
                total_diff_target_all += total_diff_target
                total_diff_baseline_all += total_diff_baseline
                total_count_all += count
                
                # Write to CSV file
                writer.writerow([target_label, avg_diff_target, avg_diff_baseline])
                
                # Flush file buffer to ensure data is written to disk
                csvfile.flush()

        # Calculate and add overall average difference
        if total_count_all > 0:
            overall_avg_diff_target = total_diff_target_all / total_count_all
            overall_avg_diff_baseline = total_diff_baseline_all / total_count_all
            
            writer.writerow(['Overall', overall_avg_diff_target, overall_avg_diff_baseline])
            
            print(f"Overall average difference (Target): {overall_avg_diff_target}")
            print(f"Overall average difference (Baseline): {overall_avg_diff_baseline}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Logits comparison for different models and token settings.")
    parser.add_argument('--model', type=str, default='resnet50', choices=['vit_b_16', 'vit_b_32', 'resnet18', 'resnet50'], help="Model to use for comparison")
    parser.add_argument('--top_n', type=int, default=20, help="Top N tokens to consider")
    parser.add_argument('--token_num', type=int, default=50, help="Number of tokens to use")
    parser.add_argument('--gpu', type=int, default=None, help="Specify GPU to use (e.g., 0 or 1). If not specified, will use any available GPU or CPU.")
    args = parser.parse_args()
    
    main(args)