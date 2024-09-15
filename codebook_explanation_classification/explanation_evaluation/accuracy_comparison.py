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

def calculate_accuracy(model, image, true_label, token_list, target_token_list, baseline_token_list, grid_size, device, preprocess):
    # Original image prediction
    input_batch = preprocess_image(image, preprocess).to(device)
    with torch.no_grad():
        original_output = model(input_batch)
    original_pred = original_output.argmax(dim=1).item()
    original_correct = (original_pred == true_label)

    # Function to create masked image
    def create_masked_image(token_list_to_mask):
        masked_image = image.copy()
        image_size = masked_image.size[0]
        patch_size = image_size // grid_size
        pixels = masked_image.load()
        
        for token_position in range(len(token_list)):
            if token_list[token_position] in token_list_to_mask:
                row = token_position // grid_size
                col = token_position % grid_size
                for i in range(patch_size):
                    for j in range(patch_size):
                        pixels[col * patch_size + i, row * patch_size + j] = (0, 0, 0)
        return masked_image

    # Our method masked image prediction
    our_masked_image = create_masked_image(target_token_list)
    our_masked_batch = preprocess_image(our_masked_image, preprocess).to(device)
    with torch.no_grad():
        our_masked_output = model(our_masked_batch)
    our_masked_pred = our_masked_output.argmax(dim=1).item()
    our_masked_correct = (our_masked_pred == true_label)

    # Baseline method masked image prediction
    baseline_masked_image = create_masked_image(baseline_token_list)
    baseline_masked_batch = preprocess_image(baseline_masked_image, preprocess).to(device)
    with torch.no_grad():
        baseline_masked_output = model(baseline_masked_batch)
    baseline_masked_pred = baseline_masked_output.argmax(dim=1).item()
    baseline_masked_correct = (baseline_masked_pred == true_label)

    return original_correct, our_masked_correct, baseline_masked_correct

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
    label_to_files = {i: [] for i in range(1000)}
    with open(test_csv, 'r', encoding='utf-8') as test_csvfile:
        reader = csv.reader(test_csvfile)
        next(reader)  # Skip header
        for row in reader:
            filename = row[0]
            label = int(row[1])
            label_to_files[label].append(filename)

    # Prepare output CSV
    output_csv = f"accuracy_comparison_{args.model}_top{args.top_n}_token{args.token_num}.csv"
    evaluation_result_path = "evaluation_results/"
    os.makedirs(evaluation_result_path, exist_ok=True)
    with open(evaluation_result_path + output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Model', 'Top N', 'Token Num', 'Original Accuracy', 'Our Masked Accuracy', 'Baseline Masked Accuracy'])

        total_original_correct = 0
        total_our_masked_correct = 0
        total_baseline_masked_correct = 0
        total_count = 0

        # Main loop for processing all labels
        for target_label in tqdm(range(1000), desc="Processing labels"):
            csv_path = f"/data2/ty45972_data2/taming-transformers/codebook_explanation_classification/results/Explanation/generated_data/label/Net1/label_activation_statistics/label_{target_label}.csv"
            baseline_path = f"/data2/ty45972_data2/taming-transformers/codebook_explanation_classification/results/Explanation/baseline_statistics/label_{target_label}.csv"

            # Load tokens
            target_token_list = load_top_tokens(csv_path, args.top_n, args.token_num)
            baseline_token_list = load_top_tokens_baseline(baseline_path, args.token_num)

            # Get file list for the current label
            npy_file_list = label_to_files[target_label]

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
                
                original_correct, our_masked_correct, baseline_masked_correct = calculate_accuracy(
                    model, image, target_label, token_list, target_token_list, baseline_token_list, 16, device, preprocess
                )
                
                total_original_correct += int(original_correct)
                total_our_masked_correct += int(our_masked_correct)
                total_baseline_masked_correct += int(baseline_masked_correct)
                total_count += 1

        # Calculate and add overall accuracy
        if total_count > 0:
            original_accuracy = total_original_correct / total_count
            our_masked_accuracy = total_our_masked_correct / total_count
            baseline_masked_accuracy = total_baseline_masked_correct / total_count
            
            writer.writerow([args.model, args.top_n, args.token_num, original_accuracy, our_masked_accuracy, baseline_masked_accuracy])
            
            print(f"Model: {args.model}")
            print(f"Top N: {args.top_n}")
            print(f"Token Num: {args.token_num}")
            print(f"Original accuracy: {original_accuracy}")
            print(f"Our masked accuracy: {our_masked_accuracy}")
            print(f"Baseline masked accuracy: {baseline_masked_accuracy}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Accuracy comparison for different models and token settings.")
    parser.add_argument('--model', type=str, required=True, choices=['vit_b_16', 'vit_b_32', 'resnet18', 'resnet50'], help="Model to use for comparison")
    parser.add_argument('--top_n', type=int, required=True, help="Top N tokens to consider")
    parser.add_argument('--token_num', type=int, required=True, help="Number of tokens to use")
    parser.add_argument('--gpu', type=int, default=None, help="Specify GPU to use (e.g., 0 or 1). If not specified, will use any available GPU or CPU.")
    args = parser.parse_args()
    
    main(args)