#!/bin/bash

# Define the Python script name
PYTHON_SCRIPT="accuracy_comparison.py"

# Define the models, top_n values, and token_num
MODELS=("vit_b_16" "vit_b_32" "resnet18" "resnet50")
TOP_N_VALUES=(10 20)
TOKEN_NUM=100

# Loop through all combinations
for MODEL in "${MODELS[@]}"; do
    for TOP_N in "${TOP_N_VALUES[@]}"; do
        echo "Running comparison for model: $MODEL, top_n: $TOP_N, token_num: $TOKEN_NUM"
        python $PYTHON_SCRIPT --model $MODEL --top_n $TOP_N --token_num $TOKEN_NUM --gpu 0
    done
done

echo "All comparisons completed!"