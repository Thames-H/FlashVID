#!/bin/bash

# FETP: Functionally Equivalent Token Pruning
# Mode: shallow (approach 1 - practical approximation)
# Runs only the first K LLM layers to approximate attention weights.

cd /workspace/home/qianjiawen/code_data/FlashVID/lmms-eval

source activate
conda activate fv-clean

# Evaluation benchmarks.
TASKS=("videomme")

# Pretrained model path.
PRETRAINED="/workspace/video_general_data_cloud/qianjiawen/ckpt/qwen/Qwen/Qwen2.5-VL-7B-Instruct"

# FETP arguments.
RETENTION_RATIOS=(0.10 0.15 0.20 0.25)
SCORING_METHOD=shallow
SHALLOW_LAYERS=4    # Number of LLM layers to run.
TARGET_LAYER=3      # Extract attention from this layer (must be < SHALLOW_LAYERS).

# Model arguments.
MAX_NUM_FRAMES=32
ATTN_IMPLEMENTATION=flash_attention_2

BASE_MODEL_ARGS="pretrained=$PRETRAINED,max_num_frames=$MAX_NUM_FRAMES,attn_implementation=$ATTN_IMPLEMENTATION"
BASE_MODEL_ARGS="$BASE_MODEL_ARGS,scoring_method=$SCORING_METHOD,shallow_layers=$SHALLOW_LAYERS,target_layer=$TARGET_LAYER"

for retention_ratio in "${RETENTION_RATIOS[@]}"; do
    echo "Running FETP-shallow with retention_ratio=${retention_ratio}"
    MODEL_ARGS="$BASE_MODEL_ARGS,retention_ratio=${retention_ratio}"
    for task in "${TASKS[@]}"; do
        echo "Evaluating task: $task"
        CUDA_VISIBLE_DEVICES=0,1,5,6 accelerate launch \
        --main_process_port 18891 \
        --num_processes 4 \
        -m lmms_eval \
        --model qwen2_5_vl_ours_v2 \
        --model_args $MODEL_ARGS \
        --tasks $task \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix "qwen2_5_vl_ours_v2_shallow" \
        --output_path /workspace/home/qianjiawen/code_data/FlashVID/lmms-eval/logs/ours_v2_shallow \
        --verbosity=DEBUG
    done
    echo "Finished running with retention_ratio=${retention_ratio}"
done
