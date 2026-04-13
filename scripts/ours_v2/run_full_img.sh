#!/bin/bash

# FETP: Functionally Equivalent Token Pruning
# Mode: full (approach 3 - theoretical upper bound)
# Runs a complete LLM forward pass to extract exact attention weights.

cd /workspace/home/qianjiawen/code_data/FlashVID/lmms-eval

source activate
conda activate fv-clean

# Evaluation benchmarks.
TASKS=("pope" "mme" "gqa")

# Pretrained model path.
PRETRAINED="/workspace/video_general_data_cloud/qianjiawen/ckpt/qwen/Qwen/Qwen2.5-VL-7B-Instruct"

# FETP arguments.
RETENTION_RATIOS=(0.05 0.1 0.2)
SCORING_METHOD=full
TARGET_LAYER=15   # Which LLM layer to extract attention from (0-indexed).

# Model arguments.
MAX_NUM_FRAMES=8
ATTN_IMPLEMENTATION=flash_attention_2
USE_APLHA=false
USE_DEVAITION=true

BASE_MODEL_ARGS="pretrained=$PRETRAINED,max_num_frames=$MAX_NUM_FRAMES,attn_implementation=$ATTN_IMPLEMENTATION,use_alpha=$USE_APLHA,use_deviation=$USE_DEVAITION"

BASE_MODEL_ARGS="$BASE_MODEL_ARGS,scoring_method=$SCORING_METHOD,target_layer=$TARGET_LAYER"

for retention_ratio in "${RETENTION_RATIOS[@]}"; do
    echo "Running FETP-full with retention_ratio=${retention_ratio}"
    MODEL_ARGS="$BASE_MODEL_ARGS,retention_ratio=${retention_ratio}"
    for task in "${TASKS[@]}"; do
        echo "Evaluating task: $task"
        CUDA_VISIBLE_DEVICES=0,1,6,7 accelerate launch \
        --main_process_port 18890 \
        --num_processes 4 \
        -m lmms_eval \
        --model qwen2_5_vl_ours_v2 \
        --model_args $MODEL_ARGS \
        --tasks $task \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix "qwen2_5_vl_ours_v2_full" \
        --output_path /workspace/home/qianjiawen/code_data/FlashVID/lmms-eval/logs/ours_v2_full \
        --verbosity=DEBUG
    done
    echo "Finished running with retention_ratio=${retention_ratio}"
done
