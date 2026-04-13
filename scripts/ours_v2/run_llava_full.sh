#!/bin/bash

# FETP for LLaVA-1.5 (HF format)
# Mode: full (approach 3)

cd /workspace/home/qianjiawen/code_data/FlashVID/lmms-eval

source activate
conda activate fv-clean

# Evaluation benchmarks.
TASKS=("mme")

# Pretrained model path.
PRETRAINED="/workspace/video_general_data_cloud/qianjiawen/ckpt/llava-hf/llava1_5-7b"

# FETP arguments.
RETENTION_RATIOS=(64 128 192)
SCORING_METHOD=full
TARGET_LAYER=15   # LLaMA-7B has 32 layers, middle = 15

# Model arguments.
ATTN_IMPLEMENTATION=flash_attention_2

BASE_MODEL_ARGS="pretrained=$PRETRAINED,attn_implementation=$ATTN_IMPLEMENTATION"
BASE_MODEL_ARGS="$BASE_MODEL_ARGS,scoring_method=$SCORING_METHOD,target_layer=$TARGET_LAYER"

for retention_ratio in "${RETENTION_RATIOS[@]}"; do
    echo "Running FETP-full on LLaVA-1.5 with retention_ratio=${retention_ratio}"
    MODEL_ARGS="$BASE_MODEL_ARGS,retention_ratio=${retention_ratio}"
    for task in "${TASKS[@]}"; do
        echo "Evaluating task: $task"
        CUDA_VISIBLE_DEVICES=0,1,6,7 accelerate launch \
        --main_process_port 18899 \
        --num_processes 4 \
        -m lmms_eval \
        --model llava_hf_ours_v2 \
        --model_args $MODEL_ARGS \
        --tasks $task \
        --batch_size 1 \
        --output_path /workspace/home/qianjiawen/code_data/FlashVID/lmms-eval/logs/llava_hf_ours_v2 \
        --verbosity=DEBUG
    done
    echo "Finished running with retention_ratio=${retention_ratio}"
done
