#!/bin/bash

# export CUDA_VISIBLE_DEVICES=0,1,5,6

cd /workspace/home/qianjiawen/code_data/FlashVID/lmms-eval

source activate

conda activate fv-clean

# Evaluation benchmarks.
TASKS=("videomme")

# Pretrained model path.
PRETRAINED="/workspace/video_general_data_cloud/qianjiawen/ckpt/qwen/Qwen/Qwen2.5-VL-7B-Instruct"

# Query-aware anchor propagation arguments.
RETENTION_RATIOS=(0.15 0.25 0.3 0.5)
SHALLOW_LAYERS=20
ALPHA=1.0
SEGMENT_THRESHOLD=0.9
MIN_SEGMENT_NUM=8
T_MATCH=0.7
LAMBDA_MIN=0.4
R_MAX=0.7
R_EXPLORE_MIN=0.15

# Model arguments.
MAX_NUM_FRAMES=32
ATTN_IMPLEMENTATION=flash_attention_2
# PIXEL_VALUES_VIDEOS=true

USE_PIXEL_SEGMENT=true

BASE_MODEL_ARGS="pretrained=$PRETRAINED,max_num_frames=$MAX_NUM_FRAMES,attn_implementation=$ATTN_IMPLEMENTATION,use_pixel_segment=$USE_PIXEL_SEGMENT"

BASE_MODEL_ARGS="$BASE_MODEL_ARGS,shallow_layers=$SHALLOW_LAYERS,alpha=$ALPHA"
BASE_MODEL_ARGS="$BASE_MODEL_ARGS,segment_threshold=$SEGMENT_THRESHOLD,min_segment_num=$MIN_SEGMENT_NUM"
BASE_MODEL_ARGS="$BASE_MODEL_ARGS,T_match=$T_MATCH,lambda_min=$LAMBDA_MIN"
BASE_MODEL_ARGS="$BASE_MODEL_ARGS,r_max=$R_MAX,r_explore_min=$R_EXPLORE_MIN"


for retention_ratio in "${RETENTION_RATIOS[@]}"; do
    echo "Running with retention_ratio=${retention_ratio}"
    MODEL_ARGS="$BASE_MODEL_ARGS,retention_ratio=${retention_ratio}"
    for task in "${TASKS[@]}"; do
        echo "Evaluating task: $task"
        CUDA_VISIBLE_DEVICES=0,1,5,6 accelerate launch \
        --main_process_port 18889 \
        --num_processes 4 \
        -m lmms_eval \
        --model qwen2_5_vl_ours \
        --model_args $MODEL_ARGS \
        --tasks $task \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix "qwen2_5_vl_ours" \
        --output_path /workspace/home/qianjiawen/code_data/FlashVID/lmms-eval/logs/ours \
        --verbosity=DEBUG
    done
    echo "Finished running with retention_ratio=${retention_ratio}"
done