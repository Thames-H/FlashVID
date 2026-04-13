#!/bin/bash

# export CUDA_VISIBLE_DEVICES=0,1,5,6

cd /workspace/home/qianjiawen/code_data/FlashVID/lmms-eval

source activate

conda activate fv-clean

# Evaluation benchmarks.
TASKS=("pope" "gqa")

# Pretrained model path.
PRETRAINED="/workspace/video_general_data_cloud/qianjiawen/ckpt/qwen/Qwen/Qwen2.5-VL-7B-Instruct"

# Random token selection arguments.
RETENTION_RATIOS=(1)

# Model arguments.
MAX_NUM_FRAMES=32
ATTN_IMPLEMENTATION=flash_attention_2
BASE_MODEL_ARGS="pretrained=$PRETRAINED,max_num_frames=$MAX_NUM_FRAMES,attn_implementation=$ATTN_IMPLEMENTATION"


for retention_ratio in "${RETENTION_RATIOS[@]}"; do
    echo "Running with retention_ratio=${retention_ratio}"
    MODEL_ARGS="$BASE_MODEL_ARGS,retention_ratio=${retention_ratio}"
    for task in "${TASKS[@]}"; do
        echo "Evaluating task: $task"
        CUDA_VISIBLE_DEVICES=0,1,6,7 accelerate launch \
        --main_process_port 18889 \
        --num_processes 4 \
        -m lmms_eval \
        --model qwen2_5_vl_random \
        --model_args $MODEL_ARGS \
        --tasks $task \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix "qwen2_5_vl_random" \
        --output_path /workspace/home/qianjiawen/code_data/FlashVID/lmms-eval/logs/random \
        --verbosity=DEBUG
    done
    echo "Finished running with retention_ratio=${retention_ratio}"
done
