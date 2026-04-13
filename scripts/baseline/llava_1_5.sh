#!/bin/bash

export HF_HOME="/workspace/video_general_data_cloud/qianjiawen/cache"
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0,1,6,7

cd /workspace/home/qianjiawen/code_data/FlashVID/lmms-eval
source activate
conda activate fv-clean
# Pretrained model path. Replace with local path if available.
PRETRAINED="/workspace/video_general_data_cloud/qianjiawen/ckpt/llava-hf/llava1_5-7b"

TASKS=("pope" "gqa")

for task in "${TASKS[@]}"; do
    echo "========== Evaluating task: $task =========="
    accelerate launch \
        --main_process_port 12346 \
        --num_processes 4 \
        -m lmms_eval \
        --model llava_hf \
        --model_args pretrained=$PRETRAINED,attn_implementation=flash_attention_2 \
        --tasks $task \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix "llava_1_5_baseline" \
        --output_path /workspace/home/qianjiawen/code_data/FlashVID/lmms-eval/logs/llava1_5 \
        --verbosity=DEBUG
done
