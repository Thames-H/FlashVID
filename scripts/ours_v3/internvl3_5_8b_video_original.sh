#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# Editable configuration. Change values here instead of exporting env vars.
CUDA_VISIBLE_DEVICES="0,1,2,3"
LMMS_EVAL_USE_CACHE="True"
LMMS_EVAL_HOME="$PROJECT_ROOT/.cache/lmms-eval"
NUM_PROCESSES=4
MAIN_PROCESS_PORT=18897
BATCH_SIZE=1
LOG_SAMPLES_SUFFIX="internvl3_5_original_8b_video"
OUTPUT_PATH="./logs/original_internvl3_5_8b_video"
TASKS=("videomme" "longvideobench_val_v")

AUTODL_MODEL_PATH="$HOME/autodl-tmp/InternVL3_5-8B-HF"
LEGACY_AUTODL_MODEL_PATH="$HOME/autodl-tmp/InternVL3_5-8B"
DEFAULT_PRETRAINED="OpenGVLab/InternVL3_5-8B-HF"
PRETRAINED="$DEFAULT_PRETRAINED"

NUM_FRAMES=8
MAX_PATCHES=12
ATTN_IMPLEMENTATION="flash_attention_2"
LOW_CPU_MEM_USAGE="true"
DEVICE_MAP=""

if [[ -d "$AUTODL_MODEL_PATH" ]]; then
    PRETRAINED="$AUTODL_MODEL_PATH"
fi

export CUDA_VISIBLE_DEVICES
export LMMS_EVAL_USE_CACHE
export LMMS_EVAL_HOME

if [[ "$NUM_PROCESSES" == "1" && -z "$DEVICE_MAP" ]]; then
    DEVICE_MAP="auto"
fi
if [[ "$NUM_PROCESSES" != "1" && "$DEVICE_MAP" == "auto" ]]; then
    echo "Error: device_map=auto requires NUM_PROCESSES=1."
    exit 1
fi

if [[ "$PRETRAINED" == "$LEGACY_AUTODL_MODEL_PATH" ]]; then
    echo "Error: '$PRETRAINED' is the original InternVL chat-format checkpoint."
    echo "internvl3_5_original expects the HF-format checkpoint:"
    echo "  - OpenGVLab/InternVL3_5-8B-HF"
    echo "  - or a local directory such as $HOME/autodl-tmp/InternVL3_5-8B-HF"
    exit 1
fi

MODEL_ARGS="pretrained=$PRETRAINED,max_patches=$MAX_PATCHES,num_frames=$NUM_FRAMES,attn_implementation=$ATTN_IMPLEMENTATION,low_cpu_mem_usage=$LOW_CPU_MEM_USAGE"
if [[ -n "$DEVICE_MAP" ]]; then
    MODEL_ARGS="$MODEL_ARGS,device_map=$DEVICE_MAP"
fi

for task in "${TASKS[@]}"; do
    echo "Running InternVL3.5-8B original video benchmark"
    echo "Evaluating task: $task"
    accelerate launch \
        --main_process_port "$MAIN_PROCESS_PORT" \
        --num_processes "$NUM_PROCESSES" \
        -m lmms_eval \
        --model internvl3_5_original \
        --model_args "$MODEL_ARGS" \
        --tasks "$task" \
        --batch_size "$BATCH_SIZE" \
        --log_samples \
        --log_samples_suffix "$LOG_SAMPLES_SUFFIX" \
        --output_path "$OUTPUT_PATH"
done
