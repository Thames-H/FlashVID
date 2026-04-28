#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# Long-video pruning ablations for LLaVA-OneVision FETP-v3.
# Defaults run stratified sampled LongVideoBench and VideoMME at 10% and 15%.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export LMMS_EVAL_USE_CACHE="${LMMS_EVAL_USE_CACHE:-True}"
export LMMS_EVAL_HOME="${LMMS_EVAL_HOME:-$PROJECT_ROOT/.cache/lmms-eval}"
export PYTHONPATH="$PROJECT_ROOT/lmms-eval${PYTHONPATH:+:$PYTHONPATH}"
export FETP_SAMPLE_SIZE="${FETP_SAMPLE_SIZE:-96}"
export FETP_SAMPLE_SEED="${FETP_SAMPLE_SEED:-42}"
export OPENCV_LOG_LEVEL="${OPENCV_LOG_LEVEL:-ERROR}"
export OPENCV_FFMPEG_LOGLEVEL="${OPENCV_FFMPEG_LOGLEVEL:-8}"
export AV_LOG_FORCE_NOCOLOR="${AV_LOG_FORCE_NOCOLOR:-1}"
export FLASHVID_SUPPRESS_DECODER_STDERR="${FLASHVID_SUPPRESS_DECODER_STDERR:-1}"

NUM_PROCESSES="${NUM_PROCESSES:-4}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-18895}"
BATCH_SIZE="${BATCH_SIZE:-1}"
CACHE_REQUESTS="${CACHE_REQUESTS:-true}"

AUTODL_MODEL_PATH="${AUTODL_MODEL_PATH:-$HOME/autodl-tmp/llava-onevision-qwen2-7b-ov-hf}"
DEFAULT_PRETRAINED="${DEFAULT_PRETRAINED:-llava-hf/llava-onevision-qwen2-7b-ov-hf}"
if [[ -d "$AUTODL_MODEL_PATH" ]]; then
    DEFAULT_PRETRAINED="$AUTODL_MODEL_PATH"
fi
PRETRAINED="${PRETRAINED:-$DEFAULT_PRETRAINED}"

if [[ -n "${TASKS_CSV:-}" ]]; then
    IFS=',' read -r -a TASKS <<< "$TASKS_CSV"
else
    TASKS=("longvideobench_val_v_sampled" "videomme_sampled")
fi

if [[ -n "${RETENTION_RATIOS_CSV:-}" ]]; then
    IFS=',' read -r -a RETENTION_RATIOS <<< "$RETENTION_RATIOS_CSV"
else
    RETENTION_RATIOS=(0.10 0.15)
fi

if [[ -n "${EXPERIMENT_SPECS_CSV:-}" ]]; then
    IFS=',' read -r -a EXPERIMENT_SPECS <<< "$EXPERIMENT_SPECS_CSV"
else
    EXPERIMENT_SPECS=(
        "baseline_topk:topk:1"
        "frame_min1:frame_aware:1"
        "frame_min2:frame_aware:2"
        "uniform_no_fes:uniform:1"
        "adaptive_gap:adaptive_topk:1"
        "frame_adaptive_min1:frame_aware_adaptive:1"
        "temporal_then_fes:temporal_then_fes:1"
    )
fi

SCORING_METHOD="${SCORING_METHOD:-full}"
SHALLOW_LAYERS="${SHALLOW_LAYERS:-4}"
TARGET_LAYER="${TARGET_LAYER:-15}"
USE_ALPHA="${USE_ALPHA:-true}"
USE_DEVIATION="${USE_DEVIATION:-true}"
TWO_STAGE="${TWO_STAGE:-false}"
TEXT_CHUNK_SIZE="${TEXT_CHUNK_SIZE:-32}"
SCORING_TEXT_MODE="${SCORING_TEXT_MODE:-benchmark_question}"
GAP_PERCENTILE="${GAP_PERCENTILE:-0.8}"
TEMPORAL_RATIO="${TEMPORAL_RATIO:-0.5}"
TOKENS_PER_FRAME="${TOKENS_PER_FRAME:-}"

MAX_FRAMES_NUM="${MAX_FRAMES_NUM:-16}"
USE_HF_VIDEO_PROCESSOR="${USE_HF_VIDEO_PROCESSOR:-false}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
DTYPE="${DTYPE:-float16}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./logs/ours_v3_llava_onevision_7b_long_video_pruning_ablation}"
LOG_SAMPLES_SUFFIX_PREFIX="${LOG_SAMPLES_SUFFIX_PREFIX:-llava_onevision_ours_v3_7b_long_video_ablation}"

REQUEST_CACHE_ARGS=()
if [[ -n "$CACHE_REQUESTS" ]]; then
    REQUEST_CACHE_ARGS=(--cache_requests "$CACHE_REQUESTS")
fi

BASE_MODEL_ARGS="pretrained=$PRETRAINED,max_frames_num=$MAX_FRAMES_NUM,use_hf_video_processor=$USE_HF_VIDEO_PROCESSOR,attn_implementation=$ATTN_IMPLEMENTATION,dtype=$DTYPE,scoring_method=$SCORING_METHOD,shallow_layers=$SHALLOW_LAYERS,target_layer=$TARGET_LAYER,use_alpha=$USE_ALPHA,use_deviation=$USE_DEVIATION,two_stage=$TWO_STAGE,text_chunk_size=$TEXT_CHUNK_SIZE,scoring_text_mode=$SCORING_TEXT_MODE"

for experiment_spec in "${EXPERIMENT_SPECS[@]}"; do
    IFS=':' read -r experiment_name pruning_policy min_keep_per_frame <<< "$experiment_spec"
    if [[ -z "${experiment_name:-}" || -z "${pruning_policy:-}" ]]; then
        echo "Invalid experiment spec: $experiment_spec" >&2
        exit 1
    fi
    min_keep_per_frame="${min_keep_per_frame:-1}"

    POLICY_ARGS="pruning_policy=$pruning_policy,min_keep_per_frame=$min_keep_per_frame,gap_percentile=$GAP_PERCENTILE,temporal_ratio=$TEMPORAL_RATIO"
    if [[ -n "$TOKENS_PER_FRAME" ]]; then
        POLICY_ARGS="$POLICY_ARGS,tokens_per_frame=$TOKENS_PER_FRAME"
    fi

    for retention_ratio in "${RETENTION_RATIOS[@]}"; do
        ratio_tag="${retention_ratio//./p}"
        model_args="$BASE_MODEL_ARGS,$POLICY_ARGS,retention_ratio=$retention_ratio"

        for task in "${TASKS[@]}"; do
            output_path="$OUTPUT_ROOT/$experiment_name/r$ratio_tag/$task"
            log_suffix="${LOG_SAMPLES_SUFFIX_PREFIX}_${experiment_name}_r${ratio_tag}_${task}"
            echo "Running $task: experiment=$experiment_name policy=$pruning_policy min_keep_per_frame=$min_keep_per_frame retention_ratio=$retention_ratio sample_size=$FETP_SAMPLE_SIZE sample_seed=$FETP_SAMPLE_SEED"

            accelerate launch \
                --main_process_port "$MAIN_PROCESS_PORT" \
                --num_processes "$NUM_PROCESSES" \
                -m lmms_eval \
                --model llava_onevision_ours_v3 \
                --model_args "$model_args" \
                --tasks "$task" \
                --batch_size "$BATCH_SIZE" \
                "${REQUEST_CACHE_ARGS[@]}" \
                --log_samples \
                --log_samples_suffix "$log_suffix" \
                --output_path "$output_path"
        done
    done
done
