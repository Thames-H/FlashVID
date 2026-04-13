#!/bin/bash
set -e

DATASET_DIR="/workspace/video_general_data_cloud/qianjiawen/datasets/Video-MME"
HF_HOME_DIR="/workspace/video_general_data_cloud/qianjiawen/hf_cache"
VIDEOMME_CACHE="${HF_HOME_DIR}/videomme"
VIDEOMME_DATA="${VIDEOMME_CACHE}/data"
MODEL_PATH="/workspace/video_general_data_cloud/qianjiawen/ckpt/qwen/Qwen/Qwen2.5-VL-7B-Instruct"
WORK_DIR="/workspace/home/qianjiawen/code_data/FlashVID"
CONDA_ENV="/root/miniconda3/envs/myenv/bin"

echo "=========================================="
echo "FlashVID + Qwen2.5-VL-7B VideoMME Eval"
echo "=========================================="

# ---- Step 1: Wait for dataset download ----
echo "[Step 1] Waiting for dataset download to complete (20 zip files)..."
while true; do
    ZIP_COUNT=$(ls "${DATASET_DIR}"/videos_chunked_*.zip 2>/dev/null | wc -l)
    echo "  $(date '+%H:%M:%S') - Downloaded ${ZIP_COUNT}/20 video zip files"
    if [ "$ZIP_COUNT" -ge 20 ]; then
        # Check if git-lfs is still running (file might not be fully written)
        if pgrep -f "git-lfs" > /dev/null 2>&1; then
            echo "  git-lfs still running, waiting for it to finish..."
            sleep 30
            continue
        fi
        echo "  All 20 zip files downloaded!"
        break
    fi
    sleep 60
done

# ---- Step 2: Extract video files ----
echo "[Step 2] Extracting video zip files to ${VIDEOMME_DATA}..."
mkdir -p "${VIDEOMME_DATA}"

for zipfile in "${DATASET_DIR}"/videos_chunked_*.zip; do
    echo "  Extracting $(basename ${zipfile})..."
    unzip -o -j "${zipfile}" -d "${VIDEOMME_DATA}" 2>/dev/null || {
        # Some zips have nested directory structure, try without -j
        unzip -o "${zipfile}" -d "${VIDEOMME_CACHE}" 2>/dev/null
    }
done

# Check how many mp4 files we got
MP4_COUNT=$(ls "${VIDEOMME_DATA}"/*.mp4 2>/dev/null | wc -l)
echo "  Extracted ${MP4_COUNT} mp4 files"

# Also check if videos ended up in a subdirectory
if [ "$MP4_COUNT" -eq 0 ]; then
    echo "  No mp4 in data/, looking for them in subdirectories..."
    # Find all mp4 files and move/link them to data/
    find "${VIDEOMME_CACHE}" -name "*.mp4" -exec mv {} "${VIDEOMME_DATA}/" \;
    MP4_COUNT=$(ls "${VIDEOMME_DATA}"/*.mp4 2>/dev/null | wc -l)
    echo "  Found and moved ${MP4_COUNT} mp4 files to data/"
fi

if [ "$MP4_COUNT" -eq 0 ]; then
    echo "ERROR: No mp4 files found after extraction!"
    exit 1
fi

# ---- Step 3: Check parquet file ----
echo "[Step 3] Setting up parquet metadata..."
PARQUET_FILE="${DATASET_DIR}/videomme/test-00000-of-00001.parquet"
if [ -f "${PARQUET_FILE}" ]; then
    echo "  Parquet file exists at ${PARQUET_FILE}"
else
    echo "  WARNING: Parquet file not found at ${PARQUET_FILE}"
fi

# ---- Step 4: Run evaluation ----
echo "[Step 4] Running FlashVID + Qwen2.5-VL-7B evaluation on VideoMME..."

export HF_HOME="${HF_HOME_DIR}"
export CUDA_VISIBLE_DEVICES=0,1,5,6

cd "${WORK_DIR}"

# Use 4 free GPUs (0, 1, 5, 6)
${CONDA_ENV}/accelerate launch \
    --main_process_port 18888 \
    --num_processes 4 \
    -m lmms_eval \
    --model qwen2_5_vl \
    --model_args pretrained=${MODEL_PATH},max_num_frames=32,attn_implementation=flash_attention_2,enable_flashvid=True,retention_ratio=0.10,do_segment=True,min_segment_num=4,complementary_segment=True,token_selection_method=attn_div,alpha=0.70,temporal_threshold=0.8,expansion=1.25,pruning_layer=20,llm_retention_ratio=0.3 \
    --tasks videomme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "qwen2_5_vl_7b_flashvid" \
    --output_path ./logs/flashvid

echo "=========================================="
echo "Evaluation complete!"
echo "Results saved to ./logs/flashvid"
echo "=========================================="
