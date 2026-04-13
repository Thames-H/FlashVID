export HF_HOME="/workspace/video_general_data_cloud/qianjiawen/cache"
cd /workspace/home/qianjiawen/code_data/FlashVID/lmms-eval
export HF_ENDPOINT=https://hf-mirror.com

source activate

conda activate fv-clean
CUDA_VISIBLE_DEVICES=0,1,6,7 accelerate launch --num_processes=4 --main_process_port=12346 -m lmms_eval \
    --model llava_mmtok \
    --model_args=pretrained=/workspace/video_general_data_cloud/qianjiawen/ckpt/llava-hf/llava1_5-7b,attn_implementation=flash_attention_2 \
    --tasks ocrbench \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "llava_motk_192" \
    --output_path /workspace/home/qianjiawen/code_data/FlashVID/lmms-eval/logs/mmtok_ocr \
    --verbosity=DEBUG