export HF_HOME="/workspace/video_general_data_cloud/qianjiawen/cache"
cd /workspace/home/qianjiawen/code_data/FlashVID/lmms-eval
export HF_ENDPOINT=https://hf-mirror.com

source activate

conda activate fv-clean
CUDA_VISIBLE_DEVICES=0,1,6,7 accelerate launch --num_processes=4 --main_process_port=12346 -m lmms_eval \
    --model qwen2_5_vl_mmtok \
    --model_args=pretrained=/workspace/video_general_data_cloud/qianjiawen/ckpt/qwen/Qwen/Qwen2.5-VL-7B-Instruct,max_pixels=12845056,attn_implementation=flash_attention_2,interleave_visuals=False \
    --tasks ocrbench \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "qwen2_5vl_mmtok_0_2" \
    --output_path /workspace/home/qianjiawen/code_data/FlashVID/lmms-eval/logs/mmtok \
    --verbosity=DEBUG