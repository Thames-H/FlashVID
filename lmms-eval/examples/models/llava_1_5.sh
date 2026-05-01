# install lmms_eval without building dependencies
cd lmms_eval;
pip install --no-deps -U -e .

# install LLaVA without building dependencies
cd LLaVA
pip install --no-deps -U -e .

# install all the requirements that require for reproduce llava results
pip install -r llava_repr_requirements.txt

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MODEL_DIR="${MODEL_DIR:-${HOME}/autodl-tmp/llavav-1.5-7b}"

# download official LLaVA-1.5 checkpoint with HF mirror
MODEL_DIR="$MODEL_DIR" bash "${PROJECT_ROOT}/scripts/download_llava_1_5.sh"

# Run and exactly reproduce llava_v1.5 results!
# mme as an example
accelerate launch --num_processes=1 -m lmms_eval --model llava   --model_args pretrained="${MODEL_DIR},device_map=auto"   --tasks mme  --batch_size 1 --log_samples --log_samples_suffix reproduce --output_path ./logs/
