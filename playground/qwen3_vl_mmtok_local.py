import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers.hf_argparser import HfArgumentParser


REPO_ROOT = Path(__file__).resolve().parents[1]
LMMS_EVAL_ROOT = REPO_ROOT / "lmms-eval"
DEFAULT_MODEL_DIRNAME = "Qwen3-VL-2B-Instruct"
DEFAULT_MODEL_PATH = REPO_ROOT / DEFAULT_MODEL_DIRNAME
DEFAULT_IMAGE_PATH = REPO_ROOT / "assets" / "method.png"
DEFAULT_VIDEO_PATH = REPO_ROOT / "assets" / "Qgr4dcsY-60.mp4"

if str(LMMS_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(LMMS_EVAL_ROOT))

from lmms_eval.models.chat.qwen3_vl_mmtok import Qwen3_VL_MMTok


def resolve_default_model_path() -> str:
    override = os.environ.get("QWEN3_VL_MODEL_PATH")
    if override:
        return str(Path(override).expanduser())
    return str(DEFAULT_MODEL_PATH)


@dataclass
class InferenceArguments:
    question: str = field(default="Describe the figure in one sentence.")
    model_path: str = field(default_factory=resolve_default_model_path)
    media_type: str = field(default="image")
    image_path: str = field(default=str(DEFAULT_IMAGE_PATH))
    video_path: str = field(default=str(DEFAULT_VIDEO_PATH))
    num_frames: int = field(default=8)
    max_new_tokens: int = field(default=128)
    retain_ratio: float = field(default=0.2)
    attn_implementation: str = field(default="sdpa")


def parse_args():
    parser = HfArgumentParser(InferenceArguments)
    (arguments,) = parser.parse_args_into_dataclasses()
    return arguments


def resolve_runtime_device():
    if torch.cuda.is_available():
        return "cuda", "auto"
    return "cpu", "cpu"


def build_messages(args: InferenceArguments):
    if args.media_type == "video":
        content = [
            {
                "type": "video",
                "video": str(Path(args.video_path)),
                "nframes": args.num_frames,
                "max_pixels": 256 * 28 * 28,
                "min_pixels": 64 * 28 * 28,
            },
            {"type": "text", "text": args.question},
        ]
    else:
        image = Image.open(args.image_path).convert("RGB")
        content = [
            {
                "type": "image",
                "image": image,
                "max_pixels": 256 * 28 * 28,
                "min_pixels": 64 * 28 * 28,
            },
            {"type": "text", "text": args.question},
        ]

    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": content},
    ]


@torch.no_grad()
def run_inference(runner: Qwen3_VL_MMTok, args: InferenceArguments) -> str:
    messages = build_messages(args)
    text = runner.processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        return_video_kwargs=True,
        image_patch_size=16,
        return_video_metadata=True,
    )

    video_metadatas = None
    if video_inputs is not None:
        video_inputs, video_metadatas = zip(*video_inputs)
        video_inputs = list(video_inputs)
        video_metadatas = list(video_metadatas)

    inputs = runner.processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        video_metadata=video_metadatas,
        do_resize=False,
        return_tensors="pt",
        **video_kwargs,
    )
    inputs.pop("token_type_ids", None)

    target_device = "cuda" if torch.cuda.is_available() and runner.device_map == "auto" else runner.device
    inputs = inputs.to(target_device)

    generated_ids = runner.model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        use_cache=runner.use_cache,
    )
    trimmed_ids = generated_ids[:, inputs.input_ids.shape[1] :]
    return runner.processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def main():
    args = parse_args()
    if not Path(args.model_path).exists():
        raise FileNotFoundError(
            f"Model path not found: {args.model_path}. "
            "Set QWEN3_VL_MODEL_PATH or place the local weights under Qwen3-VL-2B-Instruct."
        )

    if args.media_type == "image" and not Path(args.image_path).exists():
        raise FileNotFoundError(f"Image path not found: {args.image_path}")
    if args.media_type == "video" and not Path(args.video_path).exists():
        raise FileNotFoundError(f"Video path not found: {args.video_path}")

    runtime_device, runtime_device_map = resolve_runtime_device()
    runner = Qwen3_VL_MMTok(
        pretrained=args.model_path,
        device=runtime_device,
        batch_size=1,
        device_map=runtime_device_map,
        attn_implementation=args.attn_implementation,
        retain_ratio=args.retain_ratio,
    )
    answer = run_inference(runner, args)
    print(answer)


if __name__ == "__main__":
    main()
