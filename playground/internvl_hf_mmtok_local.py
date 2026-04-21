import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
from PIL import Image
from transformers.hf_argparser import HfArgumentParser


REPO_ROOT = Path(__file__).resolve().parents[1]
LMMS_EVAL_ROOT = REPO_ROOT / "lmms-eval"
DEFAULT_MODEL_DIRNAME = "InternVL3_5-8B-HF"
DEFAULT_MODEL_PATH = REPO_ROOT / DEFAULT_MODEL_DIRNAME
DEFAULT_IMAGE_PATH = REPO_ROOT / "assets" / "method.png"
DEFAULT_VIDEO_PATH = REPO_ROOT / "assets" / "Qgr4dcsY-60.mp4"

if str(LMMS_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(LMMS_EVAL_ROOT))

from lmms_eval.models.chat.internvl_hf_mmtok import InternVLHfMMTok


def resolve_default_model_path() -> str:
    override = os.environ.get("INTERNVL_HF_MODEL_PATH")
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
    target_vision_tokens: int = field(default=0)
    attn_implementation: str = field(default="sdpa")
    min_patches: int = field(default=1)
    max_patches: int = field(default=12)
    fps: float = field(default=0.0)


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
            {"type": "video", "video": str(Path(args.video_path))},
            {"type": "text", "text": args.question},
        ]
    else:
        image = Image.open(args.image_path).convert("RGB")
        content = [
            {"type": "image", "image": image},
            {"type": "text", "text": args.question},
        ]

    return [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": content},
    ]


def build_media_lists(args: InferenceArguments):
    if args.media_type == "video":
        return [], [str(Path(args.video_path))]
    return [Image.open(args.image_path).convert("RGB")], []


@torch.no_grad()
def run_inference(runner: InternVLHfMMTok, args: InferenceArguments) -> str:
    messages = build_messages(args)
    text = runner.processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    visuals, videos = build_media_lists(args)
    images_kwargs = {
        "min_patches": args.min_patches,
        "max_patches": args.max_patches,
    }
    videos_kwargs = {
        "num_frames": args.num_frames,
    }
    if args.fps > 0:
        videos_kwargs["fps"] = args.fps

    if not videos:
        videos = None

    inputs = runner.processor(
        images=visuals,
        videos=videos,
        text=text,
        return_tensors="pt",
        **images_kwargs,
        **videos_kwargs,
    ).to(runner.device, runner.model.dtype)

    generated_ids = runner.model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        use_cache=runner.use_cache,
        pad_token_id=runner.pad_token_id,
        eos_token_id=runner.eot_token_id,
    )
    trimmed_ids = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
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
            "Set INTERNVL_HF_MODEL_PATH or place the local weights under InternVL3_5-8B-HF."
        )

    if args.media_type == "image" and not Path(args.image_path).exists():
        raise FileNotFoundError(f"Image path not found: {args.image_path}")
    if args.media_type == "video" and not Path(args.video_path).exists():
        raise FileNotFoundError(f"Video path not found: {args.video_path}")

    runtime_device, runtime_device_map = resolve_runtime_device()
    runner = InternVLHfMMTok(
        pretrained=args.model_path,
        device=runtime_device,
        batch_size=1,
        device_map=runtime_device_map,
        attn_implementation=args.attn_implementation,
        min_patches=args.min_patches,
        max_patches=args.max_patches,
        num_frames=args.num_frames,
        fps=(args.fps if args.fps > 0 else None),
        retain_ratio=args.retain_ratio,
        target_vision_tokens=(args.target_vision_tokens if args.target_vision_tokens > 0 else None),
    )
    answer = run_inference(runner, args)
    print(answer)


if __name__ == "__main__":
    main()
