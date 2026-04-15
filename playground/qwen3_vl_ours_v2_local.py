import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.hf_argparser import HfArgumentParser
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel


REPO_ROOT = Path(__file__).resolve().parents[1]
LMMS_EVAL_ROOT = REPO_ROOT / "lmms-eval"
DEFAULT_REPO_ID = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_MODEL_DIRNAME = "Qwen3-VL-2B-Instruct"
if str(LMMS_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(LMMS_EVAL_ROOT))

from lmms_eval.models.chat.qwen3_vl_ours_v2 import _make_fetp_forward


@dataclass
class InferenceArguments:
    question: str = field(default="Describe the video in detail.")
    video_path: str = field(default=str(REPO_ROOT / "assets" / "Qgr4dcsY-60.mp4"))
    model_path: str = field(default=str(REPO_ROOT / DEFAULT_MODEL_DIRNAME))
    num_frames: int = field(default=16)
    max_new_tokens: int = field(default=512)
    enable_ours_v2: bool = field(default=True)
    retention_ratio: float = field(default=0.25)
    scoring_method: str = field(default="shallow")
    shallow_layers: int = field(default=4)
    target_layer: int = field(default=15)
    use_alpha: bool = field(default=True)
    use_deviation: bool = field(default=True)
    attn_implementation: str = field(default="sdpa")


def parse_args():
    parser = HfArgumentParser(InferenceArguments)
    (arguments,) = parser.parse_args_into_dataclasses()
    return arguments


def load_model(model_path: str, attn_implementation: str):
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model path not found: {model_path}. "
            f"Download {DEFAULT_REPO_ID} into {REPO_ROOT / DEFAULT_MODEL_DIRNAME} first."
        )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype="auto",
        device_map="auto",
        attn_implementation=attn_implementation,
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def patch_qwen3_vl_model(
    retention_ratio: float,
    scoring_method: str,
    shallow_layers: int,
    target_layer: int,
    use_alpha: bool,
    use_deviation: bool,
):
    Qwen3VLModel.forward = _make_fetp_forward(
        retention_ratio=retention_ratio,
        scoring_method=scoring_method,
        shallow_layers=shallow_layers,
        target_layer=target_layer,
        use_alpha=use_alpha,
        use_deviation=use_deviation,
    )


@torch.no_grad()
def run_inference(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    video_path: str,
    prompt: str,
    num_frames: int,
    max_new_tokens: int,
):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "nframes": num_frames,
                    "max_pixels": 256 * 28 * 28,
                    "min_pixels": 64 * 28 * 28,
                },
                {"type": "text", "text": prompt},
            ],
        },
    ]

    text = processor.apply_chat_template(
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

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        video_metadata=video_metadatas,
        do_resize=False,
        return_tensors="pt",
        **video_kwargs,
    )

    inputs.pop("token_type_ids", None)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = inputs.to(device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    trimmed_ids = generated_ids[:, inputs.input_ids.shape[1] :]
    return processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


def main():
    args = parse_args()

    if args.enable_ours_v2:
        patch_qwen3_vl_model(
            retention_ratio=args.retention_ratio,
            scoring_method=args.scoring_method,
            shallow_layers=args.shallow_layers,
            target_layer=args.target_layer,
            use_alpha=args.use_alpha,
            use_deviation=args.use_deviation,
        )

    model, processor = load_model(
        model_path=args.model_path,
        attn_implementation=args.attn_implementation,
    )
    model.eval()

    answer = run_inference(
        model=model,
        processor=processor,
        video_path=args.video_path,
        prompt=args.question,
        num_frames=args.num_frames,
        max_new_tokens=args.max_new_tokens,
    )
    print(answer)


if __name__ == "__main__":
    main()
