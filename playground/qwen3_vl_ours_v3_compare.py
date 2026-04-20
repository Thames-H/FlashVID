import gc
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.hf_argparser import HfArgumentParser
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel


REPO_ROOT = Path(__file__).resolve().parents[1]
LMMS_EVAL_ROOT = REPO_ROOT / "lmms-eval"
DEFAULT_REPO_ID = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_MODEL_DIRNAME = "Qwen3-VL-8B-Instruct"
DEFAULT_AUTODL_MODEL_PATH = Path.home() / "autodl-tmp" / DEFAULT_MODEL_DIRNAME
if str(LMMS_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(LMMS_EVAL_ROOT))

from lmms_eval.models.chat.qwen3_vl_ours_v2 import _make_fetp_forward
from lmms_eval.models.chat.qwen3_vl_ours_v3 import _make_fetp_anchor_forward


def resolve_default_model_path() -> str:
    override = Path(
        os.environ.get("QWEN3_VL_MODEL_PATH", "")
    ).expanduser() if os.environ.get("QWEN3_VL_MODEL_PATH") else None
    if override:
        return str(override)
    if DEFAULT_AUTODL_MODEL_PATH.exists():
        return str(DEFAULT_AUTODL_MODEL_PATH)
    return str(REPO_ROOT / DEFAULT_MODEL_DIRNAME)


@dataclass
class ComparisonArguments:
    question: str = field(default="Describe the video in detail.")
    video_path: str = field(default=str(REPO_ROOT / "assets" / "Qgr4dcsY-60.mp4"))
    model_path: str = field(default_factory=resolve_default_model_path)
    num_frames: int = field(default=16)
    max_new_tokens: int = field(default=256)
    attn_implementation: str = field(default="sdpa")
    retention_ratio: float = field(default=0.25)
    v2_scoring_method: str = field(default="shallow")
    v2_shallow_layers: int = field(default=4)
    v2_target_layer: int = field(default=15)
    v3_anchor_layers: str = field(default="9,18,27")
    v3_candidate_ratio: float = field(default=0.5)
    v3_max_score_text_tokens: int = field(default=8)
    v3_max_score_heads: int = field(default=8)
    v3_profile_reference_scoring: bool = field(default=True)
    v3_reference_scoring_method: str = field(default="shallow")


def parse_args():
    parser = HfArgumentParser(ComparisonArguments)
    (arguments,) = parser.parse_args_into_dataclasses()
    return arguments


def load_model(model_path: str, attn_implementation: str):
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model path not found: {model_path}. "
            f"Set QWEN3_VL_MODEL_PATH, place weights in {DEFAULT_AUTODL_MODEL_PATH}, "
            f"or download {DEFAULT_REPO_ID} into {REPO_ROOT / DEFAULT_MODEL_DIRNAME}."
        )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype="auto",
        device_map="auto",
        attn_implementation=attn_implementation,
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


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
    answer = processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return answer, getattr(model.model, "_fetp_last_pruning_stats", {})


def patch_v2(args: ComparisonArguments):
    Qwen3VLModel.forward = _make_fetp_forward(
        retention_ratio=args.retention_ratio,
        scoring_method=args.v2_scoring_method,
        shallow_layers=args.v2_shallow_layers,
        target_layer=args.v2_target_layer,
        use_alpha=True,
        use_deviation=True,
    )


def patch_v3(args: ComparisonArguments):
    Qwen3VLModel.forward = _make_fetp_anchor_forward(
        retention_ratio=args.retention_ratio,
        scoring_method="anchor",
        shallow_layers=args.v2_shallow_layers,
        target_layer=args.v2_target_layer,
        anchor_layers=args.v3_anchor_layers,
        candidate_ratio=args.v3_candidate_ratio,
        max_score_text_tokens=args.v3_max_score_text_tokens,
        max_score_heads=args.v3_max_score_heads,
        use_alpha=True,
        use_deviation=True,
        profile_reference_scoring=args.v3_profile_reference_scoring,
        reference_scoring_method=args.v3_reference_scoring_method,
    )


def run_variant(label: str, patch_fn, args: ComparisonArguments):
    patch_fn(args)
    model, processor = load_model(
        model_path=args.model_path,
        attn_implementation=args.attn_implementation,
    )
    model.eval()

    start = time.perf_counter()
    answer, pruning_stats = run_inference(
        model=model,
        processor=processor,
        video_path=args.video_path,
        prompt=args.question,
        num_frames=args.num_frames,
        max_new_tokens=args.max_new_tokens,
    )
    wall_time_s = time.perf_counter() - start

    result = {
        "label": label,
        "answer": answer,
        "wall_time_ms": wall_time_s * 1000.0,
        "pruning_scoring_time_ms": pruning_stats.get("pruning_scoring_time_ms"),
        "pruning_total_time_ms": pruning_stats.get("pruning_total_time_ms"),
        "pruning_reference_speedup": pruning_stats.get("pruning_reference_speedup"),
        "stats": pruning_stats,
    }

    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main():
    args = parse_args()

    v2_result = run_variant("qwen3_vl_ours_v2", patch_v2, args)
    v3_result = run_variant("qwen3_vl_ours_v3", patch_v3, args)

    comparison = {
        "qwen3_vl_ours_v2": v2_result,
        "qwen3_vl_ours_v3": v3_result,
        "speedup_vs_v2": (
            v2_result["wall_time_ms"] / v3_result["wall_time_ms"]
            if v3_result["wall_time_ms"]
            else None
        ),
        "pruning_speedup_vs_v2": (
            v2_result["pruning_scoring_time_ms"] / v3_result["pruning_scoring_time_ms"]
            if v2_result["pruning_scoring_time_ms"]
            and v3_result["pruning_scoring_time_ms"]
            else None
        ),
    }

    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
