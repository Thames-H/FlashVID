import importlib.util
import json
import tempfile
from pathlib import Path
from unittest import TestCase

import torch


def _load_visual_compare_module():
    repo_root = Path(__file__).resolve().parents[4]
    module_path = (
        repo_root / "tools" / "llava_onevision_token_pruning_visual_compare.py"
    )
    spec = importlib.util.spec_from_file_location(
        "llava_onevision_visual_compare_test",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_visual_compare = _load_visual_compare_module()


def _make_artifact(
    method: str,
    task_name: str,
    doc_id: str,
    keep_indices: torch.Tensor,
    image_preview: torch.Tensor,
    token_boxes: torch.Tensor,
    token_is_spatial: torch.Tensor,
    token_source: list[str],
    scores: dict[str, torch.Tensor],
) -> dict:
    artifact = {
        "method": method,
        "task_name": task_name,
        "doc_id": doc_id,
        "question_text": "Question?",
        "image_preview": image_preview,
        "image_size": [int(image_preview.shape[0]), int(image_preview.shape[1])],
        "token_boxes": token_boxes,
        "token_is_spatial": token_is_spatial,
        "token_source": token_source,
        "visual_compare_eligible": True,
        "visual_compare_skip_reason": None,
        "metadata": {
            "target_layer": 15,
            "n_visual_tokens_scored": int(token_boxes.shape[0]),
        },
        "selection": {
            "num_keep": int(keep_indices.numel()),
        },
    }
    artifact["selection"][
        "mmtok_keep_local" if method == "mmtok" else "fetp_keep_local"
    ] = keep_indices
    if method != "mmtok":
        artifact["selection"]["attention_only_keep_local"] = torch.tensor(
            [0, 1, 5],
            dtype=torch.long,
        )
    artifact["scores"] = scores
    artifact["visual_embeddings"] = torch.randn(token_boxes.shape[0], 6)
    return artifact


class TestLlavaOnevisionVisualCompare(TestCase):
    def test_generate_visual_compare_report_writes_outputs_from_synthetic_artifacts(
        self,
    ):
        image_preview = torch.arange(8 * 8 * 3, dtype=torch.uint8).reshape(8, 8, 3)
        token_boxes = torch.tensor(
            [
                [0, 0, 4, 4],
                [4, 0, 8, 4],
                [0, 4, 4, 8],
                [4, 4, 8, 8],
                [0, 0, 2, 2],
                [0, 0, 0, 0],
            ],
            dtype=torch.float32,
        )
        token_is_spatial = torch.tensor([1, 1, 1, 1, 1, 0], dtype=torch.bool)
        token_source = ["base", "base", "crop", "crop", "crop", "newline"]

        fetp_artifact = _make_artifact(
            method="fetp",
            task_name="gqa",
            doc_id="0",
            keep_indices=torch.tensor([1, 3, 5], dtype=torch.long),
            image_preview=image_preview,
            token_boxes=token_boxes,
            token_is_spatial=token_is_spatial,
            token_source=token_source,
            scores={
                "fetp": torch.tensor([0.1, 0.8, 0.2, 0.6, 0.4, 0.7]),
                "attention_only": torch.tensor([0.7, 0.6, 0.2, 0.1, 0.4, 0.9]),
            },
        )
        mmtok_artifact = _make_artifact(
            method="mmtok",
            task_name="gqa",
            doc_id="0",
            keep_indices=torch.tensor([0, 2, 5], dtype=torch.long),
            image_preview=image_preview,
            token_boxes=token_boxes,
            token_is_spatial=token_is_spatial,
            token_source=token_source,
            scores={
                "initial_marginal_gain": torch.tensor([0.5, 0.3, 0.9, 0.1, 0.4, 0.2]),
            },
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_root = Path(tmpdir) / "artifacts_root"
            output_dir = Path(tmpdir) / "visual_compare"
            (artifact_root / "artifacts" / "fetp").mkdir(parents=True)
            (artifact_root / "artifacts" / "mmtok").mkdir(parents=True)
            torch.save(
                fetp_artifact,
                artifact_root / "artifacts" / "fetp" / "gqa__doc0.pt",
            )
            torch.save(
                mmtok_artifact,
                artifact_root / "artifacts" / "mmtok" / "gqa__doc0.pt",
            )

            summary = _visual_compare.generate_visual_compare_report(
                artifact_root=artifact_root,
                output_dir=output_dir,
            )

            self.assertEqual(summary["num_matched_samples"], 1)
            self.assertEqual(summary["representative_sample"]["doc_id"], "0")

            summary_json = json.loads(
                (output_dir / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary_json["num_matched_samples"], 1)
            self.assertTrue((output_dir / "report.md").exists())
            self.assertTrue((output_dir / "plots" / "gqa__doc0__overview.png").exists())
            self.assertTrue((output_dir / "plots" / "gqa__doc0__structure.png").exists())
            self.assertTrue((output_dir / "plots" / "gqa__doc0__sink.png").exists())
