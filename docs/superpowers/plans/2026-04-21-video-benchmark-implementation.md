# Video Benchmark Fixed-Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Download `VideoMME` and `LongVideoBench` into fixed paths under `/root/autodl-fs`, make `lmms-eval` load them locally, and add runnable video benchmark entrypoints for `qwen3_vl`, `llava_onevision`, and `internvl3_5`.

**Architecture:** Add one dedicated downloader script for video benchmarks, switch the relevant task YAMLs from implicit remote/cache lookup to explicit fixed local paths, and update benchmark scripts so model entrypoints target the two requested video tasks only. Guard the change with test-first asset/path checks that match the repository's current AST/string-based test style.

**Tech Stack:** Python 3, `datasets`, `huggingface_hub`, `lmms-eval`, shell benchmark runners, `unittest`

---

### Task 1: Add failing fixed-path benchmark setup tests

**Files:**
- Create: `F:/FlashVID/lmms-eval/test/eval/benchmark/test_video_benchmark_setup.py`
- Test: `F:/FlashVID/lmms-eval/test/eval/benchmark/test_video_benchmark_setup.py`

- [ ] **Step 1: Write the failing test**

```python
import ast
from pathlib import Path
from unittest import TestCase


class TestVideoBenchmarkSetup(TestCase):
    def test_download_script_exists_with_expected_video_benchmark_manifest(self):
        repo_root = Path(__file__).resolve().parents[4]
        script_path = repo_root / "tools" / "download_video_benchmarks.py"

        self.assertTrue(script_path.exists(), "video benchmark download script should exist")

        tree = ast.parse(script_path.read_text(encoding="utf-8"))
        strings = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        ]

        self.assertIn("/root/autodl-fs/videomme", strings)
        self.assertIn("/root/autodl-fs/longvideobench", strings)
        self.assertIn("lmms-lab/Video-MME", strings)
        self.assertIn("longvideobench/LongVideoBench", strings)

    def test_task_configs_point_to_fixed_video_benchmark_roots(self):
        repo_root = Path(__file__).resolve().parents[4]
        file_expectations = {
            repo_root / "lmms-eval" / "lmms_eval" / "tasks" / "videomme" / "videomme.yaml": "/root/autodl-fs/videomme",
            repo_root / "lmms-eval" / "lmms_eval" / "tasks" / "longvideobench" / "longvideobench_val_v.yaml": "/root/autodl-fs/longvideobench",
            repo_root / "lmms-eval" / "lmms_eval" / "tasks" / "longvideobench" / "utils.py": "/root/autodl-fs/longvideobench",
        }

        for path, expected in file_expectations.items():
            text = path.read_text(encoding="utf-8")
            self.assertIn(expected, text, f"{path} should reference {expected}")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest F:/FlashVID/lmms-eval/test/eval/benchmark/test_video_benchmark_setup.py -v
```

Expected: `FAIL` because `tools/download_video_benchmarks.py` does not exist and task files still point to old paths.

- [ ] **Step 3: Write minimal implementation**

Do not implement production code yet. Only create the test file exactly as written above.

- [ ] **Step 4: Run test to verify it fails**

Run:

```bash
python -m pytest F:/FlashVID/lmms-eval/test/eval/benchmark/test_video_benchmark_setup.py -v
```

Expected: `FAIL` with missing script or missing string assertions.

- [ ] **Step 5: Commit**

```bash
git add F:/FlashVID/lmms-eval/test/eval/benchmark/test_video_benchmark_setup.py
git commit -m "test: add fixed-path video benchmark setup coverage"
```

### Task 2: Add failing video benchmark script asset tests

**Files:**
- Create: `F:/FlashVID/lmms-eval/test/eval/benchmark/test_video_benchmark_scripts.py`
- Test: `F:/FlashVID/lmms-eval/test/eval/benchmark/test_video_benchmark_scripts.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path
from unittest import TestCase


class TestVideoBenchmarkScripts(TestCase):
    def test_qwen_and_llava_scripts_target_requested_video_tasks(self):
        repo_root = Path(__file__).resolve().parents[4]
        paths = [
            repo_root / "scripts" / "qwen3_vl.sh",
            repo_root / "scripts" / "baseline" / "qwen3_vl.sh",
            repo_root / "scripts" / "llava_ov.sh",
            repo_root / "scripts" / "baseline" / "llava_ov.sh",
        ]

        for path in paths:
            text = path.read_text(encoding="utf-8")
            self.assertIn('"videomme"', text, f"{path} should target videomme")
            self.assertIn('"longvideobench_val_v"', text, f"{path} should target longvideobench_val_v")

    def test_internvl_video_scripts_exist_and_use_expected_models(self):
        repo_root = Path(__file__).resolve().parents[4]
        baseline_script = repo_root / "scripts" / "baseline" / "internvl3_5_video.sh"
        ours_script = repo_root / "scripts" / "ours_v3" / "internvl3_5_8b_video.sh"

        self.assertTrue(baseline_script.exists(), "InternVL3.5 baseline video script should exist")
        self.assertTrue(ours_script.exists(), "InternVL3.5 optimized video script should exist")

        baseline_text = baseline_script.read_text(encoding="utf-8")
        ours_text = ours_script.read_text(encoding="utf-8")

        self.assertIn("--model internvl3_5", baseline_text)
        self.assertIn("modality=video", baseline_text)
        self.assertIn('"videomme"', baseline_text)
        self.assertIn('"longvideobench_val_v"', baseline_text)

        self.assertIn("--model internvl3_5_ours_v3", ours_text)
        self.assertIn('"videomme"', ours_text)
        self.assertIn('"longvideobench_val_v"', ours_text)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest F:/FlashVID/lmms-eval/test/eval/benchmark/test_video_benchmark_scripts.py -v
```

Expected: `FAIL` because the InternVL scripts do not exist yet and some current scripts do not contain both requested tasks.

- [ ] **Step 3: Write minimal implementation**

Do not modify scripts yet. Only create the test file exactly as written above.

- [ ] **Step 4: Run test to verify it fails**

Run:

```bash
python -m pytest F:/FlashVID/lmms-eval/test/eval/benchmark/test_video_benchmark_scripts.py -v
```

Expected: `FAIL` with missing-file or missing-string assertions.

- [ ] **Step 5: Commit**

```bash
git add F:/FlashVID/lmms-eval/test/eval/benchmark/test_video_benchmark_scripts.py
git commit -m "test: add video benchmark script coverage"
```

### Task 3: Implement the fixed-path video benchmark downloader

**Files:**
- Create: `F:/FlashVID/tools/download_video_benchmarks.py`
- Test: `F:/FlashVID/lmms-eval/test/eval/benchmark/test_video_benchmark_setup.py`

- [ ] **Step 1: Write the failing test**

Use the Task 1 test file without modification. That test is already the failing contract for this downloader.

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest F:/FlashVID/lmms-eval/test/eval/benchmark/test_video_benchmark_setup.py -v
```

Expected: `FAIL` until the downloader exists and contains the required roots and dataset identifiers.

- [ ] **Step 3: Write minimal implementation**

Create `F:/FlashVID/tools/download_video_benchmarks.py` with this structure:

```python
import argparse
import shutil
import zipfile
from pathlib import Path

from datasets import DatasetDict, load_dataset
from huggingface_hub import snapshot_download


VIDEOMME_ROOT = Path("/root/autodl-fs/videomme")
VIDEOMME_HF = VIDEOMME_ROOT / "videomme_hf"
VIDEOMME_CACHE = VIDEOMME_ROOT / "videomme"
VIDEOMME_DATA = VIDEOMME_CACHE / "data"
VIDEOMME_SUBTITLE = VIDEOMME_CACHE / "subtitle"
VIDEOMME_REPO = "lmms-lab/Video-MME"

LONGVIDEOBENCH_ROOT = Path("/root/autodl-fs/longvideobench")
LONGVIDEOBENCH_HF = LONGVIDEOBENCH_ROOT / "longvideobench_hf"
LONGVIDEOBENCH_VIDEOS = LONGVIDEOBENCH_ROOT / "videos"
LONGVIDEOBENCH_SUBTITLES = LONGVIDEOBENCH_ROOT / "subtitles"
LONGVIDEOBENCH_REPO = "longvideobench/LongVideoBench"


def _prepare_videomme(force_download: bool) -> None:
    source_dir = Path(
        snapshot_download(
            repo_id=VIDEOMME_REPO,
            repo_type="dataset",
            local_dir=str(VIDEOMME_ROOT),
            force_download=force_download,
        )
    )
    VIDEOMME_DATA.mkdir(parents=True, exist_ok=True)
    VIDEOMME_SUBTITLE.mkdir(parents=True, exist_ok=True)

    for zip_path in source_dir.glob("videos_chunked_*.zip"):
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(VIDEOMME_DATA)

    parquet_path = source_dir / "videomme" / "test-00000-of-00001.parquet"
    dataset = load_dataset("parquet", data_files=str(parquet_path))
    DatasetDict({"test": dataset["train"]}).save_to_disk(str(VIDEOMME_HF))

    if not any(VIDEOMME_DATA.rglob("*.mp4")):
        raise RuntimeError("VideoMME preparation finished without any extracted mp4 files")


def _prepare_longvideobench(force_download: bool) -> None:
    source_dir = Path(
        snapshot_download(
            repo_id=LONGVIDEOBENCH_REPO,
            repo_type="dataset",
            local_dir=str(LONGVIDEOBENCH_ROOT),
            force_download=force_download,
        )
    )
    dataset = load_dataset(LONGVIDEOBENCH_REPO)
    dataset.save_to_disk(str(LONGVIDEOBENCH_HF))

    for video_dir in source_dir.rglob("videos"):
        if video_dir.is_dir():
            shutil.copytree(video_dir, LONGVIDEOBENCH_VIDEOS, dirs_exist_ok=True)
    for subtitle_dir in source_dir.rglob("subtitles"):
        if subtitle_dir.is_dir():
            shutil.copytree(subtitle_dir, LONGVIDEOBENCH_SUBTITLES, dirs_exist_ok=True)

    if not LONGVIDEOBENCH_VIDEOS.exists() or not any(LONGVIDEOBENCH_VIDEOS.rglob("*")):
        raise RuntimeError("LongVideoBench preparation finished without any local video assets")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", nargs="+", default=["videomme", "longvideobench"])
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    selected = {name.lower() for name in args.benchmarks}
    if "videomme" in selected:
        _prepare_videomme(force_download=args.force_download)
    if "longvideobench" in selected:
        _prepare_longvideobench(force_download=args.force_download)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python -m pytest F:/FlashVID/lmms-eval/test/eval/benchmark/test_video_benchmark_setup.py -v
```

Expected: the downloader-existence assertions pass, while path assertions for task files still fail until Task 4.

- [ ] **Step 5: Commit**

```bash
git add F:/FlashVID/tools/download_video_benchmarks.py F:/FlashVID/lmms-eval/test/eval/benchmark/test_video_benchmark_setup.py
git commit -m "feat: add fixed-path video benchmark downloader"
```

### Task 4: Switch task configs to fixed local roots

**Files:**
- Modify: `F:/FlashVID/lmms-eval/lmms_eval/tasks/videomme/videomme.yaml`
- Modify: `F:/FlashVID/lmms-eval/lmms_eval/tasks/longvideobench/longvideobench_val_v.yaml`
- Modify: `F:/FlashVID/lmms-eval/lmms_eval/tasks/longvideobench/utils.py`
- Test: `F:/FlashVID/lmms-eval/test/eval/benchmark/test_video_benchmark_setup.py`

- [ ] **Step 1: Write the failing test**

Reuse the existing failing assertions in `test_video_benchmark_setup.py`.

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest F:/FlashVID/lmms-eval/test/eval/benchmark/test_video_benchmark_setup.py -v
```

Expected: `FAIL` because the YAML and utils still reference old roots.

- [ ] **Step 3: Write minimal implementation**

Update `videomme.yaml` to:

```yaml
dataset_path: /root/autodl-fs/videomme/videomme_hf
dataset_kwargs:
  load_from_disk: True
  cache_dir: /root/autodl-fs/videomme/videomme
  video: True
```

Update `longvideobench_val_v.yaml` to:

```yaml
dataset_path: /root/autodl-fs/longvideobench/longvideobench_hf
dataset_kwargs:
  load_from_disk: True
  cache_dir: /root/autodl-fs/longvideobench
  video: True
  force_download: False
  local_files_only: True
```

Update `longvideobench/utils.py` to replace the `HF_HOME`-derived root with a fixed-root constant and direct cache-dir usage:

```python
LONGVIDEOBENCH_ROOT = "/root/autodl-fs/longvideobench"


def longvideobench_doc_to_visual_v(doc):
    cache_dir = LONGVIDEOBENCH_ROOT
    video_path = os.path.join(cache_dir, "videos", doc["video_path"])
    return [video_path]
```

Keep the rest of the logic intact. Do not refactor unrelated task helpers.

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python -m pytest F:/FlashVID/lmms-eval/test/eval/benchmark/test_video_benchmark_setup.py -v
```

Expected: `PASS`.

- [ ] **Step 5: Commit**

```bash
git add F:/FlashVID/lmms-eval/lmms_eval/tasks/videomme/videomme.yaml F:/FlashVID/lmms-eval/lmms_eval/tasks/longvideobench/longvideobench_val_v.yaml F:/FlashVID/lmms-eval/lmms_eval/tasks/longvideobench/utils.py
git commit -m "feat: point video benchmarks to fixed autodl paths"
```

### Task 5: Add and update runnable video benchmark scripts

**Files:**
- Modify: `F:/FlashVID/scripts/qwen3_vl.sh`
- Modify: `F:/FlashVID/scripts/baseline/qwen3_vl.sh`
- Modify: `F:/FlashVID/scripts/llava_ov.sh`
- Modify: `F:/FlashVID/scripts/baseline/llava_ov.sh`
- Create: `F:/FlashVID/scripts/baseline/internvl3_5_video.sh`
- Create: `F:/FlashVID/scripts/ours_v3/internvl3_5_8b_video.sh`
- Test: `F:/FlashVID/lmms-eval/test/eval/benchmark/test_video_benchmark_scripts.py`

- [ ] **Step 1: Write the failing test**

Reuse the test file from Task 2 without changes.

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest F:/FlashVID/lmms-eval/test/eval/benchmark/test_video_benchmark_scripts.py -v
```

Expected: `FAIL` because the InternVL video scripts do not exist and current scripts are not yet normalized.

- [ ] **Step 3: Write minimal implementation**

Normalize the existing task arrays so they contain only the requested tasks:

```bash
TASKS=("videomme" "longvideobench_val_v")
```

Create `scripts/baseline/internvl3_5_video.sh` with:

```bash
#!/bin/bash
set -euo pipefail

TASKS=("videomme" "longvideobench_val_v")
PRETRAINED="${PRETRAINED:-OpenGVLab/InternVL3_5-8B}"
NUM_FRAME="${NUM_FRAME:-32}"
MODEL_ARGS="pretrained=$PRETRAINED,modality=video,num_frame=$NUM_FRAME"

for task in "${TASKS[@]}"; do
  accelerate launch \
    --main_process_port 18892 \
    --num_processes "${NUM_PROCESSES:-4}" \
    -m lmms_eval \
    --model internvl3_5 \
    --model_args "$MODEL_ARGS" \
    --tasks "$task" \
    --batch_size "${BATCH_SIZE:-1}" \
    --log_samples \
    --log_samples_suffix "internvl3_5_video" \
    --output_path "${OUTPUT_PATH:-./logs/internvl3_5_video}"
done
```

Create `scripts/ours_v3/internvl3_5_8b_video.sh` with:

```bash
#!/bin/bash
set -euo pipefail

TASKS=("videomme" "longvideobench_val_v")
PRETRAINED="${PRETRAINED:-OpenGVLab/InternVL3_5-8B-HF}"
NUM_FRAMES="${NUM_FRAMES:-32}"
MODEL_ARGS="pretrained=$PRETRAINED,num_frames=$NUM_FRAMES,retention_ratio=${RETENTION_RATIO:-0.10}"

for task in "${TASKS[@]}"; do
  accelerate launch \
    --main_process_port 18893 \
    --num_processes "${NUM_PROCESSES:-4}" \
    -m lmms_eval \
    --model internvl3_5_ours_v3 \
    --model_args "$MODEL_ARGS" \
    --tasks "$task" \
    --batch_size "${BATCH_SIZE:-1}" \
    --log_samples \
    --log_samples_suffix "internvl3_5_ours_v3_video" \
    --output_path "${OUTPUT_PATH:-./logs/internvl3_5_ours_v3_video}"
done
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python -m pytest F:/FlashVID/lmms-eval/test/eval/benchmark/test_video_benchmark_scripts.py -v
```

Expected: `PASS`.

- [ ] **Step 5: Commit**

```bash
git add F:/FlashVID/scripts/qwen3_vl.sh F:/FlashVID/scripts/baseline/qwen3_vl.sh F:/FlashVID/scripts/llava_ov.sh F:/FlashVID/scripts/baseline/llava_ov.sh F:/FlashVID/scripts/baseline/internvl3_5_video.sh F:/FlashVID/scripts/ours_v3/internvl3_5_8b_video.sh F:/FlashVID/lmms-eval/test/eval/benchmark/test_video_benchmark_scripts.py
git commit -m "feat: add fixed-path video benchmark runners"
```

### Task 6: Run targeted verification and smoke checks

**Files:**
- Test: `F:/FlashVID/lmms-eval/test/eval/benchmark/test_video_benchmark_setup.py`
- Test: `F:/FlashVID/lmms-eval/test/eval/benchmark/test_video_benchmark_scripts.py`

- [ ] **Step 1: Write the failing test**

No new test file is needed. This task verifies that the earlier test contracts remain green.

- [ ] **Step 2: Run test to verify current status**

Run:

```bash
python -m pytest F:/FlashVID/lmms-eval/test/eval/benchmark/test_video_benchmark_setup.py F:/FlashVID/lmms-eval/test/eval/benchmark/test_video_benchmark_scripts.py -v
```

Expected: `PASS`.

- [ ] **Step 3: Write minimal implementation**

No new code. Run one lightweight script smoke check to validate CLI composition without launching full benchmark work:

```bash
bash F:/FlashVID/scripts/baseline/internvl3_5_video.sh
```

If the environment is too heavy for a full launch, replace the body run with:

```bash
python -m lmms_eval --tasks videomme --limit 1 --model qwen3_vl --model_args pretrained=Qwen/Qwen3-VL-8B-Instruct,max_num_frames=4 --batch_size 1 --output_path ./logs/smoke
```

- [ ] **Step 4: Run verification to confirm outcome**

Run:

```bash
python -m pytest F:/FlashVID/lmms-eval/test/eval/benchmark/test_video_benchmark_setup.py F:/FlashVID/lmms-eval/test/eval/benchmark/test_video_benchmark_scripts.py -v
```

Expected: `PASS`.

- [ ] **Step 5: Commit**

```bash
git add F:/FlashVID/tools/download_video_benchmarks.py F:/FlashVID/lmms-eval/lmms_eval/tasks/videomme/videomme.yaml F:/FlashVID/lmms-eval/lmms_eval/tasks/longvideobench/longvideobench_val_v.yaml F:/FlashVID/lmms-eval/lmms_eval/tasks/longvideobench/utils.py F:/FlashVID/scripts/qwen3_vl.sh F:/FlashVID/scripts/baseline/qwen3_vl.sh F:/FlashVID/scripts/llava_ov.sh F:/FlashVID/scripts/baseline/llava_ov.sh F:/FlashVID/scripts/baseline/internvl3_5_video.sh F:/FlashVID/scripts/ours_v3/internvl3_5_8b_video.sh F:/FlashVID/lmms-eval/test/eval/benchmark/test_video_benchmark_setup.py F:/FlashVID/lmms-eval/test/eval/benchmark/test_video_benchmark_scripts.py
git commit -m "feat: add fixed-path video benchmark support"
```
