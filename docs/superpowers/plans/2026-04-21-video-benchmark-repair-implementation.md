# Video Benchmark Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone `tools/repair_video_benchmarks.py` utility that checks and repairs missing source files, extracted assets, and local HF datasets for `VideoMME` and `LongVideoBench`.

**Architecture:** Add one standalone repair script that mirrors the repository's fixed benchmark path contract and ModelScope file manifests, plus one new static benchmark asset test file that drives the script shape through TDD. The script will separate pure checks from repair actions so `--check-only` and repair mode share the same validation logic.

**Tech Stack:** Python 3, `datasets.load_from_disk`, `wget`, `tarfile`, `zipfile`, `pathlib`, `pytest`

---

### Task 1: Add Static Repair Script Asset Test

**Files:**
- Create: `F:/FlashVID/lmms-eval/test/eval/benchmark/test_video_benchmark_repair.py`
- Reference: `F:/FlashVID/lmms-eval/test/eval/benchmark/test_video_benchmark_setup.py`
- Reference: `F:/FlashVID/tools/download_video_benchmarks.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path
from unittest import TestCase


class TestVideoBenchmarkRepairScript(TestCase):
    def test_repair_script_exists_with_expected_cli_and_manifests(self):
        repo_root = Path(__file__).resolve().parents[4]
        script_path = repo_root / "tools" / "repair_video_benchmarks.py"

        self.assertTrue(script_path.exists(), "video benchmark repair script should exist")

        text = script_path.read_text(encoding="utf-8")

        self.assertIn("--check-only", text)
        self.assertIn("--force-rebuild-hf", text)
        self.assertIn("/root/autodl-fs/videomme", text)
        self.assertIn("/root/autodl-fs/longvideobench", text)
        self.assertIn("videomme", text)
        self.assertIn("longvideobench", text)
        self.assertIn("load_from_disk", text)
        self.assertIn("wget", text)
        self.assertIn("videos_chunked_20.zip", text)
        self.assertIn("videos.tar.part.be", text)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest F:\FlashVID\lmms-eval\test\eval\benchmark\test_video_benchmark_repair.py -v`

Expected: FAIL with `video benchmark repair script should exist`

- [ ] **Step 3: Write the minimal placeholder script to satisfy the existence check only**

```python
import argparse


VIDEOMME_ROOT = "/root/autodl-fs/videomme"
LONGVIDEOBENCH_ROOT = "/root/autodl-fs/longvideobench"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--force-rebuild-hf", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    parse_args()
```

- [ ] **Step 4: Run test to verify it still fails for missing manifest details**

Run: `python -m pytest F:\FlashVID\lmms-eval\test\eval\benchmark\test_video_benchmark_repair.py -v`

Expected: FAIL on one of the missing string assertions such as `load_from_disk`, `wget`, `videos_chunked_20.zip`, or `videos.tar.part.be`

- [ ] **Step 5: Commit**

```bash
git add lmms-eval/test/eval/benchmark/test_video_benchmark_repair.py tools/repair_video_benchmarks.py
git commit -m "Add failing repair script asset test"
```

### Task 2: Build Script Skeleton With Shared Benchmark Specs and Check Mode

**Files:**
- Modify: `F:/FlashVID/tools/repair_video_benchmarks.py`
- Test: `F:/FlashVID/lmms-eval/test/eval/benchmark/test_video_benchmark_repair.py`

- [ ] **Step 1: Expand the failing test to require structured benchmark specs and status output hooks**

```python
from pathlib import Path
from unittest import TestCase


class TestVideoBenchmarkRepairScript(TestCase):
    def test_repair_script_exists_with_expected_cli_and_manifests(self):
        repo_root = Path(__file__).resolve().parents[4]
        script_path = repo_root / "tools" / "repair_video_benchmarks.py"

        self.assertTrue(script_path.exists(), "video benchmark repair script should exist")

        text = script_path.read_text(encoding="utf-8")

        self.assertIn("--check-only", text)
        self.assertIn("--force-rebuild-hf", text)
        self.assertIn("BenchmarkSpec", text)
        self.assertIn("check_source_files", text)
        self.assertIn("check_extracted_assets", text)
        self.assertIn("check_hf_dataset", text)
        self.assertIn("load_from_disk", text)
        self.assertIn("videos_chunked_20.zip", text)
        self.assertIn("videos.tar.part.be", text)
```

- [ ] **Step 2: Run test to verify it fails on missing structure**

Run: `python -m pytest F:\FlashVID\lmms-eval\test\eval\benchmark\test_video_benchmark_repair.py -v`

Expected: FAIL because `BenchmarkSpec` and check helper names are not present yet

- [ ] **Step 3: Replace the placeholder script with a real check-mode skeleton**

```python
import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

from datasets import load_from_disk


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    dataset_id: str
    root: Path
    source_root: Path
    hf_root: Path
    required_source_files: tuple[str, ...]
    required_asset_dirs: tuple[Path, ...]
    video_archives: tuple[str, ...]


VIDEOMME_SPEC = BenchmarkSpec(
    name="videomme",
    dataset_id="AI-ModelScope/Video-MME",
    root=Path("/root/autodl-fs/videomme"),
    source_root=Path("/root/autodl-fs/videomme/source"),
    hf_root=Path("/root/autodl-fs/videomme/videomme_hf"),
    required_source_files=(
        "videomme/test-00000-of-00001.parquet",
        "videos_chunked_20.zip",
        "subtitle.zip",
    ),
    required_asset_dirs=(
        Path("/root/autodl-fs/videomme/videomme/data"),
        Path("/root/autodl-fs/videomme/videomme/subtitle"),
    ),
    video_archives=(
        "videos_chunked_01.zip",
        "videos_chunked_20.zip",
    ),
)

LONGVIDEOBENCH_SPEC = BenchmarkSpec(
    name="longvideobench",
    dataset_id="AI-ModelScope/LongVideoBench",
    root=Path("/root/autodl-fs/longvideobench"),
    source_root=Path("/root/autodl-fs/longvideobench/source"),
    hf_root=Path("/root/autodl-fs/longvideobench/longvideobench_hf"),
    required_source_files=(
        "validation-00000-of-00001.parquet",
        "videos.tar.part.be",
        "subtitles.tar",
    ),
    required_asset_dirs=(
        Path("/root/autodl-fs/longvideobench/videos"),
        Path("/root/autodl-fs/longvideobench/subtitles"),
    ),
    video_archives=(
        "videos.tar.part.aa",
        "videos.tar.part.be",
    ),
)


def check_source_files(spec: BenchmarkSpec) -> list[str]:
    missing = []
    for relative_path in spec.required_source_files:
        if not (spec.source_root / relative_path).exists():
            missing.append(relative_path)
    return missing


def check_extracted_assets(spec: BenchmarkSpec) -> list[str]:
    missing = []
    for asset_dir in spec.required_asset_dirs:
        if not asset_dir.exists() or not any(asset_dir.rglob("*")):
            missing.append(str(asset_dir))
    return missing


def check_hf_dataset(spec: BenchmarkSpec) -> str | None:
    if not spec.hf_root.exists():
        return str(spec.hf_root)
    try:
        load_from_disk(str(spec.hf_root))
    except Exception:
        return str(spec.hf_root)
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest F:\FlashVID\lmms-eval\test\eval\benchmark\test_video_benchmark_repair.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tools/repair_video_benchmarks.py lmms-eval/test/eval/benchmark/test_video_benchmark_repair.py
git commit -m "Add repair script check-mode skeleton"
```

### Task 3: Implement Source File Repair and Download Hooks

**Files:**
- Modify: `F:/FlashVID/tools/repair_video_benchmarks.py`
- Test: `F:/FlashVID/lmms-eval/test/eval/benchmark/test_video_benchmark_repair.py`
- Reference: `F:/FlashVID/tools/download_video_benchmarks.py`

- [ ] **Step 1: Extend the test to require source-repair hooks and `wget` download helpers**

```python
from pathlib import Path
from unittest import TestCase


class TestVideoBenchmarkRepairScript(TestCase):
    def test_repair_script_exists_with_expected_cli_and_manifests(self):
        repo_root = Path(__file__).resolve().parents[4]
        script_path = repo_root / "tools" / "repair_video_benchmarks.py"

        self.assertTrue(script_path.exists(), "video benchmark repair script should exist")

        text = script_path.read_text(encoding="utf-8")

        self.assertIn("repair_missing_source_files", text)
        self.assertIn("_download_with_wget", text)
        self.assertIn("wget", text)
        self.assertIn("MODELSCOPE_DATASET_API_ROOT", text)
        self.assertIn("AI-ModelScope/Video-MME", text)
        self.assertIn("AI-ModelScope/LongVideoBench", text)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest F:\FlashVID\lmms-eval\test\eval\benchmark\test_video_benchmark_repair.py -v`

Expected: FAIL because the source repair helpers are not implemented yet

- [ ] **Step 3: Add explicit ModelScope manifest constants and source repair helpers**

```python
MODELSCOPE_DATASET_API_ROOT = "https://www.modelscope.cn/api/v1/datasets/"


def _require_wget() -> None:
    if shutil.which("wget") is None:
        raise RuntimeError("wget is required to repair missing video benchmark source files")


def _build_modelscope_file_url(dataset_id: str, file_path: str) -> str:
    quoted_path = quote(file_path, safe="/")
    return f"{MODELSCOPE_DATASET_API_ROOT}{dataset_id}/repo?Revision=master&FilePath={quoted_path}"


def _download_with_wget(url: str, target_path: Path) -> None:
    command = [
        "wget",
        "--continue",
        "--tries=3",
        "--timeout=30",
        "-O",
        str(target_path),
        url,
    ]
    subprocess.run(command, check=True)


def repair_missing_source_files(spec: BenchmarkSpec, missing_files: list[str]) -> None:
    if not missing_files:
        return
    _require_wget()
    spec.source_root.mkdir(parents=True, exist_ok=True)
    for relative_path in missing_files:
        target_path = spec.source_root / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        _download_with_wget(
            _build_modelscope_file_url(spec.dataset_id, relative_path),
            target_path,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest F:\FlashVID\lmms-eval\test\eval\benchmark\test_video_benchmark_repair.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tools/repair_video_benchmarks.py lmms-eval/test/eval/benchmark/test_video_benchmark_repair.py
git commit -m "Add source repair hooks for video benchmarks"
```

### Task 4: Implement Extracted Asset Repair and HF Dataset Rebuild

**Files:**
- Modify: `F:/FlashVID/tools/repair_video_benchmarks.py`
- Test: `F:/FlashVID/lmms-eval/test/eval/benchmark/test_video_benchmark_repair.py`
- Reference: `F:/FlashVID/tools/download_video_benchmarks.py`

- [ ] **Step 1: Extend the test to require extracted-asset repair and HF rebuild hooks**

```python
from pathlib import Path
from unittest import TestCase


class TestVideoBenchmarkRepairScript(TestCase):
    def test_repair_script_exists_with_expected_cli_and_manifests(self):
        repo_root = Path(__file__).resolve().parents[4]
        script_path = repo_root / "tools" / "repair_video_benchmarks.py"

        self.assertTrue(script_path.exists(), "video benchmark repair script should exist")

        text = script_path.read_text(encoding="utf-8")

        self.assertIn("repair_extracted_assets", text)
        self.assertIn("repair_hf_dataset", text)
        self.assertIn("tarfile", text)
        self.assertIn("zipfile", text)
        self.assertIn("load_dataset", text)
        self.assertIn("save_to_disk", text)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest F:\FlashVID\lmms-eval\test\eval\benchmark\test_video_benchmark_repair.py -v`

Expected: FAIL because the asset repair and HF rebuild helpers are not implemented yet

- [ ] **Step 3: Implement benchmark-specific asset repair and HF rebuild helpers**

```python
def repair_extracted_assets(spec: BenchmarkSpec) -> None:
    if spec.name == "videomme":
        data_dir = spec.root / "videomme" / "data"
        subtitle_dir = spec.root / "videomme" / "subtitle"
        data_dir.mkdir(parents=True, exist_ok=True)
        subtitle_dir.mkdir(parents=True, exist_ok=True)
        for archive_name in spec.video_archives:
            archive_path = spec.source_root / archive_name
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(data_dir)
        subtitle_archive = spec.source_root / "subtitle.zip"
        if subtitle_archive.exists():
            with zipfile.ZipFile(subtitle_archive) as archive:
                archive.extractall(subtitle_dir)
        return

    merged_archive = spec.source_root / "videos.tar"
    with merged_archive.open("wb") as merged:
        for part_name in spec.video_archives:
            with (spec.source_root / part_name).open("rb") as part_file:
                shutil.copyfileobj(part_file, merged)
    with tarfile.open(merged_archive) as archive:
        archive.extractall(spec.root)
    subtitles_archive = spec.source_root / "subtitles.tar"
    if subtitles_archive.exists():
        with tarfile.open(subtitles_archive) as archive:
            archive.extractall(spec.root)


def repair_hf_dataset(spec: BenchmarkSpec) -> None:
    if spec.name == "videomme":
        parquet_paths = sorted((spec.source_root / "videomme").glob("*.parquet"))
        dataset = load_dataset("parquet", data_files={"test": [str(path) for path in parquet_paths]})
        DatasetDict({"test": dataset["test"]}).save_to_disk(str(spec.hf_root))
        return

    dataset = load_dataset(
        "parquet",
        data_files={
            "validation": str(spec.source_root / "validation-00000-of-00001.parquet"),
            "test": str(spec.source_root / "test-00000-of-00001.parquet"),
        },
    )
    dataset.save_to_disk(str(spec.hf_root))
```

- [ ] **Step 4: Add orchestration for `--check-only` and `--force-rebuild-hf`**

```python
def repair_benchmark(spec: BenchmarkSpec, *, check_only: bool, force_rebuild_hf: bool) -> str:
    missing_source = check_source_files(spec)
    missing_assets = check_extracted_assets(spec)
    hf_error = check_hf_dataset(spec)

    if check_only:
        return "OK" if not missing_source and not missing_assets and hf_error is None else "CHECK_ONLY_WITH_GAPS"

    if missing_source:
        repair_missing_source_files(spec, missing_source)
    if missing_assets:
        repair_extracted_assets(spec)
    if force_rebuild_hf or hf_error is not None:
        repair_hf_dataset(spec)

    remaining_source = check_source_files(spec)
    remaining_assets = check_extracted_assets(spec)
    remaining_hf = check_hf_dataset(spec)
    if remaining_source or remaining_assets or remaining_hf is not None:
        return "FAILED"
    return "REPAIRED" if missing_source or missing_assets or hf_error is not None or force_rebuild_hf else "OK"
```

- [ ] **Step 5: Run the repair-script test**

Run: `python -m pytest F:\FlashVID\lmms-eval\test\eval\benchmark\test_video_benchmark_repair.py -v`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tools/repair_video_benchmarks.py lmms-eval/test/eval/benchmark/test_video_benchmark_repair.py
git commit -m "Add asset and hf dataset repair flow"
```

### Task 5: Run Regression Checks For Benchmark Utilities

**Files:**
- Verify only: `F:/FlashVID/tools/repair_video_benchmarks.py`
- Verify only: `F:/FlashVID/tools/download_video_benchmarks.py`
- Verify only: `F:/FlashVID/lmms-eval/test/eval/benchmark/test_video_benchmark_setup.py`
- Verify only: `F:/FlashVID/lmms-eval/test/eval/benchmark/test_video_benchmark_scripts.py`
- Verify only: `F:/FlashVID/lmms-eval/test/eval/benchmark/test_video_benchmark_repair.py`
- Verify only: `F:/FlashVID/lmms-eval/test/eval/qwen3_vl/test_local_qwen3_vl_ours_v2_assets.py`

- [ ] **Step 1: Run the new repair-script test suite**

Run: `python -m pytest F:\FlashVID\lmms-eval\test\eval\benchmark\test_video_benchmark_repair.py -v`

Expected: PASS

- [ ] **Step 2: Run the existing benchmark setup and script tests**

Run: `python -m pytest F:\FlashVID\lmms-eval\test\eval\benchmark\test_video_benchmark_setup.py F:\FlashVID\lmms-eval\test\eval\benchmark\test_video_benchmark_scripts.py -v`

Expected: PASS

- [ ] **Step 3: Run the existing qwen3 asset regression**

Run: `python -m pytest F:\FlashVID\lmms-eval\test\eval\qwen3_vl\test_local_qwen3_vl_ours_v2_assets.py -v`

Expected: PASS

- [ ] **Step 4: Run the full targeted regression batch**

Run: `python -m pytest F:\FlashVID\lmms-eval\test\eval\benchmark\test_video_benchmark_setup.py F:\FlashVID\lmms-eval\test\eval\benchmark\test_video_benchmark_scripts.py F:\FlashVID\lmms-eval\test\eval\benchmark\test_video_benchmark_repair.py F:\FlashVID\lmms-eval\test\eval\qwen3_vl\test_local_qwen3_vl_ours_v2_assets.py -v`

Expected: PASS with all selected tests green

- [ ] **Step 5: Commit**

```bash
git add tools/repair_video_benchmarks.py lmms-eval/test/eval/benchmark/test_video_benchmark_repair.py
git commit -m "Finalize video benchmark repair utility"
```

## Self-Review

- Spec coverage:
  - standalone script boundary is implemented by `tools/repair_video_benchmarks.py` in Tasks 1-4
  - three-layer check and repair model is implemented in Tasks 2-4
  - `--check-only` and `--force-rebuild-hf` are covered in Tasks 1, 2, and 4
  - benchmark-specific behavior for `VideoMME` and `LongVideoBench` is covered in Task 4
  - static local-only tests are covered in Tasks 1-5
- Placeholder scan:
  - no placeholder markers or “similar to previous task” shortcuts remain
  - each code-changing step contains concrete code snippets
  - each verification step contains exact commands and expected outcomes
- Type consistency:
  - `BenchmarkSpec`, `check_source_files`, `check_extracted_assets`, `check_hf_dataset`, `repair_missing_source_files`, `repair_extracted_assets`, `repair_hf_dataset`, and `repair_benchmark` are named consistently across tasks
