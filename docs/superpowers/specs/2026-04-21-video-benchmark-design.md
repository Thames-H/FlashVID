# Video Benchmark Fixed-Path Design

## Summary

This design standardizes two video benchmarks, `VideoMME` and `LongVideoBench`, onto fixed server paths under `/root/autodl-fs`, and wires three evaluation model lines to use those benchmarks for video inference:

- `qwen3_vl`
- `llava_onevision`
- `internvl3_5`

The implementation intentionally avoids introducing a configurable benchmark root. The path contract is fixed and explicit.

## Goals

- Download `VideoMME` into `/root/autodl-fs/videomme`
- Download `LongVideoBench` into `/root/autodl-fs/longvideobench`
- Make `lmms-eval` resolve both benchmarks from those fixed paths
- Ensure `qwen3_vl`, `llava_onevision`, and `internvl3_5` have runnable video benchmark entrypoints
- Keep the change localized to benchmark setup, task configs, and evaluation scripts

## Non-Goals

- No environment-variable-based benchmark root
- No generic benchmark registry
- No refactor of unrelated image benchmark paths
- No change to benchmark task semantics beyond path resolution and local dataset preparation

## Fixed Filesystem Layout

The server-side benchmark layout is fixed as follows.

### VideoMME

- `/root/autodl-fs/videomme/videomme_hf`
- `/root/autodl-fs/videomme/videomme`
- `/root/autodl-fs/videomme/videomme/data`
- `/root/autodl-fs/videomme/videomme/subtitle`

`videomme_hf` stores a local Hugging Face `save_to_disk` dataset used by `load_from_disk: True`.

`videomme/data` stores extracted video files such as `*.mp4`.

`videomme/subtitle` stores subtitle files when needed by subtitle variants.

### LongVideoBench

- `/root/autodl-fs/longvideobench/longvideobench_hf`
- `/root/autodl-fs/longvideobench/videos`
- `/root/autodl-fs/longvideobench/subtitles`

`longvideobench_hf` stores a local Hugging Face `save_to_disk` dataset used by `load_from_disk: True`.

`videos` and `subtitles` store the local assets referenced by task utils.

## Download Strategy

Add a new helper script:

- `tools/download_video_benchmarks.py`

The script will support downloading either benchmark independently or both together.

### VideoMME download flow

The script uses the existing benchmark source already referenced in the repository:

- dataset id: `lmms-lab/Video-MME`

The script will:

1. Download the dataset metadata and any packaged benchmark assets needed to construct a local evaluation copy.
2. Materialize a local Hugging Face dataset under `/root/autodl-fs/videomme/videomme_hf`.
3. Ensure videos are placed under `/root/autodl-fs/videomme/videomme/data`.
4. Ensure subtitle files, if present, are placed under `/root/autodl-fs/videomme/videomme/subtitle`.
5. Fail fast if metadata exists but no video files are available after preparation.

This keeps `videomme` aligned with the current repository expectation that actual video files live under a sibling cache directory, while removing the old `/workspace/...` hardcoded path.

### LongVideoBench download flow

The script uses the benchmark source already referenced in task YAML:

- dataset id: `longvideobench/LongVideoBench`

The script will:

1. Download the upstream benchmark from Hugging Face.
2. Save a local `DatasetDict` to `/root/autodl-fs/longvideobench/longvideobench_hf`.
3. Place referenced video assets under `/root/autodl-fs/longvideobench/videos`.
4. Place subtitle assets under `/root/autodl-fs/longvideobench/subtitles`.
5. Fail fast if metadata exists but the expected video directory is empty.

## Task Configuration Changes

### VideoMME

Update `lmms-eval/lmms_eval/tasks/videomme/videomme.yaml` so it points to the fixed local paths:

- `dataset_path: /root/autodl-fs/videomme/videomme_hf`
- `dataset_kwargs.load_from_disk: True`
- `dataset_kwargs.cache_dir: /root/autodl-fs/videomme/videomme`

Any related VideoMME variants that inherit or duplicate the same benchmark root should be checked for consistency.

The existing `videomme/utils.py` logic already derives actual video paths from the YAML `cache_dir`, so its behavior can remain mostly intact once the YAML root is corrected.

### LongVideoBench

Update `lmms-eval/lmms_eval/tasks/longvideobench/longvideobench_val_v.yaml` and the sibling LongVideoBench YAMLs to use local `load_from_disk: True` instead of remote HF lookup:

- `dataset_path: /root/autodl-fs/longvideobench/longvideobench_hf`
- `dataset_kwargs.load_from_disk: True`
- `dataset_kwargs.cache_dir: /root/autodl-fs/longvideobench`

Update `lmms-eval/lmms_eval/tasks/longvideobench/utils.py` to stop deriving asset paths from `HF_HOME`. It should instead use the absolute `cache_dir` from the YAML and then resolve:

- `videos/`
- `subtitles/`

This makes `LongVideoBench` consistent with the fixed-path requirement and eliminates hidden dependence on global Hugging Face cache settings.

## Model Integration Design

No new video decoding framework will be introduced. The implementation will reuse the current model wrappers that already support video inputs.

### Qwen3-VL

Use the existing chat-template model:

- model name: `qwen3_vl`
- implementation: `lmms-eval/lmms_eval/models/chat/qwen3_vl.py`

This wrapper already supports video inputs and `max_num_frames`. The required work is script-level: ensure benchmark tasks and fixed-path benchmark setup are aligned.

### LLaVA-OneVision

Use the existing simple model:

- model name: `llava_onevision`
- implementation: `lmms-eval/lmms_eval/models/simple/llava_onevision.py`

This wrapper already supports video evaluation through `max_frames_num`. The required work is to keep the benchmark task list aligned with the fixed local benchmark paths.

### InternVL3.5 baseline

Use the existing simple model for the baseline video path:

- model name: `internvl3_5`
- implementation: `lmms-eval/lmms_eval/models/simple/internvl3_5.py`
- underlying video-capable logic: `lmms-eval/lmms_eval/models/simple/internvl3.py`

The baseline script must pass `modality=video` so the simple wrapper uses its video path.

### InternVL3.5 optimized variant

Keep the existing FETP/FlashVID-style variant on the HF/chat path:

- model name: `internvl3_5_ours_v3`
- implementation: `lmms-eval/lmms_eval/models/chat/internvl3_5_ours_v3.py`

This wrapper already accepts `videos=` input. The work here is to add a video benchmark script that targets the fixed local benchmark layout.

## Script Changes

### Existing scripts to update

- `scripts/qwen3_vl.sh`
- `scripts/baseline/qwen3_vl.sh`
- `scripts/llava_ov.sh`
- `scripts/baseline/llava_ov.sh`

These scripts should explicitly target only the requested video tasks during this change:

- `videomme`
- `longvideobench_val_v`

They should remain benchmark runners only and should not embed dataset download logic.

### New scripts to add

- `scripts/baseline/internvl3_5_video.sh`
- `scripts/ours_v3/internvl3_5_8b_video.sh`

Baseline script expectations:

- uses `--model internvl3_5`
- passes `modality=video`
- uses the appropriate non-HF checkpoint default for InternVL3.5 baseline

Optimized script expectations:

- uses `--model internvl3_5_ours_v3`
- uses the HF-format checkpoint expected by the current chat/FETP implementation

Both scripts should target:

- `videomme`
- `longvideobench_val_v`

## Error Handling

The download helper must fail early with actionable errors.

Required checks:

- target root exists or is creatable
- local `save_to_disk` dataset directory exists after preparation
- video asset directory exists
- video asset directory is not empty

For `VideoMME`, if metadata preparation succeeds but no videos exist under `/root/autodl-fs/videomme/videomme/data`, the script must exit non-zero.

For `LongVideoBench`, if metadata preparation succeeds but no videos exist under `/root/autodl-fs/longvideobench/videos`, the script must exit non-zero.

## Testing Strategy

This work will follow test-first changes.

### New benchmark setup test

Add:

- `lmms-eval/test/eval/benchmark/test_video_benchmark_setup.py`

This test should check:

- `tools/download_video_benchmarks.py` exists
- the script contains `/root/autodl-fs/videomme`
- the script contains `/root/autodl-fs/longvideobench`
- the script references `lmms-lab/Video-MME`
- the script references `longvideobench/LongVideoBench`
- VideoMME task config points to `/root/autodl-fs/videomme`
- LongVideoBench task config points to `/root/autodl-fs/longvideobench`

### New script asset tests

Add tests that assert:

- baseline and optimized InternVL video scripts exist
- Qwen3-VL and LLaVA-OneVision video scripts contain `videomme`
- Qwen3-VL and LLaVA-OneVision video scripts contain `longvideobench_val_v`
- InternVL baseline video script uses `--model internvl3_5`
- InternVL optimized video script uses `--model internvl3_5_ours_v3`

These tests should follow the repository's current AST/string-constant style used in benchmark and asset tests.

## Implementation Order

1. Add failing benchmark setup tests.
2. Add failing script asset tests.
3. Add `tools/download_video_benchmarks.py`.
4. Update `videomme` task YAML to fixed paths.
5. Update `longvideobench` task YAML and `utils.py` to fixed paths.
6. Update existing Qwen3-VL and LLaVA-OneVision video scripts.
7. Add InternVL baseline and optimized video scripts.
8. Run targeted tests.
9. Run a minimal dry-run benchmark command with `--limit` if local environment permits.

## Risks

- `VideoMME` asset packaging may not be identical to the current ad hoc server layout, so the downloader must explicitly verify the final local directory structure.
- `LongVideoBench` may include metadata that references asset paths needing normalization when copied into `/root/autodl-fs/longvideobench`.
- `internvl3_5` baseline and `internvl3_5_ours_v3` do not use the same checkpoint format, so scripts must keep their defaults clearly separated.

## Decision Record

- Use fixed absolute benchmark paths instead of a configurable root.
- Reuse existing video-capable model wrappers instead of introducing another abstraction layer.
- Normalize both benchmarks onto local `load_from_disk: True` datasets to reduce runtime dependence on remote HF resolution.
