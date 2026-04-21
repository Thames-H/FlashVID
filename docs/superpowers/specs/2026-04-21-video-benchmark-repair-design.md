# Video Benchmark Repair Design

## Summary

This design adds a standalone repair utility for the fixed-path video benchmarks used by this repository:

- `VideoMME`
- `LongVideoBench`

The new tool does not replace the existing download helper. It focuses on local health checks and targeted repair of partially missing benchmark state after interrupted downloads, incomplete extraction, or damaged local Hugging Face `save_to_disk` datasets.

The tool will be implemented as:

- `tools/repair_video_benchmarks.py`

## Goals

- Check both video benchmarks under their fixed server roots
- Detect missing or damaged source archives and metadata
- Detect missing or damaged extracted assets such as videos and subtitles
- Detect missing or unreadable local HF datasets created with `save_to_disk`
- Repair only the missing or damaged layers instead of forcing a full re-download
- Reuse the same fixed path contract already established in the repository

## Non-Goals

- No refactor of `tools/download_video_benchmarks.py`
- No new configurable benchmark root
- No new remote source abstraction beyond the current fixed ModelScope file lists
- No online validation of remote file manifests during tests
- No attempt to repair unrelated image benchmarks

## Fixed Filesystem Contract

The repair tool must use the same fixed filesystem layout already expected by `lmms-eval`.

### VideoMME

- `/root/autodl-fs/videomme/source`
- `/root/autodl-fs/videomme/videomme`
- `/root/autodl-fs/videomme/videomme/data`
- `/root/autodl-fs/videomme/videomme/subtitle`
- `/root/autodl-fs/videomme/videomme_hf`

### LongVideoBench

- `/root/autodl-fs/longvideobench/source`
- `/root/autodl-fs/longvideobench/videos`
- `/root/autodl-fs/longvideobench/subtitles`
- `/root/autodl-fs/longvideobench/longvideobench_hf`

The repair tool must never move these roots or infer them from environment variables.

## Tool Boundary

The repository will keep two separate tools with clear responsibilities:

- `tools/download_video_benchmarks.py`
  Performs initial benchmark preparation from the fixed remote source definitions.
- `tools/repair_video_benchmarks.py`
  Performs local check-and-repair for incomplete or damaged benchmark state.

The repair tool may reuse the same fixed file lists and reconstruction rules as the downloader, but it should not absorb initial-download responsibilities beyond downloading only the files required to repair local gaps.

## Repair Model

Each benchmark is treated as three independent but ordered layers.

### Layer 1: Source Files

Checks the raw downloaded files required to reconstruct local assets.

Examples:

- `VideoMME`
  - metadata parquet
  - `videos_chunked_*.zip`
  - `subtitle.zip`
- `LongVideoBench`
  - parquet files
  - metadata json files
  - `videos.tar.part.*`
  - `subtitles.tar`

### Layer 2: Extracted Assets

Checks local extracted assets that task utils consume at runtime.

Examples:

- `VideoMME`
  - `videomme/data/*.mp4`
  - `videomme/subtitle/*`
- `LongVideoBench`
  - `videos/*`
  - `subtitles/*`

### Layer 3: Local HF Dataset

Checks local `save_to_disk` datasets used by task YAMLs.

Examples:

- `videomme_hf`
- `longvideobench_hf`

## Repair Order

Repair always proceeds from the lowest layer upward:

1. Repair missing source files
2. Repair missing extracted assets
3. Repair missing or unreadable HF dataset

This order prevents rebuilding downstream layers from incomplete upstream inputs.

## CLI Design

The repair tool keeps a narrow CLI:

```bash
python tools/repair_video_benchmarks.py --benchmarks videomme longvideobench
```

Supported arguments:

- `--benchmarks videomme longvideobench`
  Select one or both benchmarks. Default is both.
- `--check-only`
  Report status only. Do not modify local files.
- `--force-rebuild-hf`
  Rebuild the local HF dataset even if `load_from_disk` succeeds.

Default mode is check-and-repair.

## Benchmark-Specific Behavior

### VideoMME

#### Source checks

The tool validates that these required source files exist:

- `source/videomme/test-00000-of-00001.parquet`
- `source/videos_chunked_01.zip` through `source/videos_chunked_20.zip`
- `source/subtitle.zip`

If any are missing, the tool downloads only the missing files.

#### Extracted asset checks

The tool validates:

- `videomme/data` exists and contains at least one `*.mp4`
- `videomme/subtitle` exists and contains extracted subtitle files when `subtitle.zip` is present

Repair behavior:

- If videos are missing but source zip files exist, re-extract only from local zip archives
- If subtitle files are missing but `subtitle.zip` exists, re-extract only `subtitle.zip`
- If required source archives are also missing, repair source first and then re-extract

#### HF dataset checks

The tool validates:

- `videomme_hf` exists
- `datasets.load_from_disk(videomme_hf)` succeeds

Repair behavior:

- Rebuild from the local parquet file under `source/videomme/`
- Do not re-download if source metadata already exists

### LongVideoBench

#### Source checks

The tool validates that these required source files exist:

- `source/validation-00000-of-00001.parquet`
- `source/test-00000-of-00001.parquet`
- `source/lvb_val.json`
- `source/lvb_test_wo_gt.json`
- `source/subtitles.tar`
- `source/videos.tar.part.aa` through `source/videos.tar.part.be`

If any are missing, the tool downloads only the missing files.

#### Extracted asset checks

The tool validates:

- `videos` exists and is non-empty
- `subtitles` exists and is non-empty

Repair behavior:

- If `videos` is missing and all part files exist, merge parts into `source/videos.tar` and re-extract
- If part files are incomplete, download only the missing parts before merge and extraction
- If `subtitles` is missing and `subtitles.tar` exists, re-extract from local archive only

#### HF dataset checks

The tool validates:

- `longvideobench_hf` exists
- `datasets.load_from_disk(longvideobench_hf)` succeeds

Repair behavior:

- Rebuild from the local validation and test parquet files already under `source`
- Do not re-download if local parquet files are already present

## Implementation Structure

The tool should be organized into small focused units rather than a single large script body.

### Benchmark specification layer

Define a small local spec structure for each benchmark that captures:

- benchmark name
- root paths
- required source files
- extracted asset directories
- HF dataset directory
- benchmark-specific rebuild hooks

This prevents file manifests from being scattered across unrelated helper functions.

### Pure check functions

Use side-effect-free checks that return structured status for:

- source file completeness
- extracted asset completeness
- HF dataset readability

These checks should be reusable by both `--check-only` and repair mode.

### Repair functions

Use separate repair helpers for:

- missing source files
- extracted assets
- HF dataset rebuild

Repair helpers should accept the check results so they only perform the minimum required work.

### CLI orchestration

`main()` should only:

- parse args
- select benchmarks
- run checks
- conditionally run repairs
- print a concise summary
- exit non-zero if required state is still broken after repair

## Output Design

Output should be concise and operator-oriented.

Examples:

- `OK videomme source`
- `REPAIRED videomme extracted videos`
- `REPAIRED longvideobench hf dataset`
- `MISSING longvideobench source file: videos.tar.part.ak`
- `FAILED videomme hf dataset rebuild`

At the end, print one final summary per selected benchmark:

- `OK`
- `CHECK_ONLY_WITH_GAPS`
- `REPAIRED`
- `FAILED`

## Error Handling

The repair tool must fail fast on unrecoverable prerequisites.

Required checks:

- `wget` exists before any attempted source repair
- required root directories are creatable
- local archive extraction errors surface immediately
- HF dataset rebuild failures surface immediately

The tool should not silently ignore:

- unreadable archive files
- incomplete merged tar parts
- failed `load_from_disk`
- empty video directories after a claimed repair

## Testing Strategy

Implementation should follow test-first changes.

### New test file

Add a new benchmark asset test covering the repair script, for example:

- `lmms-eval/test/eval/benchmark/test_video_benchmark_repair.py`

### What the tests should assert

- `tools/repair_video_benchmarks.py` exists
- the script supports `--check-only`
- the script supports `--force-rebuild-hf`
- the script references both fixed benchmark roots
- the script references both benchmark names
- the script includes repair entrypoints for source files, extracted assets, and HF dataset rebuild
- the script includes `load_from_disk` validation logic
- the script includes `wget` repair support
- the script includes `videos_chunked_20.zip` and `videos.tar.part.be` in its fixed source manifests

Tests should remain static and local:

- no network access
- no real archive extraction
- no large-file fixtures

This matches the repository's existing benchmark asset test style.

## Risks

- The current fixed ModelScope file lists may drift from upstream over time. The repair tool therefore intentionally shares the repository's existing hard-coded contract rather than attempting dynamic remote discovery.
- A non-empty directory is only a minimum completeness check. The repair tool will detect obvious corruption and missing state, but it is not intended to verify every individual media file against upstream checksums.
- `load_from_disk` success confirms dataset readability, not semantic correctness of every row. This is acceptable for the intended operational purpose of recovering interrupted local setup.

## Decision Record

- Use a standalone repair script rather than expanding the downloader CLI
- Reuse the fixed-path contract already adopted for video benchmarks
- Repair incrementally by layer instead of forcing full benchmark re-download
- Keep tests static and structural to avoid network-dependent CI
