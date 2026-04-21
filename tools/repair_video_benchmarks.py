import argparse
import os
import subprocess
import tarfile
import zipfile
import shutil
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

try:
    from datasets import DatasetDict, load_dataset, load_from_disk
except ModuleNotFoundError:  # pragma: no cover - optional dependency fallback
    class DatasetDict(dict):
        def save_to_disk(self, path: Path) -> None:
            raise ModuleNotFoundError("datasets is required to save_to_disk") from None

    def load_dataset(*args, **kwargs):
        raise ModuleNotFoundError("datasets is required to load_dataset") from None

    def load_from_disk(path: Path):
        raise ModuleNotFoundError("datasets is required to load_from_disk") from None


MODELSCOPE_DATASET_API_ROOT = "https://www.modelscope.cn/api/v1/datasets/"


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    dataset_id: str
    root: Path
    source_root: Path
    hf_root: Path
    required_source_files: tuple[str, ...]
    required_asset_dirs: tuple[str, ...]
    video_archives: tuple[str, ...]


VIDEOMME_SPEC = BenchmarkSpec(
    name="videomme",
    dataset_id="AI-ModelScope/Video-MME",
    root=Path("/root/autodl-fs/videomme"),
    source_root=Path("/root/autodl-fs/videomme/source"),
    hf_root=Path("/root/autodl-fs/videomme/videomme_hf"),
    required_source_files=(
        "videomme/test-00000-of-00001.parquet",
        "videos_chunked_01.zip",
        "videos_chunked_02.zip",
        "videos_chunked_03.zip",
        "videos_chunked_04.zip",
        "videos_chunked_05.zip",
        "videos_chunked_06.zip",
        "videos_chunked_07.zip",
        "videos_chunked_08.zip",
        "videos_chunked_09.zip",
        "videos_chunked_10.zip",
        "videos_chunked_11.zip",
        "videos_chunked_12.zip",
        "videos_chunked_13.zip",
        "videos_chunked_14.zip",
        "videos_chunked_15.zip",
        "videos_chunked_16.zip",
        "videos_chunked_17.zip",
        "videos_chunked_18.zip",
        "videos_chunked_19.zip",
        "videos_chunked_20.zip",
        "subtitle.zip",
    ),
    required_asset_dirs=(
        "videomme/data",
        "videomme/subtitle",
    ),
    video_archives=(
        "videos_chunked_01.zip",
        "videos_chunked_02.zip",
        "videos_chunked_03.zip",
        "videos_chunked_04.zip",
        "videos_chunked_05.zip",
        "videos_chunked_06.zip",
        "videos_chunked_07.zip",
        "videos_chunked_08.zip",
        "videos_chunked_09.zip",
        "videos_chunked_10.zip",
        "videos_chunked_11.zip",
        "videos_chunked_12.zip",
        "videos_chunked_13.zip",
        "videos_chunked_14.zip",
        "videos_chunked_15.zip",
        "videos_chunked_16.zip",
        "videos_chunked_17.zip",
        "videos_chunked_18.zip",
        "videos_chunked_19.zip",
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
        "test-00000-of-00001.parquet",
        "lvb_val.json",
        "lvb_test_wo_gt.json",
        "videos.tar.part.aa",
        "videos.tar.part.ab",
        "videos.tar.part.ac",
        "videos.tar.part.ad",
        "videos.tar.part.ae",
        "videos.tar.part.af",
        "videos.tar.part.ag",
        "videos.tar.part.ah",
        "videos.tar.part.ai",
        "videos.tar.part.aj",
        "videos.tar.part.ak",
        "videos.tar.part.al",
        "videos.tar.part.am",
        "videos.tar.part.an",
        "videos.tar.part.ao",
        "videos.tar.part.ap",
        "videos.tar.part.aq",
        "videos.tar.part.ar",
        "videos.tar.part.as",
        "videos.tar.part.at",
        "videos.tar.part.au",
        "videos.tar.part.av",
        "videos.tar.part.aw",
        "videos.tar.part.ax",
        "videos.tar.part.ay",
        "videos.tar.part.az",
        "videos.tar.part.ba",
        "videos.tar.part.bb",
        "videos.tar.part.bc",
        "videos.tar.part.bd",
        "videos.tar.part.be",
        "subtitles.tar",
    ),
    required_asset_dirs=(
        "videos",
        "subtitles",
    ),
    video_archives=(
        "videos.tar.part.aa",
        "videos.tar.part.ab",
        "videos.tar.part.ac",
        "videos.tar.part.ad",
        "videos.tar.part.ae",
        "videos.tar.part.af",
        "videos.tar.part.ag",
        "videos.tar.part.ah",
        "videos.tar.part.ai",
        "videos.tar.part.aj",
        "videos.tar.part.ak",
        "videos.tar.part.al",
        "videos.tar.part.am",
        "videos.tar.part.an",
        "videos.tar.part.ao",
        "videos.tar.part.ap",
        "videos.tar.part.aq",
        "videos.tar.part.ar",
        "videos.tar.part.as",
        "videos.tar.part.at",
        "videos.tar.part.au",
        "videos.tar.part.av",
        "videos.tar.part.aw",
        "videos.tar.part.ax",
        "videos.tar.part.ay",
        "videos.tar.part.az",
        "videos.tar.part.ba",
        "videos.tar.part.bb",
        "videos.tar.part.bc",
        "videos.tar.part.bd",
        "videos.tar.part.be",
    ),
)

DEFAULT_BENCHMARKS = {
    spec.name: spec
    for spec in (VIDEOMME_SPEC, LONGVIDEOBENCH_SPEC)
}


def _require_wget() -> None:
    if shutil.which("wget") is None:
        raise RuntimeError("wget is required to repair source files")


def _build_modelscope_file_url(dataset_id: str, file_path: str) -> str:
    quoted_path = quote(file_path, safe="/")
    return f"{MODELSCOPE_DATASET_API_ROOT}{dataset_id}/repo?Revision=master&FilePath={quoted_path}"


def _download_with_wget(url: str, target_path: Path) -> None:
    _require_wget()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "wget",
        "--continue",
        "--tries=3",
        "--timeout=30",
        "-O",
        str(target_path),
        url,
    ]
    token = os.environ.get("MODELSCOPE_API_TOKEN") or os.environ.get("MODELSCOPE_TOKEN")
    if token:
        command.insert(1, f"--header=Authorization: Bearer {token}")
    subprocess.run(command, check=True)


def check_source_files(spec: BenchmarkSpec) -> list[str]:
    missing = []
    for relative_path in spec.required_source_files:
        path = spec.source_root / relative_path
        if not path.exists():
            missing.append(str(path))
    return missing


def check_extracted_assets(spec: BenchmarkSpec) -> list[str]:
    missing = []
    for relative_path in spec.required_asset_dirs:
        path = spec.root / relative_path
        if not path.exists() or not path.is_dir() or not any(path.iterdir()):
            missing.append(str(path))
    return missing


def check_hf_dataset(spec: BenchmarkSpec) -> str | None:
    try:
        load_from_disk(spec.hf_root)
    except ModuleNotFoundError as exc:
        return f"datasets unavailable: {exc}"
    except (FileNotFoundError, PermissionError, OSError) as exc:
        return str(exc)
    return None


def repair_missing_source_files(spec: BenchmarkSpec, missing_files: list[str]) -> None:
    missing_paths = {Path(path) for path in missing_files}
    for relative_path in spec.required_source_files:
        target_path = spec.source_root / relative_path
        if target_path in missing_paths and not target_path.exists():
            url = _build_modelscope_file_url(spec.dataset_id, relative_path)
            _download_with_wget(url, target_path)


def _extract_zip_archive(archive_path: Path, target_dir: Path) -> None:
    if not archive_path.exists():
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(target_dir)


def repair_extracted_assets(spec: BenchmarkSpec) -> None:
    if spec.name == "videomme":
        data_dir = spec.root / "videomme" / "data"
        subtitle_dir = spec.root / "videomme" / "subtitle"
        data_dir.mkdir(parents=True, exist_ok=True)
        subtitle_dir.mkdir(parents=True, exist_ok=True)
        for archive_name in spec.video_archives:
            _extract_zip_archive(spec.source_root / archive_name, data_dir)
        _extract_zip_archive(spec.source_root / "subtitle.zip", subtitle_dir)
        return

    if spec.name == "longvideobench":
        videos_tar = spec.source_root / "videos.tar"
        videos_tar.parent.mkdir(parents=True, exist_ok=True)
        with videos_tar.open("wb") as merged:
            for archive_name in spec.video_archives:
                archive_path = spec.source_root / archive_name
                if archive_path.exists():
                    merged.write(archive_path.read_bytes())
        spec.root.mkdir(parents=True, exist_ok=True)
        if videos_tar.exists():
            with tarfile.open(videos_tar, mode="r:") as archive:
                archive.extractall(spec.root)
        subtitles_tar = spec.source_root / "subtitles.tar"
        if subtitles_tar.exists():
            with tarfile.open(subtitles_tar, mode="r:") as archive:
                archive.extractall(spec.root)
        return

    raise ValueError(f"unsupported benchmark: {spec.name}")


def _load_parquet_dataset(files_by_split: dict[str, list[str]]):
    return load_dataset("parquet", data_files=files_by_split)


def repair_hf_dataset(spec: BenchmarkSpec) -> None:
    temp_hf_root = spec.hf_root.with_name(f"{spec.hf_root.name}.tmp")
    if temp_hf_root.exists():
        shutil.rmtree(temp_hf_root)

    if spec.name == "videomme":
        parquet_files = sorted(
            str(path)
            for path in (spec.source_root / "videomme").glob("*.parquet")
        )
        dataset = _load_parquet_dataset({"test": parquet_files})
        hf_dataset = DatasetDict({"test": dataset["test"]})
        spec.hf_root.parent.mkdir(parents=True, exist_ok=True)
        hf_dataset.save_to_disk(temp_hf_root)
        if spec.hf_root.exists():
            shutil.rmtree(spec.hf_root)
        temp_hf_root.replace(spec.hf_root)
        return

    if spec.name == "longvideobench":
        validation_files = sorted(
            str(path)
            for path in spec.source_root.glob("validation*.parquet")
        )
        test_files = sorted(
            str(path)
            for path in spec.source_root.glob("test*.parquet")
        )
        dataset = _load_parquet_dataset(
            {
                "validation": validation_files,
                "test": test_files,
            }
        )
        spec.hf_root.parent.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(temp_hf_root)
        if spec.hf_root.exists():
            shutil.rmtree(spec.hf_root)
        temp_hf_root.replace(spec.hf_root)
        return

    raise ValueError(f"unsupported benchmark: {spec.name}")


def repair_benchmark(
    spec: BenchmarkSpec,
    *,
    check_only: bool,
    force_rebuild_hf: bool,
) -> str:
    missing_source_files = check_source_files(spec)
    missing_assets = check_extracted_assets(spec)
    hf_error = check_hf_dataset(spec)

    if check_only:
        return "CHECK_ONLY_WITH_GAPS" if (missing_source_files or missing_assets or hf_error) else "OK"

    repaired = False
    if missing_source_files:
        repair_missing_source_files(spec, missing_source_files)
        repaired = True
    if missing_assets:
        repair_extracted_assets(spec)
        repaired = True
    if force_rebuild_hf or hf_error is not None:
        repair_hf_dataset(spec)
        repaired = True

    missing_source_files = check_source_files(spec)
    missing_assets = check_extracted_assets(spec)
    hf_error = check_hf_dataset(spec)
    if missing_source_files or missing_assets or hf_error:
        return "FAILED"
    return "REPAIRED" if repaired else "OK"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair video benchmark assets.")
    parser.add_argument(
        "--benchmarks",
        nargs="*",
        default=sorted(DEFAULT_BENCHMARKS),
        choices=sorted(DEFAULT_BENCHMARKS),
        help="Benchmarks to inspect or repair.",
    )
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--force-rebuild-hf", action="store_true")
    return parser.parse_args()


def _resolve_benchmarks(benchmark_names: list[str]) -> list[BenchmarkSpec]:
    return [DEFAULT_BENCHMARKS[name] for name in benchmark_names]


def _format_check_result(spec: BenchmarkSpec) -> list[str]:
    messages = []
    for missing_path in check_source_files(spec):
        messages.append(f"[{spec.name}] missing source file: {missing_path}")
    for missing_path in check_extracted_assets(spec):
        messages.append(f"[{spec.name}] missing extracted asset: {missing_path}")
    hf_error = check_hf_dataset(spec)
    if hf_error is not None:
        messages.append(f"[{spec.name}] hf dataset check failed: {hf_error}")
    return messages


def main() -> int:
    args = parse_args()
    benchmarks = _resolve_benchmarks(args.benchmarks)

    exit_code = 0
    for spec in benchmarks:
        status = repair_benchmark(
            spec,
            check_only=args.check_only,
            force_rebuild_hf=args.force_rebuild_hf,
        )
        print(f"[{spec.name}] {status}")
        if status in {"FAILED", "CHECK_ONLY_WITH_GAPS"}:
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
