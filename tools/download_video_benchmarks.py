import argparse
import shutil
import zipfile
from pathlib import Path

from datasets import DatasetDict, load_dataset
from huggingface_hub import snapshot_download


VIDEOMME_ROOT = Path("/root/autodl-fs/videomme")
VIDEOMME_SOURCE = VIDEOMME_ROOT / "source"
VIDEOMME_HF = VIDEOMME_ROOT / "videomme_hf"
VIDEOMME_CACHE = VIDEOMME_ROOT / "videomme"
VIDEOMME_DATA = VIDEOMME_CACHE / "data"
VIDEOMME_SUBTITLE = VIDEOMME_CACHE / "subtitle"
VIDEOMME_REPO = "lmms-lab/Video-MME"

LONGVIDEOBENCH_ROOT = Path("/root/autodl-fs/longvideobench")
LONGVIDEOBENCH_SOURCE = LONGVIDEOBENCH_ROOT / "source"
LONGVIDEOBENCH_HF = LONGVIDEOBENCH_ROOT / "longvideobench_hf"
LONGVIDEOBENCH_VIDEOS = LONGVIDEOBENCH_ROOT / "videos"
LONGVIDEOBENCH_SUBTITLES = LONGVIDEOBENCH_ROOT / "subtitles"
LONGVIDEOBENCH_REPO = "longvideobench/LongVideoBench"
LONGVIDEOBENCH_CACHE = LONGVIDEOBENCH_ROOT / ".hf_cache"


def _reset_dir(path: Path, force_download: bool) -> None:
    if path.exists() and force_download:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _replace_dir(path: Path, force_download: bool) -> None:
    if path.exists():
        if force_download:
            shutil.rmtree(path)
        else:
            return
    path.mkdir(parents=True, exist_ok=True)


def _save_dataset_dict(dataset: DatasetDict, target_dir: Path, force_download: bool) -> None:
    if target_dir.exists():
        if force_download:
            shutil.rmtree(target_dir)
        else:
            return
    dataset.save_to_disk(str(target_dir))


def _copy_matching_dirs(source_root: Path, names: tuple[str, ...], target_dir: Path) -> None:
    for name in names:
        for directory in source_root.rglob(name):
            if directory.is_dir():
                shutil.copytree(directory, target_dir, dirs_exist_ok=True)


def _extract_video_archives(source_root: Path, target_dir: Path) -> None:
    archives = sorted(source_root.rglob("videos_chunked_*.zip"))
    for archive in archives:
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target_dir)


def _prepare_videomme(force_download: bool) -> None:
    _reset_dir(VIDEOMME_ROOT, force_download=False)
    _reset_dir(VIDEOMME_SOURCE, force_download)
    _reset_dir(VIDEOMME_CACHE, force_download=False)
    _replace_dir(VIDEOMME_DATA, force_download)
    _replace_dir(VIDEOMME_SUBTITLE, force_download)

    source_dir = Path(
        snapshot_download(
            repo_id=VIDEOMME_REPO,
            repo_type="dataset",
            local_dir=str(VIDEOMME_SOURCE),
            force_download=force_download,
        )
    )

    _extract_video_archives(source_dir, VIDEOMME_DATA)
    _copy_matching_dirs(source_dir, ("subtitle", "subtitles"), VIDEOMME_SUBTITLE)

    parquet_paths = sorted((source_dir / "videomme").glob("*.parquet"))
    if not parquet_paths:
        raise RuntimeError("VideoMME metadata parquet files were not found after download")

    dataset = load_dataset("parquet", data_files={"test": [str(path) for path in parquet_paths]})
    _save_dataset_dict(DatasetDict({"test": dataset["test"]}), VIDEOMME_HF, force_download)

    if not any(VIDEOMME_DATA.rglob("*.mp4")):
        raise RuntimeError("VideoMME preparation finished without any extracted mp4 files")


def _prepare_longvideobench(force_download: bool) -> None:
    _reset_dir(LONGVIDEOBENCH_ROOT, force_download=False)
    _reset_dir(LONGVIDEOBENCH_SOURCE, force_download)
    _reset_dir(LONGVIDEOBENCH_CACHE, force_download)
    _replace_dir(LONGVIDEOBENCH_VIDEOS, force_download)
    _replace_dir(LONGVIDEOBENCH_SUBTITLES, force_download)

    source_dir = Path(
        snapshot_download(
            repo_id=LONGVIDEOBENCH_REPO,
            repo_type="dataset",
            local_dir=str(LONGVIDEOBENCH_SOURCE),
            force_download=force_download,
        )
    )

    dataset = load_dataset(LONGVIDEOBENCH_REPO, cache_dir=str(LONGVIDEOBENCH_CACHE))
    _save_dataset_dict(dataset, LONGVIDEOBENCH_HF, force_download)

    _copy_matching_dirs(source_dir, ("videos",), LONGVIDEOBENCH_VIDEOS)
    _copy_matching_dirs(source_dir, ("subtitle", "subtitles"), LONGVIDEOBENCH_SUBTITLES)

    if not LONGVIDEOBENCH_VIDEOS.exists() or not any(LONGVIDEOBENCH_VIDEOS.rglob("*")):
        raise RuntimeError("LongVideoBench preparation finished without any local video assets")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["videomme", "longvideobench"],
        help="Benchmarks to download. Supported values: videomme, longvideobench.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Remove existing prepared assets and download again.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = {name.lower() for name in args.benchmarks}
    unsupported = selected.difference({"videomme", "longvideobench"})
    if unsupported:
        raise ValueError(f"Unsupported benchmarks requested: {', '.join(sorted(unsupported))}")

    if "videomme" in selected:
        _prepare_videomme(force_download=args.force_download)
    if "longvideobench" in selected:
        _prepare_longvideobench(force_download=args.force_download)


if __name__ == "__main__":
    main()
