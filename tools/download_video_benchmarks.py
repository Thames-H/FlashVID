import argparse
import os
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path
from urllib.parse import quote

from datasets import DatasetDict, load_dataset


MODELSCOPE_DATASET_API_ROOT = "https://www.modelscope.cn/api/v1/datasets/"

VIDEOMME_ROOT = Path("/root/autodl-fs/videomme")
VIDEOMME_SOURCE = VIDEOMME_ROOT / "source"
VIDEOMME_HF = VIDEOMME_ROOT / "videomme_hf"
VIDEOMME_CACHE = VIDEOMME_ROOT / "videomme"
VIDEOMME_DATA = VIDEOMME_CACHE / "data"
VIDEOMME_SUBTITLE = VIDEOMME_CACHE / "subtitle"
VIDEOMME_REPO = "AI-ModelScope/Video-MME"
VIDEOMME_PARQUET_FILES = ("videomme/test-00000-of-00001.parquet",)
VIDEOMME_ARCHIVE_FILES = (
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
)
VIDEOMME_EXTRA_ARCHIVE_FILES = ("subtitle.zip",)

LONGVIDEOBENCH_ROOT = Path("/root/autodl-fs/longvideobench")
LONGVIDEOBENCH_SOURCE = LONGVIDEOBENCH_ROOT / "source"
LONGVIDEOBENCH_HF = LONGVIDEOBENCH_ROOT / "longvideobench_hf"
LONGVIDEOBENCH_VIDEOS = LONGVIDEOBENCH_ROOT / "videos"
LONGVIDEOBENCH_SUBTITLES = LONGVIDEOBENCH_ROOT / "subtitles"
LONGVIDEOBENCH_REPO = "AI-ModelScope/LongVideoBench"
LONGVIDEOBENCH_PARQUET_FILES = (
    "validation-00000-of-00001.parquet",
    "test-00000-of-00001.parquet",
)
LONGVIDEOBENCH_METADATA_FILES = (
    "lvb_val.json",
    "lvb_test_wo_gt.json",
)
LONGVIDEOBENCH_VIDEO_ARCHIVE_FILES = (
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
)
LONGVIDEOBENCH_EXTRA_ARCHIVE_FILES = ("subtitles.tar",)


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


def _require_wget() -> None:
    if shutil.which("wget") is None:
        raise RuntimeError("wget is required to download video benchmark archives")


def _build_modelscope_file_url(dataset_id: str, file_path: str) -> str:
    quoted_path = quote(file_path, safe="/")
    return f"{MODELSCOPE_DATASET_API_ROOT}{dataset_id}/repo?Revision=master&FilePath={quoted_path}"


def _download_with_wget(url: str, target_path: Path, force_download: bool) -> None:
    if target_path.exists() and not force_download:
        return

    if force_download and target_path.exists():
        target_path.unlink()

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


def _download_modelscope_files(dataset_id: str, files: tuple[str, ...], target_root: Path, force_download: bool) -> list[Path]:
    downloaded_files: list[Path] = []
    for file_path in files:
        url = _build_modelscope_file_url(dataset_id, file_path)
        target_path = target_root / file_path
        print(f"[download] {dataset_id}:{file_path} -> {target_path}")
        _download_with_wget(url, target_path, force_download)
        downloaded_files.append(target_path)
    return downloaded_files


def _extract_zip(archive_path: Path, target_dir: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(target_dir)


def _extract_tar(archive_path: Path, target_dir: Path) -> None:
    with tarfile.open(archive_path) as archive:
        archive.extractall(target_dir)


def _merge_archive_parts(part_paths: tuple[Path, ...], target_path: Path, force_download: bool) -> None:
    if target_path.exists() and not force_download:
        return

    if force_download and target_path.exists():
        target_path.unlink()

    with target_path.open("wb") as merged:
        for part_path in part_paths:
            with part_path.open("rb") as part_file:
                shutil.copyfileobj(part_file, merged)


def _prepare_videomme(force_download: bool) -> None:
    _require_wget()
    _reset_dir(VIDEOMME_ROOT, force_download=False)
    _reset_dir(VIDEOMME_SOURCE, force_download)
    _reset_dir(VIDEOMME_CACHE, force_download=False)
    _replace_dir(VIDEOMME_DATA, force_download)
    _replace_dir(VIDEOMME_SUBTITLE, force_download)

    _download_modelscope_files(VIDEOMME_REPO, VIDEOMME_PARQUET_FILES, VIDEOMME_SOURCE, force_download)
    _download_modelscope_files(VIDEOMME_REPO, VIDEOMME_EXTRA_ARCHIVE_FILES, VIDEOMME_SOURCE, force_download)
    archive_paths = _download_modelscope_files(VIDEOMME_REPO, VIDEOMME_ARCHIVE_FILES, VIDEOMME_SOURCE, force_download)

    for archive_path in archive_paths:
        _extract_zip(archive_path, VIDEOMME_DATA)

    subtitle_archive = VIDEOMME_SOURCE / "subtitle.zip"
    if subtitle_archive.exists():
        _extract_zip(subtitle_archive, VIDEOMME_SUBTITLE)

    parquet_paths = sorted((VIDEOMME_SOURCE / "videomme").glob("*.parquet"))
    if not parquet_paths:
        raise RuntimeError("VideoMME metadata parquet files were not found after download")

    dataset = load_dataset("parquet", data_files={"test": [str(path) for path in parquet_paths]})
    _save_dataset_dict(DatasetDict({"test": dataset["test"]}), VIDEOMME_HF, force_download)

    if not any(VIDEOMME_DATA.rglob("*.mp4")):
        raise RuntimeError("VideoMME preparation finished without any extracted mp4 files")


def _prepare_longvideobench(force_download: bool) -> None:
    _require_wget()
    _reset_dir(LONGVIDEOBENCH_ROOT, force_download=False)
    _reset_dir(LONGVIDEOBENCH_SOURCE, force_download)
    _replace_dir(LONGVIDEOBENCH_VIDEOS, force_download)
    _replace_dir(LONGVIDEOBENCH_SUBTITLES, force_download)

    _download_modelscope_files(LONGVIDEOBENCH_REPO, LONGVIDEOBENCH_PARQUET_FILES, LONGVIDEOBENCH_SOURCE, force_download)
    _download_modelscope_files(LONGVIDEOBENCH_REPO, LONGVIDEOBENCH_METADATA_FILES, LONGVIDEOBENCH_SOURCE, force_download)
    _download_modelscope_files(LONGVIDEOBENCH_REPO, LONGVIDEOBENCH_EXTRA_ARCHIVE_FILES, LONGVIDEOBENCH_SOURCE, force_download)
    part_paths = _download_modelscope_files(LONGVIDEOBENCH_REPO, LONGVIDEOBENCH_VIDEO_ARCHIVE_FILES, LONGVIDEOBENCH_SOURCE, force_download)

    merged_video_archive = LONGVIDEOBENCH_SOURCE / "videos.tar"
    _merge_archive_parts(tuple(part_paths), merged_video_archive, force_download)
    _extract_tar(merged_video_archive, LONGVIDEOBENCH_ROOT)

    subtitles_archive = LONGVIDEOBENCH_SOURCE / "subtitles.tar"
    if subtitles_archive.exists():
        _extract_tar(subtitles_archive, LONGVIDEOBENCH_ROOT)

    validation_parquet = LONGVIDEOBENCH_SOURCE / "validation-00000-of-00001.parquet"
    test_parquet = LONGVIDEOBENCH_SOURCE / "test-00000-of-00001.parquet"
    dataset = load_dataset(
        "parquet",
        data_files={
            "validation": str(validation_parquet),
            "test": str(test_parquet),
        },
    )
    _save_dataset_dict(dataset, LONGVIDEOBENCH_HF, force_download)

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
