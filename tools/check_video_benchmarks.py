import argparse

from repair_video_benchmarks import (
    DEFAULT_BENCHMARKS,
    BenchmarkSpec,
    check_extracted_assets,
    check_hf_dataset,
    check_source_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check video benchmark integrity.")
    parser.add_argument(
        "--benchmarks",
        nargs="*",
        default=sorted(DEFAULT_BENCHMARKS),
        choices=sorted(DEFAULT_BENCHMARKS),
        help="Benchmarks to inspect.",
    )
    return parser.parse_args()


def _check_benchmark(spec: BenchmarkSpec) -> list[str]:
    messages = []

    missing_source_files = check_source_files(spec)
    if missing_source_files:
        messages.append(f"missing source files: {len(missing_source_files)}")
        messages.extend(f"  - {path}" for path in missing_source_files)

    missing_assets = check_extracted_assets(spec)
    if missing_assets:
        messages.append(f"missing extracted assets: {len(missing_assets)}")
        messages.extend(f"  - {path}" for path in missing_assets)

    hf_error = check_hf_dataset(spec)
    if hf_error is not None:
        messages.append(f"hf dataset check failed: {hf_error}")

    return messages


def main() -> int:
    args = parse_args()
    exit_code = 0

    for name in args.benchmarks:
        spec = DEFAULT_BENCHMARKS[name]
        messages = _check_benchmark(spec)
        if not messages:
            print(f"[{spec.name}] OK")
            continue

        print(f"[{spec.name}] GAPS")
        for message in messages:
            print(message)
        exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
