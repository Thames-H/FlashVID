"""Deterministic stratified sampling helpers for quick benchmark smoke tests."""

from __future__ import annotations

import math
import os
import random
from collections import defaultdict
from typing import Iterable, List, Sequence

import datasets
from loguru import logger as eval_logger


def _get_sample_target(total: int, default_size: int) -> int:
    ratio_raw = os.getenv("FETP_SAMPLE_RATIO", "").strip()
    size_raw = os.getenv("FETP_SAMPLE_SIZE", "").strip()

    if ratio_raw:
        ratio = max(0.0, min(float(ratio_raw), 1.0))
        return max(1, min(total, int(math.ceil(total * ratio))))

    if size_raw:
        return max(1, min(total, int(size_raw)))

    return max(1, min(total, int(default_size)))


def _resolve_key_fields(
    dataset: datasets.Dataset,
    preferred_key_fields: Sequence[str],
) -> List[str]:
    available = set(dataset.column_names)
    return [field for field in preferred_key_fields if field in available]


def _group_key(doc: dict, key_fields: Sequence[str]):
    if not key_fields:
        return ("__all__",)
    return tuple(str(doc.get(field, "__missing__")) for field in key_fields)


def _allocate_proportional_counts(
    groups: dict,
    target_size: int,
    rng: random.Random,
) -> dict:
    total = sum(len(indices) for indices in groups.values())
    if target_size >= total:
        return {key: len(indices) for key, indices in groups.items()}

    allocations = {}
    remainders = []
    for key, indices in groups.items():
        exact = target_size * len(indices) / total
        count = int(math.floor(exact))
        if count == 0 and target_size >= len(groups):
            count = 1
        count = min(count, len(indices))
        allocations[key] = count
        remainders.append((exact - math.floor(exact), rng.random(), key))

    current = sum(allocations.values())
    if current > target_size:
        removable = [
            (allocations[key], rng.random(), key)
            for key in groups
            if allocations[key] > 0
        ]
        removable.sort(reverse=True)
        for _, _, key in removable:
            if current <= target_size:
                break
            allocations[key] -= 1
            current -= 1
    elif current < target_size:
        remainders.sort(reverse=True)
        while current < target_size:
            changed = False
            for _, _, key in remainders:
                if current >= target_size:
                    break
                if allocations[key] < len(groups[key]):
                    allocations[key] += 1
                    current += 1
                    changed = True
            if not changed:
                break

    return allocations


def stratified_sample_dataset(
    dataset: datasets.Dataset,
    *,
    preferred_key_fields: Sequence[str],
    default_size: int = 96,
    seed_env: str = "FETP_SAMPLE_SEED",
) -> datasets.Dataset:
    """Return a deterministic stratified proportional sample of a dataset."""
    total = len(dataset)
    if total == 0:
        return dataset

    target_size = _get_sample_target(total, default_size)
    if target_size >= total:
        return dataset

    seed = int(os.getenv(seed_env, "42"))
    rng = random.Random(seed)
    key_fields = _resolve_key_fields(dataset, preferred_key_fields)

    groups = defaultdict(list)
    for idx, doc in enumerate(dataset):
        groups[_group_key(doc, key_fields)].append(idx)

    allocations = _allocate_proportional_counts(groups, target_size, rng)
    sampled_indices = []
    for key in sorted(groups, key=lambda item: repr(item)):
        group_indices = list(groups[key])
        rng.shuffle(group_indices)
        sampled_indices.extend(group_indices[: allocations[key]])

    rng.shuffle(sampled_indices)
    sampled_indices = sampled_indices[:target_size]
    sampled_indices.sort()

    eval_logger.info(
        "Using stratified sample: "
        f"{len(sampled_indices)}/{total} docs, "
        f"key_fields={key_fields or ['__all__']}, "
        f"groups={len(groups)}, seed={seed}"
    )
    return dataset.select(sampled_indices)
