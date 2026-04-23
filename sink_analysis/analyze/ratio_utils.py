from __future__ import annotations


def sort_keep_ratios(values) -> list[str]:
    keep_ratios = {str(value) for value in values}
    return sorted(
        keep_ratios,
        key=lambda value: (
            1_000 if value == "full" else int(str(value).rstrip("%")),
            str(value),
        ),
    )
