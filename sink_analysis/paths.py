from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SinkAnalysisPaths:
    repo_root: Path
    root: Path
    partial_root: Path
    artifact_root: Path
    figure_root: Path
    data_root: Path
    eval_output_root: Path
    report_path: Path
    pipeline_state_path: Path

    @classmethod
    def from_repo_root(cls, repo_root: Path) -> "SinkAnalysisPaths":
        repo_root = repo_root.resolve()
        root = repo_root / "sink_analysis"
        return cls(
            repo_root=repo_root,
            root=root,
            partial_root=root / "artifacts_partial",
            artifact_root=root / "artifacts",
            figure_root=root / "figures",
            data_root=root / "data",
            eval_output_root=root / "lmms_eval_outputs",
            report_path=root / "report.md",
            pipeline_state_path=root / "pipeline_state.json",
        )

    def ensure_output_dirs(self) -> None:
        for path in (
            self.root,
            self.partial_root,
            self.artifact_root,
            self.figure_root,
            self.data_root,
            self.eval_output_root,
        ):
            path.mkdir(parents=True, exist_ok=True)
