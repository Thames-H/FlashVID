from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import datasets

from lmms_eval.api.task import ConfigurableTask, TaskConfig


class _DummyAccelerator:
    is_main_process = True

    def wait_for_everyone(self):
        return None


class TestVideoLocalLoadFromDisk(TestCase):
    def test_local_video_task_skips_snapshot_download_when_loading_from_disk(self):
        task = ConfigurableTask.__new__(ConfigurableTask)
        task.DATASET_NAME = None
        task._config = TaskConfig(test_split="validation")

        with TemporaryDirectory() as tmp:
            dataset_path = Path(tmp) / "longvideobench_hf"
            dataset_path.mkdir()
            task.DATASET_PATH = str(dataset_path)

            dataset_dict = datasets.DatasetDict(
                {
                    "validation": datasets.Dataset.from_dict(
                        {"question": ["q"], "correct_choice": [0]}
                    )
                }
            )

            dataset_kwargs = {
                "load_from_disk": True,
                "cache_dir": str(Path(tmp) / "missing_video_cache"),
                "video": True,
                "local_files_only": True,
            }

            with patch("lmms_eval.api.task.Accelerator", return_value=_DummyAccelerator()), patch(
                "lmms_eval.api.task.snapshot_download",
                side_effect=AssertionError(
                    "snapshot_download should not be called for local load_from_disk video tasks"
                ),
            ) as snapshot_mock, patch(
                "lmms_eval.api.task.datasets.load_from_disk",
                return_value=dataset_dict,
            ) as load_from_disk_mock:
                ConfigurableTask.download.__wrapped__(task, dataset_kwargs.copy())

            snapshot_mock.assert_not_called()
            load_from_disk_mock.assert_called_once_with(dataset_path=str(dataset_path))
            self.assertEqual(task.dataset["validation"][0]["question"], "q")
