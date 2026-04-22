import unittest
from types import SimpleNamespace

import torch

from lmms_eval.models.chat.internvl3_5_ours_v3 import _slice_position_embeddings
from lmms_eval.models.chat.internvl_hf import _build_internvl_processor_kwargs


class InternVLVideoKwargsTest(unittest.TestCase):
    def test_build_processor_kwargs_enables_sampling_and_size_for_video(self):
        config = SimpleNamespace(
            vision_config=SimpleNamespace(image_size=[448, 448])
        )

        images_kwargs, videos_kwargs = _build_internvl_processor_kwargs(
            model_config=config,
            min_patches=1,
            max_patches=4,
            num_frames=8,
            fps=None,
        )

        self.assertEqual(images_kwargs, {"min_patches": 1, "max_patches": 4})
        self.assertEqual(
            videos_kwargs,
            {
                "num_frames": 8,
                "do_sample_frames": True,
                "size": {"height": 448, "width": 448},
            },
        )

    def test_build_processor_kwargs_supports_fps_sampling(self):
        config = SimpleNamespace(
            vision_config=SimpleNamespace(image_size=[448, 448])
        )

        _, videos_kwargs = _build_internvl_processor_kwargs(
            model_config=config,
            min_patches=None,
            max_patches=None,
            num_frames=None,
            fps=2.0,
        )

        self.assertEqual(
            videos_kwargs,
            {
                "fps": 2.0,
                "do_sample_frames": True,
                "size": {"height": 448, "width": 448},
            },
        )

    @unittest.skipUnless(
        torch.cuda.is_available(),
        "CUDA is required to validate cross-device position slicing",
    )
    def test_slice_position_embeddings_moves_indices_to_embedding_device(self):
        cos = torch.randn(1, 12, 8, device="cuda")
        sin = torch.randn(1, 12, 8, device="cuda")
        positions = torch.tensor([1, 3, 7], device="cpu")

        sliced_cos, sliced_sin = _slice_position_embeddings(
            (cos, sin),
            positions,
        )

        self.assertEqual(sliced_cos.device.type, "cuda")
        self.assertEqual(sliced_sin.device.type, "cuda")
        self.assertEqual(tuple(sliced_cos.shape), (1, 3, 8))
        self.assertEqual(tuple(sliced_sin.shape), (1, 3, 8))


if __name__ == "__main__":
    unittest.main()
