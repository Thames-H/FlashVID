# MMTok: Multimodal Coverage Maximization for Efficient Inference of VLMs
# Paper: https://arxiv.org/abs/2508.18264

"""Core coverage-based token selection utilities bundled with FlashVID."""

from .mmtok_core import MMTokCore
from .semantic_selector import SemanticTokenSelector, greedy_merged_jit_kernel
from .text_processor import VQATextProcessor

__all__ = [
    "MMTokCore",
    "SemanticTokenSelector",
    "VQATextProcessor",
    "greedy_merged_jit_kernel",
]
