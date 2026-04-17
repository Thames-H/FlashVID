# MMTok: Multimodal Coverage Maximization for Efficient Inference of VLMs
# Paper: https://arxiv.org/abs/2508.18264

"""Coverage-based vision token subset selection used by bundled Qwen3 MMTok."""

import torch
from loguru import logger as eval_logger

from .semantic_selector import SemanticTokenSelector
from .text_processor import VQATextProcessor


class MMTokCore:
    """Select a vision-token subset that covers both text and vision content."""

    def __init__(
        self,
        target_vision_tokens=64,
        alpha=0.5,
        softmax_tv_temperature=0.02,
        softmax_vv_temperature=0.2,
        device="cuda",
        remove_padding_indices=False,
        **kwargs,
    ):
        self.device = device
        self.target_vision_tokens = target_vision_tokens
        self.alpha = alpha
        self.softmax_tv_temperature = softmax_tv_temperature
        self.softmax_vv_temperature = softmax_vv_temperature
        self.remove_padding_indices = remove_padding_indices
        self.extra_kwargs = kwargs
        self._init_processors()
        eval_logger.info(
            f"[MMTok] target_vision_tokens={self.target_vision_tokens}"
        )

    def _init_processors(self):
        self.token_selector = SemanticTokenSelector(
            target_vision_tokens=self.target_vision_tokens,
            alpha=self.alpha,
        )
        self.text_processor = VQATextProcessor(device=self.device)

    def _encode_text_with_token_pooling(self, text: str):
        enc = self._language_tokenizer(
            text.split(),
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = enc["input_ids"].to(self.device)
        with torch.no_grad():
            tok_emb = self._main_model_embed_tokens(input_ids)[0]
        start_idx = 0
        if (
            input_ids.shape[1] > 1
            and input_ids[0, 0].item() == self._language_tokenizer.bos_token_id
        ):
            start_idx = 1
        return tok_emb[start_idx:]

    def apply_selection(
        self,
        mm_projector_features,
        clip_features,
        images,
        question_text,
        text_embeds=None,
        padding_patch_indices=None,
    ):
        vision_feat = mm_projector_features
        vision_feat_clip = clip_features
        if (
            vision_feat_clip is not None
            and vision_feat_clip.shape[1] - vision_feat.shape[1] == 1
        ):
            vision_feat_clip = vision_feat_clip[:, 1:, :]

        text_for_coverage = self.text_processor.extract_keywords_simple(question_text)
        text_token_embedding = self._encode_text_with_token_pooling(text_for_coverage)

        selected_features, selected_indices = self.select_vision_tokens(
            vision_features=vision_feat,
            vision_features_clip=vision_feat_clip,
            text_token_embedding=text_token_embedding,
            padding_patch_indices_list=padding_patch_indices,
        )
        return selected_features, selected_indices

    def select_vision_tokens(
        self,
        vision_features: torch.Tensor,
        vision_features_clip: torch.Tensor,
        text_token_embedding: torch.Tensor,
        padding_patch_indices_list: list = None,
    ) -> torch.Tensor:
        if vision_features.dim() == 2:
            vision_features = vision_features.unsqueeze(0)
        if vision_features_clip.dim() == 2:
            vision_features_clip = vision_features_clip.unsqueeze(0)
        batch_size, num_tokens, _hidden_dim = vision_features.shape

        if num_tokens <= self.target_vision_tokens:
            return vision_features, [list(range(num_tokens))] * batch_size

        selected_tokens_list = []
        selected_indices_list = []

        for batch_idx in range(batch_size):
            vision_tokens = vision_features[batch_idx]
            vision_tokens_clip = vision_features_clip[batch_idx]
            if padding_patch_indices_list is not None and len(padding_patch_indices_list) > 0:
                padding_patch_indices = padding_patch_indices_list[batch_idx]
            else:
                padding_patch_indices = None

            selected_indices, selected_tokens = self.token_selector.mm_coverage_selection(
                text_token_embedding=text_token_embedding,
                vision_tokens=vision_tokens,
                vision_tokens_clip=vision_tokens_clip,
                tv_temp=self.softmax_tv_temperature,
                vv_temp=self.softmax_vv_temperature,
                padding_patch_indices=padding_patch_indices,
            )
            selected_tokens_list.append(selected_tokens)
            selected_indices_list.append(selected_indices)

        selected_tokens_batch = torch.stack(selected_tokens_list, dim=0)
        return selected_tokens_batch, selected_indices_list

    def apply_selection_preprocess_qwen(
        self,
        image_embeds,
        image_features,
        question_text,
        target_vision_tokens=None,
    ):
        if target_vision_tokens is not None:
            self.token_selector.target_vision_tokens = target_vision_tokens

        text_for_embedding = f"Question: {question_text}"
        if getattr(self, "clean_text", False):
            text_for_embedding = self.text_processor.extract_keywords_simple(
                text_for_embedding
            )

        text_token_embedding = self._encode_text_with_token_pooling(text_for_embedding)
        selected_features, selected_indices = self.select_vision_tokens(
            vision_features=image_embeds,
            vision_features_clip=image_features,
            text_token_embedding=text_token_embedding,
        )
        return selected_indices[0], selected_features[0]
