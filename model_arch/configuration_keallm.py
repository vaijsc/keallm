"""KeaLLM model configuration"""

import os
from typing import Union

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers.models.auto import CONFIG_MAPPING


logger = logging.get_logger(__name__)


class KeallmConfig(PretrainedConfig):

    model_type = "keallm"

    def __init__(
        self,
        kge_config=None,
        text_config=None,
        num_query_tokens=32,
        image_token_index=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if kge_config is None:
            kge_config = {}
            logger.info("kge_config is None. initializing the KeallmKGEConfig with default values.")
        if text_config is None:
            text_config = {}
            logger.info("text_config is None. Initializing the text config with default values (`LlamaConfig`).")

        kge_model_type = kge_config["model_type"] if "model_type" in kge_config else "bert"
        text_model_type = text_config["model_type"] if "model_type" in text_config else "llama"
        
        self.kge_config = CONFIG_MAPPING[kge_model_type](**kge_config)
        self.text_config = CONFIG_MAPPING[text_model_type](**text_config)

        self.tie_word_embeddings = self.text_config.tie_word_embeddings
        self.num_query_tokens = num_query_tokens
        self.initializer_factor = 1.0
        self.initializer_range = 0.02

    @classmethod
    def from_kge_text_configs(
        cls,
        kge_config: PretrainedConfig,
        text_config: PretrainedConfig,
        **kwargs,
    ):
        return cls(
            kge_config=kge_config.to_dict(),
            text_config=text_config.to_dict(),
            **kwargs,
        )