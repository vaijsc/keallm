# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING, Any, Dict, Optional, TypedDict
from peft import PeftModel
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForVision2Seq, AutoProcessor, AutoTokenizer

from trl import AutoModelForCausalLMWithValueHead

from ..extras.logging import get_logger
from ..extras.misc import count_parameters, skip_check_imports, try_download_model_from_ms
from .model_utils.misc import register_autoclass
from .model_utils.mod import convert_pretrained_model_to_mod, load_mod_pretrained_model
from .model_utils.unsloth import load_unsloth_pretrained_model
from .model_utils.valuehead import load_valuehead_params
from .model_utils.visual import get_image_seqlen

from .model_arch import KeallmForConditionalGeneration, KeallmConfig
from .prompt_tuning.model import get_pt_model, get_lora_model
from .patcher import patch_config, patch_model, patch_tokenizer, patch_valuehead_model


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer, ProcessorMixin

    from ..hparams import FinetuningArguments, ModelArguments


logger = get_logger(__name__)


class TokenizerModule(TypedDict):
    tokenizer: "PreTrainedTokenizer"
    processor: Optional["ProcessorMixin"]


def _get_init_kwargs(model_args: "ModelArguments") -> Dict[str, Any]:
    r"""
    Gets arguments to load config/tokenizer/model.

    Note: including inplace operation of model_args.
    """
    skip_check_imports()
    model_args.model_name_or_path = try_download_model_from_ms(model_args)
    return {
        # "pretrained_model_or_path": model_args.model_na
        "trust_remote_code": True,
        # "cache_dir": model_args.cache_dir,
        # "revision": model_args.model_revision,
        # "token": model_args.hf_hub_token,
        "device_map": "auto",
        # "torch_dtype": "float32"
    }


def load_tokenizer(model_args: "ModelArguments") -> "TokenizerModule":
    r"""
    Loads pretrained tokenizer.

    Note: including inplace operation of model_args.
    """
    init_kwargs = _get_init_kwargs(model_args)
    config = load_config(model_args)
    
    if config.model_type == "keallm":
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                config.text_config._name_or_path,
                use_fast=model_args.use_fast_tokenizer,
                split_special_tokens=model_args.split_special_tokens,
                padding_side="right",
                **init_kwargs,
            )
        except ValueError:  # try the fast one
            tokenizer = AutoTokenizer.from_pretrained(
                config.text_config._name_or_path,
                use_fast=True,
                padding_side="right",
                **init_kwargs,
            )
        
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_args.model_name_or_path,
                use_fast=model_args.use_fast_tokenizer,
                split_special_tokens=model_args.split_special_tokens,
                padding_side="right",
                **init_kwargs,
            )
        except ValueError:  # try the fast one
            tokenizer = AutoTokenizer.from_pretrained(
                model_args.model_name_or_path,
                use_fast=True,
                padding_side="right",
                **init_kwargs,
            )

    # if model_args.new_special_tokens is not None:
    #     num_added_tokens = tokenizer.add_special_tokens(
    #         dict(additional_special_tokens=model_args.new_special_tokens),
    #         replace_additional_special_tokens=False,
    #     )
    #     logger.info("Add {} to special tokens.".format(",".join(model_args.new_special_tokens)))
    #     if num_added_tokens > 0 and not model_args.resize_vocab:
    #         model_args.resize_vocab = True
    #         logger.warning("New tokens have been added, changed `resize_vocab` to True.")

    patch_tokenizer(tokenizer)

    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/auto/processing_auto.py#L324

    return {"tokenizer": tokenizer}


def load_config(model_args: "ModelArguments") -> "PretrainedConfig":
    r"""
    Loads model config.
    """
    init_kwargs = _get_init_kwargs(model_args)
    return KeallmConfig.from_pretrained(model_args.model_name_or_path, **init_kwargs)


def load_model(
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool = False,
    add_valuehead: bool = False,
) -> "PreTrainedModel":
    r"""
    Loads pretrained model.
    """
    init_kwargs = _get_init_kwargs(model_args)
    config = load_config(model_args)
    patch_config(config, tokenizer, model_args, init_kwargs, is_trainable)

    model = None
    lazy_load = False
    
    if "keallm" in model_args.model_type:
        if model_args.train_from_scratch:
            text_config = AutoConfig.from_pretrained(model_args.language_model_path)
            kge_config = AutoConfig.from_pretrained(model_args.kge_model_path)
            keallm_config = KeallmConfig.from_kge_text_configs(kge_config=kge_config, text_config=text_config)
            keallm_config.num_query_tokens = model_args.num_query_tokens
            keallm_config.hidden_size = text_config.hidden_size
            model = KeallmForConditionalGeneration(keallm_config)
        else:
            model = KeallmForConditionalGeneration.from_pretrained(model_args.model_name_or_path, **init_kwargs)
        if "lora" in model_args.model_type:
            model.language_model = get_lora_model(model_args, finetuning_args, model.language_model)
            if not model_args.train_from_scratch:
                model.from_pretrained(model_args.model_name_or_path, **init_kwargs)
    elif model_args.model_type == "pt":
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, device_map="auto")
        # print(model.active_adapters())
        if not model_args.train_from_scratch:
            model = PeftModel.from_pretrained(model, model_args.language_model_path)
        else:
            model = get_pt_model(model_args, finetuning_args, model)
    elif model_args.model_type == "lorra":
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, device_map="auto")
        # model = get_lora_model(model_args, finetuning_args, model)
        # if training_args.do_predict:
        # model.load_adapter("./save/sft/metaqa/keallm/lora")
        model.load_adapter("./save/sft/fb15k237/keallm/lora/checkpoint-50000")
    elif model_args.model_type == "freeze":
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, device_map="auto")
    else:
        raise ValueError("Not found model type")
    
    if not is_trainable:
        model.requires_grad_(False)
        for param in model.parameters():
            if param.data.dtype == torch.float32 and model_args.compute_dtype != torch.float32:
                param.data = param.data.to(model_args.compute_dtype)

        model.eval()
        setattr(model.generation_config, 'max_new_tokens', 200)
    else:
        # model.query_tokens.requires_grad_(False)
        # model.language_projection.requires_grad_(False)
        # model.language_model.requires_grad_(False)
        # model.kg_embedding_model.requires_grad_(False)
        # for name, param in self.llm_model.named_parameters():
        #     param.requires_grad = False
        model.train()

    trainable_params, all_param = count_parameters(model)
    if is_trainable:
        param_stats = "trainable params: {:,} || all params: {:,} || trainable%: {:.4f}".format(
            trainable_params, all_param, 100 * trainable_params / all_param
        )
    else:
        param_stats = "all params: {:,}".format(all_param)

    logger.info(param_stats)

    if model_args.print_param_status:
        for name, param in model.named_parameters():
            print(
                "name: {}, dtype: {}, device: {}, trainable: {}".format(
                    name, param.dtype, param.device, param.requires_grad
                )
            )

    return model
