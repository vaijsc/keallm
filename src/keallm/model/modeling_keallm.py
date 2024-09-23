from typing import Any, Optional, Tuple, Union

from torch import nn
import torch
from transformers import AutoModelForTextEncoding, AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from .configuration_keallm import KeallmConfig

@dataclass
class KeallmForConditionalGenerationModelOutput(ModelOutput):
    loss: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    kge_outputs: Optional[torch.FloatTensor] = None
    language_model_outputs: Optional[Tuple[torch.FloatTensor]] = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k]
            if k not in ["kge_outputs", "language_model_outputs"]
            else getattr(self, k).to_tuple()
            for k in self.keys()
        )


class KeallmPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = KeallmConfig
    base_model_prefix = "keallm"
    supports_gradient_checkpointing = True
    _no_split_modules = [
        # "InstructBlipQFormerEmbeddings",
        # "InstructBlipAttention",
        # "InstructBlipQFormerMultiHeadAttention",
        # "InstructBlipQFormerSelfOutput",
    ]
    _keep_in_fp32_modules = []

    # Copied from transformers.models.blip_2.modeling_blip_2.Blip2PreTrainedModel._init_weights with Blip2->InstructBlip
    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_range
        if isinstance(module, nn.Embedding) or isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=factor)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

class KeallmForConditionalGeneration(KeallmPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        
        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.kge_config.hidden_size))
        self.language_projection = nn.Linear(config.kge_config.hidden_size, config.text_config.hidden_size)
        self.language_projection.bias.data.zero_()
        
        language_model = AutoModelForCausalLM.from_pretrained(
            config.text_config._name_or_path, attn_implementation=config._attn_implementation
        )
        kg_embedding_model = AutoModelForTextEncoding.from_pretrained(config.kge_config._name_or_path)
        
        if kg_embedding_model._no_split_modules is not None:
            self._no_split_modules.extend(kg_embedding_model._no_split_modules)

        if kg_embedding_model._keep_in_fp32_modules is not None:
            self._keep_in_fp32_modules.extend(kg_embedding_model._keep_in_fp32_modules)
        
        if language_model._no_split_modules is not None:
            self._no_split_modules.extend(language_model._no_split_modules)

        if language_model._keep_in_fp32_modules is not None:
            self._keep_in_fp32_modules.extend(language_model._keep_in_fp32_modules)

        self.kg_embedding_model = kg_embedding_model
        self.language_model = language_model
        # Initialize weights and apply final processing
        # self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()
    
    def get_decoder(self):
        return self.language_model.get_decoder()
    
    def set_decoder(self, decoder):
        self.lanuage_model.set_decoder(decoder)

    def _preprocess_accelerate(self):
        r"""
        Some pre-processing hacks to make the model `accelerate` compatible. Check
        https://github.com/huggingface/transformers/pull/21707 for more details.
        """
        hf_device_map = self.hf_device_map

        if len(hf_device_map) > 1 and "language_model" not in hf_device_map and torch.cuda.device_count() > 1:
            # warn users about unexpected behavior when using multi-GPU + InstructBLIP + `accelerate`.
            logger.warning(
                "The `language_model` is not in the `hf_device_map` dictionary and you are running your script"
                " in a multi-GPU environment. this may lead to unexpected behavior when using `accelerate`."
                " Please pass a `device_map` that contains `language_model` to remove this warning."
                " Please refer to https://github.com/huggingface/blog/blob/main/accelerate-large-models.md for"
                " more details on creating a `device_map` for large models.",
            )

        if hasattr(self.language_model, "_hf_hook"):
            self.language_model._hf_hook.io_same_device = True  # For `generate` compatibility

    # def forward(
    #     self,
    #     pixel_values: torch.FloatTensor,
    #     qformer_input_ids: torch.FloatTensor,
    #     qformer_attention_mask: Optional[torch.LongTensor] = None,
    #     input_ids: Optional[torch.FloatTensor] = None,
    #     attention_mask: Optional[torch.LongTensor] = None,
    #     decoder_input_ids: Optional[torch.LongTensor] = None,
    #     decoder_attention_mask: Optional[torch.LongTensor] = None,
    #     output_attentions: Optional[bool] = None,
    #     output_hidden_states: Optional[bool] = None,
    #     labels: Optional[torch.LongTensor] = None,
    #     return_dict: Optional[bool] = None,
    #     interpolate_pos_encoding: bool = False,
    # ) -> Union[Tuple, KeallmForConditionalGenerationModelOutput]:
    
    def forward(
        self,
        kge_input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        qformer_input_ids: torch.FloatTensor,
        qformer_attention_mask: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
    ) -> Union[Tuple, KeallmForConditionalGenerationModelOutput]:
    
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size)
        
        kge_w_embeds = self.kg_embedding_model.get_input_embeddings()(kge_input_ids)
        # kge_w_embeds = vision_outputs[0]
        # difference with BLIP-2 here: we also feed the instruction prompt to the Q-Former
        query_tokens = self.query_tokens.expand(kge_w_embeds.shape[0], -1, -1)
        query_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=kge_embeds.device)

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        kge_attention_mask = torch.ones(kge_w_embeds.size()[:-1], dtype=torch.long, device=kge_embeds.device)

        
        query_outputs = self.kg_embedding_model(
            inputs_embeds=kge_w_embeds,
            attention_mask=kge_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # query_outputs = self.qformer(
        #     input_ids=qformer_input_ids,
        #     attention_mask=qformer_attention_mask,
        #     query_embeds=query_tokens,
        #     encoder_hidden_states=image_embeds,
        #     encoder_attention_mask=image_attention_mask,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )
        query_output = query_outputs[0][:, : query_tokens.size(1), :]

        # step 3: use the language model, conditioned on the query outputs and the prompt
        language_model_inputs = self.language_projection(query_output)
        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )

        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # if the model already has "image_token_index" then the input is expanded to account for image embeds
        # otherwise we expand manually by concatenating
        
        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)
        attention_mask = torch.cat(
            [language_model_attention_mask, attention_mask.to(language_model_attention_mask.device)], dim=1
        )
        
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits if return_dict else outputs[0]
        loss = None
        # we compute the loss here since we need to take into account the sequence length of the query embeds
        if labels is not None:
            labels = labels.to(logits.device)
            logits = logits[:, -labels.size(1) :, :]
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(logits.device)

            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction="mean")

            loss = loss_fct(shift_logits.view(-1, self.config.text_config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits, vision_outputs, query_outputs, outputs)
            return ((loss,) + output) if loss is not None else output

        return KeallmForConditionalGenerationModelOutput(
            loss=loss,
            logits=logits,
            kge_outputs=query_outputs,
            language_model_outputs=outputs,
        )
    
    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.FloatTensor,
        qformer_input_ids: Optional[torch.LongTensor] = None,
        qformer_attention_mask: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        interpolate_pos_encoding: bool = False,
        **generate_kwargs,
    ) -> torch.LongTensor:
        
        if hasattr(self, "hf_device_map"):
            # preprocess for `accelerate`
            self._preprocess_accelerate()

        batch_size = input_ids.shape[0]
        kge_w_embeds = self.kg_embedding_model.get_input_embeddings()(kge_input_ids)
        # kge_w_embeds = vision_outputs[0]
        # difference with BLIP-2 here: we also feed the instruction prompt to the Q-Former
        query_tokens = self.query_tokens.expand(kge_w_embeds.shape[0], -1, -1)
        query_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=kge_embeds.device)

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        kge_attention_mask = torch.ones(kge_w_embeds.size()[:-1], dtype=torch.long, device=kge_embeds.device)

        
        query_outputs = self.kg_embedding_model(
            inputs_embeds=kge_w_embeds,
            attention_mask=kge_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        language_model_inputs = self.language_projection(query_output)
        language_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )

        if input_ids is None:
            input_ids = (
                torch.LongTensor([[self.config.text_config.bos_token_id]])
                .repeat(batch_size, 1)
                .to(image_embeds.device)
            )
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        inputs_embeds = self.get_input_embeddings()(input_ids)

        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)
        attention_mask = torch.cat(
            [language_attention_mask, attention_mask.to(language_attention_mask.device)], dim=1
        )

        generate_kwargs["max_length"] = (
            generate_kwargs.get("max_length", 20) + language_model_inputs.shape[1] - 1
        )
        generate_kwargs["min_length"] = generate_kwargs.get("min_length", 0) + language_model_inputs.shape[1]

        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

        # this is a temporary workaround to be consistent with other generation models and
        # have BOS as the first token, even though under the hood we are calling LM with embeds
        bos_token_id = (
            2
            if self.config.text_config.architectures[0] == "LlamaForCausalLM"
            else self.config.text_config.bos_token_id
        )
        bos_tokens = torch.LongTensor([[bos_token_id]]).repeat(batch_size, 1).to(image_embeds.device)
        if not isinstance(outputs, torch.Tensor):
            outputs.sequences = torch.cat([bos_tokens, outputs.sequences], dim=-1)
        else:
            outputs = torch.cat([bos_tokens, outputs], dim=-1)

        return outputs