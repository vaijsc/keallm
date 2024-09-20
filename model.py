import torch
from torch import nn
from transformers import BertModel, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, BertPreTrainedModel

class KEALLM(PreTrainedModel):
    config_class = AutoModelForCausalLM.from_pretrained("facebook/bart-base").config_class # Using BART config
    base_model_prefix = "keallm"

    def __init__(
        self,
        config,
        kg_embedding_dim: int,
    ):
        super().__init__(config)
        self.llama_model = AutoModelForCausalLM.from_pretrained("facebook/bart-base", config=config)
        self.projector = nn.Linear(kg_embedding_dim, config.hidden_size)
        self.kg_token_id = self.llama_model.tokenizer.add_special_tokens({"additional_special_tokens": ["<KG>"]})["additional_special_tokens_ids"][0]

    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        kg_embedding: torch.Tensor = None,
        non_padding_positions: torch.Tensor = None,
        labels: torch.Tensor = None,
        **kwargs,
    ):
        # Get input embeddings from LLM's embedding layer
        input_embeddings = self.llama_model.model.embed_tokens(input_ids)

        # Select corresponding knowledge graph embeddings
        selected_kg_embedding = self.projector(kg_embedding)

        # Get "<KG>" token embedding
        kg_token_embedding = self.llama_model.model.embed_tokens(torch.tensor([self.kg_token_id]).to(input_ids.device))

        # Concatenate KG embeddings, "<KG>" token embedding, and input embeddings
        combined_embeddings = torch.cat(
            (
                selected_kg_embedding,
                kg_token_embedding.expand(selected_kg_embedding.size(0), -1, -1),
                input_embeddings[:, non_padding_positions, :],
            ),
            dim=1,
        )

        # Pass combined embeddings to LLM's decoder layers
        llama_outputs = self.llama_model.model.decoder(
            inputs_embeds=combined_embeddings,
            attention_mask=attention_mask,
        )

        # Modify final prediction layer to handle combined hidden states
        logits = self.llama_model.lm_head(llama_outputs.last_hidden_state)

        loss = None
        if labels is not None:
            loss = self.compute_loss(logits, labels)

        return {"loss": loss, "logits": logits}

    def compute_loss(self, logits, labels):
        loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
        return loss

class KGembeddingModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert_model = BertModel(config)
        self.init_weights() # Initialize weights

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.bert_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        return outputs.last_hidden_state[:, 0, :]

    def get_embedding(self, head: str, relation: str, tokenizer: BertTokenizer) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids = tokenizer.encode(
            f"[CLS] {head} [SEP] {relation} [SEP]", add_special_tokens=True, truncation=True, max_length=512
        )
        attention_mask = [1] * len(input_ids)
        input_ids = torch.tensor(input_ids).unsqueeze(0)
        attention_mask = torch.tensor(attention_mask).unsqueeze(0)
        embedding = self(input_ids, attention_mask)
        
        # Calculate non-padding positions
        non_padding_positions = (input_ids != tokenizer.pad_token_id).nonzero(as_tuple=True)[1]

        # Select the KG embedding 
        selected_kg_embedding = embedding[:, non_padding_positions, :]

        return selected_kg_embedding, non_padding_positions