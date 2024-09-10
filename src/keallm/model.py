from typing import Tuple
import torch
from torch import nn
from transformers import BertModel, BertTokenizer, AutoModelForCausalLM, AutoTokenizer

class KEALLM(nn.Module):
    def __init__(
        self,
        llama_model: AutoModelForCausalLM,
        llama_tokenizer: AutoTokenizer,
        kg_embedding_dim: int,
        llama_embedding_dim: int,
    ):
        super().__init__()
        self.llama_model = llama_model
        self.llama_tokenizer = llama_tokenizer
        self.projector = nn.Linear(kg_embedding_dim, llama_embedding_dim)
        self.llama_tokenizer.add_special_tokens({"kg_token": "<KG>"})
        self.kg_token_id = self.llama_tokenizer.convert_tokens_to_ids("<KG>")
        self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))

    def forward(
        self,
        query_ids: torch.Tensor,
        kg_embedding: torch.Tensor,
        non_padding_positions: torch.Tensor,
        labels: torch.Tensor = None,
    ):
        # Get input embeddings from LLM's embedding layer
        input_embeddings = self.llama_model.model.embed_tokens(query_ids)

        # Select corresponding knowledge graph embeddings
        selected_kg_embedding = self.projector(kg_embedding)

        # Get "<KG>" token embedding
        kg_token_embedding = self.llama_model.model.embed_tokens(torch.tensor([self.kg_token_id]).to(query_ids.device))

        # Concatenate knowledge graph embeddings, "<KG>" token embedding, and input embeddings
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
            inputs_embeds=combined_embeddings
        )

        # Modify final prediction layer to handle combined hidden states
        logits = self.llama_model.lm_head(llama_outputs.last_hidden_state)

        loss = None
        if labels is not None:
            # Shift labels for causal language modeling
            labels = labels[:, 1:].contiguous()
            logits = logits[:, :-1, :].contiguous()
            loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))

        return {"loss": loss, "logits": logits}

class KGembeddingModel(nn.Module):
    def __init__(self, bert_model: BertModel, tokenizer: BertTokenizer):
        super().__init__()
        self.bert_model = bert_model
        self.tokenizer = tokenizer

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.bert_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        return outputs.last_hidden_state[:, 0, :]

    def get_embedding(self, head: str, relation: str) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids = self.tokenizer.encode(
            f"[CLS] {head} [SEP] {relation} [SEP]", add_special_tokens=True
        )
        attention_mask = [1] * len(input_ids)
        input_ids = torch.tensor(input_ids).unsqueeze(0)
        attention_mask = torch.tensor(attention_mask).unsqueeze(0)
        embedding = self(input_ids, attention_mask)
        
        # Calculate non-padding positions within the get_embedding method
        non_padding_positions = (input_ids != self.tokenizer.pad_token_id).nonzero(as_tuple=True)[1]

        # Select the KG embedding based on non-padding positions
        selected_kg_embedding = embedding[:, non_padding_positions, :]

        return selected_kg_embedding, non_padding_positions