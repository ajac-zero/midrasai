import os

import torch
from transformers import (
    PaliGemmaForConditionalGeneration,
    PaliGemmaPreTrainedModel,
)


class ColPali(PaliGemmaPreTrainedModel):
    def __init__(self, config):
        super(ColPali, self).__init__(config=config)
        self.model: PaliGemmaForConditionalGeneration = (
            PaliGemmaForConditionalGeneration(config)
        )
        self.dim = 128
        self.custom_text_proj = torch.nn.Linear(
            self.model.config.text_config.hidden_size, self.dim
        )
        self.main_input_name = "doc_input_ids"

    def forward(self, *args, **kwargs):
        """
        Forward pass through Llama and the linear layer for dimensionality reduction

        Args:
        - input_ids (torch.LongTensor): The input tokens tensor.
        - attention_mask (torch.LongTensor): The attention mask tensor.

        Returns:
        - torch.Tensor: Embeddings of shape (batch_size, num_tokens, dim)
        """
        outputs = self.model(*args, output_hidden_states=True, **kwargs)
        last_hidden_states = outputs.hidden_states[
            -1
        ]  # (batch_size, sequence_length, hidden_size)
        proj = self.custom_text_proj(last_hidden_states)
        # normalize l2 norm
        proj = proj / proj.norm(dim=-1, keepdim=True)
        proj = proj * kwargs["attention_mask"].unsqueeze(-1)
        return proj

    @classmethod
    def initialize_from_path(cls, model_path: str):
        model = cls.from_pretrained(
            model_path + "/model",
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            local_files_only=True,
            token=os.getenv("HF_TOKEN"),
        ).eval()
        model.load_adapter(model_path + "/adapter")
        return model

    @classmethod
    def save_model(cls, path: str) -> None:
        model_path = path + "/model"
        adapter_path = path + "/adapter"

        os.makedirs(model_path)
        model = cls.from_pretrained(
            "google/paligemma-3b-mix-448", torch_dtype=torch.bfloat16, device_map="cuda"
        ).eval()
        model.save_pretrained(model_path)

        os.makedirs(adapter_path)
        model.load_adapter("vidore/colpali")
        model.save_pretrained(adapter_path)

    @classmethod
    def initialize(cls):
        model_name = "google/paligemma-3b-mix-448"
        adapter_name = "vidore/colpali"

        model = cls.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="cuda"
        ).eval()
        model.load_adapter(adapter_name)

        return model
