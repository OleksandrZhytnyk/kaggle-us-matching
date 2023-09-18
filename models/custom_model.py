import torch
import torch.nn as nn
from transformers import AutoModel


class DeBertav3Regressor(nn.Module):

    def __init__(self, model_path):
        super(DeBertav3Regressor, self).__init__()
        self.model_path = model_path
        self.deberta = AutoModel.from_pretrained(self.model_path, return_dict=True, output_attentions=False,
                                                 output_hidden_states=True)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.Dropout(0.3),
            nn.GELU(),
            nn.Linear(1024, 1),
        )

    def forward(self, input_id, mask, device):
        pooler = self.deberta(input_ids=input_id, attention_mask=mask)
        all_hidden_states = torch.stack(pooler["hidden_states"])
        layer_start = 22
        pooler_text = WeightedLayerPooling(24, layer_start=layer_start, layer_weights=None).to(device)
        weighted_pooling_embeddings = pooler_text(all_hidden_states)
        weighted_pooling_embeddings = weighted_pooling_embeddings[:, 0]
        linear_output = self.classifier(weighted_pooling_embeddings)
        return linear_output


class WeightedLayerPooling(nn.Module):
    def __init__(self, num_hidden_layers, layer_start: int = 15, layer_weights=None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = layer_weights if layer_weights is not None \
            else nn.Parameter(
            torch.tensor([1] * (num_hidden_layers + 1 - layer_start), dtype=torch.float)
        )

    def forward(self, all_hidden_states):
        all_layer_embedding = all_hidden_states[self.layer_start:, :, :, :]
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor * all_layer_embedding).sum(dim=0) / self.layer_weights.sum()
        return weighted_average
