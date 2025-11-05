import torch
import torch.nn as nn
from transformers import RobertaModel

class MultimodalRoberta(nn.Module):
    def __init__(self, config):
        super().__init__()
        config.hidden_size = 768
        self.roberta = RobertaModel(config)
        
        self.graph_proj = nn.Sequential(
            nn.Linear(512, config.hidden_size * 4),
            nn.LayerNorm(config.hidden_size * 4),
            nn.ReLU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size * 6),
            nn.LayerNorm(config.hidden_size * 6),
            nn.ReLU(),
            nn.Linear(config.hidden_size * 6, config.hidden_size * 6),
            nn.LayerNorm(config.hidden_size * 6),
            nn.ReLU(),
            nn.Linear(config.hidden_size * 6, config.hidden_size * 4),
            nn.LayerNorm(config.hidden_size * 4),
            nn.ReLU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size)
        )
        
        self.text_proj = nn.Sequential(
            nn.Linear(768, config.hidden_size * 4),
            nn.LayerNorm(config.hidden_size * 4),
            nn.ReLU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size * 6),
            nn.LayerNorm(config.hidden_size * 6),
            nn.ReLU(),
            nn.Linear(config.hidden_size * 6, config.hidden_size * 6),
            nn.LayerNorm(config.hidden_size * 6),
            nn.ReLU(),
            nn.Linear(config.hidden_size * 6, config.hidden_size * 4),
            nn.LayerNorm(config.hidden_size * 4),
            nn.ReLU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size)
        )
        
        self.kg_proj = nn.Sequential(
            nn.Linear(128, config.hidden_size * 4),
            nn.LayerNorm(config.hidden_size * 4),
            nn.ReLU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size * 6),
            nn.LayerNorm(config.hidden_size * 6),
            nn.ReLU(),
            nn.Linear(config.hidden_size * 6, config.hidden_size * 6),
            nn.LayerNorm(config.hidden_size * 6),
            nn.ReLU(),
            nn.Linear(config.hidden_size * 6, config.hidden_size * 4),
            nn.LayerNorm(config.hidden_size * 4),
            nn.ReLU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size)
        )

    def forward(self, input_ids, attention_mask, graph_emb=None, text_emb=None, kg_emb=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        text_roberta_emb = outputs.last_hidden_state[:, 0, :]
        text_roberta_emb = torch.nn.functional.normalize(text_roberta_emb, p=2, dim=1)
        
        graph_hidden = self.graph_proj(graph_emb)
        text_hidden = self.text_proj(text_emb)
        kg_hidden = self.kg_proj(kg_emb)
        
        combined = torch.cat([text_roberta_emb, graph_hidden, text_hidden, kg_hidden], dim=0)
        return combined
