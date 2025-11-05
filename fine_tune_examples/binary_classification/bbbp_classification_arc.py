import torch
import torch.nn as nn
from transformers import RobertaModel

class MultimodalClassificationModel(nn.Module):
    def __init__(self, pretrained_model, hidden_size=768, num_labels=1, dropout_prob=0.3):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size * 4),
            nn.Linear(hidden_size * 4, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size // 2, num_labels)
        )

    def forward(self, graph_emb, text_emb, kg_emb, input_ids, attention_mask, labels=None):
        embeddings = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            graph_emb=graph_emb,
            text_emb=text_emb,
            kg_emb=kg_emb
        )
        B = input_ids.size(0)
        combined = torch.cat([
            embeddings[0:B], embeddings[B:2*B], embeddings[2*B:3*B], embeddings[3*B:4*B]
        ], dim=1)
        logits = self.classifier(combined)

        if labels is not None:
            pos_weight = torch.tensor([1.5]).to(logits.device)
            loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(logits.view(-1), labels.float().view(-1))
            return loss, logits
        return None, logits