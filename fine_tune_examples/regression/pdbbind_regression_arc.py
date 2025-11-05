import torch
import torch.nn as nn

class MultimodalRegressionModel(nn.Module):
    def __init__(self, backbone, hidden_size=768, dropout=0.2):
        super().__init__()
        self.backbone = backbone
        self.regressor = nn.Sequential(
            nn.LayerNorm(hidden_size * 4),
            nn.Linear(hidden_size * 4, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, graph_emb, text_emb, kg_emb, input_ids, attention_mask, labels=None):
        embeds = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            graph_emb=graph_emb,
            text_emb=text_emb,
            kg_emb=kg_emb
        )
        B = input_ids.size(0)
        combined = torch.cat([embeds[0:B], embeds[B:2*B], embeds[2*B:3*B], embeds[3*B:4*B]], dim=1)
        preds = self.regressor(combined)

        loss = None
        if labels is not None:
            loss = nn.MSELoss()(preds.view(-1), labels.view(-1))
        return preds, combined