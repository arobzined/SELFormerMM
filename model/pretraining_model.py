import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from transformers import RobertaConfig, RobertaTokenizerFast, RobertaModel
from revised_losses import SINCERELoss
import random

######################################
# 1) Dummy CSV dosyası oluşturma (2 örnek - SELFIES formatında)
######################################
def create_dummy_csv(csv_path="dummy_4.csv"):
    """
    Dört örnekten oluşan CSV:
    1) Metin: [C][H][N][O], Label: 0
    2) Metin: [C][C][O][H], Label: 1
    """
    df = pd.DataFrame({
        "text": ["[C][H][N][O]", "[C][C][O][H]"],
        "label": [0, 1]
    })
    df.to_csv(csv_path, index=False, header=False)
    print(f"'{csv_path}' oluşturuldu. İçerik:")
    print(df)

######################################
# 2) Dummy Embeddings (Graph, Text, KG)
######################################
def create_dummy_embeddings():
    """
    2 örnek, her biri 4 boyutlu graph, text ve kg embedding.
    """
    graph_embeddings = np.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.9, 0.8, 0.7, 0.6]
    ], dtype=np.float32)

    text_embeddings = np.array([
        [0.4, 0.3, 0.2, 0.1],
        [0.6, 0.7, 0.8, 0.9]
    ], dtype=np.float32)

    kg_embeddings = np.array([
        [0.5, 0.5, 0.5, 0.5],
        [0.1, 0.1, 0.1, 0.1]
    ], dtype=np.float32)

    print("Graph Embeddings:", graph_embeddings)
    print("Text Embeddings:", text_embeddings)
    print("KG Embeddings:", kg_embeddings)

    return graph_embeddings, text_embeddings, kg_embeddings

######################################
# 3) CustomDataset
######################################
class CustomDataset(Dataset):
    def __init__(self, csv_path, graph_embeddings, text_embeddings, kg_embeddings, tokenizer, max_len):
        self.data = pd.read_csv(csv_path, header=None)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.graph_embeddings = graph_embeddings
        self.text_embeddings = text_embeddings
        self.kg_embeddings = kg_embeddings
        
        assert len(self.data) == len(self.graph_embeddings), \
            "CSV satır sayısı ile graph embeddings sayısı eşleşmiyor!"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.values[idx]
        text = row[0]
        
        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            truncation=True,
            padding="max_length"
        )
        input_ids = torch.tensor(encoding["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(encoding["attention_mask"], dtype=torch.long)
        graph_emb = torch.tensor(self.graph_embeddings[idx], dtype=torch.float)
        text_emb = torch.tensor(self.text_embeddings[idx], dtype=torch.float)
        kg_emb = torch.tensor(self.kg_embeddings[idx], dtype=torch.float)

        # No label here; we dynamically label within the batch
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "graph_emb": graph_emb,
            "text_emb": text_emb,
            "kg_emb": kg_emb
        }


######################################
# 4) Multimodal Model
######################################
class MultimodalRoberta(nn.Module):
    def __init__(self, config, embedding_dim, num_labels=2):
        super().__init__()
        self.roberta = RobertaModel(config)
        self.graph_proj = nn.Linear(embedding_dim, config.hidden_size)
        self.text_proj = nn.Linear(embedding_dim, config.hidden_size)
        self.kg_proj = nn.Linear(embedding_dim, config.hidden_size)
        self.combined_fc = nn.Linear(config.hidden_size * 4, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        # graph, text, kg, selformer -> her biri aynı size matris, her bir molekül için de, 4 adet aynı size matris.

    def forward(self, input_ids, attention_mask, graph_emb, text_emb, kg_emb):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        text_roberta_emb = outputs.last_hidden_state[:, 0, :]  # [B, hidden_size]
        
        # Project graph, text, and KG embeddings
        graph_hidden = self.graph_proj(graph_emb)  # [B, hidden_size]
        text_hidden = self.text_proj(text_emb)    # [B, hidden_size]
        kg_hidden = self.kg_proj(kg_emb)          # [B, hidden_size]
        
        # Concatenate all modalities: [B * modalities, hidden_size]
        combined = torch.cat([text_roberta_emb, graph_hidden, text_hidden, kg_hidden], dim=0)
        return combined

        # batch size içinde, her bir molkülü bir labela sahip oalcak şekilde işlem yaptık.
        # output size -> [batch * number of views, dimention]

######################################
# 5) Eğitim Fonksiyonu
######################################
def train_and_save_roberta_model(
    csv_path, graph_embeddings, text_embeddings, kg_embeddings, tokenizer_name="roberta-base", # bu model mi kullanılmış kontrol et.
    save_path="multimodal_roberta.pt"
):
    max_len = 16
    batch_size = 1
    num_epochs = 2
    lr = 1e-4

    config = RobertaConfig.from_pretrained(tokenizer_name)
    tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_name)

    dataset = CustomDataset(csv_path, graph_embeddings, text_embeddings, kg_embeddings, tokenizer, max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    embedding_dim = graph_embeddings.shape[1]
    model = MultimodalRoberta(config, embedding_dim=embedding_dim, num_labels=2)

    sincere_loss = SINCERELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Device:", device)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            graph_emb = batch["graph_emb"].to(device)
            text_emb = batch["text_emb"].to(device)
            kg_emb = batch["kg_emb"].to(device)
            
            # Dynamically create labels for the batch
            batch_size = input_ids.size(0)
            labels = torch.arange(batch_size, dtype=torch.long, device=device).repeat_interleave(3)  # 3 modalities

            # Forward pass
            logits = model(input_ids, attention_mask, graph_emb, text_emb, kg_emb)
            loss = sincere_loss(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print(f"Epoch {epoch+1}, Step {step+1}, Loss={loss.item():.4f}")
        print(f"Epoch {epoch+1} bitti. Avg Loss={total_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Model kaydedildi: {save_path}")

######################################
# 6) main
######################################
if __name__ == "__main__":
    csv_name = "dummy_4.csv"
    create_dummy_csv(csv_path=csv_name)
    graph_embeddings, text_embeddings, kg_embeddings = create_dummy_embeddings()
    train_and_save_roberta_model(
        csv_path=csv_name,
        graph_embeddings=graph_embeddings,
        text_embeddings=text_embeddings,
        kg_embeddings=kg_embeddings,
        tokenizer_name="roberta-base",
        save_path="multimodal_roberta.pt"
    )
    print("Tüm işlem tamamlandı!")