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
import matplotlib.pyplot as plt


def read_selfies_column(csv_path):
    try:
        df = pd.read_csv(csv_path, header=None)
        df = df.iloc[1:].reset_index(drop=True)  # Remove first row and reset index
        
        print("Updated CSV Shape:", df.shape)
        return df[[5]]
    except FileNotFoundError:
        print(f"Error: The file at path '{csv_path}' was not found.")
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty.")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def read_embeddings_from_npy(graph_path, text_path, kg_path):
    """
    Reads embeddings from .npy files.
    """
    graph_embeddings = np.load(graph_path).astype(np.float32).squeeze(axis=1)
    text_embeddings = np.load(text_path).astype(np.float32)
    kg_embeddings = np.load(kg_path).astype(np.float32)

    print("Graph Embeddings Shape:", graph_embeddings.shape)
    print("Text Embeddings Shape:", text_embeddings.shape)
    print("KG Embeddings Shape:", kg_embeddings.shape)

    return graph_embeddings, text_embeddings, kg_embeddings

######################################
# 3) CustomDataset
######################################
class CustomDataset(Dataset):
    def __init__(self, csv_path, graph_embeddings, text_embeddings, kg_embeddings, tokenizer, max_len):
        self.data = csv_path
        self.tokenizer = tokenizer
        self.max_len = max_len     # TODO: value calculated by tokenizer MUST ADD ın here
        self.graph_embeddings = graph_embeddings
        self.text_embeddings = text_embeddings
        self.kg_embeddings = kg_embeddings
        
        print(len(self.data))
        print(len(self.graph_embeddings))
        assert len(self.data) == len(self.graph_embeddings), \
            "CSV satır sayısı ile graph embeddings sayısı eşleşmiyor!"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.values[idx]
        text = row[0]
        
        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len, # TODO: value calculated by tokenizer MUST ADD ın here
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
    def __init__(self, config):
        super().__init__()
        config.hidden_size = 768
        self.roberta = RobertaModel(config)
        self.graph_proj = nn.Linear(512, config.hidden_size)
        self.text_proj = nn.Linear(768, config.hidden_size)
        self.kg_proj = nn.Linear(64, config.hidden_size)
        self.combined_fc = nn.Linear(config.hidden_size * 4, config.hidden_size)
        # graph, text, kg, selformer -> her biri aynı size matris, her bir molekül için de, 4 adet aynı size matris.

    def forward(self, input_ids, attention_mask, graph_emb=None, text_emb=None, kg_emb=None):
        # Get RoBERTa output
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        text_roberta_emb = outputs.last_hidden_state[:, 0, :]  # [B, hidden_size]


        # TODO: FIX ME
        # If graph_emb is missing (None), fill with random
        if none in graph_emb is None: # replace these none with looking up the array and replace them with randn, since we are giving the 
                                      # embeddings batch by batch, and every modality comes as a array in here, so it must look up the array.
            batch_size = input_ids.size(0)
            # Adjust this shape (512) to match your actual graph embedding dimension
            graph_emb = torch.randn(batch_size, 512, device=input_ids.device, dtype=torch.float)
        graph_hidden = self.graph_proj(graph_emb)

        # If text_emb is missing (None), fill with random
        if text_emb is None:
            batch_size = input_ids.size(0)
            # Adjust this shape (768) to match your actual text embedding dimension
            text_emb = torch.randn(batch_size, 768, device=input_ids.device, dtype=torch.float)
        text_hidden = self.text_proj(text_emb)

        # If kg_emb is missing (None), fill with random
        if kg_emb is None:
            batch_size = input_ids.size(0)
            # Adjust this shape (64) to match your actual KG embedding dimension
            kg_emb = torch.randn(batch_size, 64, device=input_ids.device, dtype=torch.float)
        kg_hidden = self.kg_proj(kg_emb)

        # Concatenate all modalities along dim=0 (to produce 4 * batch_size rows)
        combined = torch.cat([text_roberta_emb, graph_hidden, text_hidden, kg_hidden], dim=0)
        return combined


######################################
# 5) Eğitim Fonksiyonu
######################################
def train_and_save_roberta_model(
    csv_path, graph_embeddings, text_embeddings, kg_embeddings, model_file, # bu model mi kullanılmış kontrol et.
    save_path="multimodal_roberta.pt"
):
    max_len = 32
    batch_size = 20
    num_epochs = 20
    lr = 2e-5

    config = RobertaConfig.from_pretrained(model_file)
    config.output_hidden_states = True
    tokenizer = RobertaTokenizerFast.from_pretrained(model_file)

    dataset = CustomDataset(csv_path, graph_embeddings, text_embeddings, kg_embeddings, tokenizer, max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    embedding_dim = graph_embeddings.shape[1]
    print(embedding_dim)
    model = RobertaModel.from_pretrained(model_file, config=config)

    sincere_loss = SINCERELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Device:", device)

    model.train()

    loss_values = []
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
            labels = torch.arange(batch_size, dtype=torch.long, device=device).repeat(4)  # modalities [0,1,2,3,...,0,1,2,..]

            #print(labels)

            # Forward pass
            logits = model(input_ids, attention_mask, graph_emb, text_emb, kg_emb)
            loss = sincere_loss(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch+1} bitti. Avg Loss={total_loss/len(dataloader):.4f}")
        loss_values.append(total_loss / len(dataloader))
    torch.save(model.state_dict(), save_path)
    print(f"Model kaydedildi: {save_path}")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_values) + 1), loss_values, marker='o', linestyle='-', color='blue')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.xticks(range(1, len(loss_values) + 1))

    # Save the figure to a file (PNG, PDF, JPG, etc.)
    plt.savefig("loss_plot.png", dpi=300, bbox_inches='tight')


######################################
# 6) main
######################################
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", required=True, metavar="<str>", type=str, help="Name of the pretrained model.")
    args = parser.parse_args()

    model_file = args.model_file

    csv_name = "/home/g3bbmproject/main_folder/KG/kg.pt/data_with_selfies.csv"
    csv_name = read_selfies_column(csv_path=csv_name)
    
    kg_path = "/home/g3bbmproject/main_folder/traning/data/6k_kg_embed.npy"
    text_path = "/home/g3bbmproject/main_folder/traning/data/text_embeddings_6k.npy"
    graph_path = "/home/g3bbmproject/main_folder/traning/data/graph_embeddings_6k.npy"

    graph_embeddings, text_embeddings, kg_embeddings = read_embeddings_from_npy(graph_path, text_path, kg_path)

    train_and_save_roberta_model(
        csv_path=csv_name,
        graph_embeddings=graph_embeddings,
        text_embeddings=text_embeddings,
        kg_embeddings=kg_embeddings,
        model_file=model_file,
        save_path="multimodal_roberta.pt"
    )
    print("Tüm işlem tamamlandı!")