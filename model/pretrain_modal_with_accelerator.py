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
import wandb
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from torchmetrics import Metric
from torchmetrics.functional import pairwise_cosine_similarity
from sklearn.model_selection import train_test_split
import json
# add this
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
import argparse

# ----- Checkpoint ayarları -----
CHECKPOINT_ROOT = "/media/ubuntu/8TB/g3-bbm-project/checkpoints"
os.makedirs(CHECKPOINT_ROOT, exist_ok=True)

INPUT_EMBEDDINGS_PATH = '/home/g3-bbm-project/main_folder/data/MAIN_DATA'
MODEL_FILE = "/home/g3-bbm-project/main_folder/SELFormer"
TOKENIZER_FAST = "/home/g3-bbm-project/main_folder/SELFormer_github/SELFormer/data/RobertaFastTokenizer"

wandb.login()

class Silhouette(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("silhouette_score", default=torch.tensor(0.0))
        self.add_state("count", default=torch.tensor(0))

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        if len(set(target.cpu().numpy())) > 1:
            self.silhouette_score += silhouette_score(preds.cpu().numpy(), target.cpu().numpy())
            self.count += 1

    def compute(self) -> torch.Tensor:
        return self.silhouette_score / self.count

class InterAndIntraClassSimilarity(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("intra_class_similarity", default=torch.tensor(0.0))
        self.add_state("inter_class_similarity", default=torch.tensor(0.0))
        self.add_state("count", default=torch.tensor(0))

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        cos_sim = pairwise_cosine_similarity(preds, zero_diagonal=True).detach().cpu()
        same_label_mask = target.cpu().unsqueeze(1) == target.cpu().unsqueeze(0)
        upper_triangle_mask = torch.triu(torch.ones_like(same_label_mask), diagonal=1)
        intra_class_mask = same_label_mask & upper_triangle_mask
        inter_class_mask = (~same_label_mask) & upper_triangle_mask
        intra_class_sim = cos_sim[intra_class_mask]
        inter_class_sim = cos_sim[inter_class_mask]
        self.intra_class_similarity += intra_class_sim.mean()
        self.inter_class_similarity += inter_class_sim.mean()
        self.count += 1

    def compute(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inter_class_similarity / self.count, self.intra_class_similarity / self.count

class KNNAccuracy(Metric):
    def __init__(self, k: int = 20):
        self.k = k
        super().__init__()
        self.add_state("accuracy", default=torch.tensor(0.0))
        self.add_state("count", default=torch.tensor(0))

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        preds = preds.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        preds = KNeighborsClassifier(self.k, metric="cosine").fit(preds, target).predict(preds)
        self.accuracy += accuracy_score(target, preds)
        self.count += 1

    def compute(self) -> torch.Tensor:
        return self.accuracy / self.count

def save_hf_format(model, tokenizer, config, save_dir):          # ⭐️
    """
    Modelle birlikte tokenizer ve config'i Hugging Face biçiminde kaydeder.
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1) Ağırlıklar
    torch.save(model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))

    # 2) Config
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    # 3) Tokenizer
    tokenizer.save_pretrained(save_dir)
    print(f"Model HF formatında {save_dir} klasörüne kaydedildi.")


def read_selfies_column(csv_path: str,
                        column_name: str = "selfies") -> pd.DataFrame:
    """
    NaN'leri boş string ile doldurur, hepsini str'e çevirir ve 
    reset_index ile orijinal satır sayısını korur.
    """
    df = pd.read_csv(csv_path, usecols=[column_name])
    # 1) NaN → ""
    df[column_name] = df[column_name].fillna("")
    # 2) strip() vs. yalnızca boşlukları uçurup string olarak bırak
    df[column_name] = df[column_name].astype(str).str.strip()
    print(f"Loaded {len(df):,} selfie strings (NaN→\"\").")
    return df

def read_embeddings_from_npy(graph_path, text_path, kg_path):
    graph_embeddings = np.load(graph_path).astype(np.float32)
    text_embeddings = np.load(text_path).astype(np.float32)
    kg_embeddings = np.load(kg_path).astype(np.float32)
    print("Graph Embeddings Shape:", graph_embeddings.shape)
    print("Text Embeddings Shape:", text_embeddings.shape)
    print("KG Embeddings Shape:", kg_embeddings.shape)
    return graph_embeddings, text_embeddings, kg_embeddings

def create_datasets(csv_path, graph_embeddings, text_embeddings, kg_embeddings, tokenizer, max_len, test_size=0.1):
    # Split indices
    indices = list(range(len(csv_path)))
    train_idx, val_idx = train_test_split(indices, test_size=test_size, random_state=42)
    
    # Create subsets of embeddings
    train_graph = graph_embeddings[train_idx]
    train_text = text_embeddings[train_idx]
    train_kg = kg_embeddings[train_idx]
    train_csv = csv_path.iloc[train_idx]
    
    val_graph = graph_embeddings[val_idx]
    val_text = text_embeddings[val_idx]
    val_kg = kg_embeddings[val_idx]
    val_csv = csv_path.iloc[val_idx]
    
    # Create datasets
    train_dataset = CustomDataset(train_csv, train_graph, train_text, train_kg, tokenizer, max_len)
    val_dataset = CustomDataset(val_csv, val_graph, val_text, val_kg, tokenizer, max_len)
    
    return train_dataset, val_dataset    

class CustomDataset(Dataset):
    def __init__(self, csv_path, graph_embeddings, text_embeddings, kg_embeddings, tokenizer, max_len):
        self.data = csv_path
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.graph_embeddings = graph_embeddings
        self.text_embeddings = text_embeddings
        self.kg_embeddings = kg_embeddings
        assert len(self.data) == len(self.graph_embeddings), "CSV satır sayısı ile graph embeddings sayısı eşleşmiyor!"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.values[idx]
        text = row[0]
        encoding = self.tokenizer.encode_plus(
            text, max_length=self.max_len, truncation=True, padding="max_length"
        )
        return {
            "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
            "graph_emb": torch.tensor(self.graph_embeddings[idx], dtype=torch.float),
            "text_emb": torch.tensor(self.text_embeddings[idx], dtype=torch.float),
            "kg_emb": torch.tensor(self.kg_embeddings[idx], dtype=torch.float)
        }


class MultimodalRoberta(nn.Module):
    def __init__(self, config):
        super().__init__()
        config.hidden_size = 768
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        
        # Enhanced projections with hidden layers (more params)
        self.graph_proj = nn.Sequential(
            nn.Linear(512, config.hidden_size * 4),
            nn.LayerNorm(config.hidden_size * 4),  # Expanded hidden size
            nn.ReLU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size * 6),  # Expanded hidden size
            nn.LayerNorm(config.hidden_size * 6),  # Expanded hidden size
            nn.ReLU(),
            nn.Linear(config.hidden_size * 6, config.hidden_size * 6),  # Expanded hidden size
            nn.LayerNorm(config.hidden_size * 6),  # Expanded hidden size
            nn.ReLU(),
            nn.Linear(config.hidden_size * 6, config.hidden_size * 4),  # Expanded hidden size
            nn.LayerNorm(config.hidden_size * 4),  # Expanded hidden size
            nn.ReLU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size)  # Back to original size
        )
        
        self.text_proj = nn.Sequential(
            nn.Linear(768, config.hidden_size * 4),  # Expanded hidden size
            nn.LayerNorm(config.hidden_size * 4),  # Expanded hidden size
            nn.ReLU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size * 6),  # Expanded hidden size
            nn.LayerNorm(config.hidden_size * 6),  # Expanded hidden size
            nn.ReLU(),
            nn.Linear(config.hidden_size * 6, config.hidden_size * 6),  # Expanded hidden size
            nn.LayerNorm(config.hidden_size * 6), # Expanded hidden size
            nn.ReLU(),
            nn.Linear(config.hidden_size * 6, config.hidden_size * 4),  # Expanded hidden size
            nn.LayerNorm(config.hidden_size * 4),  # Expanded hidden size
            nn.ReLU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size)  # Back to original size
        )
        
        self.kg_proj = nn.Sequential(
            nn.Linear(128, config.hidden_size * 4),  # Expanded hidden size
            nn.LayerNorm(config.hidden_size * 4),  # Expanded hidden size
            nn.ReLU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size * 6),  # Expanded hidden size
            nn.LayerNorm(config.hidden_size * 6),  # Expanded hidden size
            nn.ReLU(),
            nn.Linear(config.hidden_size * 6, config.hidden_size * 6),  # Expanded hidden size
            nn.LayerNorm(config.hidden_size * 6),  # Expanded hidden size
            nn.ReLU(),
            nn.Linear(config.hidden_size * 6, config.hidden_size * 4),  # Expanded hidden size
            nn.LayerNorm(config.hidden_size * 4),  # Expanded hidden size
            nn.ReLU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size)  # Back to original size
        )

    def forward(self, input_ids, attention_mask, graph_emb=None, text_emb=None, kg_emb=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        text_roberta_emb = outputs.last_hidden_state[:, 0, :]
        text_roberta_emb = torch.nn.functional.normalize(text_roberta_emb, p=2, dim=1)
        
        # Project each modality through hidden layers
        graph_hidden = self.graph_proj(graph_emb)
        text_hidden = self.text_proj(text_emb)
        kg_hidden = self.kg_proj(kg_emb)
        
        # Concatenate along dim=0 (as per your original code)
        combined = torch.cat([text_roberta_emb, graph_hidden, text_hidden, kg_hidden], dim=0)
        return combined

def validate_model(model, dataloader, device, accelerator):
    model.eval()
    sincere_loss = SINCERELoss()
    silhouette = Silhouette().to(device) 
    inter_intra_sim = InterAndIntraClassSimilarity().to(device) 
    knn_acc = KNNAccuracy(k=5).to(device) 
    
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            graph_emb = batch["graph_emb"].to(device)
            text_emb = batch["text_emb"].to(device)
            kg_emb = batch["kg_emb"].to(device)
            
            batch_size = input_ids.size(0)
            labels = torch.arange(batch_size, dtype=torch.long, device=device).repeat(4)

            with accelerator.autocast():
                logits = model(input_ids, attention_mask, graph_emb, text_emb, kg_emb)
                # loss = sincere_loss(logits, labels)
            
            """logits = model(input_ids, attention_mask, graph_emb, text_emb, kg_emb)
            
            loss = sincere_loss(logits, labels)"""
            loss = sincere_loss(logits, labels)
            total_loss += loss.item()
            
            silhouette.update(logits, labels)
            inter_intra_sim.update(logits, labels)
            knn_acc.update(logits, labels)
    
    avg_loss = total_loss / len(dataloader)
    silhouette_score = silhouette.compute()
    inter_sim, intra_sim = inter_intra_sim.compute()
    knn_accuracy = knn_acc.compute()
    
    metrics = {
        "val_loss": avg_loss,
        "val_silhouette": silhouette_score,
        "val_inter_sim": inter_sim,
        "val_intra_sim": intra_sim,
        "val_knn_acc": knn_accuracy
    }
    
    return metrics

def scalar(t):
    """Return a python float no matter what torchmetrics hands back."""
    if torch.is_tensor(t):
        return t.mean().item()          # 1-element or n-element tensor → float
    return float(t)   

def train_and_save_roberta_model(csv_path, graph_embeddings, text_embeddings, kg_embeddings, model_file, save_path="multimodal_roberta.pt", resume_checkpoint=None):
    wandb.init(project="selformer-mm-training-4GPU", name="MAIN_TRAIN", config={
        "epochs": 1000,
        "batch_size": 40,
        "learning_rate": 2e-5,
        "model": "SELFormer"
    })

    ddp_kwargs   = DistributedDataParallelKwargs(find_unused_parameters=True)  # NEW
    accelerator  = Accelerator(
        kwargs_handlers=[ddp_kwargs],
        mixed_precision="fp16",
    )
    device = accelerator.device

    
    max_len = 512
    batch_size = 40
    num_epochs = 1000
    lr = 2e-5
    
    config = RobertaConfig.from_pretrained(model_file)
    config.output_hidden_states = True
    tokenizer = RobertaTokenizerFast.from_pretrained(TOKENIZER_FAST)
    
    # Create train and validation datasets
    train_dataset, val_dataset = create_datasets(
        csv_path, graph_embeddings, text_embeddings, kg_embeddings, 
        tokenizer, max_len
    )
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = MultimodalRoberta(config=config)
    sincere_loss = SINCERELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    '''device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    print(device)'''

    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )

    if resume_checkpoint:
        accelerator.load_state(resume_checkpoint)
        start_epoch = int(os.path.basename(resume_checkpoint).split("_")[-1])
        print(f"✅ Checkpoint yüklendi: {resume_checkpoint}, start_epoch = {start_epoch}")
    else:
        start_epoch = 0

    model.to(device)
    wandb.watch(model, log="gradients")
    
    # Initialize metrics
    '''train_silhouette = Silhouette()
    train_inter_intra_sim = InterAndIntraClassSimilarity()
    train_knn_acc = KNNAccuracy(k=5)'''

    train_silhouette = Silhouette().to(device) 
    train_inter_intra_sim = InterAndIntraClassSimilarity().to(device) 
    train_knn_acc = KNNAccuracy(k=5).to(device)
    
    best_val_loss = float('inf')

    model.train()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.log({"Trainable Parameters": trainable_params})
    
    for epoch in range(start_epoch, num_epochs):
        total_loss = 0.0
        
        # Training loop
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                graph_emb = batch["graph_emb"]
                text_emb = batch["text_emb"]
                kg_emb = batch["kg_emb"]
                
                batch_size = input_ids.size(0)
                labels = torch.arange(batch_size, dtype=torch.long, device=device).repeat(4)
                
                optimizer.zero_grad()

                with accelerator.autocast():
                    logits = model(input_ids, attention_mask, graph_emb, text_emb, kg_emb)
                """logits = model(input_ids, attention_mask, graph_emb, text_emb, kg_emb)
                loss = sincere_loss(logits, labels)"""
                loss = sincere_loss(logits, labels)
                accelerator.backward(loss)
                optimizer.step()
                
                total_loss += loss.item()
                
                # Update training metrics
                train_silhouette.update(logits, labels)
                train_inter_intra_sim.update(logits, labels)
                train_knn_acc.update(logits, labels)
                wandb.log({"Batch Loss": loss.item()})
        
        # Calculate training metrics
        avg_train_loss = total_loss / len(train_dataloader)
        '''train_silhouette_score = train_silhouette.compute()
        train_inter_sim, train_intra_sim = train_inter_intra_sim.compute()
        train_knn_accuracy = train_knn_acc.compute()'''
        
        # Validation
        '''val_metrics = validate_model(model, val_dataloader, device)'''

        train_silhouette_score = scalar(train_silhouette.compute())
        inter_sim, intra_sim   = train_inter_intra_sim.compute()
        train_inter_sim        = scalar(inter_sim)
        train_intra_sim        = scalar(intra_sim)
        train_knn_accuracy     = scalar(train_knn_acc.compute())
        val_metrics             = validate_model(model, val_dataloader, device, accelerator)
        # convert tensor values in the dict to Python numbers
        for k, v in val_metrics.items():
            if torch.is_tensor(v):
                val_metrics[k] = scalar(val_metrics[k])

        
        # Log everything to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_silhouette": train_silhouette_score,
            "train_inter_sim": train_inter_sim,
            "train_intra_sim": train_intra_sim,
            "train_knn_acc": train_knn_accuracy,
            "val_loss": val_metrics["val_loss"],
            "val_silhouette": val_metrics["val_silhouette"],
            "val_inter_sim": val_metrics["val_inter_sim"],
            "val_intra_sim": val_metrics["val_intra_sim"],
            "val_knn_acc": val_metrics["val_knn_acc"]
        })
        
        # Reset training metrics
        train_silhouette.reset()
        train_inter_intra_sim.reset()
        train_knn_acc.reset()

        if accelerator.is_main_process:
            epoch_ckpt_dir = os.path.join(
                CHECKPOINT_ROOT,
                f"epoch_{epoch+1:03d}"          # ör. epoch_001
            )
            accelerator.save_state(epoch_ckpt_dir)        # <─ tüm state
                                                          # İsterseniz HuggingFace uyumlu format da kaydedin
            save_hf_format(
                accelerator.unwrap_model(model),
                tokenizer,
                config,
                os.path.join(epoch_ckpt_dir, "hf")        # checkpoints/epoch_003/hf/
            )
            wandb.save(os.path.join(epoch_ckpt_dir, "**/*"))
        
        """# Save best model
        if accelerator.is_main_process and val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            save_hf_format(
                accelerator.unwrap_model(model), 
                tokenizer, 
                config, 
                save_path
            )
            wandb.save(os.path.join(save_path, "*"))"""

    wandb.finish()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Devam etmek için checkpoint klasörü yolu (ör. checkpoints/epoch_196)"
    )
    args = parser.parse_args()

    csv_name    = f"{INPUT_EMBEDDINGS_PATH}/SELFIES.csv"
    csv_name    = read_selfies_column(csv_path=csv_name)
    
    kg_path     = f"{INPUT_EMBEDDINGS_PATH}/KNOWLEDGE_GRAPH.npy"
    text_path   = f"{INPUT_EMBEDDINGS_PATH}/TEXT_normalized.npy"
    graph_path  = f"{INPUT_EMBEDDINGS_PATH}/GRAPH_normalized.npy"

    graph_embeddings, text_embeddings, kg_embeddings = read_embeddings_from_npy(graph_path, text_path, kg_path)

    train_and_save_roberta_model(
        csv_path=csv_name,
        graph_embeddings=graph_embeddings,
        text_embeddings=text_embeddings,
        kg_embeddings=kg_embeddings,
        model_file=MODEL_FILE,
        save_path="SELFormer_multimodal_hf",
        resume_checkpoint=args.resume_checkpoint
    )
    print("Tüm işlem tamamlandı!")