# -----------------------
# Environment Settings
# -----------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
import seaborn as sns

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaConfig, RobertaTokenizerFast
from safetensors.torch import load_file
import argparse
import evaluate
import wandb
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append("/home/g3-bbm-project/main_folder")
sys.path.append("/home/g3-bbm-project/main_folder/FineTune")

from train.finetuning.prepare_finetuning_data import smiles_to_selfies, train_val_test_split
from multimodal_roberta import MultimodalRoberta

# -----------------------
# Argument Parser
# -----------------------
parser = argparse.ArgumentParser()
parser.add_argument("--use_scaffold", type=int, default=1, help="0: random split, 1: scaffold split")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
args = parser.parse_args()

wandb.init(project="finetune-tasks", name="bbb - pretrain 266 - fine tune 100")

# -----------------------
# Config ve Tokenizer
# -----------------------
model_path = "/media/ubuntu/8TB/g3-bbm-project/checkpoints/epoch_266/hf"
weights_path = "/media/ubuntu/8TB/g3-bbm-project/checkpoints/epoch_266/model.safetensors"
config = RobertaConfig.from_pretrained(model_path)
tokenizer = RobertaTokenizerFast.from_pretrained(model_path)


def plot_umap_embeddings(model, dataset, title_prefix="Fine-tuned", dataset_name="BBBP"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    embeddings_all = []
    labels_all = []

    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            input_ids = sample["input_ids"].unsqueeze(0).to(device)
            attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
            graph_emb = sample["graph_emb"].unsqueeze(0).to(device)
            text_emb = sample["text_emb"].unsqueeze(0).to(device)
            kg_emb = sample["kg_emb"].unsqueeze(0).to(device)

            emb = model.pretrained_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                graph_emb=graph_emb,
                text_emb=text_emb,
                kg_emb=kg_emb
            )
            B = input_ids.size(0)
            combined = torch.cat([
                emb[0:B], emb[B:2*B], emb[2*B:3*B], emb[3*B:4*B]
            ], dim=1)

            embeddings_all.append(combined.squeeze(0).cpu().numpy())
            labels_all.append(sample["labels"].item())

    embeddings_all = np.vstack(embeddings_all)
    labels_all = np.array(labels_all)

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
    umap_emb = reducer.fit_transform(embeddings_all)

    plt.figure(figsize=(6, 6))
    plt.scatter(umap_emb[labels_all == 0, 0], umap_emb[labels_all == 0, 1], c='green', label='Not Penetrate', s=10)
    plt.scatter(umap_emb[labels_all == 1, 0], umap_emb[labels_all == 1, 1], c='orange', label='Penetrate', s=10)
    plt.title(f"{dataset_name} {title_prefix} Model Embeddings (UMAP Projection)")
    plt.xlabel("UMAP-1") 
    plt.ylabel("UMAP-2")
    plt.legend()
    plt.rcParams.update({'font.family': 'serif'})
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(f"/home/g3-bbm-project/main_folder/FineTune/umap_{dataset_name.lower()}_{title_prefix.lower()}.png")
    plt.show()


# -----------------------
# Pretrained Model
# -----------------------
pretrained_model = MultimodalRoberta(config)
pretrained_state = load_file(weights_path)
missing, unexpected = pretrained_model.load_state_dict(pretrained_state, strict=False)
print("🚩 missing params:", len(missing), "| unexpected params:", len(unexpected))

for name, param in pretrained_model.named_parameters():
    if any([f"encoder.layer.{i}" in name for i in [9, 10, 11]]):
        param.requires_grad = True
    else:
        param.requires_grad = False

# -----------------------
# Classification Model
# -----------------------
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

model = MultimodalClassificationModel(pretrained_model)

# -----------------------
# Evaluation Metrics
# -----------------------
acc = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1 = evaluate.load("f1")

def compute_metrics(predictions, labels):
    predictions = torch.sigmoid(predictions).squeeze().detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    preds_binary = (predictions > 0.5).astype(int)

    precision_curve, recall_curve, _ = precision_recall_curve(labels, predictions)
    sorted_pairs = sorted(zip(recall_curve, precision_curve))
    recall_sorted, precision_sorted = zip(*sorted_pairs)

    results = {
        **acc.compute(predictions=preds_binary, references=labels),
        **precision.compute(predictions=preds_binary, references=labels),
        **recall.compute(predictions=preds_binary, references=labels),
        **f1.compute(predictions=preds_binary, references=labels),
        "roc-auc": roc_auc_score(labels, predictions),
        "prc-auc": auc(recall_sorted, precision_sorted)
    }
    return results

# -----------------------
# Dataset Class
# -----------------------
class CustomFinetuneDataset(Dataset):
    def __init__(self, dataframe, graph_embeddings, text_embeddings, kg_embeddings, tokenizer, max_len):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.graph_embeddings = graph_embeddings
        self.text_embeddings = text_embeddings
        self.kg_embeddings = kg_embeddings

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            self.data.iloc[idx]["selfies"],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "graph_emb": torch.tensor(self.graph_embeddings[idx], dtype=torch.float),
            "text_emb": torch.tensor(self.text_embeddings[idx], dtype=torch.float),
            "kg_emb": torch.tensor(self.kg_embeddings[idx], dtype=torch.float),
            "labels": torch.tensor(self.data.iloc[idx]["Class"], dtype=torch.float)
        }

# -----------------------
# Data Loading
# -----------------------
csv_path = "/home/g3-bbm-project/main_folder/finetune_datasets/finetuning_datasets/backup/bbbp/bbbp_annotated.csv"
if args.use_scaffold:
    train, val, test = train_val_test_split(csv_path, 1, scaffold_split=True)
    train_df = pd.DataFrame(np.column_stack([[x[0] for x in train.smiles()], train.targets()]), columns=["smiles", "Class"])
    val_df = pd.DataFrame(np.column_stack([[x[0] for x in val.smiles()], val.targets()]), columns=["smiles", "Class"])
else:
    train_df, val_df, _ = train_val_test_split(csv_path, 1, scaffold_split=False)
    train_df = train_df.rename(columns={"target": "Class"})
    val_df = val_df.rename(columns={"target": "Class"})

print(f"📊 Train: {len(train_df)} | Val: {len(val_df)}")

train_df = smiles_to_selfies(train_df)
val_df = smiles_to_selfies(val_df)

graph = np.load("/home/g3-bbm-project/main_folder/finetune_datasets/finetuning_datasets/backup/bbbp/bbbp_annotated_graph_embedding.npy")
text = np.load("/home/g3-bbm-project/main_folder/finetune_datasets/finetuning_datasets/backup/bbbp/bbbp_text_embedding.npy")
kg = np.zeros((len(graph), 128), dtype=np.float32)

train_ds = CustomFinetuneDataset(train_df, graph[train_df.index], text[train_df.index], kg[train_df.index], tokenizer, max_len=512)
val_ds = CustomFinetuneDataset(val_df, graph[val_df.index], text[val_df.index], kg[val_df.index], tokenizer, max_len=512)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -----------------------
# Pretrained Embedding Visualization
# -----------------------
print("🔍 Plotting pretrained embeddings (train set)...")
plot_umap_embeddings(model, train_ds, title_prefix="Pretrained", dataset_name="BBBP")

# -----------------------
# Training Loop
# -----------------------
def train(model, train_loader, val_loader, device="cuda", lr=2e-5, epochs=30):
    model.to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            graph_emb = batch["graph_emb"].to(device)
            text_emb = batch["text_emb"].to(device)
            kg_emb = batch["kg_emb"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            loss, _ = model(graph_emb, text_emb, kg_emb, input_ids, attention_mask, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        train_loss_eval, train_acc_eval, train_preds_eval, train_labels_eval = evaluate_model(
            model, train_loader, device
        )
        train_metrics = compute_metrics(train_preds_eval, train_labels_eval)

        val_loss, val_acc, val_preds, val_labels = evaluate_model(
            model, val_loader, device
        )
        val_metrics = compute_metrics(val_preds, val_labels)

        print(f"\nEpoch {epoch+1}")
        print(f"  • Train Loss = {avg_train_loss:.4f}")
        for k, v in train_metrics.items():
            print(f"  • train_{k}: {v:.4f}")

        print(f"  • Val   Loss = {val_loss:.4f}")
        for k, v in val_metrics.items():
            print(f"  • val_{k}:   {v:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            "val_loss": val_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })
        
def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            graph_emb = batch["graph_emb"].to(device)
            text_emb = batch["text_emb"].to(device)
            kg_emb = batch["kg_emb"].to(device)
            labels = batch["labels"].to(device)

            loss, logits = model(graph_emb, text_emb, kg_emb, input_ids, attention_mask, labels)
            total_loss += loss.item()
            all_preds.append(logits.squeeze().cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    avg_loss = total_loss / len(loader)
    acc = ((torch.sigmoid(all_preds) > 0.5).long() == all_labels).float().mean().item()
    return avg_loss, acc, all_preds, all_labels

# -----------------------
# Execute Training
# -----------------------
train(model, train_loader, val_loader, device, lr=args.lr, epochs=args.epochs)
# -----------------------
# Modeli Kaydet
# -----------------------
save_dir = f"/home/g3-bbm-project/main_folder/FineTune/classification_codes/saved_models_LAST/BBBP"
os.makedirs(save_dir, exist_ok=True)

checkpoint_path = os.path.join(save_dir, "multimodal_BBBP_classifier_sd.pt")

torch.save(
    {
        "state_dict": model.state_dict(),
        "config": config.to_dict()
    },
    checkpoint_path
)
print("✅ Model saved:", checkpoint_path)

tokenizer.save_pretrained(save_dir)
print("✅ Tokenizer saved:", save_dir)


# -----------------------
# Finetuned Embedding Visualization
# -----------------------
print("🔍 Plotting fine-tuned embeddings (train set)...")
plot_umap_embeddings(model, train_ds, title_prefix="Fine-tuned", dataset_name="BBBP")
