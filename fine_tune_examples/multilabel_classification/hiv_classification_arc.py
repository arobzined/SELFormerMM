# -----------------------
# Environment Settings
# -----------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from transformers import RobertaConfig, RobertaTokenizerFast
from safetensors.torch import load_file
import argparse
import wandb
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from rdkit import Chem
import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import evaluate
import seaborn as sns  # UMAP grafiği için

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["TOKENIZER_PARALLELISM"] = "false"

sys.path.append("/home/g3-bbm-project/main_folder")
sys.path.append("/home/g3-bbm-project/main_folder/FineTune")

from train.finetuning.prepare_finetuning_data import smiles_to_selfies, train_val_test_split
from multimodal_roberta import MultimodalRoberta

# -----------------------
# Argparse Parameters
# -----------------------
parser = argparse.ArgumentParser()
parser.add_argument("--use_scaffold", type=int, default=1)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--epochs", type=int, default=50)
args = parser.parse_args()

wandb.init(project="finetune-tasks", name="hiv - pretrain 266 - fine tune 100")

# -----------------------
# Load Embeddings and Config
# -----------------------
graph = np.load("/home/g3-bbm-project/main_folder/finetune_datasets/finetuning_datasets/backup/hiv/hiv_annotated_graph_embedding.npy")
text  = np.load("/home/g3-bbm-project/main_folder/finetune_datasets/finetuning_datasets/backup/hiv/hiv_text_embedding.npy")
config    = RobertaConfig.from_pretrained("/media/ubuntu/8TB/g3-bbm-project/checkpoints/epoch_266/hf")
tokenizer = RobertaTokenizerFast.from_pretrained("/media/ubuntu/8TB/g3-bbm-project/checkpoints/epoch_266/hf")

# -----------------------
# Load and Filter CSV
# -----------------------
csv_path = "/home/g3-bbm-project/main_folder/finetune_datasets/finetuning_datasets/backup/hiv/hiv_annotated.csv"
raw_df = pd.read_csv(csv_path)
raw_df = raw_df.rename(columns={"HIV_active": "Class"})

print("🧪 Filtering invalid SMILES...")
print(f"🔢 Total molecules before filtering: {len(raw_df)}")

kg = np.zeros((len(raw_df), 128), dtype=np.float32)

# -----------------------
# Metrics
# -----------------------
acc       = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall    = evaluate.load("recall")
f1        = evaluate.load("f1")

def compute_metrics(predictions, labels):
    # predictions: logits tensor; labels: tensor
    probs = torch.sigmoid(predictions).detach().cpu().numpy().astype(float)
    y_true = labels.detach().cpu().numpy().astype(int)

    roc = roc_auc_score(y_true, probs) if len(np.unique(y_true)) == 2 else float("nan")

    prec, rec, thr = precision_recall_curve(y_true, probs)
    f1s = 2 * prec * rec / (prec + rec + 1e-12)
    best_idx = f1s.argmax()
    best_thr = thr[best_idx] if best_idx < len(thr) else 0.5

    preds_bin = (probs >= best_thr).astype(int)

    results = {
        "threshold": float(best_thr),
        "roc-auc": float(roc),
        "prc-auc": float(auc(rec, prec)),
        **acc.compute(predictions=preds_bin, references=y_true),
        **precision.compute(predictions=preds_bin, references=y_true),
        **recall.compute(predictions=preds_bin, references=y_true),
        **f1.compute(predictions=preds_bin, references=y_true),
    }
    return results

def is_valid_smiles(smi):
    try:
        return Chem.MolFromSmiles(smi) is not None
    except:
        return False

valid_mask = raw_df["smiles"].apply(is_valid_smiles)
filtered_df = raw_df[valid_mask].reset_index(drop=True)
valid_indices = np.where(valid_mask)[0]
print(f"✅ Valid molecules after filtering: {len(filtered_df)}")

# (opsiyonel) sınıf dağılımı bilgi amaçlı
y_all = filtered_df["Class"].astype(int).to_numpy()
print("n_pos:", (y_all==1).sum(), "n_neg:", (y_all==0).sum())

# Apply valid indices
graph = graph[valid_indices]
text  = text[valid_indices]
kg    = kg[valid_indices]

filtered_csv_path = "/home/g3-bbm-project/main_folder/FineTune/tmp/hiv_filtered.csv"
os.makedirs(os.path.dirname(filtered_csv_path), exist_ok=True)
filtered_df.to_csv(filtered_csv_path, index=False)

# -----------------------
# Train/Val Split
# -----------------------
if args.use_scaffold:
    train, val, test = train_val_test_split(filtered_csv_path, 1, scaffold_split=True)
else:
    train, val, test = train_val_test_split(filtered_csv_path, 1, scaffold_split=False)

train_df = pd.DataFrame(np.column_stack([[x[0] for x in train.smiles()], train.targets()]), columns=["smiles", "Class"])
val_df   = pd.DataFrame(np.column_stack([[x[0] for x in val.smiles()],   val.targets()]),   columns=["smiles", "Class"])

# ---- SELFIES'e çevirmeden ÖNCE orig_idx üret + yedekle ----
idx_of = {smi: i for i, smi in enumerate(filtered_df["smiles"].tolist())}
train_df["orig_idx"] = train_df["smiles"].map(idx_of).astype(int)
val_df["orig_idx"]   = val_df["smiles"].map(idx_of).astype(int)

_train_backup = train_df[["smiles", "orig_idx"]].copy()
_val_backup   = val_df[["smiles", "orig_idx"]].copy()

# ---- SELFIES dönüşümü (DAYANIKLI) ----
print("🔁 Converting SMILES to SELFIES...")

# 0) Dönüşümden ÖNCE tekil anahtar ekle
train_df["row_id"] = np.arange(len(train_df))
val_df["row_id"]   = np.arange(len(val_df))

# Yedek: dönüşüm öncesi referans değerler
_train_backup = train_df[["row_id", "smiles", "orig_idx"]].copy()
_val_backup   = val_df[["row_id", "smiles", "orig_idx"]].copy()

# 1) Dönüştür (kopya üzerinde)
train_df_conv = smiles_to_selfies(train_df.copy())
val_df_conv   = smiles_to_selfies(val_df.copy())

# 2) 'selfies' üretildi mi?
for name, d in [("train", train_df_conv), ("val", val_df_conv)]:
    if "selfies" not in d.columns:
        raise RuntimeError(f"smiles_to_selfies({name}_df) 'selfies' kolonu üretmedi.")

# 3) Güvenli yeniden kurulum fonksiyonu
def _rebuild_df(conv_df, backup_df, split_name):
    cols = conv_df.columns
    has_row_id = "row_id" in cols
    has_smiles = "smiles" in cols
    has_orig   = "orig_idx" in cols

    # a) orig_idx korunmuşsa: sadece NaN SELFIES'leri at, gerekirse smiles'ı backup'tan getir
    if has_orig:
        out = conv_df.dropna(subset=["selfies"]).reset_index(drop=True)
        if "smiles" not in out.columns and "row_id" in out.columns:
            out = out.merge(backup_df[["row_id", "smiles"]], on="row_id", how="left")
        return out

    # b) orig_idx yok ama row_id varsa: row_id ile backup’a bağla
    if has_row_id:
        out = conv_df.dropna(subset=["selfies"]).merge(backup_df, on="row_id", how="inner")
        return out.reset_index(drop=True)

    # c) row_id yok ama smiles varsa: smiles ile bağla
    if has_smiles:
        out = conv_df.dropna(subset=["selfies"]).merge(
            backup_df[["smiles", "orig_idx"]], on="smiles", how="left"
        )
        if out["orig_idx"].isna().any():
            missing = int(out["orig_idx"].isna().sum())
            raise RuntimeError(
                f"{split_name}: smiles merge sonrası {missing} satırın 'orig_idx'i bulunamadı. "
                "Dönüşüm 'smiles' değerini değiştirmiş olabilir."
            )
        return out.reset_index(drop=True)

    # d) Hiçbiri yoksa, dönüşüm fonksiyonunu row_id/smiles koruyacak şekilde güncellemek gerekir
    raise RuntimeError(
        f"{split_name}: smiles_to_selfies dönüşümü 'row_id' ve 'smiles' kolonlarını düşürmüş. "
        "Lütfen dönüşüm fonksiyonunu bu kolonları koruyacak şekilde düzenleyin."
    )

# 4) Uygula
train_df = _rebuild_df(train_df_conv, _train_backup, "train")
val_df   = _rebuild_df(val_df_conv,   _val_backup,   "val")

# 5) Bilgilendirme
bad_train = len(_train_backup) - len(train_df)
bad_val   = len(_val_backup)   - len(val_df)
if bad_train:
    print(f"⚠️ Train'de encode edilemeyen SELFIES satırı: {bad_train}")
if bad_val:
    print(f"⚠️ Val'de encode edilemeyen SELFIES satırı: {bad_val}")

# 6) pos_weight (train)
y_train = train_df["Class"].astype(int).to_numpy()
n_pos = (y_train == 1).sum()
n_neg = (y_train == 0).sum()
pos_weight_value = n_neg / max(1, n_pos)
print(f"🔧 Computed pos_weight ≈ {pos_weight_value:.2f}")


# -----------------------
# Dataset Class
# -----------------------
class CustomFinetuneDataset(Dataset):
    def __init__(self, dataframe, graph_embeddings, text_embeddings, kg_embeddings, tokenizer, max_len):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.graph_embeddings = graph_embeddings  # tüm filtreli diziler
        self.text_embeddings  = text_embeddings
        self.kg_embeddings    = kg_embeddings

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        oi  = int(row["orig_idx"])
        enc = self.tokenizer.encode_plus(
            row["selfies"],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "graph_emb": torch.tensor(self.graph_embeddings[oi], dtype=torch.float),
            "text_emb":  torch.tensor(self.text_embeddings[oi],  dtype=torch.float),
            "kg_emb":    torch.tensor(self.kg_embeddings[oi],    dtype=torch.float),
            "labels":    torch.tensor(row["Class"], dtype=torch.float)
        }

# -----------------------
# Datasets
# -----------------------
train_ds = CustomFinetuneDataset(train_df, graph, text, kg, tokenizer, max_len=512)
val_ds   = CustomFinetuneDataset(val_df,   graph, text, kg, tokenizer, max_len=512)

full_df = pd.concat([train_df, val_df], ignore_index=True)
full_dataset = CustomFinetuneDataset(full_df, graph, text, kg, tokenizer, max_len=512)

# -----------------------
# Load Model (for UMAP)
# -----------------------
pretrained_model = MultimodalRoberta(config)
weights_path_umap = "/media/ubuntu/8TB/g3-bbm-project/checkpoints/epoch_090/model.safetensors"
state_dict = load_file(weights_path_umap)
pretrained_model.load_state_dict(state_dict, strict=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrained_model.to(device)

# -----------------------
# UMAP Plot Function
# -----------------------
def plot_umap_embeddings(model, dataset, title_prefix="Fine-tuned", dataset_name="HIV"):
    model.eval()
    model.to(device)
    embeddings_all = []
    labels_all = []

    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            input_ids      = sample["input_ids"].unsqueeze(0).to(device)
            attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
            graph_emb      = sample["graph_emb"].unsqueeze(0).to(device)
            text_emb       = sample["text_emb"].unsqueeze(0).to(device)
            kg_emb         = sample["kg_emb"].unsqueeze(0).to(device)

            emb = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                graph_emb=graph_emb,
                text_emb=text_emb,
                kg_emb=kg_emb
            )

            # DİKKAT: MultimodalRoberta çıktısının [4B, H] olduğunu varsayan eski mantık
            B = input_ids.size(0)
            combined = torch.cat([
                emb[0:B], emb[B:2*B], emb[2*B:3*B], emb[3*B:4*B]
            ], dim=1)

            embeddings_all.append(combined.squeeze(0).detach().cpu().numpy())
            labels_all.append(sample["labels"].item())

    embeddings_all = np.vstack(embeddings_all)
    labels_all = np.array(labels_all)

    # PCA -> UMAP
    pca = PCA(n_components=60)
    pca_embeddings = pca.fit_transform(embeddings_all)

    reducer = umap.UMAP(n_neighbors=20, min_dist=0.1, metric="cosine", random_state=42)
    umap_emb = reducer.fit_transform(pca_embeddings)

    # Plot
    palette = sns.color_palette("Set1", n_colors=2)
    inactive_color = palette[1]
    active_color   = palette[0]

    plt.figure(figsize=(7.5, 6.5))
    plt.scatter(umap_emb[labels_all == 0, 0], umap_emb[labels_all == 0, 1], color=inactive_color, label='Inactive', s=10)
    plt.scatter(umap_emb[labels_all == 1, 0], umap_emb[labels_all == 1, 1], color=active_color,   label='Active',   s=10)
    plt.title(f"{dataset_name} {title_prefix} Model Embeddings\n(UMAP Projection)", fontsize=13, weight="bold", pad=15)
    plt.xlabel("UMAP-1", fontsize=11)
    plt.ylabel("UMAP-2", fontsize=11)
    plt.legend(title="Class", fontsize=9, title_fontsize=10)
    plt.rcParams.update({'font.family': 'serif'})
    plt.grid(False)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(f"/home/g3-bbm-project/main_folder/FineTune/umap_14aug_{dataset_name.lower()}_{title_prefix.lower()}.png")
    plt.show()

# -----------------------
# Pretrained Embedding Visualization
# -----------------------
print("🔍 Generating UMAP for pretrained model...")
# plot_umap_embeddings(pretrained_model, full_dataset, title_prefix="Pretrained", dataset_name="HIV")
print("✅ Pretrained UMAP completed.\n")

# -----------------------
# Model Definition (fine-tune head)
# -----------------------
# Load newer encoder weights
weights_path = "/media/ubuntu/8TB/g3-bbm-project/checkpoints/epoch_266/model.safetensors"
state_dict = load_file(weights_path)
pretrained_model.load_state_dict(state_dict, strict=False)

# Freeze encoder
for param in pretrained_model.parameters():
    param.requires_grad = False

class MultimodalClassificationModel(nn.Module):
    def __init__(self, pretrained_model, hidden_size=768, num_labels=1, dropout_prob=0.3, pos_weight=1.0):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.pos_weight = float(pos_weight)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size * 4),
            nn.Linear(hidden_size * 4, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
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
        # Eski davranış: embeddings [4B, H] → modality-wise concat
        B = input_ids.size(0)
        combined = torch.cat([
            embeddings[0:B], embeddings[B:2*B], embeddings[2*B:3*B], embeddings[3*B:4*B]
        ], dim=1)

        logits = self.classifier(combined)

        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.pos_weight], device=logits.device))
            loss = loss_fn(logits.view(-1), labels.float().view(-1))
            return loss, logits
        return None, logits

def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
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
            all_preds.append(logits.view(-1).cpu())
            all_labels.append(labels.view(-1).cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    avg_loss = total_loss / max(1, len(loader))
    return avg_loss, all_preds, all_labels

def train(model, train_loader, val_loader, device, lr=1e-4, epochs=25):
    os.makedirs("/home/g3-bbm-project/main_folder/FineTune/logs", exist_ok=True)
    open("/home/g3-bbm-project/main_folder/FineTune/logs/hiv_metrics_log.txt", "w").close()
    model.to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.01)

    for epoch in range(epochs):
        print(f"🧪 Epoch {epoch+1}/{epochs} started...")
        model.train()
        total_loss = 0.0

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

        avg_train_loss = total_loss / max(1, len(train_loader))

        train_loss_eval, train_preds_eval, train_labels_eval = evaluate_model(model, train_loader, device)
        train_metrics = compute_metrics(train_preds_eval, train_labels_eval)

        val_loss, val_preds, val_labels = evaluate_model(model, val_loader, device)
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
            "best_threshold_train": train_metrics["threshold"],
            "best_threshold_val":   val_metrics["threshold"],
        })

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fine_model = MultimodalClassificationModel(pretrained_model, pos_weight=pos_weight_value)

print("🚀 Starting fine-tuning...")

# Dengesizlik için WeightedRandomSampler önerilir
labels_np = train_df["Class"].astype(int).to_numpy()
class_count = np.bincount(labels_np)  # [n_neg, n_pos]
class_weight = 1.0 / (class_count + 1e-12)
sample_weight = class_weight[labels_np]
sampler = WeightedRandomSampler(sample_weight, num_samples=len(sample_weight), replacement=True)

#train_loader = DataLoader(train_ds, batch_size=16, sampler=sampler, num_workers=4, pin_memory=True)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False,  num_workers=4, pin_memory=True)

train(fine_model, train_loader, val_loader, device, lr=args.lr, epochs=args.epochs)
print("✅ Fine-tuning completed.\n")

save_dir = f"/home/g3-bbm-project/main_folder/FineTune/classification_codes/saved_models_LAST/HIV"
os.makedirs(save_dir, exist_ok=True)

checkpoint_path = os.path.join(save_dir, "multimodal_HIV_classifier_sd.pt")

torch.save(
    {
        "state_dict": fine_model.state_dict(),
        "config": config.to_dict()
    },
    checkpoint_path
)
print("✅ Model kaydedildi:", checkpoint_path)

tokenizer.save_pretrained(save_dir)
print("✅ Tokenizer da kaydedildi:", save_dir)

print("🔍 Generating UMAP for fine-tuned model...")
# plot_umap_embeddings(fine_model.pretrained_model, full_dataset, title_prefix="Fine-tuned", dataset_name="HIV")
print("✅ Fine-tuned UMAP completed.")
