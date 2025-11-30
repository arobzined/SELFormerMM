# SELFormer Multi Modal: Integrating SELFIES, Graph, Text, and Knowledge Graph Modalities for Enhanced Molecular Representation Learning

Automated computational analysis of the vast chemical space is critical for research areas such as drug discovery and material science. To address the need for compact and meaningful representations of molecular structures, recent efforts have applied natural language processing (NLP) techniques to string-based chemical notations. While most existing models rely on SMILES, its fragility and syntactic constraints often limit robustness and validity during model training. To overcome these limitations, we introduce SELFormer, a transformer-based chemical language model that employs SELFIES—a 100% valid, compact, and expressive molecular notation. Pre-trained on two million drug-like molecules and fine-tuned for downstream molecular property prediction tasks, SELFormer demonstrates superior performance compared to both graph-based and SMILES-based approaches, particularly in predicting aqueous solubility and adverse drug reactions. Visualization of its learned embeddings via dimensionality reduction further shows that even the pre-trained model can effectively capture structural differences between molecules.
Building upon this foundation, we propose Multimodal SELFormer, an extended framework that integrates four complementary molecular modalities: SELFIES sequences, molecular graph structures, natural language descriptions, and knowledge graph embeddings. Each modality is projected into a shared embedding space through modality-specific projection layers. Using supervised contrastive learning with SINCERE loss, the model learns aligned, semantically rich, and modality-invariant representations. This multimodal design enhances the model’s generalization capability and expressiveness in various classification and regression tasks, including toxicity prediction, solubility estimation, and drug-target interaction assessment. Experimental results show that Multimodal SELFormer consistently outperforms unimodal baselines and produces more informative latent spaces, demonstrating the power of multimodal fusion in molecular representation learning.

## Model Architecture

The MultimodalRoberta architecture integrates four complementary information sources—RoBERTa text embeddings, graph embeddings, auxiliary text embeddings, and knowledge-graph embeddings—into a unified multimodal representation. The model uses a standard RoBERTa encoder (hidden size 768) to extract the normalized [CLS] embedding from the input text, while three parallel projection networks transform the graph (512-dim), external text (768-dim), and KG (128-dim) embeddings into the same hidden space through deep feed-forward layers with LayerNorm and ReLU activations. Each projection network expands and contracts the embedding dimensionality (×4 → ×6 → ×6 → ×4 → ×1 of 768) to allow rich nonlinear transformation. In the final step, the model concatenates the RoBERTa text embedding with the projected graph, text, and KG embeddings along the batch dimension, producing a combined multimodal feature vector that captures structural, textual, and knowledge-based information in a single architecture.

### Data Processing
You can find all the work related to data processing in the `data` folder. It contains the steps we have taken to preprocess and prepare the data for the multimodal approach.

### Current Model
Current model that described detailly above can be found in the `model` folder. It also contains the loss function we used,and the accelerator mode where we need to use multiple GPU's since the process took so much efford, its very beneficial to check them out too.

---

## Create Environment

We used conda environment while we were working with this project, and we are recommended this environment for users. You can easily create a environment with these commands:

```
conda create -n SELFormer_env
conda activate SELFormer_env
pip install -r requirements.txt
```

## Generating Embeddings - For Pretraining Model

SELFIES notations are directly used in this task but other modalities must be embedded before passing to the pretraining phase, so embeddings can be generated using:
- For Text Embeddings: `text_embedding.py`,
- For Graph Embeddings:  `graph_embeddning.py`,
- For Knowledge Graphs Embeddings: `dmgi_model.py`.
  
Generating embeddings are valid, with this direction:
- Change the folder_path's in the code and run the codes, f.e. text embedding:

```
nohup python text_embedding.py > text_embeddings.txt 2>&1 &
```

## Pretraining the Multimodal Model

After generating your available embeddings, you can pretrain the multimodal backbone using  
`pretrain_modal_with_accelerator.py` located under the `model/` directory.

> **Important:**  
> You are not required to provide all modalities. Missing modalities can be replaced with zero-filled matrices.  
> However, every modality you supply must follow the exact same sample ordering.  
> If the ordering is inconsistent across your SELFIES file and embedding files, the model will receive mismatched inputs and pretraining quality will degrade significantly.

---

### Preparing Input Paths

Inside the pretraining script, update the following paths to match the locations of your datasets:

```python
csv_name    = f"{INPUT_EMBEDDINGS_PATH}/SELFIES.csv"
csv_name    = read_selfies_column(csv_path=csv_name)

kg_path     = f"{INPUT_EMBEDDINGS_PATH}/KNOWLEDGE_GRAPH.npy"
text_path   = f"{INPUT_EMBEDDINGS_PATH}/TEXT_normalized.npy"
graph_path  = f"{INPUT_EMBEDDINGS_PATH}/GRAPH_normalized.npy"
```
Notes:

SELFIES.csv contains the textual SELFIES strings.

All other modalities (GRAPH_normalized.npy, TEXT_normalized.npy, KNOWLEDGE_GRAPH.npy) must have the same number of rows and identical indexing.

Missing modalities must be replaced with zero matrices of appropriate shape.

### Running the Pretraining Script

To launch pretraining on a single GPU or CPU:

```python
nohup python pretrain_model.py > MAIN_TRAIN.log 2>&1 &
```

This runs the training process in the background and logs output to MAIN_TRAIN.log.

### Multi-GPU Training with Accelerate
If your environment supports distributed training, you can run pretraining on multiple GPUs:

```python
nohup setsid accelerate launch \
  --num_processes 3 \
  --mixed_precision "no" \
  pretrain_modal_with_accelerator.py > MAIN_TRAIN.log 2>&1 &
```

Adjust the settings as needed:

--num_processes: number of GPUs to use

--mixed_precision: "no", "fp16", or "bf16" depending on your hardware

## Fine Tuning

This repository includes three downstream examples built on top of SELFormer:

BBBP → Binary classification

HIV → Multi-label classification

PDBBind → Regression (continuous affinity prediction)

Each task contains:

A training script → *_{task}.py

A prediction script → *_{task}_prediction.py

Below are short descriptions and instructions for running each example.

1. BBBP — Binary Classification

The BBBP task predicts whether a molecule is blood–brain barrier permeable or not.
The model uses SELFIES tokens together with graph/text/KG embeddings.
Training logs ROC-AUC, PRC-AUC, loss curves, etc.

Train

File: bbbp_classification.py

```python
python bbbp_classification.py \
  --use_scaffold 1 \
  --lr 1e-4 \
  --epochs 100
```

Arguments

--use_scaffold {0|1} – 1 for scaffold split, 0 for random split

--lr – learning rate

--epochs – number of epochs

Note: Dataset + embedding paths inside the script are currently hard-coded.
Update them to match your directory layout.

Predict

File: bbbp_prediction.py

Predict using a CSV:

```python 
python bbbp_prediction.py \
  --model_path path/to/multimodal_BBBP_classifier_sd.pt \
  --tokenizer_path path/to/BBBP_tokenizer_dir \
  --input_file data/bbbp_test_smiles.csv \
  --output_dir outputs/bbbp_preds
```

Predict a single SMILES:

```python
python bbbp_prediction.py \
  --model_path path/to/multimodal_BBBP_classifier_sd.pt \
  --tokenizer_path path/to/BBBP_tokenizer_dir \
  --smiles "CCOC(=O)NCCC1=CNc2c1cc(OC)cc2" \
  --output_dir outputs/bbbp_single
```

Outputs

predictions.csv — predicted permeability (0/1)

embeddings.npy — learned multimodal embeddings for each sample

2. HIV — Multi-Label Classification

The HIV task predicts HIV activity labels.
The script handles invalid SMILES using RDKit, converts valid ones to SELFIES, loads graph/text/KG embeddings, and trains a classification head using BCE-with-logits with pos_weight to handle imbalance.

Train

Files:

hiv_classification.py

hiv_classification_arc.py (architecture-extended version)

```python
python hiv_classification.py \
  --use_scaffold 1 \
  --lr 3e-4 \
  --epochs 50
```

Arguments

--use_scaffold – scaffold or random split

--lr – learning rate

--epochs – number of epochs

Predict

File: hiv_prediction.py

From CSV:

```python
python hiv_prediction.py \
  --model_path path/to/hiv_classifier_sd.pt \
  --tokenizer_path path/to/HIV_tokenizer_dir \
  --input_file data/hiv_test_smiles.csv \
  --output_dir outputs/hiv_preds
```

Single SMILES:

```python
python hiv_prediction.py \
  --model_path path/to/hiv_classifier_sd.pt \
  --tokenizer_path path/to/HIV_tokenizer_dir \
  --smiles "CC1=C(C(=O)NC(=O)N1)N" \
  --output_dir outputs/hiv_single
```

Outputs

predictions.csv — multi-label predictions (sigmoid + thresholding)

embeddings.npy — multimodal latent vectors

3. PDBBind — Regression
Task Summary

The PDBBind example predicts continuous binding affinities for protein–ligand complexes.
The model uses SELFIES + external graph/text/KG embeddings and fine-tunes a regression head on top of the multimodal backbone.

Train

File: regression_pdbbind_with_wandb.py

```python
python regression_pdbbind_with_wandb.py \
  --lr 2e-5 \
  --epochs 100
```

Arguments

--lr – learning rate

--epochs – number of epochs

Predict

File: pdbbind_prediction.py

Using CSV + embeddings:

```python
python pdbbind_prediction.py \
  --model_path path/to/pdbbind_regressor_sd.pt \
  --tokenizer_path path/to/epoch_266/hf \
  --input_file data/pdbbind_test_smiles.csv \
  --graph_embs data/pdbbind_graph_embs.npy \
  --text_embs data/pdbbind_text_embs.npy \
  --kg_embs data/pdbbind_kg_embs.npy \
  --output_dir outputs/pdbbind_preds
```

Single SMILES (no external embeddings):

```python
python pdbbind_prediction.py \
  --model_path path/to/pdbbind_regressor_sd.pt \
  --tokenizer_path path/to/epoch_266/hf \
  --smiles "CC(C)CC1=CC(=O)NC(=O)N1" \
  --output_dir outputs/pdbbind_single
```

Outputs

regression_predictions.csv — continuous affinity predictions

embeddings.npy — embeddings used by the regressor
