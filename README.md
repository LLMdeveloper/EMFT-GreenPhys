# EMFT: Energy-Minimization Fine-Tuning

This repository provides the source code and datasets to reproduce the computational experiments presented in our paper. EMFT is a differentiable regularization framework designed to impose global sequence-level constraints during autoregressive language generation.

## 📂 Project Structure

```text
EMFT-GreenPhys/
├── README.md
├── requirements.txt
├── __init__.py
├── train.py                  # Unified training entry point (supports all models & variants)
├── evaluate.py               # Independent evaluation and PCA analysis script
├── data_utils.py             # Data loading, tokenization, and prompt formatting
├── trainer.py                # Custom EMFTTrainer class (handles dynamic weight scheduling)
├── losses.py                 # Loss functions (LM, Energy, Path) and EnergyHead architecture
│
├── dataset/                  # Data directory
│   ├── GreenPhys-10K_train_en.json  # Full English training set (10K)
│   ├── GreenPhys-10K_val_en.json    # Full English validation set (10K)
│   ├── GreenPhys-10K_test_en.json   # Full English test set (10K)
│   ├── GreenPhys-3.5K_train_en.json # Stratified subset used in our experiments
│   ├── GreenPhys-400_val_en.json    # Stratified subset used in our experiments
│   ├── GreenPhys-400_test_en.json   # Stratified subset used in our experiments
│   └── ... (Chinese and full versions)
│
└── outputs/                  # (Auto-generated) Directory for checkpoints, logs, and evaluation results
```

## 📊 Dataset Description (GreenPhys-10K)

The `dataset/` folder contains the **GreenPhys-10K** instruction-tuning dataset. 
- `10K` indicates the dataset contains 10,000 samples in total.
- Files ending with `_en` are in **English**, while files without the suffix are in **Chinese**.
- The `3.5K_train`, `400_val`, and `400_test` splits are **stratified samples** drawn from the full `10K` English dataset. These are the exact splits used to train and evaluate the models reported in our paper.

## 🧪 Supported Models and Ablations

The training script is fully parameterized. We evaluate EMFT across **3 base models** and **4 ablation variants**.

### Supported Models
Use the `--model` argument: `llama`, `mistral`, `qwen`.

### Ablation Variants
Use the `--variant` argument:
| Variant      | Description                                          | Weights ($\alpha, \beta, \gamma$) | Schedule    |
| :----------- | :--------------------------------------------------- | :-------------------------------- | :---------- |
| `full`       | Full EMFT framework                                  | (1.0, 0.5, 0.1)                   | Progressive |
| `nopath`     | Removes the Path Integral Regularizer                | (1.0, 0.5, 0.0)                   | Progressive |
| `noenergy`   | Removes the Energy Regression Loss                   | (1.0, 0.0, 0.1)                   | Progressive |
| `noschedule` | Uses fixed weights instead of progressive scheduling | (1.0, 0.5, 0.1)                   | Fixed       |

## 🚀 Quick Start

To train and evaluate a specific model with a specific variant, use the unified `train.py` script:

```bash
# Example 1: Train LLaMA-3-8B with the Full EMFT framework
python train.py --model llama --variant full

# Example 2: Train Mistral-7B with the NoPath ablation
python train.py --model mistral --variant nopath
```

The script will automatically perform the following:
1. Load the `3.5K` train and `400` val splits from the `dataset/` folder.
2. Train the model using QLoRA and the specified EMFT loss combination.
3. Evaluate the model on the `400` test split (sampling 200 for QCR/DRA calculation).
4. Run PCA on the hidden states to analyze the energy landscape.
5. Save all weights, `loss_history.json`, and `eval_results.json` to `./outputs/{Model}-EMFT-{Variant}/`.
