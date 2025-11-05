# MedGemma OMAMA Training Pipeline (Balanced)

Fine-tune MedGemma-4B for breast cancer classification on balanced OMAMA 256×256 dataset.

## 🚀 Quick Start

```bash
cd ~/medgemmaOMAMA/beta/run
bash run-all.sh  # Runs all 5 steps sequentially
```

## 📋 Pipeline Steps

| Step | Script | SLURM Job | Purpose |
|------|--------|-----------|---------|
| 1 | `1-dataset-balanced.py` | `1-create-dataset.slurm` | Create 50/50 balanced dataset |
| 2 | `2-loading.py` | `2-process-data.slurm` | Load & validate NPZ images + labels |
| 3 | `3-processor-balanced.py` | `3-train-balanced.slurm` | Fine-tune with LoRA (H200 GPU, 8 epochs) |
| 4 | `4-modelH-balanced.py` | `4-evaluate-balanced.slurm` | Evaluate on test set |
| 5 | `5-baseline-improved-prompt.py` | `5-baseline-comparison.slurm` | Compare vs baseline (no fine-tuning) |

**Note:** Step 6 (merge & push to HF) is excluded from git.

## 🔧 Manual Step Execution

```bash
sbatch run/1-create-dataset.slurm
sbatch run/2-process-data.slurm
sbatch run/3-train-balanced.slurm      # Requires H200 GPU
sbatch run/4-evaluate-balanced.slurm
sbatch run/5-baseline-comparison.slurm
```

## 📊 Expected Results

- **Accuracy**: 98.88%
- **Sensitivity**: 99.66%
- **Specificity**: 98.10%

## ⚙️ Setup

### 1. Clone Environment
```bash
# Create conda environment from file
conda env create -f environment.yml

# Activate environment
conda activate medgemma
```

### 2. Set HuggingFace Token


## 📁 Key Directories

- `/hpcstor6/scratch01/e/edward.gaibor001/medgemma_runs/omama256_balanced` - LoRA adapters
- `run/logs/` - SLURM output logs
