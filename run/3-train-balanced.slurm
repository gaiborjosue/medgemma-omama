#!/bin/bash
#SBATCH --job-name=train-balanced
#SBATCH -A cs_daniel.haehn
#SBATCH -p pomplun
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200
#SBATCH --mem=200G
#SBATCH -w chimera21
#SBATCH -t 2-00:00:00
#SBATCH --output=/home/edward.gaibor001/medgemmaOMAMA/beta/run/logs/3-train-%j.out
#SBATCH --error=/home/edward.gaibor001/medgemmaOMAMA/beta/run/logs/3-train-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=edward.gaibor001@umb.edu
#SBATCH --dependency=afterany:636181

# Resource optimization:
# - H200 partition (141GB VRAM for MedGemma-4B + LoRA)
# - 1 GPU (single-node training)
# - 32 CPUs for data loading
# - 120GB RAM for data pipeline
# - 8 hours max (balanced dataset ~50% size, expect ~3-4 hours)

echo "=============================================="
echo "Step 3: Train with Balanced Dataset"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "=============================================="

# Create log directory
mkdir -p /home/edward.gaibor001/medgemmaOMAMA/beta/run/logs

# GPU info
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Activate conda environment
source /home/edward.gaibor001/miniconda3/etc/profile.d/conda.sh
conda activate medgemma

# Environment for performance
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run training
cd /home/edward.gaibor001/medgemmaOMAMA/beta
python 4-modelH-balanced.py

echo ""
echo "=============================================="
echo "✅ Step 3 Complete"
echo "End time: $(date)"
echo "=============================================="
echo ""
echo "📋 Next steps:"
echo "  sbatch run/4-evaluate-balanced.slurm"
echo "  sbatch run/5-baseline-comparison.slurm"
echo "=============================================="
