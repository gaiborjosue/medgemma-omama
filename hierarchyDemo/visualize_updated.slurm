#!/bin/bash
#SBATCH --job-name=visualize-adapter
#SBATCH -A cs_daniel.haehn
#SBATCH -p pomplun
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200
#SBATCH --mem=200G
#SBATCH -w chimera21
#SBATCH -t 0-00:30:00
#SBATCH --output=/home/edward.gaibor001/medgemmaOMAMA/beta/hierarchy/logs/visualize-%j.out
#SBATCH --error=/home/edward.gaibor001/medgemmaOMAMA/beta/hierarchy/logs/visualize-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=edward.gaibor001@umb.edu

# Resource optimization:
# - H200 partition
# - 1 GPU for model loading
# - 200GB RAM for models
# - 30 min max (architecture inspection, ~2 min expected)

echo "=============================================="
echo "🔬 LoRA Adapter Architecture Visualizer"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "=============================================="

mkdir -p /home/edward.gaibor001/medgemmaOMAMA/beta/hierarchy/logs

echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Activate conda environment
source /home/edward.gaibor001/miniconda3/etc/profile.d/conda.sh
conda activate medgemma

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/edward.gaibor001/medgemmaOMAMA/beta/hierarchy
python visualize_adapter.py

echo ""
echo "=============================================="
echo "✅ Visualization Complete!"
echo "End time: $(date)"
echo "=============================================="
