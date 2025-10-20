#!/bin/bash
#SBATCH --job-name=lora-hierarchy-demo
#SBATCH -A cs_daniel.haehn
#SBATCH -p pomplun
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200
#SBATCH --mem=200G
#SBATCH -w chimera21
#SBATCH -t 0-01:00:00
#SBATCH --output=/home/edward.gaibor001/medgemmaOMAMA/beta/hierarchy/logs/demo-%j.out
#SBATCH --error=/home/edward.gaibor001/medgemmaOMAMA/beta/hierarchy/logs/demo-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=edward.gaibor001@umb.edu

# Resource optimization:
# - H200 partition (need GPU for model loading + inference)
# - 1 GPU sufficient for demo
# - 200GB RAM to load base model + adapter + test data
# - 1 hour max (demo with 5 samples, ~10 min expected)

echo "=============================================="
echo "🎯 LoRA Adapter Hierarchy Demo"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "=============================================="
echo ""
echo "This demo shows:"
echo "  1. Base model conversational capabilities"
echo "  2. Adapter classification capabilities"
echo "  3. Hierarchical decision-making"
echo "=============================================="

# Create logs directory
mkdir -p /home/edward.gaibor001/medgemmaOMAMA/beta/hierarchyDemo/logs

# GPU info
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Activate conda environment
source /home/edward.gaibor001/miniconda3/etc/profile.d/conda.sh
conda activate medgemma

# Set environment variables for performance
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run the demo
cd /home/edward.gaibor001/medgemmaOMAMA/beta/hierarchy
echo "Running hierarchical adapter demo..."
echo ""
python demo_adapter_hierarchy.py
