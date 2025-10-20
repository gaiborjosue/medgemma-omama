#!/bin/bash
#SBATCH --job-name=balanced-dataset
#SBATCH --partition=DGXA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/edward.gaibor001/medgemmaOMAMA/beta/run/logs/1-dataset-%j.out
#SBATCH --error=/home/edward.gaibor001/medgemmaOMAMA/beta/run/logs/1-dataset-%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=edward.gaibor001@umb.edu

# Resource optimization:
# - A100 partition (no GPU needed, but has fast CPUs)
# - 16 CPUs for parallel data processing
# - 64GB RAM (dataset loading + undersampling)
# - 2 hours max (should finish in ~30 minutes)

echo "=============================================="
echo "Step 1: Create Balanced Dataset (50/50)"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=============================================="

# Create log directory
mkdir -p /home/edward.gaibor001/medgemmaOMAMA/beta/run/logs

# Activate conda environment
source /home/edward.gaibor001/miniconda3/etc/profile.d/conda.sh
conda activate medgemma

# Run dataset balancing
cd /home/edward.gaibor001/medgemmaOMAMA/beta
python 1-dataset-balanced.py

echo ""
echo "=============================================="
echo "✅ Step 1 Complete"
echo "End time: $(date)"
echo "=============================================="
echo ""
echo "📋 Next step:"
echo "  sbatch run/2-process-data.slurm"
echo "=============================================="
