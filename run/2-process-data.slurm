#!/bin/bash
#SBATCH --job-name=process-balanced
#SBATCH --partition=DGXA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/edward.gaibor001/medgemmaOMAMA/beta/run/logs/2-process-%j.out
#SBATCH --error=/home/edward.gaibor001/medgemmaOMAMA/beta/run/logs/2-process-%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=edward.gaibor001@umb.edu

# Resource optimization:
# - A100 partition (minimal GPU usage, mostly CPU)
# - 16 CPUs for parallel processing
# - 64GB RAM (loading processor + dataset)
# - 1 hour max (should finish in ~15 minutes)

echo "=============================================="
echo "Step 2: Process Balanced Dataset"
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

# Run data processing
cd /home/edward.gaibor001/medgemmaOMAMA/beta
python 3-processor-balanced.py

echo ""
echo "=============================================="
echo "✅ Step 2 Complete"
echo "End time: $(date)"
echo "=============================================="
echo ""
echo "📋 Next step:"
echo "  sbatch run/3-train-balanced.slurm"
echo "=============================================="
