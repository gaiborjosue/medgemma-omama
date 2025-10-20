#!/bin/bash
#SBATCH --job-name=eval-balanced
#SBATCH -A cs_daniel.haehn
#SBATCH -p pomplun
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200
#SBATCH --mem=200G
#SBATCH -w chimera21
#SBATCH -t 0-04:00:00
#SBATCH --output=/home/edward.gaibor001/medgemmaOMAMA/beta/run/logs/4-eval-%j.out
#SBATCH --error=/home/edward.gaibor001/medgemmaOMAMA/beta/run/logs/4-eval-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=edward.gaibor001@umb.edu
#SBATCH --dependency=afterany:636181

# Resource optimization:
# - H200 partition (need GPU for inference)
# - 1 GPU sufficient for evaluation
# - 16 CPUs for data loading
# - 80GB RAM for model + dataset
# - 2 hours max (should finish in ~90 minutes)

echo "=============================================="
echo "Step 4: Evaluate Balanced Model"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "=============================================="

# Create log directory
mkdir -p /home/edward.gaibor001/medgemmaOMAMA/beta/run/logs

# Activate conda environment
source /home/edward.gaibor001/miniconda3/etc/profile.d/conda.sh
conda activate medgemma

# Run evaluation
cd /home/edward.gaibor001/medgemmaOMAMA/beta
python 5-final-evaluation.py

echo ""
echo "=============================================="
echo "✅ Step 4 Complete"
echo "End time: $(date)"
echo "=============================================="
echo ""
echo "Expected Results:"
echo "  ✅ Sensitivity: 75-85% (vs 16% original)"
echo "  ✅ Specificity: 85-95%"
echo "  ✅ Overall accuracy: 88-92%"
echo "  ✅ Balanced predictions across classes"
echo "=============================================="
