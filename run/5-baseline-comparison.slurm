#!/bin/bash
#SBATCH --job-name=baseline-prompts
#SBATCH -A cs_daniel.haehn
#SBATCH -p pomplun
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200
#SBATCH --mem=200G
#SBATCH -w chimera21
#SBATCH -t 2-12:00:00
#SBATCH --output=/home/edward.gaibor001/medgemmaOMAMA/beta/run/logs/5-baseline-%j.out
#SBATCH --error=/home/edward.gaibor001/medgemmaOMAMA/beta/run/logs/5-baseline-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=edward.gaibor001@umb.edu
#SBATCH --dependency=afterany:636181

# Resource optimization:
# - H200 partition (need GPU for inference)
# - 1 GPU sufficient for evaluation
# - 16 CPUs for data loading
# - 80GB RAM for model + dataset
# - 12 hours max (7 prompts × ~90 min each ≈ 10.5 hours)

echo "=============================================="
echo "Step 5: Baseline Prompt Comparison"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "=============================================="
echo ""
echo "Testing 7 different prompts:"
echo "  1. original"
echo "  2. constrained"
echo "  3. few_shot"
echo "  4. role_based"
echo "  5. ultra_strict"
echo "  6. medical_context"
echo "  7. binary_choice"
echo "=============================================="

# Create log directory
mkdir -p /home/edward.gaibor001/medgemmaOMAMA/beta/run/logs

# Activate conda environment
source /home/edward.gaibor001/miniconda3/etc/profile.d/conda.sh
conda activate medgemma

# Run baseline comparison
cd /home/edward.gaibor001/medgemmaOMAMA/beta
python 5-baseline-improved-prompt.py

echo ""
echo "=============================================="
echo "✅ Step 5 Complete"
echo "End time: $(date)"
echo "=============================================="
echo ""
echo "Results saved in:"
echo "  /hpcstor6/scratch01/e/edward.gaibor001/medgemma_runs/baseline_prompt_comparison/"
echo "=============================================="
