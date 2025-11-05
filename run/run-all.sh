#!/bin/bash
# Master script to run entire balanced training pipeline with dependencies

set -e  # Exit on error

echo "=============================================="
echo "🚀 BALANCED MEDGEMMA TRAINING PIPELINE"
echo "=============================================="
echo "This script will submit all 5 steps with proper dependencies"
echo ""
echo "Expected total time: ~6-8 hours"
echo "  Step 1: Dataset creation     ~30 min"
echo "  Step 2: Data processing      ~15 min"
echo "  Step 3: Model training       ~3-4 hours"
echo "  Step 4: Evaluation           ~90 min"
echo "  Step 5: Baseline comparison  ~10.5 hours (optional)"
echo "  Step 6: Merge & Push to HF   ~30 min (optional)"
echo "=============================================="
echo ""

# Create logs directory
mkdir -p logs

# Submit step 1
echo "📋 Submitting Step 1: Create balanced dataset..."
JOB1=$(sbatch --parsable 1-create-dataset.slurm)
echo "  ✓ Job ID: $JOB1"

# Submit step 2 (depends on step 1)
echo "📋 Submitting Step 2: Process data (after step 1)..."
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 2-process-data.slurm)
echo "  ✓ Job ID: $JOB2"

# Submit step 3 (depends on step 2)
echo "📋 Submitting Step 3: Train model (after step 2)..."
JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 3-train-balanced.slurm)
echo "  ✓ Job ID: $JOB3"

# Submit step 4 (depends on step 3)
echo "📋 Submitting Step 4: Evaluate model (after step 3)..."
JOB4=$(sbatch --parsable --dependency=afterok:$JOB3 4-evaluate-balanced.slurm)
echo "  ✓ Job ID: $JOB4"

# Ask about step 5
echo ""
read -p "❓ Run Step 5 (baseline comparison, ~10.5 hours)? [y/N]: " -n 1 -r
echo ""
RUN_BASELINE=$REPLY

# Ask about step 6
echo ""
read -p "❓ Run Step 6 (merge & push to Hugging Face)? [y/N]: " -n 1 -r
echo ""
RUN_PUSH=$REPLY

# Submit step 5 if requested
if [[ $RUN_BASELINE =~ ^[Yy]$ ]]; then
    echo "📋 Submitting Step 5: Baseline comparison (optional)..."
    JOB5=$(sbatch --parsable 5-baseline-comparison.slurm)
    echo "  ✓ Job ID: $JOB5"
fi

# Submit step 6 if requested (depends on step 4)
if [[ $RUN_PUSH =~ ^[Yy]$ ]]; then
    echo "📋 Submitting Step 6: Merge & push to HF (after step 4)..."
    JOB6=$(sbatch --parsable --dependency=afterok:$JOB4 6-merge-push.slurm)
    echo "  ✓ Job ID: $JOB6"
fi

echo ""
echo "=============================================="
echo "✅ JOBS SUBMITTED"
echo "=============================================="
echo "Job IDs:"
echo "  Step 1 (dataset):   $JOB1"
echo "  Step 2 (process):   $JOB2 (after $JOB1)"
echo "  Step 3 (train):     $JOB3 (after $JOB2)"
echo "  Step 4 (evaluate):  $JOB4 (after $JOB3)"
if [[ $RUN_BASELINE =~ ^[Yy]$ ]]; then
    echo "  Step 5 (baseline):  $JOB5 (independent)"
fi
if [[ $RUN_PUSH =~ ^[Yy]$ ]]; then
    echo "  Step 6 (push HF):   $JOB6 (after $JOB4)"
fi

echo ""
echo "=============================================="
echo "📊 MONITOR PROGRESS"
echo "=============================================="
echo "Queue status:"
echo "  squeue -u edward.gaibor001"
echo ""
echo "Live logs:"
echo "  tail -f logs/1-dataset-$JOB1.out    # Step 1"
echo "  tail -f logs/2-process-$JOB2.out    # Step 2"
echo "  tail -f logs/3-train-$JOB3.out      # Step 3"
echo "  tail -f logs/4-eval-$JOB4.out       # Step 4"
if [[ $RUN_BASELINE =~ ^[Yy]$ ]]; then
    echo "  tail -f logs/5-baseline-$JOB5.out   # Step 5"
fi
if [[ $RUN_PUSH =~ ^[Yy]$ ]]; then
    echo "  tail -f logs/6-merge-push-$JOB6.out # Step 6"
fi
echo ""
echo "=============================================="
echo "🎯 EXPECTED RESULTS"
echo "=============================================="
echo "After Step 4 completes (~6-8 hours):"
echo "  ✅ Sensitivity: 75-85% (vs 16% original)"
echo "  ✅ Specificity: 85-95%"
echo "  ✅ Balanced cancer detection"
echo "  ✅ Clinically deployable model"
echo "=============================================="
