#!/usr/bin/env python3
"""
Baseline evaluation with IMPROVED PROMPTS to get better zero-shot performance.

Tests multiple prompt strategies:
1. Original (generic classification prompt)
2. Constrained (forces format compliance)
3. Few-shot (provides examples)
4. Role-based (radiologist persona)
5. Ultra-strict (minimal instructions)

Goal: Find the best prompt for fair baseline comparison.
"""
import os
import json
import re
from datetime import datetime
from typing import List, Tuple, Dict
from collections import Counter
import torch
from datasets import load_from_disk, ClassLabel
import evaluate
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    GenerationConfig,
)

# ---------- paths ----------
DATA_DIR = "/hpcstor6/scratch01/e/edward.gaibor001/omamadata256/hf_proc_messages_balanced"
BASE_ID  = "google/medgemma-4b-it"   # baseline (gated—needs token/login)
RESULTS_DIR = "/hpcstor6/scratch01/e/edward.gaibor001/medgemma_runs"
DEBUG_LOG_FILE = os.path.join(RESULTS_DIR, "baseline_prompt_evaluation_debug.jsonl")

# keep HF caches off home
os.environ.setdefault("HF_DATASETS_CACHE", "/hpcstor6/scratch01/e/edward.gaibor001/.hf_cache/datasets")
os.environ.setdefault("TRANSFORMERS_CACHE", "/hpcstor6/scratch01/e/edward.gaibor001/.hf_cache/transformers")

# ---------- load data ----------
data = load_from_disk(DATA_DIR)
test_data = data["validation"]

CLASS_NAMES = test_data.features["label"].names
assert CLASS_NAMES == ["NonCancer", "Cancer"], f"Got {CLASS_NAMES}"
LABEL_FEATURE: ClassLabel = test_data.features["label"]

# Verify label distribution
print("\n" + "="*60)
print("📊 DATASET DISTRIBUTION")
print("="*60)
label_counts = Counter(test_data["label"])
for label_idx, count in sorted(label_counts.items()):
    label_name = CLASS_NAMES[label_idx]
    pct = count / len(test_data) * 100
    print(f"  {label_name}: {count:,} samples ({pct:.1f}%)")
print(f"  Total: {len(test_data):,} samples")
print("="*60 + "\n")

# ---------- MULTIPLE PROMPT STRATEGIES ----------
PROMPTS = {
    "original": """Classify this mammogram.
A: NonCancer
B: Cancer""",

    "constrained": """You are an expert radiologist analyzing a mammogram.

Classify this image as ONE of the following:
A: NonCancer
B: Cancer

Respond with ONLY the letter and label (e.g., "A: NonCancer" or "B: Cancer").
Do NOT add explanations or additional text.

Your classification:""",

    "few_shot": """Classify this mammogram image.

Examples of correct responses:
- If normal tissue: "A: NonCancer"
- If suspicious mass detected: "B: Cancer"
- If calcifications concerning: "B: Cancer"
- If dense but normal: "A: NonCancer"

Your classification for this image:""",

    "role_based": """You are an expert radiologist with 20 years of experience in mammography screening.

Task: Classify this mammogram as NonCancer or Cancer.

Instructions:
1. Analyze the image for masses, calcifications, architectural distortion
2. Respond with EXACTLY: "A: NonCancer" or "B: Cancer"
3. Do not include explanations

Classification:""",

    "ultra_strict": """Classify this mammogram.

Answer format (choose one):
A: NonCancer
B: Cancer

Answer:""",

    "medical_context": """Medical context: You are reviewing a screening mammogram.

Question: Does this mammogram show signs of malignancy?

Options:
A: NonCancer (benign/normal findings)
B: Cancer (malignant/suspicious findings)

Your answer (format "X: Label"):""",

    "binary_choice": """Analyze this mammogram and select the correct classification:

[ ] A: NonCancer
[ ] B: Cancer

Selected (format "X: Label"):"""
}

# ---------- metrics ----------
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

REFERENCES = test_data["label"]

def compute_metrics(preds: List[int], prompt_name: str, verbose=True) -> Dict:
    """Compute comprehensive metrics for medical AI evaluation."""
    valid_preds = [p if p in (0,1) else -1 for p in preds]
    valid_mask = np.array(valid_preds) != -1
    filtered_preds = np.array(valid_preds)[valid_mask]
    filtered_refs = np.array(REFERENCES)[valid_mask]
    
    if len(filtered_preds) == 0:
        print(f"⚠️  WARNING: All predictions invalid for prompt '{prompt_name}'!")
        return {"accuracy": 0.0, "sensitivity": 0.0, "specificity": 0.0}
    
    acc = accuracy_metric.compute(predictions=filtered_preds, references=filtered_refs)
    f1  = f1_metric.compute(predictions=filtered_preds, references=filtered_refs, average="weighted")
    
    cm = confusion_matrix(filtered_refs, filtered_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        filtered_refs, filtered_preds, labels=[0, 1], zero_division=0
    )
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    n_invalid = len(preds) - len(filtered_preds)
    invalid_rate = n_invalid / len(preds) if len(preds) > 0 else 0
    
    metrics = {
        **acc, 
        **f1,
        "precision_noncancer": float(precision[0]),
        "recall_noncancer": float(recall[0]),
        "f1_noncancer": float(f1_per_class[0]),
        "support_noncancer": int(support[0]),
        
        "precision_cancer": float(precision[1]),
        "recall_cancer": float(recall[1]),
        "f1_cancer": float(f1_per_class[1]),
        "support_cancer": int(support[1]),
        
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv": float(ppv),
        "npv": float(npv),
        
        "false_positive_rate": float(false_positive_rate),
        "false_negative_rate": float(false_negative_rate),
        
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        
        "n_invalid_predictions": int(n_invalid),
        "invalid_prediction_rate": float(invalid_rate),
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"PROMPT: {prompt_name.upper()}")
        print(f"{'='*60}")
        print(f"\n📊 Performance:")
        print(f"  Accuracy:    {metrics['accuracy']:.2%}")
        print(f"  Sensitivity: {sensitivity:.2%}  ← Cancer detection")
        print(f"  Specificity: {specificity:.2%}  ← NonCancer detection")
        print(f"  F1-Score:    {metrics['f1']:.4f}")
        print(f"  Invalid:     {n_invalid}/{len(preds)} ({invalid_rate:.2%})")
        
        print(f"\n🎯 Confusion Matrix:")
        print(f"               NonCancer  Cancer")
        print(f"  Actual NonC    {tn:5d}    {fp:5d}")
        print(f"         Cancer  {fn:5d}    {tp:5d}")
        
        print(f"\n⚠️  Error Analysis:")
        print(f"  False Positive Rate: {false_positive_rate:.2%}  ({fp} false alarms)")
        print(f"  False Negative Rate: {false_negative_rate:.2%}  ({fn} missed cancers)")
        
        print(f"\n💡 Assessment:")
        if sensitivity > 0.85 and specificity > 0.85:
            print(f"  🌟 EXCELLENT: Both sensitivity and specificity >85%")
        elif sensitivity > 0.75 and specificity > 0.75:
            print(f"  ✅ GOOD: Balanced performance >75%")
        elif sensitivity > 0.60:
            print(f"  ✓  ACCEPTABLE: Sensitivity >60%")
        else:
            print(f"  ❌ POOR: Sensitivity <60% - not suitable for screening")
        print(f"{'='*60}\n")
    
    return metrics

# ---------- LENIENT PARSER WITH NEGATION HANDLING ----------
def postprocess_text_to_label_lenient(text: str, true_label: int = None, debug_log: list = None) -> Tuple[int, str]:
    """Lenient parser with negation handling."""
    raw_text = text
    t = text.strip().lower()
    
    # Priority 1: Exact format patterns
    if re.search(r'b\s*:?\s*cancer', t):
        if debug_log is not None:
            debug_log.append({
                "raw_output": raw_text, "normalized": t, "predicted": 1, "true_label": true_label,
                "reason": "matched_pattern_b_cancer", "correct": true_label == 1 if true_label is not None else None
            })
        return 1, "matched_pattern_b_cancer"
    
    if re.search(r'a\s*:?\s*noncancer', t):
        if debug_log is not None:
            debug_log.append({
                "raw_output": raw_text, "normalized": t, "predicted": 0, "true_label": true_label,
                "reason": "matched_pattern_a_noncancer", "correct": true_label == 0 if true_label is not None else None
            })
        return 0, "matched_pattern_a_noncancer"
    
    # Priority 2: Negation patterns (BEFORE generic cancer matching)
    negation_patterns = [
        r'non-?cancerous',
        r'no\s+(?:obvious\s+)?(?:signs?\s+of\s+)?cancer',
        r'not\s+cancer',
        r'no\s+(?:evidence|indication)\s+of\s+cancer',
        r'does\s+not\s+(?:appear|seem)\s+(?:to\s+(?:be|have)\s+)?cancer',
        r'doesn?\'?t\s+suggest\s+(?:malignancy|cancer)',
        r'negative\s+for\s+cancer',
        r'normal.*no.*cancer',
        r'absence\s+of.*(?:masses|cancer)',
        r'normal\s+finding',
        r'benign',
    ]
    
    for pattern in negation_patterns:
        if re.search(pattern, t):
            if debug_log is not None:
                debug_log.append({
                    "raw_output": raw_text, "normalized": t, "predicted": 0, "true_label": true_label,
                    "reason": f"negation: {pattern}", "correct": true_label == 0 if true_label is not None else None
                })
            return 0, f"negation: {pattern}"
    
    # Priority 3: NonCancer substring
    if "noncancer" in t:
        if debug_log is not None:
            debug_log.append({
                "raw_output": raw_text, "normalized": t, "predicted": 0, "true_label": true_label,
                "reason": "substring_noncancer", "correct": true_label == 0 if true_label is not None else None
            })
        return 0, "substring_noncancer"
    
    # Priority 4: Positive cancer indicators
    positive_patterns = [
        r'(?:suspicious|possible|probable|likely)\s+(?:\w+\s+){0,3}cancer',
        r'(?:suggests?|indicates?)\s+(?:\w+\s+){0,3}(?:malignancy|cancer)',
        r'classification\s+is\s+(?:\*\*)?b\s*:?\s*cancer',
        r'therefore.*b\s*:?\s*cancer',
    ]
    
    for pattern in positive_patterns:
        if re.search(pattern, t):
            if debug_log is not None:
                debug_log.append({
                    "raw_output": raw_text, "normalized": t, "predicted": 1, "true_label": true_label,
                    "reason": f"positive: {pattern}", "correct": true_label == 1 if true_label is not None else None
                })
            return 1, f"positive: {pattern}"
    
    # Priority 5: Generic cancer substring
    if "cancer" in t:
        if debug_log is not None:
            debug_log.append({
                "raw_output": raw_text, "normalized": t, "predicted": 1, "true_label": true_label,
                "reason": "substring_cancer", "correct": true_label == 1 if true_label is not None else None
            })
        return 1, "substring_cancer"
    
    # Priority 6: Letter patterns
    if re.search(r'\bb\b', t):
        if debug_log is not None:
            debug_log.append({
                "raw_output": raw_text, "normalized": t, "predicted": 1, "true_label": true_label,
                "reason": "matched_letter_b", "correct": true_label == 1 if true_label is not None else None
            })
        return 1, "matched_letter_b"
    
    if re.search(r'\ba\b', t):
        if debug_log is not None:
            debug_log.append({
                "raw_output": raw_text, "normalized": t, "predicted": 0, "true_label": true_label,
                "reason": "matched_letter_a", "correct": true_label == 0 if true_label is not None else None
            })
        return 0, "matched_letter_a"
    
    # Failed to parse
    if debug_log is not None:
        debug_log.append({
            "raw_output": raw_text, "normalized": t, "predicted": -1, "true_label": true_label,
            "reason": "unparseable", "correct": None
        })
    return -1, "unparseable"

# ---------- CUSTOM PROMPT BUILDER ----------
def build_prompts_with_custom_text(custom_prompt: str) -> List[str]:
    """Build prompts using custom text instead of the messages format."""
    prompts = []
    for ex in test_data:
        # Build a simple user message with custom prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": custom_prompt},
                ],
            },
        ]
        # Apply chat template
        prompt_text = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        prompts.append(prompt_text)
    return prompts

# ---------- batch predict ----------
def batch_predict(prompts, images, model, processor, *, batch_size=32, device="cuda", 
                  dtype=torch.bfloat16, debug_log=None, **gen_kwargs):
    """Batch prediction with debug logging."""
    preds = []
    raw_outputs = []
    
    for i in range(0, len(prompts), batch_size):
        texts = prompts[i: i+batch_size]
        imgs  = [[im] for im in images[i: i+batch_size]]
        enc = processor(text=texts, images=imgs, padding=True, return_tensors="pt").to(device)
        
        for k, v in enc.items():
            if torch.is_floating_point(v):
                enc[k] = v.to(dtype)

        plen = enc["input_ids"].shape[1]
        
        with torch.inference_mode():
            out = model.generate(
                **enc,
                max_new_tokens=40,
                do_sample=False,
                **gen_kwargs
            )
        
        for idx, seq in enumerate(out):
            ans = processor.decode(seq[plen:], skip_special_tokens=True)
            raw_outputs.append(ans)
            true_label = REFERENCES[i + idx]
            pred, reason = postprocess_text_to_label_lenient(ans, true_label, debug_log)
            preds.append(pred)
    
    return preds, raw_outputs

# ---------- Sample outputs ----------
def show_sample_outputs(raw_outputs, predictions, references, prompt_name, n_samples=3):
    """Show sample outputs for each category."""
    print(f"\n{'='*60}")
    print(f"SAMPLE OUTPUTS - {prompt_name.upper()}")
    print(f"{'='*60}")
    
    correct_nc = [(i, raw_outputs[i]) for i in range(len(raw_outputs)) 
                  if predictions[i] == references[i] == 0]
    correct_c = [(i, raw_outputs[i]) for i in range(len(raw_outputs)) 
                 if predictions[i] == references[i] == 1]
    
    print(f"\n✅ Correct NonCancer predictions ({len(correct_nc)} total):")
    for i, (idx, output) in enumerate(correct_nc[:n_samples]):
        print(f"  [{i+1}] {repr(output[:150])}")
    
    print(f"\n✅ Correct Cancer predictions ({len(correct_c)} total):")
    for i, (idx, output) in enumerate(correct_c[:n_samples]):
        print(f"  [{i+1}] {repr(output[:150])}")
    
    print(f"{'='*60}\n")

# ---------- load baseline model ----------
print("\n" + "="*60)
print("🔵 LOADING BASELINE MODEL")
print("="*60)
processor = AutoProcessor.from_pretrained(BASE_ID)
tok = processor.tokenizer
tok.padding_side = "left"

base = AutoModelForImageTextToText.from_pretrained(
    BASE_ID,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
).to("cuda").eval()

gen_cfg = GenerationConfig.from_pretrained(BASE_ID)
gen_cfg.update(
    do_sample=False,
    top_k=None,
    top_p=None,
    cache_implementation="dynamic",
)
base.generation_config = gen_cfg
base.config.pad_token_id = tok.pad_token_id
base.generation_config.pad_token_id = tok.pad_token_id

images = test_data["image"]
print(f"Model loaded. Eval set: {len(test_data)} samples")
print("="*60 + "\n")

# ---------- TEST ALL PROMPTS ----------
all_results = {}
all_debug_logs = {}

for prompt_name, prompt_text in PROMPTS.items():
    print(f"\n{'='*80}")
    print(f"🧪 TESTING PROMPT: {prompt_name.upper()}")
    print(f"{'='*80}")
    print(f"\nPrompt text:\n{'-'*60}\n{prompt_text}\n{'-'*60}\n")
    
    # Build prompts with this custom text
    prompts = build_prompts_with_custom_text(prompt_text)
    
    # Run inference
    debug_log = []
    preds, raw_outputs = batch_predict(
        prompts, images, base, processor,
        batch_size=32,
        debug_log=debug_log
    )
    
    # Compute metrics
    metrics = compute_metrics(preds, prompt_name, verbose=True)
    
    # Show sample outputs
    show_sample_outputs(raw_outputs, preds, REFERENCES, prompt_name, n_samples=3)
    
    # Store results
    all_results[prompt_name] = {
        "metrics": metrics,
        "prompt_text": prompt_text,
        "raw_outputs_sample": raw_outputs[:10]  # Save first 10 for inspection
    }
    all_debug_logs[prompt_name] = debug_log

# ---------- COMPARISON SUMMARY ----------
print("\n" + "="*80)
print("📊 PROMPT COMPARISON SUMMARY")
print("="*80)

# Sort by sensitivity (most important for cancer detection)
sorted_prompts = sorted(
    all_results.items(),
    key=lambda x: x[1]["metrics"]["sensitivity"],
    reverse=True
)

print(f"\n{'Prompt':<20} {'Accuracy':<12} {'Sensitivity':<12} {'Specificity':<12} {'Invalid%':<10}")
print("-" * 80)
for prompt_name, result in sorted_prompts:
    m = result["metrics"]
    print(f"{prompt_name:<20} {m['accuracy']:>10.2%}  {m['sensitivity']:>10.2%}  "
          f"{m['specificity']:>10.2%}  {m['invalid_prediction_rate']:>8.2%}")

# Best prompt
best_prompt = sorted_prompts[0][0]
best_metrics = sorted_prompts[0][1]["metrics"]

print(f"\n{'='*80}")
print(f"🏆 BEST PROMPT: {best_prompt.upper()}")
print(f"{'='*80}")
print(f"  Accuracy:    {best_metrics['accuracy']:.2%}")
print(f"  Sensitivity: {best_metrics['sensitivity']:.2%}  ← Cancer detection")
print(f"  Specificity: {best_metrics['specificity']:.2%}  ← NonCancer detection")
print(f"  F1-Score:    {best_metrics['f1']:.4f}")
print(f"  Invalid:     {best_metrics['invalid_prediction_rate']:.2%}")

print(f"\n💡 Recommendation:")
if best_metrics['sensitivity'] > 0.85 and best_metrics['specificity'] > 0.85:
    print(f"  ✅ This prompt gives excellent baseline performance!")
    print(f"  ✅ Use this for fair comparison with fine-tuned model.")
elif best_metrics['sensitivity'] > 0.75:
    print(f"  ✓  Good performance - suitable for baseline comparison.")
else:
    print(f"  ⚠️  All prompts show suboptimal performance.")
    print(f"  ⚠️  Fine-tuning may still be necessary for clinical deployment.")

print(f"\n📝 Best prompt text:")
print(f"{'-'*80}")
print(sorted_prompts[0][1]["prompt_text"])
print(f"{'-'*80}\n")

# ---------- SAVE RESULTS ----------
results_file = os.path.join(
    RESULTS_DIR,
    f"baseline_prompt_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
)

results_data = {
    "timestamp": datetime.now().isoformat(),
    "model": BASE_ID,
    "n_samples": len(test_data),
    "class_distribution": {CLASS_NAMES[k]: int(v) for k, v in label_counts.items()},
    "prompts_tested": list(PROMPTS.keys()),
    "results": {
        name: {
            "prompt_text": result["prompt_text"],
            "metrics": result["metrics"]
        }
        for name, result in all_results.items()
    },
    "best_prompt": {
        "name": best_prompt,
        "text": sorted_prompts[0][1]["prompt_text"],
        "metrics": best_metrics
    }
}

with open(results_file, 'w') as f:
    json.dump(results_data, f, indent=2)

print(f"📁 Results saved to: {results_file}")

# Save debug logs
for prompt_name, debug_log in all_debug_logs.items():
    debug_file = DEBUG_LOG_FILE.replace(".jsonl", f"_{prompt_name}.jsonl")
    with open(debug_file, 'w') as f:
        for entry in debug_log:
            f.write(json.dumps(entry) + '\n')
    print(f"   Debug log: {debug_file}")

print(f"\n✅ Baseline prompt evaluation complete!\n")
