#!/usr/bin/env python3
"""
Adapter Architecture Visualizer

This script demonstrates how LoRA adapters work at a technical level,
showing the weight structure and how adapters modify the base model.
"""

import os
import torch
from transformers import AutoModelForImageTextToText
from peft import PeftModel, LoraConfig

# Paths
BASE_MODEL_ID = "google/medgemma-4b-it"
ADAPTER_PATH = "/hpcstor6/scratch01/e/edward.gaibor001/medgemma_runs/omama256_balanced"
HF_CACHE = "/hpcstor6/scratch01/e/edward.gaibor001/.hf_cache"

os.environ.setdefault("HF_HOME", HF_CACHE)

def count_parameters(model):
    """Count trainable and total parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def format_number(num):
    """Format large numbers for readability"""
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return str(num)

print("\n" + "="*80)
print(" LoRA ADAPTER ARCHITECTURE VISUALIZER")
print("="*80 + "\n")

# Load base model
print("Loading base model...")
base_model = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
print("✓ Loaded\n")

# Count base model parameters
total_base, _ = count_parameters(base_model)
print("="*80)
print(" BASE MODEL STRUCTURE")
print("="*80)
print(f"\n📊 Total Parameters: {format_number(total_base)} ({total_base:,})")
print(f"💾 Approximate Size: {total_base * 2 / (1024**3):.2f} GB (bfloat16)")
print("\n🔒 Status: All parameters are FROZEN during LoRA training")
print("   → Original capabilities preserved!")
print()

# Show some layer names
print("📋 Sample Layer Names:")
layer_count = 0
for name, param in base_model.named_parameters():
    if layer_count < 10:
        print(f"   {name}: {param.shape}")
        layer_count += 1
    else:
        break
print(f"   ... ({len(list(base_model.named_parameters()))} total layers)\n")

# Load model with adapter
print("="*80)
print(" LOADING LORA ADAPTER")
print("="*80 + "\n")

print(f"Adapter Path: {ADAPTER_PATH}")
print("Loading adapter on top of base model...\n")

adapter_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
print("✓ Adapter loaded\n")

# Count parameters
total_with_adapter, trainable_adapter = count_parameters(adapter_model)

print("="*80)
print(" ADAPTER STRUCTURE")
print("="*80)
print(f"\n📊 Total Parameters: {format_number(total_with_adapter)} ({total_with_adapter:,})")
print(f"🎯 Trainable (LoRA): {format_number(trainable_adapter)} ({trainable_adapter:,})")
print(f"🔒 Frozen (Base):    {format_number(total_with_adapter - trainable_adapter)}")
print()
print(f"💡 Adapter Size: Only {trainable_adapter/total_base*100:.3f}% of base model!")
print(f"💾 Adapter Storage: ~{trainable_adapter * 2 / (1024**2):.1f} MB")
print()

# Show adapter layers
print("📋 Adapter Layers Added:")
adapter_layers = []
for name, param in adapter_model.named_parameters():
    if "lora" in name.lower() and param.requires_grad:
        adapter_layers.append((name, param.shape))

print(f"\n   Found {len(adapter_layers)} LoRA adapter layers:\n")
for name, shape in adapter_layers[:10]:
    print(f"   ✨ {name}: {shape}")
if len(adapter_layers) > 10:
    print(f"   ... ({len(adapter_layers) - 10} more)\n")

# Explain LoRA math
print("\n" + "="*80)
print(" HOW LORA WORKS - THE MATH")
print("="*80)
print("""
Original Layer (Frozen):
   output = W × input
   where W is a large weight matrix (e.g., 4096 × 4096 = 16.7M parameters)

LoRA Adapter (Trainable):
   output = W × input + (B × A) × input
                        └─────┬─────┘
                          LoRA adapter
   where:
   • A is a small matrix: 4096 × r (r = rank, typically 8-64)
   • B is a small matrix: r × 4096
   • B × A approximates the weight update

Example with rank=16:
   Original W: 4096 × 4096 = 16,777,216 parameters
   LoRA (A+B): (4096×16) + (16×4096) = 131,072 parameters
   Reduction: 99.2% fewer parameters! 🎉

Why This Works:
   • Most weight updates are "low-rank" (can be approximated by smaller matrices)
   • We only train A and B, keeping W frozen
   • At inference: output = W × input + α × (B × A) × input
     where α is a scaling factor (lora_alpha / lora_rank)
""")

print("="*80)
print(" ADAPTER CONFIGURATION")
print("="*80)
print("\nYour LoRA config (from training):")
print("""
   • Rank (r): 16
   • Alpha: 16
   • Dropout: 0.05
   • Target modules: all-linear
   • Task type: CAUSAL_LM
   • Modules saved: lm_head, embed_tokens
""")

print("\n💡 What this means:")
print("   • Rank 16 = good balance of capacity vs. efficiency")
print("   • Alpha 16 = scaling factor for adapter contribution")
print("   • All-linear = applies LoRA to all linear layers")
print("   • lm_head/embed_tokens also fine-tuned for classification\n")

# Memory comparison
print("="*80)
print(" MEMORY EFFICIENCY COMPARISON")
print("="*80)
print("\n❌ Fine-tuning entire model:")
print(f"   • All {format_number(total_base)} parameters trainable")
print(f"   • Optimizer states: 3× model size = {total_base * 6 / (1024**3):.1f} GB")
print(f"   • Gradients: 1× model size = {total_base * 2 / (1024**3):.1f} GB")
print(f"   • Total GPU memory: ~{total_base * 8 / (1024**3):.1f} GB")
print()
print("✅ LoRA fine-tuning:")
print(f"   • Only {format_number(trainable_adapter)} parameters trainable")
print(f"   • Optimizer states: 3× adapter = {trainable_adapter * 6 / (1024**3):.2f} GB")
print(f"   • Gradients: 1× adapter = {trainable_adapter * 2 / (1024**3):.2f} GB")
print(f"   • Total GPU memory: ~{(total_base * 2 + trainable_adapter * 8) / (1024**3):.1f} GB")
print(f"   • Savings: ~{(1 - (total_base * 2 + trainable_adapter * 8)/(total_base * 8))*100:.1f}%")
print()

# Deployment comparison
print("="*80)
print(" DEPLOYMENT SCENARIOS")
print("="*80)
print("\n🏥 Scenario 1: Single Task")
print("   • Base model: 8 GB")
print("   • 1 Adapter: 10 MB")
print("   • Total: ~8 GB")
print()
print("🏥 Scenario 2: Multiple Tasks (Your Use Case!)")
print("   • Base model: 8 GB")
print("   • Classification adapter: 10 MB")
print("   • Explanation adapter: 10 MB")
print("   • Report generation adapter: 10 MB")
print("   • Total: ~8.03 GB")
print()
print("   Compare to fine-tuning 3 separate models: 24 GB!")
print("   💾 Space savings: 66%")
print()

print("="*80)
print(" SWITCHING BETWEEN MODES")
print("="*80)
print("""
# Load base model once
base = AutoModelForImageTextToText.from_pretrained("medgemma-4b-it")

# Mode 1: Classification
classifier = PeftModel.from_pretrained(base, "classification_adapter")
output = classifier.generate(...)  # "B: Cancer"
classifier.unload()  # Remove adapter

# Mode 2: Conversational (base model restored)
output = base.generate(...)  # Detailed explanation

# Mode 3: Load different adapter
explainer = PeftModel.from_pretrained(base, "explanation_adapter")
output = explainer.generate(...)  # Structured findings
explainer.unload()

⚡ Fast switching: Only loading/unloading ~10MB adapter!
""")

print("="*80)
print(" SUMMARY")
print("="*80)
print(f"""
✨ Your LoRA Adapter Stats:
   • Base model size: {format_number(total_base)} parameters
   • Adapter size: {format_number(trainable_adapter)} parameters
   • Ratio: {trainable_adapter/total_base*100:.3f}%
   • Training time: ~{trainable_adapter/total_base:.1%} of full fine-tuning
   • Storage: Only {trainable_adapter * 2 / (1024**2):.1f} MB per adapter

🎯 Key Benefits:
   1. Preserves base model capabilities (conversational)
   2. Adds specialized knowledge (classification)
   3. Memory efficient (99% smaller than full model)
   4. Fast training (only updates {trainable_adapter/total_base:.1%} of parameters)
   5. Modular (train multiple adapters for different tasks)

💡 Your Hierarchy:
   Base Model (Conversational) ← 4B params
        ↓
   + Classification Adapter ← {format_number(trainable_adapter)} params
        ↓
   = Specialized Classifier that can still access base knowledge!
""")
print("="*80 + "\n")
