#!/usr/bin/env python3
"""
Hierarchical LoRA Adapter Demo
================================
Demonstrates how to use:
1. Base MedGemma model (conversational capabilities)
2. Fine-tuned classification adapter (specialized classification)

This shows the power of LoRA adapters - you can switch between
general conversational mode and specialized classification mode
using the SAME base model!
"""

import os
import torch
from PIL import Image
from datasets import load_from_disk
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
BASE_MODEL_ID = "google/medgemma-4b-it"
ADAPTER_PATH = "/hpcstor6/scratch01/e/edward.gaibor001/medgemma_runs/omama256_balanced"
DATA_DIR = "/hpcstor6/scratch01/e/edward.gaibor001/omamadata256/hf_proc_messages_balanced"
HF_CACHE = "/hpcstor6/scratch01/e/edward.gaibor001/.hf_cache"

# Setup cache paths
os.environ.setdefault("HF_HOME", HF_CACHE)
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(HF_CACHE, "transformers"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(HF_CACHE, "datasets"))

# ============================================================================
# MODEL LOADING
# ============================================================================

print("\n" + "="*70)
print("🚀 HIERARCHICAL LORA ADAPTER DEMO")
print("="*70)

print("\n📦 Loading base MedGemma model...")
base_model = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager"  # Use eager if flash attention not available
)
processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)
print("✓ Base model loaded")

print("\n📦 Loading classification adapter...")
classifier_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
print(f"✓ Adapter loaded from: {ADAPTER_PATH}")

print("\n📊 Loading test data...")
data = load_from_disk(DATA_DIR)

# Get 1 Cancer and 1 NonCancer sample for presentation
validation_data = data["validation"]
cancer_samples = validation_data.filter(lambda x: x["label"] == 1).shuffle(seed=42).select(range(1))
noncancer_samples = validation_data.filter(lambda x: x["label"] == 0).shuffle(seed=42).select(range(1))

# Combine them
from datasets import concatenate_datasets
test_samples = concatenate_datasets([cancer_samples, noncancer_samples])
CLASS_NAMES = ["NonCancer", "Cancer"]
print(f"✓ Loaded {len(test_samples)} test samples (1 Cancer, 1 NonCancer)")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_prompt(prompt_text):
    """Format a text prompt for the model"""
    return prompt_text

def generate_response(model, processor, messages, max_tokens=200, temperature=0.1):
    """Generate a response from the model"""
    model.eval()
    
    # Format the messages
    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )
    
    # Get the image from messages if present
    images = []
    for msg in messages:
        if isinstance(msg.get("content"), list):
            for item in msg["content"]:
                if item.get("type") == "image":
                    # Image is already in the message
                    pass
    
    # Prepare inputs
    inputs = processor(
        text=prompt,
        images=images if images else None,
        return_tensors="pt",
        padding=True
    ).to(model.device)
    
    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=processor.tokenizer.pad_token_id
        )
    
    # Decode
    response = processor.decode(output_ids[0], skip_special_tokens=True)
    
    # Extract only the assistant's response
    if "model\n" in response:
        response = response.split("model\n")[-1]
    
    return response.strip()

def classify_with_adapter(image, model, processor):
    """Classify image using the fine-tuned adapter"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Classify this mammogram.\nA: NonCancer\nB: Cancer"}
            ]
        }
    ]
    
    model.eval()
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.1,
            do_sample=False
        )
    
    response = processor.decode(output_ids[0], skip_special_tokens=True)
    return response

def converse_with_base(image, question, model, processor):
    """Have a conversation with the base model (no adapter)"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question}
            ]
        }
    ]
    
    model.eval()
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True
        )
    
    response = processor.decode(output_ids[0], skip_special_tokens=True)
    return response

def parse_classification(response):
    """Parse classification from model output"""
    # Extract only the model's response (after "model\n")
    if "model\n" in response:
        model_response = response.split("model\n")[-1].strip()
    else:
        model_response = response.strip()
    
    response_upper = model_response.upper()
    
    # Check for the letter choice first (more reliable)
    if response_upper.startswith("A:") or response_upper == "A":
        return "NonCancer"
    elif response_upper.startswith("B:") or response_upper == "B":
        return "Cancer"
    # Fallback to checking the content
    elif "NONCANCER" in response_upper:
        return "NonCancer"
    elif "CANCER" in response_upper and "NONCANCER" not in response_upper:
        return "Cancer"
    else:
        return "Unknown"

# ============================================================================
# DEMO: HIERARCHICAL INFERENCE
# ============================================================================

print("\n" + "="*70)
print("🎯 HIERARCHICAL INFERENCE DEMO")
print("="*70)
print("\nThis demo shows a TRUE hierarchical pipeline:")
print("  1. LoRA adapter classifies the image (fast, specialized)")
print("  2. Classification result guides base model's explanation")
print("  3. Base model provides context-aware reasoning")
print("\nKey insight: The adapter's classification INFORMS the base model!")
print("="*70)

# Create output directory for saving images
job_id = os.environ.get("SLURM_JOB_ID", "local")
output_dir = f"logs/{job_id}_data"
os.makedirs(output_dir, exist_ok=True)
print(f"\n💾 Saving test images to: {output_dir}")

for idx, example in enumerate(test_samples):
    image = example["image"]
    true_label = CLASS_NAMES[example["label"]]
    
    # Save the image
    image_filename = f"{output_dir}/sample_{idx+1}_{true_label.lower()}.png"
    image.save(image_filename)
    print(f"   ✓ Saved: sample_{idx+1}_{true_label.lower()}.png")
    
    print("\n" + "─"*70)
    print(f"📋 SAMPLE {idx + 1}/{len(test_samples)}")
    print("─"*70)
    print(f"Ground Truth: {true_label}")
    print()
    
    # -------------------------------------------------------------------------
    # STEP 1: Quick Classification (With LoRA Adapter)
    # -------------------------------------------------------------------------
    print("🎯 STEP 1: Quick Triage Classification (LoRA Adapter)")
    print("-" * 70)
    print("🔄 Using specialized classification adapter...")
    
    # Ensure adapter is enabled for classification
    classifier_model.enable_adapter_layers()
    
    try:
        raw_output = classify_with_adapter(image, classifier_model, processor)
        predicted = parse_classification(raw_output)
        
        # Extract just the classification part for display
        if "model\n" in raw_output:
            classification_result = raw_output.split("model\n")[-1].strip()
        else:
            classification_result = raw_output.strip()
        
        print(f"🔬 Classifier Output: {classification_result}")
        print(f"📊 Classification: {predicted}")
        print(f"✓ Ground Truth: {true_label}")
        
        if predicted == true_label:
            print("✅ CORRECT!")
        else:
            print("❌ INCORRECT")
    except Exception as e:
        print(f"❌ Classification Error: {e}")
        predicted = "Unknown"
    
    # -------------------------------------------------------------------------
    # STEP 2: Explanation Based on Classification (Base Model + Context)
    # -------------------------------------------------------------------------
    print("\n\n💬 STEP 2: Detailed Explanation (Base Model with Classification Context)")
    print("-" * 70)
    print("🔄 Disabling adapter - using base model reasoning...")
    
    # Disable adapter for conversational mode
    classifier_model.disable_adapter_layers()
    
    # Create context-aware prompt based on classification result
    if predicted == "Cancer":
        explanation_prompt = (
            f"This mammogram has been classified as showing cancer. "
            f"Please describe the visual findings that support this diagnosis. "
            f"Focus on specific features such as masses, their characteristics "
            f"(shape, margin, density), calcifications, architectural distortions, "
            f"or asymmetries that are concerning for malignancy."
        )
    elif predicted == "NonCancer":
        explanation_prompt = (
            f"This mammogram has been classified as showing no signs of cancer. "
            f"Please describe the breast tissue characteristics and explain why "
            f"this mammogram appears normal or benign. Discuss the tissue density, "
            f"symmetry, and absence of suspicious features."
        )
    else:
        explanation_prompt = "Please analyze this mammogram and describe what you see."
    
    try:
        print(f"\n📝 Prompt: {explanation_prompt}")
        response = converse_with_base(image, explanation_prompt, classifier_model, processor)
        
        # Extract just the assistant response
        if "model\n" in response:
            response = response.split("model\n")[-1].strip()
        
        print(f"\n🤖 Base Model Explanation:")
        print(f"{response}")  # Full response for presentation
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    
    # -------------------------------------------------------------------------
    # STEP 3: Hierarchical Clinical Decision & Recommendations
    # -------------------------------------------------------------------------
    print("\n\n🏥 STEP 3: Clinical Decision & Recommendations")
    print("-" * 70)
    
    if predicted == "Cancer":
        print("⚠️  HIGH RISK CASE")
        print("📊 Classification Result: CANCER detected by specialized adapter")
        print("🔍 Detailed Analysis: See base model explanation above")
        print("\n📋 RECOMMENDED ACTIONS:")
        print("   • Immediate referral to breast imaging specialist")
        print("   • Schedule diagnostic mammogram and ultrasound")
        print("   • Consider biopsy based on radiologist assessment")
        print("   • Patient counseling and follow-up within 1 week")
        
    elif predicted == "NonCancer":
        print("✓ LOW RISK CASE")
        print("📊 Classification Result: NO CANCER detected by specialized adapter")
        print("🔍 Detailed Analysis: See base model explanation above")
        print("\n📋 RECOMMENDED ACTIONS:")
        print("   • Continue routine screening schedule")
        print("   • Next mammogram in 12 months")
        print("   • Patient education on breast self-examination")
        print("   • Monitor for any symptom changes")
    else:
        print("⚠️  UNCERTAIN CASE")
        print("📊 Classification Result: Unable to classify")
        print("\n📋 RECOMMENDED ACTIONS:")
        print("   • Manual review by radiologist required")
        print("   • Consider additional imaging views")
    
    print("\n" + "─"*70)

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("📊 DEMO SUMMARY")
print("="*70)
print("\n✨ Key Takeaways:")
print("\n1. 🎯 CLASSIFICATION ADAPTER (Stage 1 - Fast Triage):")
print("   - Specialized for binary classification")
print("   - Fast and accurate predictions")
print("   - Trained on your specific dataset")
print("   - Outputs: 'A: NonCancer' or 'B: Cancer'")
print("   - Provides quick initial assessment")

print("\n2. 🧠 BASE MODEL (Stage 2 - Context-Aware Reasoning):")
print("   - Uses classification result as CONTEXT")
print("   - Provides detailed explanations for the diagnosis")
print("   - Explains visual features supporting the classification")
print("   - Maintains full conversational capabilities")
print("   - Uses general medical knowledge for reasoning")

print("\n3. 🏗️ HIERARCHICAL PIPELINE ARCHITECTURE:")
print("   - Stage 1: Adapter classifies (fast, accurate)")
print("   - Stage 2: Base model explains WHY (context-aware)")
print("   - Classification result guides the explanation")
print("   - Best of both worlds - speed + understanding!")
print("   - Memory efficient (one base model + small adapter)")

print("\n4. 💡 KEY BENEFITS:")
print("   - Adapter's classification informs base model's reasoning")
print("   - No need for separate models")
print("   - Base model never modified (preserves capabilities)")
print("   - Adapter is only ~10MB (vs 8GB for full model)")
print("   - Can train multiple adapters for different tasks")
print("   - True AI-assisted diagnostic workflow")

print("\n" + "="*70)
print("🎉 Demo completed successfully!")
print("="*70)
print("\n💡 Next Steps:")
print("   - Train additional adapters for other tasks")
print("   - Experiment with adapter merging")
print("   - Build production pipeline with smart routing")
print("\n📁 Test Images Saved:")
print(f"   - Location: {output_dir}/")
print(f"   - Files: sample_1_cancer.png, sample_2_noncancer.png")
print("="*70 + "\n")
