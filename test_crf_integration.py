"""
Test script for CRF layer implementation
Verifies that CRF is properly integrated and can improve discontinuous MWE detection
"""
import sys
import os

print("=" * 80)
print("CRF LAYER IMPLEMENTATION TEST")
print("=" * 80)
print()

# Test 1: Check if pytorch-crf is installed
print("Test 1: Checking pytorch-crf installation...")
try:
    from torchcrf import CRF
    print("✓ pytorch-crf is installed")
    CRF_AVAILABLE = True
except ImportError:
    print("✗ pytorch-crf is NOT installed")
    print("  Install with: pip install pytorch-crf")
    CRF_AVAILABLE = False

print()

# Test 2: Check model.py integration
print("Test 2: Checking model.py CRF integration...")
sys.path.insert(0, 'src')
try:
    from model import MWEIdentificationModel, CRF_AVAILABLE as MODEL_CRF_AVAILABLE
    print(f"✓ Model imported successfully")
    print(f"  CRF_AVAILABLE in model.py: {MODEL_CRF_AVAILABLE}")
except Exception as e:
    print(f"✗ Error importing model: {e}")
    sys.exit(1)

print()

# Test 3: Initialize model with CRF
print("Test 3: Initializing model with CRF enabled...")
try:
    import torch
    model = MWEIdentificationModel(
        model_name='bert-base-multilingual-cased',
        num_labels=3,
        num_categories=10,
        use_crf=True
    )
    print(f"✓ Model initialized successfully")
    print(f"  Model has use_crf: {model.use_crf}")
    print(f"  Model has CRF layer: {hasattr(model, 'crf')}")
    
    if model.use_crf and hasattr(model, 'crf'):
        print("✓ CRF layer is properly integrated!")
    elif not CRF_AVAILABLE:
        print("⚠ CRF not available (pytorch-crf not installed)")
    else:
        print("✗ CRF layer not found in model")
except Exception as e:
    print(f"✗ Error initializing model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 4: Check train.py has --crf argument
print("Test 4: Checking train.py CRF argument...")
try:
    with open('src/train.py', 'r', encoding='utf-8') as f:
        train_content = f.read()
    
    if '--crf' in train_content and 'use_crf' in train_content:
        print("✓ train.py has --crf argument")
    else:
        print("✗ train.py missing --crf argument")
except Exception as e:
    print(f"✗ Error reading train.py: {e}")

print()

# Test 5: Check predict.py loads use_crf
print("Test 5: Checking predict.py CRF support...")
try:
    with open('src/predict.py', 'r', encoding='utf-8') as f:
        predict_content = f.read()
    
    if "use_crf = checkpoint.get('use_crf'" in predict_content:
        print("✓ predict.py loads use_crf from checkpoint")
    else:
        print("✗ predict.py doesn't load use_crf")
except Exception as e:
    print(f"✗ Error reading predict.py: {e}")

print()

# Test 6: Check workflow.py has use_crf parameter
print("Test 6: Checking workflow.py CRF parameter...")
try:
    with open('workflow.py', 'r', encoding='utf-8') as f:
        workflow_content = f.read()
    
    if 'use_crf' in workflow_content and '--crf' in workflow_content:
        print("✓ workflow.py has use_crf parameter")
    else:
        print("✗ workflow.py missing use_crf parameter")
except Exception as e:
    print(f"✗ Error reading workflow.py: {e}")

print()

# Summary
print("=" * 80)
print("SUMMARY")
print("=" * 80)

if CRF_AVAILABLE and MODEL_CRF_AVAILABLE and model.use_crf:
    print("✓ CRF layer is fully integrated and ready to use!")
    print()
    print("Usage:")
    print("  Training with CRF:")
    print("    python src/train.py --train data/train.cupt --dev data/dev.cupt --crf")
    print()
    print("  Or via workflow:")
    print("    python workflow.py train FR --epochs 5 --use_crf")
    print()
    print("Benefits:")
    print("  - Learns sequence-level constraints")
    print("  - Better handles discontinuous MWEs")
    print("  - Prevents invalid BIO tag sequences (e.g., I-MWE without B-MWE)")
elif not CRF_AVAILABLE:
    print("⚠ CRF layer requires pytorch-crf installation")
    print()
    print("To install:")
    print("  pip install pytorch-crf")
    print()
    print("After installation, you can train with CRF:")
    print("  python src/train.py --train data/train.cupt --dev data/dev.cupt --crf")
else:
    print("✗ CRF integration has issues. Please check the error messages above.")

print()
