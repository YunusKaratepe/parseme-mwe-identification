# 🎯 PARSEME 2.0 - MWE Identification (Subtask 1)

Multilingual multiword expression (MWE) identification system using BERT-based token classification with optional POS feature injection for the PARSEME 2.0 shared task.

## 🏆 Latest Results

**Multilingual Model (PL+FR+EL)** - 2 epochs with 10% training data:
- **Test F1**: 65.49%
- **Test Precision**: 70.22%
- **Test Recall**: 61.36%
- **Validation F1**: 63.97%

**French Baseline** - 3 epochs (no POS):
- **Test F1**: 74.37%
- **Category Accuracy**: 71.18%

## ⚡ Quick Start

### 1. Install Dependencies

**With CUDA (GPU) - Recommended:**
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install transformers numpy tqdm scikit-learn
```

**CPU Only:**
```bash
pip install torch transformers numpy tqdm scikit-learn
```

**Verify CUDA:**
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### 2. Train a Model

**Single Language:**
```bash
python workflow.py train FR --epochs 3 --batch_size 16
```

**Multilingual Model:**
```bash
python workflow.py train FR PL EL --epochs 3 --batch_size 16 --multilingual
```

**With POS Features (Innovation):**
```bash
python workflow.py train FR --epochs 3 --batch_size 16 --pos
```

**Quick Experiment (10% data):**
```bash
python workflow.py train FR PL EL --epochs 1 --sample_ratio 0.1 --multilingual
```

### 3. Generate Predictions

**For All Languages in Model:**
```bash
python generate_submission.py --model models/multilingual_FR+PL+EL/best_model.pt
```

**For Specific Languages:**
```bash
python generate_submission.py --model models/multilingual_FR+PL+EL/best_model.pt --lang FR PL EL
```

**For All 17 Available Languages:**
```bash
python generate_submission.py --model models/multilingual_FR+PL+EL/best_model.pt --lang all
```

### 4. Validate Submission

```bash
python validate_submission.py
```

## 🌍 Key Features

- **🎯 Multi-Task Learning**: Dual-head architecture (BIO tagging + MWE category classification)
- **🔬 POS Feature Injection**: Optional 128-dim POS embeddings (18 Universal tags)
- **🌐 Multilingual Training**: Train single model on multiple languages combined
- **⚖️ Language-Balanced Splitting**: Each language split 50/50 for validation/test
- **📊 Data Sampling**: Use `--sample_ratio` for quick performance analysis
- **🚀 CUDA Support**: Automatic GPU acceleration
- **📦 Submission Pipeline**: Automated prediction generation and validation
- **17 Languages Supported**: FR, PL, EL, PT, RO, SL, SR, SV, UK, NL, EGY, KA, GRC, JA, HE, LV, FA

---

## 🔧 Complete Workflow

### Step 1: Training

The training process creates a model directory with three key files:

```bash
# Train with default settings
python workflow.py train FR PL EL --epochs 3 --multilingual

# Train with POS features
python workflow.py train FR --epochs 3 --pos
```

**Output Directory Structure:**
```
models/multilingual_FR+PL+EL/
├── best_model.pt           # Model checkpoint
├── training_history.json   # Loss/metrics per epoch
└── info.json               # Training metadata
```

**info.json contents:**
```json
{
  "languages": ["FR", "PL", "EL"],
  "sample_ratio": 1.0,
  "epochs": 3,
  "lr": 2e-05,
  "batch_size": 16,
  "use_pos": false,
  "model_name": "bert-base-multilingual-cased",
  "best_f1": 0.7437,
  "num_labels": 3,
  "num_categories": 19,
  "seed": 42,
  "max_length": 512
}
```

### Step 2: Generate Predictions

The `generate_submission.py` script handles prediction generation for multiple languages:

```bash
# Auto-detect languages from model (reads info.json)
python generate_submission.py --model models/multilingual_FR+PL+EL/best_model.pt

# Generate for specific languages only
python generate_submission.py --model models/multilingual_FR+PL+EL/best_model.pt --lang FR PL

# Generate for ALL 17 available languages
python generate_submission.py --model models/multilingual_FR+PL+EL/best_model.pt --lang all

# Skip zip creation (only generate predictions)
python generate_submission.py --model models/multilingual_FR+PL+EL/best_model.pt --no-zip
```

**Output Structure:**
```
predictions/
├── FR/test.system.cupt
├── PL/test.system.cupt
├── EL/test.system.cupt
└── ...
submission.zip              # Ready for submission
```

### Step 3: Validate Predictions

Before submitting, validate all predictions at once:

```bash
python validate_submission.py
```

**Expected Output:**
```
================================================================================
                    PARSEME 2.0 - Submission Validation
================================================================================

Found 17 prediction files to validate:
  - FR: predictions/FR/test.system.cupt
  - PL: predictions/PL/test.system.cupt
  ...

✅ PASSED: FR
✅ PASSED: PL
...

================================================================================
VALIDATION SUMMARY
================================================================================
Total: 17 files
Passed: 17
Failed: 0

🎉 All predictions passed validation!
   Your submission is ready!
```

### Step 4: Submit

Your `submission.zip` is ready! It contains:
```
FR/test.system.cupt
PL/test.system.cupt
EL/test.system.cupt
...
```

---

## 🌍 Supported Languages

Train on any of these 17 languages:

| Code | Language | Code | Language |
|------|----------|------|----------|
| FR | French | PL | Polish |
| EL | Greek | PT | Portuguese |
| RO | Romanian | SL | Slovene |
| SR | Serbian | SV | Swedish |
| UK | Ukrainian | NL | Dutch |
| EGY | Egyptian | KA | Georgian |
| GRC | Ancient Greek | JA | Japanese |
| HE | Hebrew | LV | Latvian |
| FA | Persian | | |

---

## 📊 Model Architecture

- **Base Model**: bert-base-multilingual-cased (110M parameters)
- **Task**: Multi-task learning
  - **Head 1**: BIO tagging (O, B-MWE, I-MWE)
  - **Head 2**: MWE category classification (19 categories)
- **Innovation**: Optional POS feature injection (128-dim embeddings, 18 Universal tags)
- **Features**: BERT (768-dim) + optional POS (128-dim) = 896-dim combined
- **Optimizer**: AdamW with linear warmup
- **Training**: Language-balanced validation/test splits
- **Categories**: VID, LVC.full, LVC.cause, IAV, IRV, MVC, VPC.full, VPC.semi, LS.ICV, MWEP, MWV, PART, IDIOM, and more

## 🔧 Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 5 | Number of training epochs |
| `--batch_size` | 8 | Training batch size |
| `--lr` | 2e-5 | Learning rate for AdamW optimizer |
| `--sample_ratio` | 1.0 | Fraction of training data (0.0-1.0) |
| `--multilingual` | False | Train single model on multiple languages |
| `--pos` | False | Enable POS feature injection (innovation) |
| `--model_name` | bert-base-multilingual-cased | Base model (HuggingFace ID or local path) |
| `--output` | auto | Output directory (auto: models/LANG or models/multilingual_LANG1+LANG2) |

## 📝 Advanced Usage Examples

### Training Variations

```bash
# Single language with POS features
python workflow.py train FR --epochs 5 --batch_size 16 --pos

# Large-scale multilingual
python workflow.py train FR PL EL PT RO SL --epochs 3 --multilingual --batch_size 32

# Quick experiment with 20% data
python workflow.py train FR PL --epochs 1 --sample_ratio 0.2 --multilingual

# Custom output directory
python workflow.py train FR --epochs 3 --output experiments/french_v1

# Custom base model
python workflow.py train FR --epochs 3 --model_name ./my_custom_model
```

### Prediction Variations

```bash
# Generate for languages not in training set
python generate_submission.py --model models/FR/best_model.pt --lang FR PT ES

# Single language prediction (manual)
python src/predict.py \
    --model models/FR/best_model.pt \
    --input 2.0/subtask1/FR/test.blind.cupt \
    --output predictions/FR/test.system.cupt

# Validate single language (manual)
python 2.0/subtask1/tools/parseme_validate.py --lang FR predictions/FR/test.system.cupt
```

## 📁 Model Output Files

After training, your model directory contains:

| File | Description |
|------|-------------|
| `best_model.pt` | PyTorch checkpoint (model weights, label mappings, optimizer state) |
| `training_history.json` | Epoch-by-epoch metrics (loss, F1, precision, recall) |
| `info.json` | Training metadata (languages, hyperparameters, results) |

**info.json** is especially useful for:
- 📋 Documenting model configurations
- 🔄 Reproducing experiments
- 📊 Comparing different training runs
- 📝 Submission documentation

## 🔍 Understanding Results

### During Training
```
Epoch 1/3
Train Loss: 0.2345, Dev Loss: 0.1876
Dev F1: 72.45%, Precision: 75.12%, Recall: 69.98%
✓ New best F1: 72.45%
```

### In training_history.json
```json
[
  {
    "epoch": 1,
    "train_loss": 0.2345,
    "dev_loss": 0.1876,
    "dev_f1": 0.7245,
    "dev_precision": 0.7512,
    "dev_recall": 0.6998
  }
]
```

### In info.json
```json
{
  "languages": ["FR"],
  "best_f1": 0.7437,
  "epochs": 3,
  "use_pos": false
}
```

---

## 🎯 Submission Format

### Requirements Checklist

✅ **File Format**: `test.system.cupt` in CUPT format  
✅ **Structure**: `LANG/test.system.cupt` (one file per language)  
✅ **Column 11**: MWE annotations (format: `MWE_ID:CATEGORY` or `MWE_ID`)  
✅ **Validation**: Must pass `parseme_validate.py`  
✅ **Zip Contents**: Only language folders with prediction files  

### CUPT Format Example

```
# sentence_id = FR_train_001
1	Les	le	DET	...	*
2	droits	droit	NOUN	...	1:VID
3	de	de	ADP	...	1
4	l'	le	DET	...	1
5	homme	homme	NOUN	...	1

# sentence_id = FR_train_002
1	Il	il	PRON	...	*
2	fait	faire	VERB	...	2:LVC.full
3	attention	attention	NOUN	...	2
4	.	.	PUNCT	...	*
```

**Key Points:**
- First token of MWE: `MWE_ID:CATEGORY` (e.g., `1:VID`)
- Continuation tokens: `MWE_ID` only (e.g., `1`)
- Non-MWE tokens: `*`
- MWE IDs are sentence-local (reset each sentence)

---

## 🐛 Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
python workflow.py train FR --batch_size 4

# Use CPU instead
# (Edit workflow.py to remove .cuda() calls)
```

### Import Errors
```bash
# Reinstall dependencies
pip install --upgrade torch transformers numpy tqdm scikit-learn
```

### Validation Failures
```bash
# Check individual language
python 2.0/subtask1/tools/parseme_validate.py --lang FR predictions/FR/test.system.cupt

# Common issues:
# - Wrong filename (must be test.system.cupt)
# - Wrong column for MWE annotations (must be column 11)
# - Invalid MWE format (use MWE_ID:CATEGORY or MWE_ID)
```

### Model Loading Issues
```bash
# Check if model file exists
ls models/multilingual_FR+PL+EL/best_model.pt

# Check info.json for model configuration
cat models/multilingual_FR+PL+EL/info.json
```

---

## 📚 Resources

- **PARSEME 2.0 Shared Task**: https://unidive.lisn.upsaclay.fr/doku.php?id=other-events:parseme-st
- **Annotation Guidelines**: https://parsemefr.lis-lab.fr/parseme-st-guidelines/2.0/
- **CUPT Format Specification**: https://gitlab.com/parseme/corpora/-/wikis/CUPT-format
- **HuggingFace Transformers**: https://huggingface.co/docs/transformers/
- **BERT Multilingual**: https://huggingface.co/bert-base-multilingual-cased

---

## 📄 License

This project is part of the PARSEME 2.0 shared task. Please refer to the official PARSEME guidelines for data usage and citation requirements.

