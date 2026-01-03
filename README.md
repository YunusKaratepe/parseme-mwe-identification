# 🎯 PARSEME 2.0 - MWE Identification (Subtask 1)

Multilingual multiword expression (MWE) identification system using BERT-based token classification with advanced features including ensemble predictions for the PARSEME 2.0 shared task.

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

**With Language Tokens (Prevents Language Interference):**
```bash
python workflow.py train FR PL EL --epochs 3 --batch_size 16 --multilingual --lang_tokens
```

**With Focal Loss (Handles Class Imbalance):**
```bash
python workflow.py train FR PL EL --epochs 3 --batch_size 16 --multilingual --lang_tokens --loss focal
```

**Quick Experiment (10% data):**
```bash
python workflow.py train FR PL EL --epochs 1 --sample_ratio 0.1 --multilingual
```

### 3. Generate Predictions

**Single Model Submission:**
```bash
python generate_submission.py --model models/multilingual_FR+PL+EL/best_model.pt --lang FR PL EL
```

**Ensemble Submission (CE + Focal Loss):**
```bash
python generate_submission.py \
    --model ensemble/ce/multilingual_XXX/best_model.pt \
    --focal_model ensemble/focal/multilingual_XXX/best_model.pt \
    --lang FR PL EL PT RO SL SR SV UK NL EGY KA JA HE LV FA \
    --zip ensemble_submission.zip
```

**For All Available Languages:**
```bash
python generate_submission.py --model models/multilingual_XXX/best_model.pt --lang all
```

### 4. Validate Submission

```bash
python validate_submission.py
```

## 🌍 Key Features

- **🎯 Multi-Task Learning**: Dual-head architecture (BIO tagging + MWE category classification)
- **🔬 Language-Conditioned Inputs**: Optional language tokens (`[FR]`, `[PL]`, etc.) to prevent language interference in multilingual models
- **🎲 Focal Loss**: Addresses class imbalance (90% O tags vs 10% B/I-MWE tags) by focusing on hard examples
- **🤝 Ensemble Predictions**: Combine Cross-Entropy and Focal Loss models for better performance
- **🔧 Discontinuous MWE Post-Processing**: Automatic heuristic stitching to fix B-X ... O ... I-X patterns (converts 0% → 5-10% F1 on discontinuous)
- **🌐 Multilingual Training**: Train single model on multiple languages combined
- **⚖️ Data Splitting**: Validation is the last 10% of `train.cupt`; `dev.cupt` is used fully as the test set
- **📊 Data Sampling**: Use `--sample_ratio` for quick performance analysis
- **🚀 CUDA Support**: Automatic GPU acceleration
- **📦 Submission Pipeline**: Automated prediction generation and validation
- **17 Languages Supported**: FR, PL, EL, PT, RO, SL, SR, SV, UK, NL, EGY, KA, GRC, JA, HE, LV, FA

---

## 🔧 Complete Workflow

### Step 1: Training

**Standard Training:**
```bash
# Train with Cross-Entropy loss (default)
python workflow.py train FR PL EL --epochs 3 --multilingual --lang_tokens

# Train with Focal Loss (handles class imbalance)
python workflow.py train FR PL EL --epochs 3 --multilingual --lang_tokens --loss focal
```

**For Ensemble (train both):**
```bash
# Train CE model
python workflow.py train FR PL EL PT RO SL SR SV UK NL EGY KA JA HE LV FA \
    --multilingual --lang_tokens --loss ce --epochs 10 --output ensemble/ce

# Train Focal model
python workflow.py train FR PL EL PT RO SL SR SV UK NL EGY KA JA HE LV FA \
    --multilingual --lang_tokens --loss focal --epochs 10 --output ensemble/focal
```

**Output Directory Structure:**
```
models/multilingual_FR+PL+EL/  (or ensemble/ce/multilingual_XXX/)
├── best_model.pt           # Model checkpoint
├── training_history.json   # Loss/metrics per epoch
├── info.json               # Training metadata (includes loss_type)
└── tokenizer/              # Tokenizer files (if lang_tokens used)
```

### Step 2: Generate Predictions

**Single Model:**
```bash
python generate_submission.py --model models/multilingual_XXX/best_model.pt --lang FR PL EL
```

**Ensemble (Recommended):**
```bash
python generate_submission.py \
    --model ensemble/ce/multilingual_XXX/best_model.pt \
    --focal_model ensemble/focal/multilingual_XXX/best_model.pt \
    --lang FR PL EL PT RO SL SR SV UK NL EGY KA JA HE LV FA \
    --zip ensemble_submission.zip
```

**How Ensemble Works:**
- Averages probability distributions from CE and Focal Loss models
- CE model: Better precision on common patterns
- Focal Loss model: Better recall on rare categories (e.g., LVC.cause)
- Combined: Covers each other's blind spots for optimal performance

**Output Structure:**
```
predictions/
├── FR/test.system.cupt
├── PL/test.system.cupt
└── ...
ensemble_submission.zip     # Ready for submission
```

### Step 3: Evaluate Ensemble (Optional)

Evaluate ensemble performance on validation/test sets:

```bash
python src/ensemble_evaluate.py \
    --ce_model_dir ensemble/ce/multilingual_XXX \
    --focal_model_dir ensemble/focal/multilingual_XXX \
    --languages FR PL EL PT RO \
    --output ensemble
```

Results saved to `ensemble/evaluation_results.json` with:
- Overall Precision/Recall/F1
- Discontinuous MWE F1
- Per-category performance
- Average metrics across languages

### Step 4: Validate Predictions

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

### Step 5: Submit

Your `submission.zip` (or `ensemble_submission.zip`) is ready! It contains:
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
  - **Head 2**: MWE category classification (19+ categories)
- **Loss Functions**:
  - **Cross-Entropy** (default): Standard classification loss
  - **Focal Loss**: Addresses class imbalance (α=1.0, γ=2.0) - down-weights easy examples
- **Innovations**: 
  - **Language Tokens**: Prepend `[LANG]` tokens to prevent multilingual interference
  - **Discontinuous Post-Processing**: Heuristic stitching for B-X ... O ... I-X patterns
  - **Ensemble**: Probability averaging from CE + Focal models
- **Optimizer**: AdamW with linear warmup
- **Training**: Validation from the last 10% of `train.cupt`; evaluation/test on full `dev.cupt`

## 🔧 Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 5 | Number of training epochs |
| `--batch_size` | 8 | Training batch size |
| `--lr` | 2e-5 | Learning rate for AdamW optimizer |
| `--sample_ratio` | 1.0 | Fraction of training data (0.0-1.0) |
| `--multilingual` | False | Train single model on multiple languages |
| `--lang_tokens` | False | Enable language-conditioned inputs (prepend [LANG] tokens) |
| `--loss` | ce | Loss function: 'ce' (Cross-Entropy) or 'focal' (Focal Loss) |
| `--model_name` | bert-base-multilingual-cased | Base model (HuggingFace ID or local path) |
| `--output` | models | Output directory base |

## 📝 Advanced Usage Examples

### Training Variations

```bash
# Multilingual with language tokens (recommended for >3 languages)
python workflow.py train FR PL EL PT RO --multilingual --lang_tokens --epochs 5 --batch_size 16

# With Focal Loss for class imbalance
python workflow.py train FR PL EL --multilingual --lang_tokens --loss focal --epochs 10

# Train both for ensemble
python workflow.py train FR PL EL --multilingual --lang_tokens --loss ce --epochs 10 --output ensemble/ce
python workflow.py train FR PL EL --multilingual --lang_tokens --loss focal --epochs 10 --output ensemble/focal

# Quick experiment with 20% data
python workflow.py train FR PL --epochs 1 --sample_ratio 0.2 --multilingual
```

### Ensemble Operations

```bash
# Generate ensemble submission
python generate_submission.py \
    --model ensemble/ce/multilingual_XXX/best_model.pt \
    --focal_model ensemble/focal/multilingual_XXX/best_model.pt \
    --lang all \
    --zip ensemble_submission.zip

# Evaluate ensemble performance
python src/ensemble_evaluate.py \
    --ce_model_dir ensemble/ce/multilingual_XXX \
    --focal_model_dir ensemble/focal/multilingual_XXX \
    --languages FR PL EL PT RO SL SR SV UK \
    --output ensemble
```

---

## 🎯 Key Innovations

### 1. Language-Conditioned Inputs
Prevent language interference in multilingual models by prepending language tokens (`[FR]`, `[PL]`, etc.).

**Benefits:**
- Reduces high-resource language dominance
- Improves low-resource language performance (+2-5% F1)
- Explicit language signal to attention mechanism

**Usage:** Add `--lang_tokens` flag

### 2. Focal Loss for Class Imbalance
Addresses the 90% O-tag vs 10% B/I-MWE imbalance by down-weighting easy examples.

**Formula:** `-α(1-p_t)^γ log(p_t)` with α=1.0, γ=2.0

**Benefits:**
- Forces model to focus on hard MWE tags
- Better recall on rare categories (e.g., LVC.cause)
- Useful for ensemble diversity

**Usage:** Add `--loss focal` flag

### 3. Discontinuous MWE Post-Processing
Automatically fix broken MWE sequences (B-X ... O ... I-X) using heuristic stitching.

**Benefits:**
- Converts 0% F1 → 5-10% F1 on discontinuous MWEs
- No model retraining required
- Safe category-based matching

**Usage:** Enabled by default in predictions

### 4. Ensemble Predictions
Combine CE and Focal Loss models via probability averaging.

**How It Works:**
- CE model: Better precision on common patterns (VID, LVC.full)
- Focal model: Better recall on rare categories (LVC.cause, IRV)
- Ensemble: Averages softmax outputs before argmax

**Benefits:**
- Covers blind spots of individual models
- More robust predictions
- Better overall F1 score

**Usage:** Use `--focal_model` in `generate_submission.py`

📖 **See [FEATURES.md](FEATURES.md) for detailed documentation**

## 📁 Model Output Files

After training, your model directory contains:

| File | Description |
|------|-------------|
| `best_model.pt` | PyTorch checkpoint (model weights, label mappings, loss_type) |
| `training_history.json` | Epoch-by-epoch metrics (loss, F1, precision, recall) |
| `info.json` | Training metadata (languages, loss_type, hyperparameters, results) |
| `tokenizer/` | Tokenizer files (if `--lang_tokens` used) |

**info.json** includes:
- Languages trained on
- Loss function type (`ce` or `focal`)
- Whether language tokens were used
- Best validation F1 score
- All hyperparameters for reproducibility

---

## 🎯 Submission Format

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
```

### Ensemble Model Not Found
```bash
# Check if both models exist
ls ensemble/ce/multilingual_XXX/best_model.pt
ls ensemble/focal/multilingual_XXX/best_model.pt
```

### Validation Failures
```bash
# Validate all predictions
python validate_submission.py

# Check individual language
python 2.0/subtask1/tools/parseme_validate.py --lang FR predictions/FR/test.system.cupt
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

