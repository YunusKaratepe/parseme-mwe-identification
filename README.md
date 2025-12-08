# 🎯 PARSEME 2.0 - MWE Identification (Subtask 1)

Multilingual multiword expression (MWE) identification system using BERT-based token classification for the PARSEME 2.0 shared task.

## 🏆 Latest Results

**Multilingual Model (PL+FR+EL)** - 2 epochs with 10% training data:
- **Test F1**: 65.49%
- **Test Precision**: 70.22%
- **Test Recall**: 61.36%
- **Validation F1**: 63.97%

## ⚡ Quick Start

### Install Dependencies

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

### Train Single Language
```bash
python workflow.py train PL --epochs 3 --batch_size 16
```

### Train Multilingual Model
```bash
python workflow.py train PL FR EL --epochs 3 --batch_size 16 --multilingual
```

### Quick Analysis (10% data sample)
```bash
python workflow.py train PL FR EL --epochs 1 --sample_ratio 0.1 --multilingual
```

## 🌍 Key Features

- **Multilingual Training**: Train single model on multiple languages combined
- **Language-Balanced Splitting**: Each language split 50/50 for validation/test
- **Data Sampling**: Use `--sample_ratio` for quick performance analysis
- **CUDA Support**: Automatic GPU acceleration
- **17 Languages Supported**: FR, PL, EL, PT, RO, SL, SR, SV, UK, NL, EGY, KA, GRC, JA, HE, LV, FA

## 📝 Usage Examples

```bash
# Train on multiple languages separately
python workflow.py train PL FR EL --epochs 3 --batch_size 16

# Train single multilingual model
python workflow.py train PL FR EL --epochs 3 --multilingual

# Quick test with 20% of data
python workflow.py train FR PL EL PT RO --epochs 1 --sample_ratio 0.2 --multilingual

# Generate predictions
python workflow.py predict FR
    --train 2.0/subtask1/FR/train.cupt \
    --dev 2.0/subtask1/FR/dev.cupt \
    --output models/FR \
    --epochs 5 \
    --batch_size 8 \
    --lr 2e-5
```

#### Prediction
```bash
python src/predict.py \
    --model models/FR/best_model.pt \
    --input 2.0/subtask1/FR/test.blind.cupt \
    --output predictions/FR/test.cupt
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

**Example**: Train on Polish
```bash
python workflow.py train PL --epochs 5
python workflow.py predict PL
python workflow.py visualize PL
```

---

```

## 📊 Model Architecture

- **Base Model**: bert-base-multilingual-cased (110M parameters)
- **Task**: Token classification (BIO tagging: O, B-MWE, I-MWE)
- **Optimizer**: AdamW with linear warmup
- **Training**: Language-balanced validation/test splits
- **Supports**: 18+ MWE categories (VID, LVC, MVC, NID, AdjID, etc.)

## 🔧 Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 5 | Number of training epochs |
| `--batch_size` | 8 | Training batch size |
| `--sample_ratio` | 1.0 | Fraction of training data (0.0-1.0) |
| `--multilingual` | False | Train single model on multiple languages |
| `--lr` | 2e-5 | Learning rate |

## 📚 Resources

- **PARSEME 2.0**: https://unidive.lisn.upsaclay.fr/doku.php?id=other-events:parseme-st
- **Guidelines**: https://parsemefr.lis-lab.fr/parseme-st-guidelines/2.0/
- **CUPT Format**: https://gitlab.com/parseme/corpora/-/wikis/CUPT-format
