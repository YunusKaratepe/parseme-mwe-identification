# PARSEME 2.0 Subtask 1: MWE Identification

This project provides a complete training pipeline for the PARSEME 2.0 shared task on multiword expression (MWE) identification.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train a Model

Train on French data (fastest to test):

```bash
python src/train.py --train 2.0/subtask1/FR/train.cupt --dev 2.0/subtask1/FR/dev.cupt --output models/FR --epochs 3 --batch_size 8
```

Or use the batch script (Windows):

```bash
run_training.bat
```

### 3. Make Predictions

```bash
python src/predict.py --model models/FR/best_model.pt --input 2.0/subtask1/FR/test.blind.cupt --output predictions/FR/test.cupt
```

### 4. Evaluate (using official tools)

```bash
python 2.0/subtask1/tools/parseme_evaluate.py --gold 2.0/subtask1/FR/test.cupt --pred predictions/FR/test.cupt
```

## Project Structure

```
src/
├── data_loader.py    # CUPT file parser and BIO tag converter
├── model.py          # Transformer-based token classification model
├── train.py          # Training script
└── predict.py        # Inference script

2.0/subtask1/         # Training data for all languages
├── FR/               # French data
├── PL/               # Polish data
├── EL/               # Greek data
└── ...               # Other languages

models/               # Saved model checkpoints
predictions/          # Generated predictions
```

## Model Architecture

- **Base Model**: Multilingual BERT (bert-base-multilingual-cased)
- **Task**: Token classification with BIO tagging
- **Labels**: O (outside MWE), B-MWE (beginning of MWE), I-MWE (inside MWE)
- **Features**: Subword tokenization with label alignment

## Training Parameters

Default parameters (can be adjusted via command line):

- **Model**: bert-base-multilingual-cased
- **Epochs**: 5 (use 3 for quick testing)
- **Batch Size**: 16 (use 8 for lower memory)
- **Learning Rate**: 2e-5
- **Max Length**: 512 tokens
- **Optimizer**: AdamW with linear warmup

## Available Languages

Training data is available for 17 languages:
- Dutch (NL), Egyptian (EGY), French (FR), Georgian (KA), Ancient Greek (GRC)
- Modern Greek (EL), Japanese (JA), Hebrew (HE), Latvian (LV), Persian (FA)
- Polish (PL), Brazilian Portuguese (PT), Romanian (RO), Serbian (SR)
- Slovene (SL), Swedish (SV), Ukrainian (UK)

## MWE Types

The model identifies MWEs of all syntactic types:
- **VID**: Verbal idioms (e.g., "break the ice")
- **LVC**: Light verb constructions (e.g., "take a walk")
- **NID**: Nominal idioms (e.g., "hot dog")
- **AdjID**: Adjectival idioms
- **AdvID**: Adverbial idioms
- **ConjID**: Functional MWEs

## Performance Metrics

The model reports:
- **Precision**: Percentage of predicted MWEs that are correct
- **Recall**: Percentage of gold MWEs that are found
- **F1 Score**: Harmonic mean of precision and recall

## Training on Multiple Languages

To train on a different language, change the data paths:

```bash
# Polish
python src/train.py --train 2.0/subtask1/PL/train.cupt --dev 2.0/subtask1/PL/dev.cupt --output models/PL

# Greek
python src/train.py --train 2.0/subtask1/EL/train.cupt --dev 2.0/subtask1/EL/dev.cupt --output models/EL
```

## Tips for Better Results

1. **Use language-specific models** when available (e.g., CamemBERT for French)
2. **Increase epochs** to 5-10 for better convergence
3. **Adjust batch size** based on your GPU memory
4. **Try different learning rates** (1e-5 to 5e-5)
5. **Use ensemble methods** combining multiple models

## Troubleshooting

### Out of Memory Errors
- Reduce `--batch_size` (try 4 or 8)
- Reduce `--max_length` (try 256)

### Slow Training
- Use GPU if available (automatic detection)
- Reduce number of epochs for testing

### Low Performance
- Train for more epochs
- Try a language-specific BERT model
- Ensure data is loaded correctly (check data_loader.py output)

## References

- [PARSEME Shared Task 2.0](https://unidive.lisn.upsaclay.fr/doku.php?id=other-events:parseme-st)
- [CUPT Format Documentation](https://gitlab.com/parseme/corpora/-/wikis/CUPT-format)
- [Annotation Guidelines](https://parsemefr.lis-lab.fr/parseme-st-guidelines/2.0/)
