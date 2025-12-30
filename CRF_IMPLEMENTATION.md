# CRF Layer Implementation - Summary

## What was implemented?

Added **Conditional Random Fields (CRF)** layer as an optional component to improve discontinuous MWE detection.

## Changes Made

### 1. **requirements.txt**
- Added `pytorch-crf>=0.7.2`

### 2. **src/model.py**
- Added CRF import with availability check
- Added `use_crf` parameter to `MWEIdentificationModel.__init__()`
- Added CRF layer initialization when enabled
- Modified forward pass to use CRF loss when enabled
- Modified `predict_mwe_tags()` to use CRF decoding when enabled

### 3. **src/train.py**
- Added `--crf` command line argument
- Added `use_crf` parameter to `train_mwe_model()`
- Save `use_crf` in model checkpoint
- Save `use_crf` in info.json

### 4. **src/predict.py**
- Load `use_crf` from checkpoint
- Pass `use_crf` to model initialization
- Display CRF status in logs

### 5. **workflow.py**
- Added `use_crf` parameter to `train_model()`
- Pass `--crf` flag to train.py when enabled

## How to Use

### Training with CRF (single language):
```bash
python src/train.py --train 2.0/subtask1/FR/train.cupt \
                    --dev 2.0/subtask1/FR/dev.cupt \
                    --output models/FR_crf \
                    --crf \
                    --epochs 10
```

### Training with CRF (multilingual):
```bash
python src/train.py --train 2.0/subtask1/FR/train.cupt,2.0/subtask1/PL/train.cupt \
                    --dev 2.0/subtask1/FR/dev.cupt,2.0/subtask1/PL/dev.cupt \
                    --output models/multilingual_crf \
                    --crf \
                    --lang_tokens \
                    --epochs 10
```

### Via workflow script:
```python
# In workflow.py, pass use_crf=True
train_model(['FR', 'PL'], use_crf=True, epochs=10)
```

### Prediction (automatic):
```bash
python src/predict.py --model models/FR_crf/best_model.pt \
                      --input 2.0/subtask1/FR/dev.cupt \
                      --output predictions/FR/dev.system.cupt
```
The model will automatically use CRF decoding if it was trained with CRF.

## Benefits of CRF Layer

1. **Sequence-level constraints**: CRF learns valid BIO tag transitions
   - Prevents invalid sequences like I-MWE without preceding B-MWE
   - Enforces proper MWE boundaries

2. **Better discontinuous MWE detection**: 
   - CRF can learn that after O tokens, I-MWE is valid if part of same category
   - Helps predict patterns like: B-MWE → I-MWE → O → O → I-MWE

3. **Global optimization**:
   - Token-level classification: each token predicted independently
   - CRF: finds globally optimal sequence considering all transitions

## Expected Improvements

Based on NER/sequence labeling literature:

- **Continuous MWEs**: +1-2% F1 (already high, less room for improvement)
- **Discontinuous MWEs**: +10-20% F1 (currently 0%, significant potential)
- **Overall F1**: +3-5% expected improvement

## Technical Details

### Without CRF (Current):
```python
# Each token classified independently
for token in sentence:
    bio_tag = argmax(classifier(token_features))
    
# Problem: No constraints between tags
# Can predict: O → I-MWE (invalid!)
# Can predict: B-MWE → O → O (assumes MWE ended)
```

### With CRF:
```python
# All tokens classified jointly
best_sequence = crf.decode(all_token_features)

# Learns transition scores:
# B-MWE → I-MWE: high score ✓
# O → I-MWE: low score (but not impossible if part of discontinuous)
# B-MWE → B-MWE: low score ✗

# Finds globally optimal sequence
```

### CRF Loss Function:
```python
# Without CRF: Cross-Entropy per token
loss = sum(CrossEntropy(pred_i, gold_i) for i in tokens)

# With CRF: Negative log likelihood of full sequence
loss = -log P(gold_sequence | features)
     = -log( exp(score(gold_sequence)) / Z )
where Z = sum over all possible sequences (computed efficiently via forward algorithm)
```

## Testing

Run test suite:
```bash
python test_crf_integration.py
```

Expected output:
```
✓ pytorch-crf is installed
✓ Model imported successfully
✓ CRF layer is properly integrated!
✓ train.py has --crf argument
✓ predict.py loads use_crf from checkpoint
✓ workflow.py has use_crf parameter
```

## Next Steps

1. **Train a model with CRF**:
   ```bash
   python src/train.py --train 2.0/subtask1/FR/train.cupt \
                       --dev 2.0/subtask1/FR/dev.cupt \
                       --output models/FR_crf \
                       --crf --epochs 10 --batch_size 16
   ```

2. **Compare results**:
   - Train baseline (without CRF)
   - Train with CRF
   - Compare discontinuous MWE F1 scores

3. **Full multilingual training with CRF**:
   ```bash
   # Train on all 16 languages with CRF
   python src/train.py --train [all train files] \
                       --dev [all dev files] \
                       --output models/multilingual_crf \
                       --crf --lang_tokens --epochs 10
   ```

## Backward Compatibility

✅ Fully backward compatible:
- Models without CRF continue to work
- `use_crf` defaults to `False` if not in checkpoint
- No breaking changes to existing code
