# CRF Discontinuous MWE Fix

## Problem Analysis

### Initial Implementation
CRF was added to improve discontinuous MWE detection, but evaluation showed **0% F1 score on discontinuous MWEs** even with CRF enabled.

### Root Cause
The issue wasn't with CRF itself, but with how it interacts with the BIO tagging scheme:

1. **Training data contains discontinuous MWEs** (892 in FR alone):
   - Example: `en (83) ... possession (85)` with gap at position 84
   - Encoded as: `B-MWE ... O ... I-MWE` in BIO format

2. **CRF learns transition constraints from data**:
   - Standard BIO: `O → I-MWE` should be INVALID
   - But discontinuous MWEs REQUIRE: `O → I-MWE` 
   - CRF treats this as rare/invalid → heavily penalizes it

3. **Result**: CRF blocks the very pattern needed for discontinuous MWEs!
   - Postprocessing looks for `B-MWE ... O ... I-MWE` patterns
   - But CRF prevents model from generating them
   - **0 discontinuous MWEs in all predictions**

### Why This Happens
- CRF learns from training data that `O → I-MWE` is rare compared to valid transitions
- Even though discontinuous MWEs exist in training, they're outnumbered by continuous ones
- CRF's transition matrix heavily penalizes `O → I-MWE` transitions
- During decoding, CRF never allows resuming an MWE after a gap

## Solution Implemented

### 1. Transition Matrix Initialization (`_init_crf_for_discontinuous`)

Modified CRF initialization to explicitly allow discontinuous patterns:

```python
def _init_crf_for_discontinuous(self):
    """Initialize CRF to support O -> I-MWE transitions"""
    with torch.no_grad():
        # Allow O (0) -> I-MWE (2) for discontinuous MWEs
        self.crf.transitions.data[2, 0] = -1.0  # Slightly penalized but ALLOWED
        
        # Encourage I-MWE -> I-MWE continuity  
        self.crf.transitions.data[2, 2] = 1.0   # Encouraged
```

**Key insight**: 
- We don't force `O → I-MWE` to be highly probable
- We just prevent CRF from blocking it entirely
- CRF still learns optimal weights from data
- But now it has freedom to generate discontinuous patterns

### 2. Unicode Fix
Fixed Windows encoding issue with checkmark character:
- Changed `✓` to `[CRF]` in print statements
- Prevents `UnicodeEncodeError` on Windows terminals

### 3. Tensor Indexing Fix
Fixed prediction decoding for CRF vs non-CRF:
- CRF decode returns 1D tensor: `[seq_len]`
- Standard argmax returns 2D tensor: `[batch_size, seq_len]`
- Added conditional indexing: `bio_predictions[idx]` vs `bio_predictions[0, idx]`

## Expected Impact

### Before Fix
- Continuous MWEs: ~62% F1
- Discontinuous MWEs: **0% F1**
- Total: ~55% F1

### After Fix (Expected)
- CRF can now generate discontinuous patterns
- Postprocessing will find and preserve them
- Discontinuous F1 should improve from 0% to 10-20%+
- Overall F1 may improve by 1-3%

### Trade-offs
- Relaxed constraints may slightly reduce precision on continuous MWEs
- But massive recall improvement on discontinuous should outweigh this
- Net effect: better overall performance

## How to Test

### 1. Retrain model with fixed CRF:
```bash
python src/train.py --train 2.0/subtask1/FR/train.cupt \
                    --dev 2.0/subtask1/FR/dev.cupt \
                    --output models/pos-crf-fixed \
                    --crf --use_pos \
                    --epochs 10
```

### 2. Generate predictions:
```bash
python predict_multi.py
```

### 3. Check for discontinuous MWEs:
```bash
python check_all_discontinuous.py
```

**Look for**: Non-zero discontinuous counts!

### 4. Evaluate:
```bash
python eval_multi.py
```

**Compare**:
- Section 3: "Continuity & Token Count Statistics"
- Discontinuous F1 should be > 0.0

## Alternative Approaches Considered

1. **BIOES tagging** - Still can't handle discontinuous naturally
2. **BILOU + gap tags** - Complex, requires changing training data format
3. **Multi-label classification** - Major architectural change
4. **Disable CRF entirely** - Loses sequence modeling benefits
5. **Relaxed CRF constraints** ← **Chosen: Best balance**

## Files Modified

1. `src/model.py`:
   - Added `_init_crf_for_discontinuous()` method
   - Modified CRF initialization in `__init__()`
   - Fixed tensor indexing in `predict_mwe_tags()`
   - Fixed Unicode print statements

## Notes

- **Existing trained models** with CRF have this issue
- **Need to retrain** to benefit from the fix
- Models without `--crf` flag are unaffected
- Postprocessing still active for both CRF and non-CRF models

## References

- Discontinuous MWE analysis: `check_all_discontinuous.py`
- Data loader BIO conversion: `src/data_loader.py` lines 200-205
- Postprocessing logic: `src/postprocess_discontinuous.py`
- CRF library: `pytorch-crf` (torchcrf)
