# POS Tag Injection: "Free Lunch" Innovation

**Date**: December 9, 2025  
**Innovation**: Morphosyntactic Feature Injection  
**Cost**: Minimal (~128 parameters per POS tag)  
**Expected Gain**: +3-5% F1

---

## Concept

MWEs are highly syntactic (e.g., LVC = Verb + Noun, VID = Verb + Particle). The .cupt format provides Universal POS tags (VERB, NOUN, ADP, etc.) in column 4. Standard BERT ignores this - it has to learn grammar from scratch.

**Innovation**: Explicitly feed POS tags into the model via embedding layer.

---

## Implementation

### Architecture Changes

**Before**:
```
Input → BERT → Hidden [768] → Classifier
```

**After**:
```
Input → BERT → Hidden [768]
             ↓
      POS Embedding [128]
             ↓
    Concatenate [896] → Classifier
```

### Code Changes

#### 1. Model (`src/model.py`)
- Added `num_pos_tags` and `use_pos` parameters
- Added `self.pos_embedding = nn.Embedding(num_pos_tags, 128)`
- Forward pass concatenates BERT features with POS embeddings
- Classification heads take combined features (768 + 128 = 896 dim)

#### 2. Data Loader (`src/data_loader.py`)
- Already extracts POS tags from column 4
- Added `get_pos_mapping()` method
- Returns mapping for 18 Universal POS tags + padding

#### 3. Tokenizer (`src/model.py` - MWETokenizer)
- Updated `tokenize_and_align_labels()` to handle `pos_tags` and `pos_to_id`
- POS tags aligned to subword tokens (same as labels)
- Subwords inherit POS from first subword

#### 4. Training (`src/train.py`)
- Dataset now takes `pos_to_id` parameter
- `__getitem__` returns `pos_ids` tensor
- Training/eval loops pass `pos_ids` to model
- Checkpoints save `pos_to_id` mapping and `use_pos` flag

#### 5. Prediction (`src/predict.py`)
- Loads `pos_to_id` from checkpoint
- Backward compatible (works with old models without POS)
- Displays "POS feature injection: ENABLED/DISABLED"

---

## Universal POS Tags (18 tags)

From Universal Dependencies:
1. **<PAD>** - Padding/unknown
2. **ADJ** - Adjective
3. **ADP** - Adposition (preposition)
4. **ADV** - Adverb
5. **AUX** - Auxiliary verb
6. **CCONJ** - Coordinating conjunction
7. **DET** - Determiner
8. **INTJ** - Interjection
9. **NOUN** - Noun
10. **NUM** - Numeral
11. **PART** - Particle
12. **PRON** - Pronoun
13. **PROPN** - Proper noun
14. **PUNCT** - Punctuation
15. **SCONJ** - Subordinating conjunction
16. **SYM** - Symbol
17. **VERB** - Verb
18. **X** - Other

---

## Why This Works

### 1. Syntactic Patterns
MWE categories have strong POS signatures:
- **LVC.full**: VERB + NOUN ("take a walk", "make a decision")
- **VID**: VERB + PART ("pick up", "give in")
- **NID**: NOUN + (NOUN|ADJ) ("hot dog", "red tape")
- **AdpID**: ADP + NOUN ("by heart", "on purpose")

### 2. Removes Ambiguity
Word "walk" can be NOUN or VERB:
- "take a **walk**" (NOUN) → LVC candidate
- "I **walk**" (VERB) → Not LVC

POS helps model distinguish these instantly.

### 3. Cross-lingual Transfer
Universal POS tags are consistent across languages.
A multilingual model can learn "VERB + NOUN = potential LVC" pattern that applies to all languages.

### 4. Minimal Cost
- Embedding size: 18 tags × 128 dim = 2,304 parameters
- BERT has 110M parameters
- Cost: **0.002%** increase
- Speed impact: Negligible (one embedding lookup per token)

---

## Expected Performance Gains

### Conservative Estimate: +2-3% F1
- Faster convergence (fewer epochs needed)
- Better category prediction (+5-8% accuracy)
- Especially helpful for small datasets

### Optimistic Estimate: +4-6% F1
- If model was struggling to learn grammar
- Multilingual models benefit more (shared POS across languages)
- Helps with rare MWE types

### Real-world Examples (from literature):
- BERT + POS for NER: +1.5-2.5% F1
- BERT + POS for chunking: +2-4% F1
- BERT + POS for dependency parsing: +3-5% UAS

---

## Training New Model

### Command (same as before):
```bash
python src/train.py \
    --train_file 2.0/subtask1/FR/train.cupt \
    --dev_file 2.0/subtask1/FR/dev.cupt \
    --test_file 2.0/subtask1/FR/test.cupt \
    --model_name bert-base-multilingual-cased \
    --output_dir models/FR_with_pos \
    --epochs 3 \
    --batch_size 16
```

**No changes needed!** POS injection is automatically enabled.

### To Disable POS (for comparison):
Edit `src/train.py` line 280:
```python
model = MWEIdentificationModel(
    model_name, 
    num_labels=len(label_to_id), 
    num_categories=len(category_to_id),
    num_pos_tags=len(pos_to_id),
    use_pos=False  # Disable POS injection
)
```

---

## Validation

### Check if POS is enabled:
```bash
python src/predict.py --model models/FR/best_model.pt ...
# Output will show: "POS feature injection: ENABLED"
```

### Test POS alignment:
```python
from src.data_loader import CUPTDataLoader
from src.model import MWETokenizer

loader = CUPTDataLoader()
data = loader.read_cupt_file('2.0/subtask1/FR/train.cupt')
pos_to_id = loader.get_pos_mapping(data)

tokenizer = MWETokenizer('bert-base-multilingual-cased')
sent = data[0]
result = tokenizer.tokenize_and_align_labels(
    sent['tokens'], sent['mwe_tags'], {'O': 0, 'B-MWE': 1, 'I-MWE': 2},
    sent['mwe_categories'], loader.get_category_mapping(),
    sent['pos_tags'], pos_to_id
)

print("POS IDs:", result['pos_ids'])
```

---

## Comparison Experiment

### Setup:
1. Train **without** POS: 
   - Modify code to set `use_pos=False`
   - Train for 3 epochs
   
2. Train **with** POS:
   - Default settings
   - Train for 3 epochs

3. Compare:
   - Final F1 scores
   - Category accuracy
   - Convergence speed (epoch 1 vs epoch 3 improvement)

### Expected Results:
```
Without POS:
- Epoch 1: F1 = 65%
- Epoch 3: F1 = 72%

With POS:
- Epoch 1: F1 = 68% (+3%)
- Epoch 3: F1 = 76% (+4%)
```

---

## Other "Free Lunch" Features to Consider

### 1. **Lemma Features** (Column 3)
- Similar to POS, minimal cost
- Helps with morphological variants
- "run", "running", "ran" → same lemma

### 2. **Dependency Relations** (Column 8)
- Some MWEs have specific dependency patterns
- LVC: "obj" or "obl" relation
- Cost: ~50 relation types × 128 dim

### 3. **Token Position**
- Distance from start/end of sentence
- Some MWEs appear in specific positions
- Cost: Tiny (positional embeddings already in BERT)

---

## Limitations

### 1. Not a Silver Bullet
- Won't help if POS tags are wrong in data
- Some MWEs are purely lexical (not syntactic)
- Gains depend on how well BERT already learned grammar

### 2. Language-Specific POS
- Some languages have extended tagsets
- Need to map to Universal POS first
- Already handled by UD format

### 3. Prediction Requires POS
- At test time, need POS tags in input
- Fortunately, .cupt format always provides them
- For raw text, would need POS tagger first

---

## Conclusion

**POS tag injection is a "free lunch"** because:
1. ✅ Data already has POS tags (column 4)
2. ✅ Minimal computational cost (<0.01% parameters)
3. ✅ No preprocessing needed
4. ✅ Expected +3-5% F1 improvement
5. ✅ Helps model learn faster
6. ✅ Especially useful for multilingual models

**Implementation complete** - ready to train and evaluate!
