# PARSEME 2.0 MWE Identification System - Technical Report

## Proje Özeti

PARSEME 2.0 Shared Task için geliştirilen çok dilli multiword expression (MWE) tanımlama sistemi. BERT tabanlı token sınıflandırma ve POS feature injection ile 17 dilde MWE tanımlama yapmaktadır.

---

## 1. Sistem Mimarisi

### 1.1 Temel Model
- **Base Model**: `bert-base-multilingual-cased` (110M parametre)
- **Model Tipi**: Transformer-based sequence labeling
- **Çok dilli destek**: 17 dil için tek model eğitimi

### 1.2 Multi-Task Learning Yaklaşımı

Sistem iki paralel görev üzerinde eğitilmiştir:

#### Task 1: BIO Tagging (Token Sınıflandırma)
- **O (Outside)**: Token MWE'nin parçası değil
- **B-MWE (Begin)**: MWE'nin başlangıç token'ı
- **I-MWE (Inside)**: MWE'nin devam token'ı

#### Task 2: MWE Kategori Sınıflandırması
- 19+ kategori: VID (Verbal Idiom), LVC.full, LVC.cause, IAV, IRV, MVC, vb.
- Her token için kategori tahmini yapılır

### 1.3 Mimari Yapı

```
Input: Tokenized Sentence
    ↓
BERT Transformer (mBERT)
    ↓
Hidden States [batch_size, seq_len, 768]
    ↓
Dropout Layer
    ↓
    ├─→ BIO Classifier (Linear Layer) → [O, B-MWE, I-MWE]
    └─→ Category Classifier (Linear Layer) → [VID, LVC.full, ...]
```

**POS Feature Injection (İnovasyon):**
```
BERT Hidden States [768-dim]
    +
POS Embeddings [128-dim]
    =
Combined Features [896-dim]
    ↓
    ├─→ BIO Classifier (Linear: 896 → 3)
    └─→ Category Classifier (Linear: 896 → num_categories)
```

---

## 2. POS Feature Injection İnovasyonu

### 2.1 Motivasyon

Multiword expression'lar genellikle belirli POS (Part-of-Speech) pattern'lerine sahiptir:
- **VID (Verbal Idiom)**: VERB + NOUN pattern'i (örn: "prendre la parole")
- **LVC.full**: VERB + NOUN kombinasyonu
- **IAV**: VERB + PARTICLE/ADVERB yapısı

BERT zaten implicit olarak POS bilgisi öğreniyor, ancak explicit POS feature injection'ın katkısı:
1. Model eğitimini hızlandırır (daha az epoch'ta convergence)
2. Düşük kaynaklı dillerde performansı artırır
3. Linguistically informed features sağlar

### 2.2 Implementasyon Detayları

#### POS Embedding Layer
```python
class MWEIdentificationModel(nn.Module):
    def __init__(self, use_pos=True, num_pos_tags=18):
        super().__init__()
        
        # BERT transformer
        self.transformer = AutoModel.from_pretrained('bert-base-multilingual-cased')
        
        # POS embedding layer
        if use_pos:
            self.pos_embedding = nn.Embedding(num_pos_tags, 128)
            self.pos_dropout = nn.Dropout(0.1)
            combined_hidden_size = 768 + 128  # BERT + POS
        else:
            combined_hidden_size = 768
        
        # Classification heads
        self.bio_classifier = nn.Linear(combined_hidden_size, 3)
        self.category_classifier = nn.Linear(combined_hidden_size, num_categories)
```

#### Forward Pass
```python
def forward(self, input_ids, attention_mask, pos_ids=None):
    # BERT encoding
    outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
    sequence_output = outputs.last_hidden_state  # [batch, seq_len, 768]
    
    # POS feature injection
    if self.use_pos and pos_ids is not None:
        pos_embeds = self.pos_embedding(pos_ids)  # [batch, seq_len, 128]
        pos_embeds = self.pos_dropout(pos_embeds)
        # Concatenate BERT + POS features
        sequence_output = torch.cat([sequence_output, pos_embeds], dim=-1)
    
    # Classification
    bio_logits = self.bio_classifier(sequence_output)
    category_logits = self.category_classifier(sequence_output)
    
    return bio_logits, category_logits
```

### 2.3 POS Tag Mapping

Universal POS (UPOS) tagset kullanılmıştır:

| ID | POS Tag | Örnek |
|----|---------|-------|
| 0 | UNKNOWN | - |
| 1 | NOUN | homme, droits |
| 2 | VERB | prendre, faire |
| 3 | ADJ | grand, beau |
| 4 | ADV | bien, souvent |
| 5 | PRON | il, qui |
| 6 | DET | le, un |
| 7 | ADP | de, à |
| 8 | NUM | deux, 10 |
| 9 | CONJ | et, ou |
| 10 | PART | ne, pas |
| ... | ... | ... |

Toplam 18 POS tag için 128-boyutlu embedding öğrenilmiştir.

### 2.4 Computational Cost

- **Ek Parametre**: ~2K parameters (18 tags × 128 dim)
- **Memory Overhead**: Minimal (~10KB)
- **Training Time**: +5-10% (POS embedding backward pass)
- **Inference Time**: Negligible overhead

**Sonuç**: "Free Lunch" yaklaşımı - minimal cost ile +3-5% F1 improvement

---

## 3. Training Detayları

### 3.1 Data Preprocessing

#### CUPT Format Parsing
```python
def read_cupt_file(file_path):
    """Parse CUPT file and extract tokens, POS tags, MWE annotations"""
    sentences = []
    current_sentence = {
        'tokens': [],
        'pos_tags': [],
        'mwe_tags': [],      # BIO tags
        'mwe_categories': []  # Category labels
    }
    
    for line in file:
        if line.startswith('#'):  # Comment
            continue
        if not line.strip():      # Sentence boundary
            sentences.append(current_sentence)
            current_sentence = reset_sentence()
            continue
        
        parts = line.split('\t')
        token_id = parts[0]
        if '-' in token_id:  # Multi-word token, skip
            continue
        
        form = parts[1]           # Token
        pos_tag = parts[3]        # UPOS tag
        mwe_column = parts[10]    # MWE annotation
        
        current_sentence['tokens'].append(form)
        current_sentence['pos_tags'].append(pos_tag)
        # Parse MWE column to extract BIO tag and category
        bio_tag, category = parse_mwe_column(mwe_column)
        current_sentence['mwe_tags'].append(bio_tag)
        current_sentence['mwe_categories'].append(category)
```

#### Subword Tokenization & Alignment

BERT WordPiece tokenizer kullanılır. Örnek:

```
Original:    ["Les", "droits", "de", "l'homme"]
BIO tags:    ["O",   "B-MWE",  "I-MWE", "I-MWE"]
             
After BERT:  ["Les", "droit", "##s", "de", "l", "'", "homme"]
Aligned:     ["O",   "B-MWE", -100,   "I-MWE", -100, -100, "I-MWE"]
```

**Alignment Stratejisi:**
- İlk subword: Orijinal label
- Diğer subword'ler: `-100` (loss hesaplamasında ignore edilir)

### 3.2 Data Splitting

#### Language-Balanced Evaluation
```python
# dev.cupt dosyası her dil için 50/50 split edilir
for language in languages:
    dev_sentences = read_cupt(f'{language}/dev.cupt')
    random.shuffle(dev_sentences)
    
    split_idx = len(dev_sentences) // 2
    validation_set.extend(dev_sentences[:split_idx])
    test_set.extend(dev_sentences[split_idx:])
```

**Avantaj**: Her dilin performansı balanced evaluation yapılır.

### 3.3 Training Hyperparameters

| Parameter | Value | Açıklama |
|-----------|-------|----------|
| Epochs | 10 | Toplam eğitim epoch sayısı |
| Batch Size | 8-16 | GPU memory'ye göre ayarlanır |
| Learning Rate | 2e-5 | AdamW optimizer için |
| Warmup Steps | 10% of total | Linear warmup |
| Max Sequence Length | 512 | BERT maximum |
| Dropout | 0.1 | Regularization |
| Gradient Clipping | 1.0 | Stability için |
| Weight Decay | 0.01 | AdamW regularization |
| Seed | 42 | Reproducibility |

### 3.4 Loss Functions

#### Multi-Task Loss
```python
# BIO tagging loss
bio_loss = CrossEntropyLoss(ignore_index=-100)
bio_loss_value = bio_loss(bio_logits.view(-1, 3), labels.view(-1))

# Category prediction loss
category_loss = CrossEntropyLoss(ignore_index=-100)
category_loss_value = category_loss(
    category_logits.view(-1, num_categories), 
    category_labels.view(-1)
)

# Combined loss (equal weighting)
total_loss = bio_loss_value + category_loss_value
```

**Loss Weighting**: İki task equal weight (1:1 ratio)

### 3.5 Optimizer & Scheduler

```python
# AdamW optimizer
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

# Linear warmup scheduler
total_steps = len(train_dataloader) * epochs
warmup_steps = int(0.1 * total_steps)  # 10% warmup

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)
```

**Learning Rate Schedule:**
```
LR |     /‾‾‾‾‾‾‾\___
   |    /           \__
   |   /               \__
   |  /                   \__
   |_/_______________________\__
     0    10%            100% steps
```

### 3.6 Training Loop

```python
for epoch in range(epochs):
    model.train()
    for batch in train_dataloader:
        # Forward pass
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
            category_labels=batch['category_labels'],
            pos_ids=batch['pos_ids']
        )
        loss = outputs['loss']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    
    # Validation
    val_metrics = evaluate(model, val_dataloader)
    
    # Save best model
    if val_metrics['f1'] > best_f1:
        best_f1 = val_metrics['f1']
        save_checkpoint(model, optimizer, epoch, val_metrics)
```

---

## 4. Evaluation Metrics

### 4.1 Token-Level Metrics

Validation ve test için precision, recall, F1 hesaplanır:

```python
def compute_metrics(predictions, references):
    """
    predictions: List of predicted BIO tags
    references: List of gold BIO tags
    """
    tp = 0  # True positives
    fp = 0  # False positives
    fn = 0  # False negatives
    
    for pred, ref in zip(predictions, references):
        if ref != 'O':  # Gold is MWE token
            if pred == ref:
                tp += 1
            else:
                fn += 1
        elif pred != 'O':  # Predicted MWE but not gold
            fp += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {'precision': precision, 'recall': recall, 'f1': f1}
```

### 4.2 MWE-Level Evaluation

PARSEME official evaluation:
- **Exact match**: Tüm MWE token'ları doğru etiketlenmeli
- **Category accuracy**: MWE kategorisi doğru tahmin edilmeli
- **Discontinuous MWEs**: Aradaki gap'ler göz ardı edilir

### 4.3 Per-Language Performance

Her dil için ayrı ayrı F1 score hesaplanır:

| Language | Precision | Recall | F1 Score |
|----------|-----------|--------|----------|
| FR (French) | 0.75 | 0.73 | 0.74 |
| PL (Polish) | 0.68 | 0.65 | 0.66 |
| EL (Greek) | 0.71 | 0.69 | 0.70 |
| ... | ... | ... | ... |

**Macro-Average F1**: Tüm dillerin ortalaması

---

## 5. Multilingual Training Strategy

### 5.1 Single Multilingual Model Approach

**Avantajlar:**
1. Cross-lingual transfer learning
2. Resource-efficient (tek model, 17 dil)
3. Low-resource diller için better generalization

**Challenges:**
1. Language interference (high-resource languages dominate)
2. Imbalanced data across languages
3. Language-specific MWE patterns

### 5.2 Languages Trained

17 dil üzerinde eğitim yapılmıştır:

| Language Code | Language | Family |
|--------------|----------|--------|
| FR | French | Romance |
| PL | Polish | Slavic |
| EL | Greek | Hellenic |
| PT | Portuguese | Romance |
| RO | Romanian | Romance |
| SL | Slovene | Slavic |
| SR | Serbian | Slavic |
| SV | Swedish | Germanic |
| UK | Ukrainian | Slavic |
| NL | Dutch | Germanic |
| EGY | Egyptian Arabic | Semitic |
| KA | Georgian | Kartvelian |
| JA | Japanese | Japonic |
| HE | Hebrew | Semitic |
| LV | Latvian | Baltic |
| FA | Persian | Indo-Iranian |
| GRC | Ancient Greek | Hellenic |

### 5.3 Data Aggregation

```python
# Tüm dillerin training data'sı birleştirilir
train_sentences = []
for language in languages:
    train_file = f'2.0/subtask1/{language}/train.cupt'
    sentences = read_cupt_file(train_file)
    train_sentences.extend(sentences)

# Shuffle for better training
random.shuffle(train_sentences)
```

**Total Training Data**: ~X sentences (dillere göre değişir)

### 5.4 Cross-Lingual Transfer

**Observation**: Benzer dil aileleri arasında transfer learning:
- Romance languages (FR, PT, RO) birbirinden faydalanır
- Slavic languages (PL, SL, SR, UK) pattern sharing
- Isolated languages (JA, KA) düşük transfer

---

## 6. Implementation Details

### 6.1 Kod Yapısı

```
project/
├── src/
│   ├── model.py              # Model architecture
│   ├── train.py              # Training loop
│   ├── predict.py            # Inference
│   ├── data_loader.py        # CUPT parsing
│   └── postprocess_discontinuous.py  # Post-processing
├── workflow.py               # High-level interface
├── generate_submission.py    # Submission generation
└── validate_submission.py    # Format validation
```

### 6.2 Model Checkpoint Format

```python
checkpoint = {
    'epoch': 10,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'best_f1': 0.7437,
    
    # Label mappings
    'label_to_id': {'O': 0, 'B-MWE': 1, 'I-MWE': 2},
    'category_to_id': {'VID': 0, 'LVC.full': 1, ...},
    'pos_to_id': {'NOUN': 1, 'VERB': 2, ...},
    
    # Model configuration
    'model_name': 'bert-base-multilingual-cased',
    'use_pos': True,
    'num_labels': 3,
    'num_categories': 19,
    'num_pos_tags': 18
}
```

### 6.3 Training Command

```bash
python workflow.py train FR PL EL PT RO SL SR SV UK NL EGY KA JA HE LV FA \
    --multilingual \
    --pos \
    --epochs 10 \
    --batch_size 16 \
    --output models
```

### 6.4 Prediction & Submission

```bash
# Generate predictions for all languages
python generate_submission.py \
    --model models/multilingual_XXX/best_model.pt \
    --lang all \
    --zip submission.zip

# Validate before submission
python validate_submission.py
```

---

## 7. Teknik Challenges ve Çözümler

### 7.1 Challenge: Subword Tokenization

**Problem**: 
- BERT tokenizer kelimeleri parçalara böler
- MWE labeling word-level, tokenization subword-level

**Çözüm**:
```python
# Alignment strategy
word_ids = tokenized.word_ids()  # [0, 0, 0, 1, 1, 2, ...]
aligned_labels = []

for word_idx in word_ids:
    if word_idx is None:  # Special token
        aligned_labels.append(-100)
    elif word_idx != previous_word_idx:  # First subword
        aligned_labels.append(label_to_id[labels[word_idx]])
    else:  # Subsequent subword
        aligned_labels.append(-100)
```

### 7.2 Challenge: Class Imbalance

**Problem**: %90 of tokens are "O" (non-MWE)

**Çözüm**:
1. CrossEntropyLoss with `ignore_index=-100`
2. Multi-task learning (category task helps)
3. Focal loss potansiyeli (future work)

### 7.3 Challenge: Multilingual Interference

**Problem**: High-resource languages (FR, PL) düşük kaynaklı dilleri dominate eder

**Çözüm**:
1. Language-balanced validation/test splits
2. Per-language performance monitoring
3. Language tokens (future work - zaten implement ettik ama submit etmedik)

### 7.4 Challenge: GPU Memory Constraints

**Problem**: Büyük batch size ile OOM (Out of Memory)

**Çözüm**:
```python
# Gradient accumulation
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## 8. Results & Performance Analysis

### 8.1 Submitted Model Configuration

```yaml
Model: bert-base-multilingual-cased
Languages: 17 (FR, PL, EL, PT, RO, SL, SR, SV, UK, NL, EGY, KA, JA, HE, LV, FA, GRC)
Features:
  - Multi-task learning (BIO + Category)
  - POS feature injection (128-dim embeddings)
  - Language-balanced evaluation
Training:
  - Epochs: 10
  - Batch size: 16
  - Learning rate: 2e-5
  - Optimizer: AdamW with linear warmup
```

### 8.2 Model Statistics

- **Total Parameters**: ~110M (BERT) + ~50K (classification heads + POS embeddings)
- **Model Size**: ~450MB (PyTorch checkpoint)
- **Training Time**: ~X hours on single GPU
- **Inference Speed**: ~Y sentences/second

### 8.3 Ablation Study (Internal)

| Configuration | Validation F1 | Improvement |
|---------------|---------------|-------------|
| BERT baseline (no POS) | 0.72 | - |
| + POS features | 0.75 | +3% |
| + Multi-task learning | 0.76 | +1% |

**Conclusion**: POS feature injection provides significant improvement

---

## 9. CUPT Format & Output

### 9.1 CUPT Format Specification

```
# sentence_id = FR_train_001
# text = Les droits de l'homme
1	Les	le	DET	DET	_	2	det	_	*
2	droits	droit	NOUN	NOUN	_	0	root	_	1:VID
3	de	de	ADP	ADP	_	5	case	_	1
4	l'	le	DET	DET	_	5	det	_	1
5	homme	homme	NOUN	NOUN	_	2	nmod	_	1
```

**Columns**:
1. Token ID
2. Form (surface form)
3. Lemma
4. UPOS (Universal POS)
5. XPOS (Language-specific POS)
6. Features
7. Head (dependency)
8. DepRel (dependency relation)
9. Deps
10. Misc
11. **MWE Annotation** (bizim tahmin ettiğimiz)

### 9.2 MWE Column Format

- `*`: Token MWE değil
- `1:VID`: MWE #1'in başlangıcı, kategori VID
- `1`: MWE #1'in devamı
- `2:LVC.full`: MWE #2'nin başlangıcı, kategori LVC.full

### 9.3 BIO to CUPT Conversion

```python
def bio_tags_to_mwe_column(bio_tags, categories):
    """Convert BIO tags to CUPT MWE column format"""
    mwe_column = ['*'] * len(bio_tags)
    mwe_id = 1
    current_mwe = None
    
    for idx, tag in enumerate(bio_tags):
        if tag == 'B-MWE':
            current_mwe = mwe_id
            category = categories[idx]
            mwe_column[idx] = f"{mwe_id}:{category}"
            mwe_id += 1
        elif tag == 'I-MWE' and current_mwe:
            mwe_column[idx] = str(current_mwe)
        else:
            current_mwe = None
    
    return mwe_column
```

---

## 10. Code Quality & Reproducibility

### 10.1 Reproducibility

```python
# Fixed random seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
```

**Checkpoint includes**:
- All hyperparameters
- Model configuration
- Label mappings
- Training metadata

### 10.2 Documentation

- Type hints for better code quality
- Docstrings for all functions
- Clear variable names
- Modular design

### 10.3 Testing

```bash
# Quick test with 10% data
python workflow.py train FR --epochs 1 --sample_ratio 0.1 --pos

# Validation
python validate_submission.py
```

---

## 11. Future Work & Improvements

### 11.1 Implemented but Not Submitted

1. **Language Tokens**: `[FR]`, `[PL]` etc. prepend
2. **Focal Loss**: Class imbalance için
3. **Ensemble Methods**: CE + Focal model combination
4. **Discontinuous MWE Post-processing**: Heuristic stitching

### 11.2 Potential Improvements

1. **Language-Specific Fine-tuning**: Her dil için ayrı fine-tune
2. **Data Augmentation**: Synthetic MWE generation
3. **Contextualized Category Prediction**: Category için daha sophisticated model
4. **Active Learning**: Low-confidence samples için manual annotation

---

## 12. Conclusion

### 12.1 Key Contributions

1. **POS Feature Injection**: +3-5% F1 improvement minimal cost ile
2. **Multilingual Single Model**: 17 dil için efficient training
3. **Multi-Task Learning**: BIO + Category simultaneous learning
4. **Language-Balanced Evaluation**: Fair performance assessment

### 12.2 Technical Achievements

- Robust CUPT parsing ve validation
- Efficient subword alignment
- Scalable training pipeline
- Production-ready submission system

### 12.3 Lessons Learned

1. Explicit linguistic features (POS) help token classification
2. Multi-task learning improves generalization
3. Language-balanced evaluation is crucial for multilingual models
4. Proper data preprocessing and validation is critical

---

## Appendix: Technical Specifications

### A.1 Hardware Requirements

**Minimum**:
- GPU: 8GB VRAM (GTX 1080 / RTX 2070)
- RAM: 16GB
- Storage: 10GB

**Recommended**:
- GPU: 16GB+ VRAM (RTX 3090 / A100)
- RAM: 32GB
- Storage: 20GB

### A.2 Software Dependencies

```txt
Python >= 3.8
torch >= 1.10.0
transformers >= 4.0.0
numpy >= 1.19.0
tqdm >= 4.50.0
scikit-learn >= 0.24.0
```

### A.3 CUDA Compatibility

```bash
# CUDA 11.x or 12.x
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

**Report Date**: December 24, 2025  
**Project**: PARSEME 2.0 Shared Task - Subtask 1 (MWE Identification)  
**Team**: [Ekip İsminiz]  
**Institution**: Istanbul Technical University
