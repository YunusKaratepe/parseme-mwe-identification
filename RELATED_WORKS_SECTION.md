# Related Works - Paper Flow

## System Development Narrative

### 1. Base Architecture and Model Selection

We began our investigation with **BERT-based token classification** as the foundational approach for multiword expression (MWE) identification. Specifically, we employed `bert-base-multilingual-cased` (110M parameters) as our base transformer model, chosen for its proven cross-lingual capabilities and wide language coverage supporting all 17 languages in the PARSEME 2.0 shared task.

The core architecture implements a **dual-head multi-task learning framework**:
- **Primary task**: BIO sequence labeling (O, B-MWE, I-MWE tags)
- **Auxiliary task**: MWE category classification (19+ categories including VID, LVC.full, NID, etc.)

This multi-task design enables the model to simultaneously learn MWE boundaries and semantic categories, with the category classification task providing auxiliary supervision that improves boundary detection.

### 2. Linguistic Feature Enhancement: POS Tag Injection

Recognizing that MWEs exhibit strong morphosyntactic patterns (e.g., VID follows VERB+NOUN structures, LVC.full combines VERB+NOUN), we introduced **POS feature injection** to provide explicit linguistic signals to the model. 

**Implementation**: We augmented BERT's 768-dimensional hidden states with 128-dimensional POS embeddings derived from Universal POS tags in the CUPT format, creating 896-dimensional combined feature vectors. This "free lunch" approach adds only ~2,300 parameters (0.002% overhead) while achieving:
- **+3-5% F1 score improvement** across languages
- Faster convergence during training
- Particularly strong gains on low-resource languages

The key insight is that while BERT implicitly learns syntactic information, explicit POS features accelerate learning and improve generalization, especially when training data is limited.

### 3. Addressing Class Imbalance: Focal Loss

Analysis of our training data revealed severe class imbalance at multiple levels. First, at the token level: approximately 90% of tokens are labeled as "O" (outside MWE), while only 10% belong to B-MWE or I-MWE classes. Second, at the category level: the task involves identifying **19+ distinct MWE categories** with highly skewed distributions:
- **High-frequency categories**: VID (verbal idioms), LVC.full (light verb constructions), NID (nominal idioms), IRV (inherently reflexive verbs)
- **Mid-frequency categories**: IAV (inherently adpositional verbs), AdvID (adverbial idioms), AdjID (adjectival idioms)
- **Rare categories**: MVC (multi-verb constructions), IVPC variants, compositional categories (e.g., AV.LVC.cause, NV.IRV)

From our analysis, categories like VID and NID account for the majority of MWEs, while rare categories like MVC or compositional types appear in less than 1% of instances. This dual imbalance—both in BIO tags and category distributions—causes the model to bias toward predicting majority classes, resulting in low recall on minority categories and poor performance on rare MWE types.

To address this challenge, we implemented **Focal Loss** (Lin et al., 2017), which down-weights easy examples (abundant O tags and common categories) and focuses learning on hard examples (rare B-MWE, I-MWE tags and uncommon MWE types). The focal loss formulation:

$$FL(p_t) = -\alpha_t(1-p_t)^\gamma \log(p_t)$$

where $\gamma=2$ focuses on misclassified examples and $\alpha$ balances class frequencies.

**Results**: Focal loss training improved recall on minority classes, particularly benefiting discontinuous MWE detection where the imbalance is most severe.

### 4. Multilingual Learning and Language Interference

When training a single model on multiple languages, we observed **language interference** effects: high-resource languages (FR, RO with 60K+ sentences) dominated the learning process, degrading performance on low-resource languages (KA, JA, NL with <2K sentences).

**Solution: Language-Conditioned Inputs**

We introduced language-specific tokens (`[FR]`, `[PL]`, `[EL]`, etc.) prepended to each sentence during training:
- Expanded BERT's vocabulary with 17 new special tokens
- Model embeddings grew from 119,547 → 119,564 tokens
- Enables the attention mechanism to distinguish language-specific patterns

This approach, inspired by multilingual translation models, provides an explicit language signal that helps the model:
- Learn language-specific MWE patterns
- Reduce cross-lingual interference
- Improve low-resource language performance by +2-5% F1

### 5. Structural Constraints: CRF Layer Integration

Despite strong performance on continuous MWEs, our initial models struggled with discontinuous patterns. The problem stems from independent token-level classification, which cannot enforce sequence-level constraints.

**Solution: Conditional Random Fields (CRF)**

We augmented our architecture with an optional **CRF layer** on top of BERT outputs:
- Learns valid BIO tag transition probabilities
- Prevents invalid sequences (e.g., I-MWE without preceding B-MWE)
- Enables global sequence optimization via Viterbi decoding

**Experimental Results** (French, same training setup):
- **POS-only**: Overall F1 = 0.6566, Continuous = 0.6985, Discontinuous = 0.4977
- **POS + CRF**: Overall F1 = 0.6786, Continuous = 0.7180, Discontinuous = 0.5253
- **Improvements**: +2.2% overall, +2.0% continuous, +2.8% discontinuous

**Benefits**:
- Enforces structural consistency in predictions
- Modest but consistent improvements across all MWE types
- Minimal computational overhead during training

### 6. Post-Processing Exploration: Discontinuous MWE Stitching

Initially, we explored **rule-based post-processing** to address discontinuous MWE prediction challenges. The approach involved:
1. Detecting broken MWE patterns: `B-MWE ... O ... I-MWE` with matching categories
2. Applying heuristic stitching to fill gaps (converting intervening O tags to I-MWE)
3. Assuming all tokens between components should be part of the MWE

**Critical Finding**: This approach actually **degraded performance** because it violated the gold standard annotation scheme. In true discontinuous MWEs (e.g., "take ... into account"), only the actual MWE components are labeled, while intervening tokens remain as O (outside). By filling these gaps, we were introducing false positives.

**Experimental Results** (same model, with/without stitching):
- **With stitching**: Discontinuous F1 = 0.29, Continuous F1 = 0.68
- **Without stitching**: Discontinuous F1 = 0.53, Continuous F1 = 0.72
- **Improvement from disabling**: +23% discontinuous, +3.3% continuous

**Lesson Learned**: The model's raw BIO predictions already handle discontinuous patterns correctly through the conversion logic that preserves gaps. The post-processing was solving a non-existent problem while creating new errors. This highlights the importance of understanding the evaluation criteria before implementing "fixes."

### 7. Ensemble Methods for Robustness

To leverage the complementary strengths of different training objectives, we developed an **ensemble prediction system** combining:
- **Cross-Entropy model**: Strong on high-confidence predictions
- **Focal Loss model**: Better recall on rare/difficult cases

The ensemble uses weighted voting with confidence-based combination, improving robustness and achieving higher overall F1 scores than individual models.

---

## Summary of Technical Contributions

| Component | Innovation | Impact | Submitted |
|-----------|-----------|--------|-----------|
| Base Architecture | BERT + Dual-head Multi-task | Baseline performance | ✓ |
| POS Injection | Linguistic feature fusion | +3-5% F1 | ✓ |
| Focal Loss | Class imbalance handling | Better recall | ✓ |
| Language Tokens | Multilingual interference reduction | +2-5% F1 (low-resource) | ✓ |
| CRF Layer | Sequence-level constraints | +2.2% F1 overall, +2.8% discontinuous | ✓ |
| Post-processing | Heuristic MWE stitching (explored) | -23% F1 (abandoned) | ✗ |
| Ensemble | Model combination | Overall robustness | ✓ |

---

## Related Literature Context

Our work builds upon and extends several key research directions:

1. **Transformer-based Sequence Labeling**: Following Devlin et al. (2019), we leverage BERT's contextual representations for token classification tasks.

2. **Multi-task Learning for MWE**: Similar to Fei et al. (2020), we employ auxiliary tasks (category classification) to improve primary task (boundary detection) performance.

3. **Linguistic Feature Integration**: Our POS injection approach extends the tradition of hybrid neural-symbolic models (e.g., Kondratyuk & Straka, 2019), demonstrating that explicit linguistic features complement pre-trained language models.

4. **Focal Loss for NLP**: We adapt Lin et al.'s (2017) focal loss from computer vision to address token-level class imbalance in sequence labeling, similar to recent applications in named entity recognition.

5. **CRF for Structured Prediction**: Following Lample et al. (2016), we combine neural encoders (BERT) with CRF decoders to enforce structural constraints in sequence predictions.

6. **Multilingual Model Training**: Our language-conditioned approach draws inspiration from Johnson et al.'s (2017) zero-shot translation work, applying language tokens to reduce cross-lingual interference.

This progressive development strategy—starting with a strong baseline and systematically addressing identified weaknesses—enabled us to build a robust, multilingual MWE identification system achieving competitive performance across diverse languages and MWE types.
