# Related Works

## 1. Base Models and Base Architecture

We began our investigation with **BERT-based token classification** as the foundational approach for multiword expression (MWE) identification. Specifically, we employed `bert-base-multilingual-cased` (110M parameters) as our base transformer model, chosen for its proven cross-lingual capabilities and wide language coverage supporting all 17 languages in the PARSEME 2.0 shared task.

* **Primary task**: BIO sequence labeling (O, B-MWE, I-MWE tags)
* **Secondary task**: MWE category classification (19+ categories including VID, LVC.full, NID, etc.)

This multi-task design enables the model to simultaneously learn MWE boundaries and semantic categories, with the category classification task providing auxiliary supervision that improves boundary detection.

## 2. Linguistic Feature Enhancement: POS Tag Injection

- Recognizing that MWEs exhibit strong morphosyntactic patterns (e.g., VID follows VERB+NOUN structures, LVC.full combines VERB+NOUN), we introduced **POS feature injection** to provide explicit linguistic signals to the model.

- The key insight is that while BERT implicitly learns syntactic information, explicit POS features accelerate learning and improve generalization, especially when training data is limited.

## 3. Structural Constraints: CRF Layer Integration for Discontinuous MWEs

Despite strong performance on continuous MWEs, our initial models struggled with discontinuous patterns. The problem stems from independent token-level classification, which cannot enforce sequence-level constraints.

**Solution: Conditional Random Fields (CRF)**

We augmented our architecture with an optional **CRF layer** on top of BERT outputs:

* Learns valid BIO tag transition probabilities
* Prevents invalid sequences (e.g., I-MWE without preceding B-MWE)
* Enables global sequence optimization via Viterbi decoding

## 4. Addressing Class Imbalance: Focal Loss

Analysis of our training data revealed severe class imbalance at multiple levels. First, at the token level: approximately 90% of tokens are labeled as "O" (outside MWE), while only 10% belong to B-MWE or I-MWE classes. Second, at the category level: the task involves identifying **19+ distinct MWE categories** with highly skewed distributions.

From our analysis, categories like VID and NID account for the majority of MWEs, while rare categories like MVC or compositional types appear in less than 1% of instances. This dual imbalance—both in BIO tags and category distributions—causes the model to bias toward predicting majority classes, resulting in low recall on minority categories and poor performance on rare MWE types.

To address this challenge, we implemented **Focal Loss** (Lin et al., 2017), which down-weights easy examples (abundant O tags and common categories) and focuses learning on hard examples (rare B-MWE, I-MWE tags and uncommon MWE types).

## 5. Improving Multilingual Learning with Language Tags

When training a single model on multiple languages, we observed **language interference** effects: high-resource languages (FR, RO with 60K+ sentences) dominated the learning process, degrading performance on low-resource languages.

**Solution: Language-Conditioned Inputs**

We introduced language-specific tokens (`[FR]`, `[PL]`, `[EL]`, etc.) prepended to each sentence during training:

* Expanded BERT's vocabulary with 17 new special tokens
* Model embeddings grew from 119,547 → 119,564 tokens
* Enables the attention mechanism to distinguish language-specific patterns

This approach, inspired by multilingual translation models, provides an explicit language signal that helps the model:

* Learn language-specific MWE patterns
* Reduce cross-lingual interference

# 

## 6. Ensemble Methods for Robustness

To leverage the complementary strengths of different training objectives, we developed an **ensemble prediction system** combining:

* **Cross-Entropy model**: Strong on high-confidence predictions
* **Focal Loss model**: Better recall on rare/difficult cases
