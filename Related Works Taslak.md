# Related Works

## 1. Base Models and Base Architecture

We began our investigation with **BERT-based token classification** as the foundational approach for multiword expression (MWE) identification. Specifically, we employed `bert-base-multilingual-cased` (110M parameters) as our base transformer model, chosen for its proven cross-lingual capabilities and wide language coverage supporting all 17 languages in the PARSEME 2.0 shared task.

* **Primary task**: BIO sequence labeling (O, B-MWE, I-MWE tags)
* **Secondary task**: MWE category classification (19+ categories including VID, LVC.full, NID, etc.)

This multi-task design enables the model to simultaneously learn MWE boundaries and semantic categories, with the category classification task providing auxiliary supervision that improves boundary detection.

## 2. Linguistic Feature Enhancement: POS Tag Injection

- Recognizing that MWEs exhibit strong morphosyntactic patterns (e.g., VID follows VERB+NOUN structures, LVC.full combines VERB+NOUN), we introduced **POS feature injection** to provide explicit linguistic signals to the model.

- The key insight is that while BERT implicitly learns syntactic information, explicit POS features accelerate learning and improve generalization, especially when training data is limited.

## 3. Addressing Class Imbalance: Focal Loss

- Analysis of our training data revealed severe class imbalance: approximately 90% of tokens are labeled as "O" (outside MWE), while only 10% belong to B-MWE or I-MWE classes. This imbalance causes the model to bias toward predicting the majority class, resulting in low recall.




