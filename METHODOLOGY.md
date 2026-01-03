# 3. Methodology

Subtask 1, which requires identifying MWEs and assigning their categories in sentences provided in the CUPT format. Each sentence is a sequence of tokens with morphological and syntactic annotations. A key property of the data is the presence of both **continuous MWEs**  and **discontinuous MWEs**, and the strong imbalance between non-MWE tokens and MWE tokens

## 3.1 Submitted System (Official Architecture)

We formulate MWE detection as **token-level sequence prediction**, and we unify identification and typing under a shared encoder with task-specific output heads.

### 3.1.1 Backbone Encoder

Our submitted system uses a multilingual pretrained transformer encoder, specifically `bert-base-multilingual-cased`, to produce contextual token representations. Given an input token sequence x=(x1​,…,xn​), the encoder produces contextual embeddings.

### 3.1.2 Two-Head Token Prediction

Token prediction is modeled to have two-heads: BIO Identification + Category Classification 


