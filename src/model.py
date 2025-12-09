"""
Transformer-based MWE identification model for PARSEME 2.0
Uses BERT/RoBERTa for token classification with BIO tagging
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import List, Dict, Tuple, Optional


class MWEIdentificationModel(nn.Module):
    """Multi-task token classification model for MWE identification and categorization"""
    
    def __init__(self, model_name: str = 'bert-base-multilingual-cased', num_labels: int = 3, 
                 num_categories: int = 1, dropout: float = 0.1, num_pos_tags: int = 18, 
                 use_pos: bool = True):
        super().__init__()
        
        self.model_name = model_name
        self.num_labels = num_labels
        self.num_categories = num_categories
        self.use_pos = use_pos
        
        # Load pretrained transformer
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # POS tag embeddings ("free lunch" feature injection)
        if self.use_pos:
            self.pos_embedding = nn.Embedding(num_pos_tags, 128)  # Small embedding dim
            self.pos_dropout = nn.Dropout(dropout)
            # Combined hidden size = BERT hidden + POS embedding
            combined_hidden_size = self.config.hidden_size + 128
        else:
            combined_hidden_size = self.config.hidden_size
        
        # Shared dropout
        self.dropout = nn.Dropout(dropout)
        
        # Multi-task classification heads (take combined features)
        self.bio_classifier = nn.Linear(combined_hidden_size, num_labels)
        self.category_classifier = nn.Linear(combined_hidden_size, num_categories)
        
    def forward(self, input_ids, attention_mask, labels=None, category_labels=None, pos_ids=None):
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            labels: [batch_size, seq_len] (optional, BIO labels for training)
            category_labels: [batch_size, seq_len] (optional, category labels for training)
            pos_ids: [batch_size, seq_len] (optional, POS tag IDs for feature injection)
        
        Returns:
            dict with loss, bio_logits, category_logits
        """
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get hidden states
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        sequence_output = self.dropout(sequence_output)
        
        # Inject POS tag features ("free lunch" innovation)
        if self.use_pos and pos_ids is not None:
            pos_embeds = self.pos_embedding(pos_ids)  # [batch_size, seq_len, 128]
            pos_embeds = self.pos_dropout(pos_embeds)
            # Concatenate BERT features with POS features
            sequence_output = torch.cat([sequence_output, pos_embeds], dim=-1)
        
        # Predict BIO tags
        bio_logits = self.bio_classifier(sequence_output)  # [batch_size, seq_len, num_labels]
        
        # Predict MWE categories
        category_logits = self.category_classifier(sequence_output)  # [batch_size, seq_len, num_categories]
        
        loss = None
        if labels is not None and category_labels is not None:
            # BIO tagging loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            bio_loss = loss_fct(bio_logits.view(-1, self.num_labels), labels.view(-1))
            
            # Category prediction loss
            category_loss = loss_fct(category_logits.view(-1, self.num_categories), category_labels.view(-1))
            
            # Combined loss (equal weighting)
            loss = bio_loss + category_loss
        
        return {
            'loss': loss, 
            'bio_logits': bio_logits,
            'category_logits': category_logits
        }


class MWETokenizer:
    """Tokenizer wrapper that handles subword alignment for MWE tagging"""
    
    def __init__(self, model_name: str = 'bert-base-multilingual-cased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def tokenize_and_align_labels(
        self, 
        tokens: List[str], 
        labels: List[str],
        label_to_id: Dict[str, int],
        categories: Optional[List[str]] = None,
        category_to_id: Optional[Dict[str, int]] = None,
        pos_tags: Optional[List[str]] = None,
        pos_to_id: Optional[Dict[str, int]] = None,
        max_length: int = 512
    ) -> Dict:
        """
        Tokenize words and align BIO labels, categories, and POS tags to subword tokens
        
        Args:
            tokens: List of words
            labels: List of BIO tags (same length as tokens)
            label_to_id: Mapping from label string to ID
            categories: List of MWE categories (optional)
            category_to_id: Mapping from category string to ID (optional)
            pos_tags: List of POS tags (optional, for feature injection)
            pos_to_id: Mapping from POS tag to ID (optional)
            max_length: Maximum sequence length
        
        Returns:
            Dictionary with input_ids, attention_mask, aligned labels, category labels, and POS IDs
        """
        # Tokenize with word alignment
        tokenized = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Align labels and POS tags to subword tokens
        word_ids = tokenized.word_ids(batch_index=0)
        aligned_labels = []
        aligned_categories = []
        aligned_pos = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens get -100 (ignored in loss) or 0 for POS
                aligned_labels.append(-100)
                aligned_categories.append(-100)
                aligned_pos.append(0 if pos_tags and pos_to_id else -100)
            elif word_idx != previous_word_idx:
                # First subword of a word gets the original label
                aligned_labels.append(label_to_id[labels[word_idx]])
                if categories and category_to_id:
                    aligned_categories.append(category_to_id[categories[word_idx]])
                else:
                    aligned_categories.append(-100)
                # POS tag for this word
                if pos_tags and pos_to_id:
                    aligned_pos.append(pos_to_id.get(pos_tags[word_idx], 0))  # 0 = unknown POS
                else:
                    aligned_pos.append(-100)
            else:
                # Subsequent subwords get -100 (only first subword is labeled)
                # But inherit POS from first subword
                aligned_labels.append(-100)
                aligned_categories.append(-100)
                if pos_tags and pos_to_id:
                    aligned_pos.append(pos_to_id.get(pos_tags[word_idx], 0))
                else:
                    aligned_pos.append(-100)
            
            previous_word_idx = word_idx
        
        tokenized['labels'] = torch.tensor([aligned_labels])
        tokenized['category_labels'] = torch.tensor([aligned_categories])
        tokenized['pos_ids'] = torch.tensor([aligned_pos])
        
        return tokenized
    
    def batch_tokenize_and_align(
        self,
        batch_tokens: List[List[str]],
        batch_labels: List[List[str]],
        label_to_id: Dict[str, int],
        batch_categories: Optional[List[List[str]]] = None,
        category_to_id: Optional[Dict[str, int]] = None,
        batch_pos_tags: Optional[List[List[str]]] = None,
        pos_to_id: Optional[Dict[str, int]] = None,
        max_length: int = 512
    ) -> Dict:
        """Tokenize and align labels for a batch of sentences"""
        
        all_input_ids = []
        all_attention_mask = []
        all_labels = []
        all_category_labels = []
        all_pos_ids = []
        
        for i, (tokens, labels) in enumerate(zip(batch_tokens, batch_labels)):
            categories = batch_categories[i] if batch_categories else None
            pos_tags = batch_pos_tags[i] if batch_pos_tags else None
            tokenized = self.tokenize_and_align_labels(
                tokens, labels, label_to_id, categories, category_to_id, 
                pos_tags, pos_to_id, max_length
            )
            all_input_ids.append(tokenized['input_ids'])
            all_attention_mask.append(tokenized['attention_mask'])
            all_labels.append(tokenized['labels'])
            all_category_labels.append(tokenized['category_labels'])
            all_pos_ids.append(tokenized['pos_ids'])
        
        return {
            'input_ids': torch.cat(all_input_ids, dim=0),
            'attention_mask': torch.cat(all_attention_mask, dim=0),
            'labels': torch.cat(all_labels, dim=0),
            'pos_ids': torch.cat(all_pos_ids, dim=0),
            'category_labels': torch.cat(all_category_labels, dim=0)
        }


def predict_mwe_tags(
    model: MWEIdentificationModel,
    tokenizer: MWETokenizer,
    tokens: List[str],
    id_to_label: Dict[int, str],
    id_to_category: Dict[int, str],
    device: torch.device
) -> Tuple[List[str], List[str]]:
    """
    Predict MWE tags and categories for a sentence
    
    Args:
        model: Trained MWE identification model
        tokenizer: MWE tokenizer
        tokens: List of words
        id_to_label: Mapping from label ID to BIO tag string
        id_to_category: Mapping from category ID to category string
        device: torch device
    
    Returns:
        Tuple of (bio_tags, categories)
    """
    model.eval()
    
    # Create dummy labels for tokenization
    dummy_labels = ['O'] * len(tokens)
    label_to_id = {'O': 0, 'B-MWE': 1, 'I-MWE': 2}
    
    # Tokenize
    tokenized = tokenizer.tokenize_and_align_labels(tokens, dummy_labels, label_to_id)
    
    input_ids = tokenized['input_ids'].to(device)
    attention_mask = tokenized['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        bio_predictions = torch.argmax(outputs['bio_logits'], dim=-1)
        category_predictions = torch.argmax(outputs['category_logits'], dim=-1)
    
    # Get word-level predictions (skip special tokens and subwords)
    word_ids = tokenized.word_ids(batch_index=0)
    word_bio_tags = []
    word_categories = []
    previous_word_idx = None
    
    for idx, word_idx in enumerate(word_ids):
        if word_idx is not None and word_idx != previous_word_idx:
            bio_pred_id = bio_predictions[0, idx].item()
            cat_pred_id = category_predictions[0, idx].item()
            word_bio_tags.append(id_to_label[bio_pred_id])
            word_categories.append(id_to_category[cat_pred_id])
            previous_word_idx = word_idx
    
    return word_bio_tags, word_categories


if __name__ == '__main__':
    # Test model initialization
    print("Testing MWE Identification Model...")
    
    model = MWEIdentificationModel()
    print(f"Model initialized with {model.num_labels} labels")
    print(f"Using transformer: {model.model_name}")
    
    # Test tokenizer
    tokenizer = MWETokenizer()
    tokens = ["This", "is", "a", "test", "sentence"]
    labels = ["O", "O", "O", "B-MWE", "I-MWE"]
    label_to_id = {"O": 0, "B-MWE": 1, "I-MWE": 2}
    
    tokenized = tokenizer.tokenize_and_align_labels(tokens, labels, label_to_id)
    print(f"\nTokenized input shape: {tokenized['input_ids'].shape}")
    print(f"Labels shape: {tokenized['labels'].shape}")
    
    print("\nModel ready for training!")
