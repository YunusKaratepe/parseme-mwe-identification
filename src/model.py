"""
Transformer-based MWE identification model for PARSEME 2.0
Uses BERT/RoBERTa for token classification with BIO tagging
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import List, Dict, Tuple, Optional


class MWEIdentificationModel(nn.Module):
    """Token classification model for MWE identification"""
    
    def __init__(self, model_name: str = 'bert-base-multilingual-cased', num_labels: int = 3, dropout: float = 0.1):
        super().__init__()
        
        self.model_name = model_name
        self.num_labels = num_labels
        
        # Load pretrained transformer
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask, labels=None):
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            labels: [batch_size, seq_len] (optional, for training)
        
        Returns:
            loss (if labels provided), logits
        """
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get hidden states
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Apply dropout and classifier
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)  # [batch_size, seq_len, num_labels]
        
        loss = None
        if labels is not None:
            # Calculate cross-entropy loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return {'loss': loss, 'logits': logits}


class MWETokenizer:
    """Tokenizer wrapper that handles subword alignment for MWE tagging"""
    
    def __init__(self, model_name: str = 'bert-base-multilingual-cased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def tokenize_and_align_labels(
        self, 
        tokens: List[str], 
        labels: List[str],
        label_to_id: Dict[str, int],
        max_length: int = 512
    ) -> Dict:
        """
        Tokenize words and align BIO labels to subword tokens
        
        Args:
            tokens: List of words
            labels: List of BIO tags (same length as tokens)
            label_to_id: Mapping from label string to ID
            max_length: Maximum sequence length
        
        Returns:
            Dictionary with input_ids, attention_mask, and aligned labels
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
        
        # Align labels to subword tokens
        word_ids = tokenized.word_ids(batch_index=0)
        aligned_labels = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens get -100 (ignored in loss)
                aligned_labels.append(-100)
            elif word_idx != previous_word_idx:
                # First subword of a word gets the original label
                aligned_labels.append(label_to_id[labels[word_idx]])
            else:
                # Subsequent subwords get -100 (only first subword is labeled)
                aligned_labels.append(-100)
            
            previous_word_idx = word_idx
        
        tokenized['labels'] = torch.tensor([aligned_labels])
        
        return tokenized
    
    def batch_tokenize_and_align(
        self,
        batch_tokens: List[List[str]],
        batch_labels: List[List[str]],
        label_to_id: Dict[str, int],
        max_length: int = 512
    ) -> Dict:
        """Tokenize and align labels for a batch of sentences"""
        
        all_input_ids = []
        all_attention_mask = []
        all_labels = []
        
        for tokens, labels in zip(batch_tokens, batch_labels):
            tokenized = self.tokenize_and_align_labels(tokens, labels, label_to_id, max_length)
            all_input_ids.append(tokenized['input_ids'])
            all_attention_mask.append(tokenized['attention_mask'])
            all_labels.append(tokenized['labels'])
        
        return {
            'input_ids': torch.cat(all_input_ids, dim=0),
            'attention_mask': torch.cat(all_attention_mask, dim=0),
            'labels': torch.cat(all_labels, dim=0)
        }


def predict_mwe_tags(
    model: MWEIdentificationModel,
    tokenizer: MWETokenizer,
    tokens: List[str],
    id_to_label: Dict[int, str],
    device: torch.device
) -> List[str]:
    """
    Predict MWE tags for a sentence
    
    Args:
        model: Trained MWE identification model
        tokenizer: MWE tokenizer
        tokens: List of words
        id_to_label: Mapping from label ID to string
        device: torch device
    
    Returns:
        List of predicted BIO tags
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
        predictions = torch.argmax(outputs['logits'], dim=-1)
    
    # Get word-level predictions (skip special tokens and subwords)
    word_ids = tokenized.word_ids(batch_index=0)
    word_predictions = []
    previous_word_idx = None
    
    for idx, word_idx in enumerate(word_ids):
        if word_idx is not None and word_idx != previous_word_idx:
            pred_id = predictions[0, idx].item()
            word_predictions.append(id_to_label[pred_id])
            previous_word_idx = word_idx
    
    return word_predictions


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
