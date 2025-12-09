"""
Training script for PARSEME 2.0 MWE identification
"""
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from typing import List, Dict
import json
from datetime import datetime

from data_loader import CUPTDataLoader
from model import MWEIdentificationModel, MWETokenizer


class MWEDataset(Dataset):
    """PyTorch dataset for MWE identification with categories and POS tags"""
    
    def __init__(self, sentences: List[Dict], tokenizer: MWETokenizer, label_to_id: Dict[str, int], 
                 category_to_id: Dict[str, int], pos_to_id: Dict[str, int] = None, max_length: int = 512):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.category_to_id = category_to_id
        self.pos_to_id = pos_to_id
        self.max_length = max_length
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        tokens = sentence['tokens']
        labels = sentence['mwe_tags']
        categories = sentence['mwe_categories']
        pos_tags = sentence.get('pos_tags', None)
        
        # Tokenize and align labels
        tokenized = self.tokenizer.tokenize_and_align_labels(
            tokens, labels, self.label_to_id, categories, self.category_to_id, 
            pos_tags, self.pos_to_id, self.max_length
        )
        
        return {
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'labels': tokenized['labels'].squeeze(0),
            'category_labels': tokenized['category_labels'].squeeze(0),
            'pos_ids': tokenized['pos_ids'].squeeze(0)
        }


def compute_metrics(predictions: List[List[str]], references: List[List[str]]) -> Dict[str, float]:
    """
    Compute precision, recall, F1 for MWE identification
    
    Args:
        predictions: List of predicted tag sequences
        references: List of gold tag sequences
    
    Returns:
        Dictionary with precision, recall, F1
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for pred_tags, ref_tags in zip(predictions, references):
        for pred, ref in zip(pred_tags, ref_tags):
            if ref != 'O':
                if pred == ref:
                    true_positives += 1
                else:
                    false_negatives += 1
            elif pred != 'O':
                false_positives += 1
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }





def evaluate_model(model, dataloader, id_to_label, id_to_category, device):
    """Evaluate model on validation set"""
    model.eval()
    
    all_predictions = []
    all_references = []
    all_category_predictions = []
    all_category_references = []
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            category_labels = batch['category_labels'].to(device)
            pos_ids = batch['pos_ids'].to(device)
            
            outputs = model(input_ids, attention_mask, labels, category_labels, pos_ids)
            
            if outputs['loss'] is not None:
                total_loss += outputs['loss'].item()
                num_batches += 1
            
            bio_predictions = torch.argmax(outputs['bio_logits'], dim=-1)
            category_predictions = torch.argmax(outputs['category_logits'], dim=-1)
            
            # Convert to labels (skip padding)
            for i in range(bio_predictions.shape[0]):
                pred_tags = []
                ref_tags = []
                pred_cats = []
                ref_cats = []
                
                for j in range(bio_predictions.shape[1]):
                    if labels[i, j] != -100:
                        pred_tags.append(id_to_label[bio_predictions[i, j].item()])
                        ref_tags.append(id_to_label[labels[i, j].item()])
                        pred_cats.append(id_to_category[category_predictions[i, j].item()])
                        ref_cats.append(id_to_category[category_labels[i, j].item()])
                
                all_predictions.append(pred_tags)
                all_references.append(ref_tags)
                all_category_predictions.append(pred_cats)
                all_category_references.append(ref_cats)
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    metrics = compute_metrics(all_predictions, all_references)
    
    # Compute category accuracy (only for MWE tokens, not O)
    category_correct = 0
    category_total = 0
    for pred_cats, ref_cats, ref_tags in zip(all_category_predictions, all_category_references, all_references):
        for pc, rc, rt in zip(pred_cats, ref_cats, ref_tags):
            if rt != 'O':  # Only count MWE tokens
                category_total += 1
                if pc == rc:
                    category_correct += 1
    
    category_acc = category_correct / category_total if category_total > 0 else 0
    metrics['loss'] = avg_loss
    metrics['category_accuracy'] = category_acc
    
    return metrics


def train_mwe_model(
    train_file: str,
    dev_file: str,
    output_dir: str = 'models',
    model_name: str = 'bert-base-multilingual-cased',
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 512,
    seed: int = 42,
    sample_ratio: float = 1.0,
    use_pos: bool = False
):
    """
    Train MWE identification model
    
    Args:
        train_file: Path to training .cupt file
        dev_file: Path to development .cupt file
        output_dir: Directory to save model checkpoints
        model_name: Pretrained transformer model name
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        max_length: Maximum sequence length
        seed: Random seed
        sample_ratio: Ratio of training data to use (0.0-1.0, default: 1.0)
    """
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading training data...")
    data_loader = CUPTDataLoader()
    
    # Handle multiple training files (for multilingual training)
    train_files = train_file.split(',')
    train_sentences = []
    for tf in train_files:
        tf = tf.strip()
        sentences = data_loader.read_cupt_file(tf)
        print(f"  Loaded {len(sentences)} sentences from {tf}")
        train_sentences.extend(sentences)
    print(f"Total training sentences: {len(train_sentences)}")
    
    # Sample training data if requested
    if sample_ratio < 1.0:
        import random
        random.seed(seed)
        sample_size = int(len(train_sentences) * sample_ratio)
        train_sentences = random.sample(train_sentences, sample_size)
        print(f"Sampled {sample_ratio*100:.1f}% of training data: {len(train_sentences)} sentences")
    
    print("\nLoading development data...")
    # Handle multiple dev files (for multilingual training)
    dev_files = dev_file.split(',')
    
    import random
    random.seed(seed)
    
    val_sentences = []
    test_sentences = []
    
    for df in dev_files:
        df = df.strip()
        sentences = data_loader.read_cupt_file(df)
        print(f"  Loaded {len(sentences)} sentences from {df}")
        
        # Count MWEs in this language's dev set
        mwe_count = sum(1 for s in sentences for tag in s['mwe_tags'] if tag != 'O')
        
        # Shuffle each language's dev set separately
        random.shuffle(sentences)
        
        # Split each language 50/50 into validation and test
        split_idx = len(sentences) // 2
        lang_val = sentences[:split_idx]
        lang_test = sentences[split_idx:]
        
        # Count MWEs in val and test splits
        val_mwe_count = sum(1 for s in lang_val for tag in s['mwe_tags'] if tag != 'O')
        test_mwe_count = sum(1 for s in lang_test for tag in s['mwe_tags'] if tag != 'O')
        
        val_sentences.extend(lang_val)
        test_sentences.extend(lang_test)
        print(f"    -> Split into {len(lang_val)} val (MWEs: {val_mwe_count}) and {len(lang_test)} test (MWEs: {test_mwe_count})")
    
    print(f"\nTotal validation sentences: {len(val_sentences)}")
    val_total_mwe = sum(1 for s in val_sentences for tag in s['mwe_tags'] if tag != 'O')
    print(f"Total validation MWE tokens: {val_total_mwe}")
    
    print(f"Total test sentences: {len(test_sentences)}")
    test_total_mwe = sum(1 for s in test_sentences for tag in s['mwe_tags'] if tag != 'O')
    print(f"Total test MWE tokens: {test_total_mwe}")
    
    # Get label mappings
    label_to_id = data_loader.get_label_mapping()
    id_to_label = {v: k for k, v in label_to_id.items()}
    print(f"\nLabels: {label_to_id}")
    print(f"MWE categories found: {sorted(data_loader.mwe_categories)}")
    
    # Get category mappings
    category_to_id = data_loader.get_category_mapping()
    id_to_category = {v: k for k, v in category_to_id.items()}
    print(f"\nCategory labels ({len(category_to_id)}): {list(category_to_id.keys())[:10]}...")  # Show first 10
    
    # Get POS tag mappings (for feature injection) if enabled
    if use_pos:
        pos_to_id = data_loader.get_pos_mapping(train_sentences)
        print(f"\n✓ POS feature injection ENABLED")
        print(f"  POS tags ({len(pos_to_id)}): {list(pos_to_id.keys())}")
    else:
        pos_to_id = None
        print(f"\n✗ POS feature injection DISABLED (use --pos to enable)")
    
    # Initialize tokenizer and model
    print(f"\nInitializing model: {model_name}")
    tokenizer = MWETokenizer(model_name)
    model = MWEIdentificationModel(
        model_name, 
        num_labels=len(label_to_id), 
        num_categories=len(category_to_id),
        num_pos_tags=len(pos_to_id) if pos_to_id else 18,
        use_pos=use_pos
    )
    model.to(device)
    
    # Create datasets
    train_dataset = MWEDataset(train_sentences, tokenizer, label_to_id, category_to_id, pos_to_id, max_length)
    val_dataset = MWEDataset(val_sentences, tokenizer, label_to_id, category_to_id, pos_to_id, max_length)
    test_dataset = MWEDataset(test_sentences, tokenizer, label_to_id, category_to_id, pos_to_id, max_length)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    print(f"Total training steps: {total_steps}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}\n")
    
    best_f1 = -1  # Initialize to -1 so first epoch always saves
    training_history = []
    
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        print("-" * 50)
        
        # Training
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc="Training")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            category_labels = batch['category_labels'].to(device)
            pos_ids = batch['pos_ids'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask, labels, category_labels, pos_ids)
            loss = outputs['loss']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = total_loss / len(train_dataloader)
        
        # Evaluation on validation set
        print("\nEvaluating on validation set...")
        val_metrics = evaluate_model(model, val_dataloader, id_to_label, id_to_category, device)
        
        print(f"\nEpoch {epoch + 1} Results:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Val Precision: {val_metrics['precision']:.4f}")
        print(f"  Val Recall: {val_metrics['recall']:.4f}")
        print(f"  Val F1: {val_metrics['f1']:.4f}")
        print(f"  Val Category Acc: {val_metrics['category_accuracy']:.4f}")
        
        # Save training history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': val_metrics['loss'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_f1': val_metrics['f1'],
            'val_category_accuracy': val_metrics['category_accuracy']
        })
        
        # Save best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            print(f"\nNew best F1: {best_f1:.4f} - Saving model...")
            
            model_path = os.path.join(output_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'label_to_id': label_to_id,
                'category_to_id': category_to_id,
                'pos_to_id': pos_to_id,
                'model_name': model_name,
                'use_pos': use_pos
            }, model_path)
            
            # Save tokenizer
            tokenizer.tokenizer.save_pretrained(os.path.join(output_dir, 'tokenizer'))
        
        print()
    
    # Save training history
    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # Load the best model for final test evaluation (if it exists)
    model_path = os.path.join(output_dir, 'best_model.pt')
    if os.path.exists(model_path):
        print("\nLoading best model for test evaluation...")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Evaluating on test set...")
    else:
        print("\n⚠️  Warning: No model was saved (validation F1 never improved)")
        print("Evaluating on test set with final epoch model...")
    
    # Final evaluation on test set
    test_metrics = evaluate_model(model, test_dataloader, id_to_label, id_to_category, device)
    print(f"\nFinal Test Results:")
    print(f"  Test Precision: {test_metrics['precision']:.4f}")
    print(f"  Test Recall: {test_metrics['recall']:.4f}")
    print(f"  Test F1: {test_metrics['f1']:.4f}")
    print(f"  Test Category Acc: {test_metrics['category_accuracy']:.4f}")
    
    # Save test results
    test_results = {
        'test_precision': test_metrics['precision'],
        'test_recall': test_metrics['recall'],
        'test_f1': test_metrics['f1'],
        'test_loss': test_metrics['loss'],
        'test_category_accuracy': test_metrics['category_accuracy'],
        'best_val_f1': best_f1
    }
    test_path = os.path.join(output_dir, 'test_results.json')
    with open(test_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print("\nTraining completed!")
    print(f"Best Validation F1 score: {best_f1:.4f}")
    print(f"Test Category Accuracy: {test_metrics['category_accuracy']:.4f}")
    print(f"Model saved to: {output_dir}")
    
    return model, best_f1


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train MWE identification model')
    parser.add_argument('--train', type=str, required=True, help='Path to training .cupt file')
    parser.add_argument('--dev', type=str, required=True, help='Path to development .cupt file')
    parser.add_argument('--output', type=str, default='models', help='Output directory for models')
    parser.add_argument('--model_name', type=str, default='bert-base-multilingual-cased', 
                       help='Pretrained model name')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--sample_ratio', type=float, default=1.0, 
                       help='Ratio of training data to use (0.0-1.0, default: 1.0)')
    parser.add_argument('--pos', action='store_true', 
                       help='Enable POS tag feature injection (improves performance by 3-5%%)')
    
    args = parser.parse_args()
    
    train_mwe_model(
        train_file=args.train,
        dev_file=args.dev,
        output_dir=args.output,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
        seed=args.seed,
        sample_ratio=args.sample_ratio,
        use_pos=args.pos
    )
