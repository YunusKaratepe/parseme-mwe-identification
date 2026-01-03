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
        language = sentence.get('language', None)
        
        # Tokenize and align labels
        tokenized = self.tokenizer.tokenize_and_align_labels(
            tokens, labels, self.label_to_id, categories, self.category_to_id, 
            pos_tags, self.pos_to_id, self.max_length, language
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
    use_pos: bool = False,
    use_lang_tokens: bool = False,
    loss_type: str = 'ce',
    use_crf: bool = False,
    resume_from: str = None
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
        use_pos: Enable POS feature injection
        use_lang_tokens: Enable language-conditioned inputs (prepend [LANG] tokens)
        loss_type: Loss function type - 'ce' for CrossEntropy, 'focal' for Focal Loss
        use_crf: Enable CRF layer for BIO tagging (improves discontinuous MWE detection)
        resume_from: Path to checkpoint to resume training from (optional)
    """
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check if resuming from checkpoint
    start_epoch = 0
    resume_checkpoint = None
    if resume_from and os.path.exists(resume_from):
        print(f"\n🔄 Resuming training from checkpoint: {resume_from}")
        resume_checkpoint = torch.load(resume_from, map_location=device)
        print(f"   Previous best F1: {resume_checkpoint.get('best_f1', 0):.4f}")
        print(f"   Checkpoint epoch: {resume_checkpoint.get('epoch', 0)}")
        # Override settings from checkpoint
        model_name = resume_checkpoint['model_name']
        use_pos = resume_checkpoint.get('use_pos', False)
        use_lang_tokens = resume_checkpoint.get('use_lang_tokens', False)
        loss_type = resume_checkpoint.get('loss_type', 'ce')
        use_crf = resume_checkpoint.get('use_crf', False)
        print(f"   Loaded settings: pos={use_pos}, lang_tokens={use_lang_tokens}, loss={loss_type}, crf={use_crf}")
    elif resume_from:
        print(f"⚠️  Warning: Resume checkpoint not found: {resume_from}")
        print(f"   Starting fresh training instead...")
    
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

    # Split training into train/validation: last 10% as validation (no shuffling)
    total_train = len(train_sentences)
    if total_train >= 2:
        val_size = max(1, int(total_train * 0.10))
    else:
        val_size = 0

    if val_size > 0:
        val_sentences = train_sentences[-val_size:]
        train_sentences = train_sentences[:-val_size]
    else:
        val_sentences = []

    print(f"\nTraining/Validation split (from train.cupt):")
    print(f"  Train sentences: {len(train_sentences)}")
    print(f"  Validation sentences (last 10%): {len(val_sentences)}")
    
    print("\nLoading development data (used as TEST set)...")
    # Handle multiple dev files (for multilingual training)
    dev_files = dev_file.split(',')

    test_sentences = []
    for df in dev_files:
        df = df.strip()
        sentences = data_loader.read_cupt_file(df)
        print(f"  Loaded {len(sentences)} sentences from {df}")
        test_sentences.extend(sentences)

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
    print(f"Loss function: {loss_type.upper()} ({'Focal Loss' if loss_type == 'focal' else 'Cross-Entropy'})")
    if use_crf:
        print(f"CRF Layer: ENABLED (improves discontinuous MWE detection)")
    else:
        print(f"CRF Layer: DISABLED (use --crf to enable)")
    
    tokenizer = MWETokenizer(model_name, use_lang_tokens=use_lang_tokens)
    
    # Use checkpoint's label mappings if resuming
    if resume_checkpoint:
        label_to_id = resume_checkpoint['label_to_id']
        category_to_id = resume_checkpoint['category_to_id']
        pos_to_id = resume_checkpoint.get('pos_to_id', pos_to_id)
        print(f"   Using label mappings from checkpoint")
    
    model = MWEIdentificationModel(
        model_name, 
        num_labels=len(label_to_id), 
        num_categories=len(category_to_id),
        num_pos_tags=len(pos_to_id) if pos_to_id else 18,
        use_pos=use_pos,
        loss_type=loss_type,
        use_crf=use_crf
    )
    
    # Load model state if resuming
    if resume_checkpoint:
        model.load_state_dict(resume_checkpoint['model_state_dict'])
        print(f"✓ Loaded model weights from checkpoint")
    
    # Resize token embeddings if language tokens were added
    if use_lang_tokens:
        model.transformer.resize_token_embeddings(len(tokenizer.get_tokenizer()))
        print(f"✓ Language-conditioned inputs ENABLED")
        print(f"  Resized embeddings to {len(tokenizer.get_tokenizer())} tokens")
    else:
        print(f"✗ Language-conditioned inputs DISABLED (use --lang_tokens to enable)")
    
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
    
    # Load optimizer state if resuming
    if resume_checkpoint and 'optimizer_state_dict' in resume_checkpoint:
        optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
        print(f"✓ Loaded optimizer state from checkpoint")
    
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
    
    best_f1 = resume_checkpoint.get('best_f1', -1) if resume_checkpoint else -1
    start_epoch = resume_checkpoint.get('epoch', -1) + 1 if resume_checkpoint else 0
    
    # Load training history if resuming
    training_history = []
    if resume_checkpoint and os.path.exists(os.path.join(output_dir, 'training_history.json')):
        try:
            with open(os.path.join(output_dir, 'training_history.json'), 'r') as f:
                training_history = json.load(f)
            print(f"✓ Loaded training history ({len(training_history)} previous epochs)")
        except:
            pass
    
    if resume_checkpoint:
        print(f"\n🔄 Continuing from epoch {start_epoch + 1} (best F1 so far: {best_f1:.4f})")
        epochs = start_epoch + epochs  # Train for additional epochs
        print(f"   Will train until epoch {epochs}\n")
    
    for epoch in range(start_epoch, epochs):
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
            
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            model_path = os.path.join(output_dir, 'best_model.pt')
            
            try:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_f1': best_f1,
                    'label_to_id': label_to_id,
                    'category_to_id': category_to_id,
                    'pos_to_id': pos_to_id,
                    'model_name': model_name,
                    'use_pos': use_pos,
                    'use_lang_tokens': use_lang_tokens,
                    'loss_type': loss_type,
                    'use_crf': use_crf
                }, model_path)
                print(f"✓ Model saved successfully")
                
                # Save tokenizer
                tokenizer_dir = os.path.join(output_dir, 'tokenizer')
                os.makedirs(tokenizer_dir, exist_ok=True)
                tokenizer.tokenizer.save_pretrained(tokenizer_dir)
                print(f"✓ Tokenizer saved")
            except Exception as e:
                print(f"⚠️  ERROR saving model: {e}")
                import traceback
                traceback.print_exc()
        
        print()
    
    # Save training history
    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # Save training info
    # Extract language codes from train_file paths
    train_file_list = [tf.strip() for tf in train_file.split(',')]
    languages = []
    for tf in train_file_list:
        # Extract language code from path like "2.0/subtask1/FR/train.cupt"
        parts = tf.split('/')
        if len(parts) >= 3:
            lang = parts[-2]  # Get parent directory name
            if lang not in languages:
                languages.append(lang)
    
    training_info = {
        'languages': languages,
        'sample_ratio': sample_ratio,
        'epochs': epochs,
        'lr': learning_rate,
        'batch_size': batch_size,
        'use_pos': use_pos,
        'use_lang_tokens': use_lang_tokens,
        'loss_type': loss_type,
        'use_crf': use_crf,
        'model_name': model_name,
        'best_f1': best_f1,
        'num_labels': len(label_to_id),
        'num_categories': len(category_to_id),
        'seed': seed,
        'max_length': max_length
    }
    
    info_path = os.path.join(output_dir, 'info.json')
    with open(info_path, 'w') as f:
        json.dump(training_info, f, indent=2)
    print(f"✓ Training info saved to: {info_path}")
    
    # Load the best model for final test evaluation (if it exists)
    model_path = os.path.join(output_dir, 'best_model.pt')
    
    if os.path.exists(model_path):
        print("\nLoading best model for test evaluation...")
        try:
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✓ Best model loaded successfully")
            print("Evaluating on test set...")
        except Exception as e:
            print(f"⚠️  ERROR loading model: {e}")
            import traceback
            traceback.print_exc()
            print("Evaluating on test set with final epoch model...")
    else:
        print(f"\n⚠️  Warning: Model file not found at {model_path}")
        print("Possible reasons:")
        print("  - Validation F1 never improved from initial value (-1)")
        print("  - Model saving failed")
        print("  - output_dir parameter changed between save and load")
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
    parser.add_argument('--lang_tokens', action='store_true',
                       help='Enable language-conditioned inputs (prepend [LANG] tokens to prevent language interference)')
    parser.add_argument('--loss', type=str, default='ce', choices=['ce', 'focal'],
                       help='Loss function: ce (CrossEntropy) or focal (Focal Loss for class imbalance)')
    parser.add_argument('--crf', action='store_true',
                       help='Enable CRF layer for BIO tagging (improves discontinuous MWE detection)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from (e.g., models/FR/best_model.pt)')
    
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
        use_pos=args.pos,
        use_lang_tokens=args.lang_tokens,
        loss_type=args.loss,
        use_crf=args.crf,
        resume_from=args.resume
    )
