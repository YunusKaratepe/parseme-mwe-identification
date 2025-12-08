"""
Summary script - display training results and generate report
"""
import json
import os
import sys


def print_training_summary(output_dir='models/FR'):
    """Print training summary and results"""
    
    print("=" * 70)
    print("PARSEME 2.0 SUBTASK 1: MWE IDENTIFICATION - TRAINING SUMMARY")
    print("=" * 70)
    print()
    
    # Load training history
    history_file = os.path.join(output_dir, 'training_history.json')
    
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        print(f"Training on: French (FR)")
        print(f"Number of epochs: {len(history)}")
        print()
        
        print("-" * 70)
        print("EPOCH-BY-EPOCH RESULTS")
        print("-" * 70)
        print()
        
        for entry in history:
            epoch = entry['epoch']
            train_loss = entry['train_loss']
            dev_loss = entry['dev_loss']
            precision = entry['dev_precision']
            recall = entry['dev_recall']
            f1 = entry['dev_f1']
            
            print(f"Epoch {epoch}:")
            print(f"  Training Loss:   {train_loss:.4f}")
            print(f"  Dev Loss:        {dev_loss:.4f}")
            print(f"  Dev Precision:   {precision:.4f} ({precision*100:.2f}%)")
            print(f"  Dev Recall:      {recall:.4f} ({recall*100:.2f}%)")
            print(f"  Dev F1 Score:    {f1:.4f} ({f1*100:.2f}%)")
            print()
        
        # Best results
        best_epoch = max(history, key=lambda x: x['dev_f1'])
        print("=" * 70)
        print("BEST RESULTS")
        print("=" * 70)
        print()
        print(f"Best epoch: {best_epoch['epoch']}")
        print(f"Best F1 Score: {best_epoch['dev_f1']:.4f} ({best_epoch['dev_f1']*100:.2f}%)")
        print(f"Precision: {best_epoch['dev_precision']:.4f}")
        print(f"Recall: {best_epoch['dev_recall']:.4f}")
        print()
        
    else:
        print(f"Training history not found at: {history_file}")
        print()
    
    # Check for model file
    model_file = os.path.join(output_dir, 'best_model.pt')
    if os.path.exists(model_file):
        print(f"✓ Model saved: {model_file}")
        file_size = os.path.getsize(model_file) / (1024 * 1024)  # MB
        print(f"  Size: {file_size:.1f} MB")
    else:
        print(f"✗ Model not found: {model_file}")
    print()
    
    # Check for predictions
    pred_file = 'predictions/FR/test.cupt'
    if os.path.exists(pred_file):
        print(f"✓ Predictions generated: {pred_file}")
        
        # Count predictions
        mwe_count = 0
        sentence_count = 0
        with open(pred_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    sentence_count += 1
                elif not line.startswith('#'):
                    parts = line.split('\t')
                    if len(parts) >= 11 and parts[10] != '*':
                        mwe_count += 1
        
        print(f"  Sentences: {sentence_count}")
        print(f"  MWE tokens predicted: {mwe_count}")
    else:
        print(f"✗ Predictions not found: {pred_file}")
    print()
    
    print("=" * 70)
    print("MODEL INFORMATION")
    print("=" * 70)
    print()
    print("Architecture: Transformer-based token classification")
    print("Base Model: bert-base-multilingual-cased")
    print("Task: Sequence labeling with BIO tagging")
    print("Labels: O (outside), B-MWE (begin), I-MWE (inside)")
    print()
    
    print("=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print()
    print("1. Improve the model:")
    print("   - Train for more epochs (5-10)")
    print("   - Use larger batch size if GPU allows")
    print("   - Try language-specific BERT (e.g., CamemBERT for French)")
    print()
    print("2. Train on other languages:")
    print("   python src/train.py --train 2.0/subtask1/PL/train.cupt \\")
    print("                       --dev 2.0/subtask1/PL/dev.cupt \\")
    print("                       --output models/PL")
    print()
    print("3. Make predictions on other test sets:")
    print("   python src/predict.py --model models/FR/best_model.pt \\")
    print("                         --input 2.0/subtask1/FR/test.blind.cupt \\")
    print("                         --output predictions/FR/test.cupt")
    print()
    print("4. Evaluate with official metrics (when gold data is available):")
    print("   python 2.0/subtask1/tools/parseme_evaluate.py \\")
    print("          --gold 2.0/subtask1/FR/test.cupt \\")
    print("          --pred predictions/FR/test.cupt")
    print()
    print("=" * 70)


if __name__ == '__main__':
    output_dir = 'models/FR'
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    
    print_training_summary(output_dir)
