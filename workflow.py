#!/usr/bin/env python
"""
Complete workflow for PARSEME 2.0 Subtask 1 - MWE Identification
This script provides an end-to-end pipeline for training and prediction
"""
import argparse
import os
import sys


def print_banner():
    print("\n" + "=" * 80)
    print(" " * 20 + "PARSEME 2.0 - MWE IDENTIFICATION")
    print(" " * 25 + "Complete Workflow")
    print("=" * 80 + "\n")


def print_options():
    print("Available commands:")
    print()
    print("1. TRAIN      - Train a new model on specified language")
    print("2. PREDICT    - Generate predictions using trained model")
    print("3. VISUALIZE  - View prediction examples")
    print("4. SUMMARY    - Show training results and statistics")
    print("5. QUICK      - Quick training on French (2 epochs for testing)")
    print("6. FULL       - Full training on French (10 epochs)")
    print()


def train_model(languages, epochs=5, batch_size=8, sample_ratio=1.0, multilingual=False, use_pos=False, output_base='models'):
    """Train model on specified language(s)"""
    # Handle both single language string and list of languages
    if isinstance(languages, str):
        languages = [languages]
    
    # Multilingual mode: train single model on all languages combined
    if multilingual:
        print(f"\n🌍 Training MULTILINGUAL model on {len(languages)} languages...")
        print(f"   Languages: {', '.join(languages)}")
        print(f"   Epochs: {epochs}, Batch size: {batch_size}, Sample ratio: {sample_ratio*100:.1f}%\n")
        
        # Collect all train and dev files
        train_files = []
        dev_files = []
        missing_languages = []
        
        for language in languages:
            train_file = f"2.0/subtask1/{language}/train.cupt"
            dev_file = f"2.0/subtask1/{language}/dev.cupt"
            
            if not os.path.exists(train_file):
                print(f"⚠️  Warning: Training file not found for {language}: {train_file}")
                missing_languages.append(language)
                continue
            
            if not os.path.exists(dev_file):
                print(f"⚠️  Warning: Dev file not found for {language}: {dev_file}")
                missing_languages.append(language)
                continue
            
            train_files.append(train_file)
            dev_files.append(dev_file)
        
        if not train_files:
            print("❌ Error: No valid language files found")
            return False
        
        if missing_languages:
            print(f"⚠️  Skipping languages: {', '.join(missing_languages)}\n")
        
        # Determine output directory
        # If output_base looks like a complete path (contains multilingual_ or language code at end),
        # use it as-is. Otherwise, construct the path automatically.
        if 'multilingual_' in output_base or output_base.endswith(tuple(languages)):
            # User provided complete path
            output_dir = output_base
        else:
            # Construct path automatically
            lang_code = '+'.join([lang for lang in languages if lang not in missing_languages])
            output_dir = f"{output_base}/multilingual_{lang_code}"
        
        # Join files with comma separator for train.py
        train_files_str = ','.join(train_files)
        dev_files_str = ','.join(dev_files)
        
        cmd = f"python src/train.py --train {train_files_str} --dev {dev_files_str} --output {output_dir} --epochs {epochs} --batch_size {batch_size} --sample_ratio {sample_ratio}"
        if use_pos:
            cmd += " --pos"
        result = os.system(cmd)
        
        if result == 0:
            print(f"\n✅ Successfully trained multilingual model")
            print(f"   Model saved to: {output_dir}")
            return True
        else:
            print(f"\n❌ Multilingual training failed")
            return False
    
    # Standard mode: train separate model for each language
    else:
        success_count = 0
        failed_languages = []
        
        for language in languages:
            print(f"\n🚀 Training model on {language}...")
            print(f"   Epochs: {epochs}, Batch size: {batch_size}, Sample ratio: {sample_ratio*100:.1f}%\n")
            
            train_file = f"2.0/subtask1/{language}/train.cupt"
            dev_file = f"2.0/subtask1/{language}/dev.cupt"
            output_dir = f"{output_base}/{language}"
            
            if not os.path.exists(train_file):
                print(f"❌ Error: Training file not found: {train_file}")
                failed_languages.append(language)
                continue
            
            if not os.path.exists(dev_file):
                print(f"❌ Error: Dev file not found: {dev_file}")
                failed_languages.append(language)
                continue
            
            cmd = f"python src/train.py --train {train_file} --dev {dev_file} --output {output_dir} --epochs {epochs} --batch_size {batch_size} --sample_ratio {sample_ratio}"
            if use_pos:
                cmd += " --pos"
            result = os.system(cmd)
            
            if result == 0:
                success_count += 1
                print(f"✅ Successfully trained model for {language}")
            else:
                failed_languages.append(language)
                print(f"❌ Training failed for {language}")
        
        # Print summary
        print("\n" + "=" * 80)
        print(f"Training Summary: {success_count}/{len(languages)} languages completed successfully")
        if failed_languages:
            print(f"Failed languages: {', '.join(failed_languages)}")
        print("=" * 80 + "\n")
        
        return success_count > 0


def predict(language):
    """Generate predictions for specified language"""
    print(f"\n🔮 Generating predictions for {language}...\n")
    
    model_file = f"models/{language}/best_model.pt"
    input_file = f"2.0/subtask1/{language}/test.blind.cupt"
    output_dir = f"predictions/{language}"
    output_file = f"{output_dir}/test.cupt"
    
    if not os.path.exists(model_file):
        print(f"❌ Error: Model not found: {model_file}")
        print(f"   Please train the model first using: workflow.py train {language}")
        return False
    
    if not os.path.exists(input_file):
        print(f"❌ Error: Test file not found: {input_file}")
        return False
    
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = f"python src/predict.py --model {model_file} --input {input_file} --output {output_file}"
    os.system(cmd)
    
    return True


def visualize(language, num_examples=10):
    """Visualize predictions"""
    print(f"\n👁️  Visualizing predictions for {language}...\n")
    
    pred_file = f"predictions/{language}/test.cupt"
    
    if not os.path.exists(pred_file):
        print(f"❌ Error: Predictions not found: {pred_file}")
        print(f"   Please generate predictions first using: workflow.py predict {language}")
        return False
    
    cmd = f"python src/visualize.py {pred_file} {num_examples}"
    os.system(cmd)
    
    return True


def show_summary(language):
    """Show training summary"""
    print(f"\n📊 Training summary for {language}...\n")
    
    output_dir = f"models/{language}"
    
    if not os.path.exists(output_dir):
        print(f"❌ Error: Model directory not found: {output_dir}")
        print(f"   Please train the model first using: workflow.py train {language}")
        return False
    
    cmd = f"python src/summary.py {output_dir}"
    os.system(cmd)
    
    return True


def quick_test():
    """Quick training on French for testing (2 epochs)"""
    print("\n⚡ Quick test training on French (2 epochs)...\n")
    return train_model("FR", epochs=2, batch_size=4)


def full_training(language="FR"):
    """Full training with recommended settings"""
    print(f"\n🎯 Full training on {language} (10 epochs)...\n")
    return train_model(language, epochs=10, batch_size=8)


def main():
    parser = argparse.ArgumentParser(
        description="PARSEME 2.0 MWE Identification Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test on French (2 epochs)
  python workflow.py quick
  
  # Train on Polish for 5 epochs
  python workflow.py train PL --epochs 5
  
  # Train on multiple languages at once
  python workflow.py train PL FR EL --epochs 3 --batch_size 16
  
  # Quick analysis with 50% of training data
  python workflow.py train PL FR EL --epochs 1 --sample_ratio 0.5
  
  # Train single MULTILINGUAL model on multiple languages
  python workflow.py train PL FR EL --epochs 3 --multilingual
  
  # Train multilingual model on all languages with 20% sample
  python workflow.py train FR PL EL PT RO SL SR SV UK --epochs 1 --sample_ratio 0.2 --multilingual
  
  # Full training on French (10 epochs)
  python workflow.py full FR
  
  # Generate predictions
  python workflow.py predict FR
  
  # Visualize results
  python workflow.py visualize FR
  
  # Show training summary
  python workflow.py summary FR

Available languages:
  FR (French), PL (Polish), EL (Greek), PT (Portuguese),
  RO (Romanian), SL (Slovene), SR (Serbian), SV (Swedish),
  UK (Ukrainian), NL (Dutch), EGY (Egyptian), KA (Georgian),
  GRC (Ancient Greek), JA (Japanese), HE (Hebrew), LV (Latvian), FA (Persian)
        """
    )
    
    parser.add_argument('command', choices=['train', 'predict', 'visualize', 'summary', 'quick', 'full'],
                       help='Command to execute')
    parser.add_argument('languages', nargs='*', default=['FR'],
                       help='Language code(s) - can specify multiple languages (default: FR)')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs (default: 5)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size (default: 8)')
    parser.add_argument('--sample_ratio', type=float, default=1.0,
                       help='Ratio of training data to use (0.0-1.0, default: 1.0 = 100%%)')
    parser.add_argument('--multilingual', action='store_true',
                       help='Train single multilingual model on all specified languages combined')
    parser.add_argument('--pos', action='store_true',
                       help='Enable POS tag feature injection for improved performance')
    parser.add_argument('--output', type=str, default='models',
                       help='Base output directory for saving models (default: models)')
    parser.add_argument('--examples', type=int, default=10,
                       help='Number of examples to visualize (default: 10)')
    
    args = parser.parse_args()
    
    print_banner()
    
    if args.command == 'train':
        train_model(args.languages, args.epochs, args.batch_size, args.sample_ratio, args.multilingual, args.pos, args.output)
    elif args.command == 'predict':
        # For predict, visualize, summary - use first language if multiple specified
        language = args.languages[0] if args.languages else 'FR'
        predict(language)
    elif args.command == 'visualize':
        language = args.languages[0] if args.languages else 'FR'
        visualize(language, args.examples)
    elif args.command == 'summary':
        language = args.languages[0] if args.languages else 'FR'
        show_summary(language)
    elif args.command == 'quick':
        quick_test()
    elif args.command == 'full':
        language = args.languages[0] if args.languages else 'FR'
        full_training(language)


if __name__ == '__main__':
    main()
