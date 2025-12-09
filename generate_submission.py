"""
Generate predictions for all languages and create submission zip file
"""
import os
import sys
import zipfile
import argparse
from pathlib import Path
import subprocess


def get_all_available_languages(base_data_dir):
    """Get all available languages from the data directory"""
    all_langs = []
    if os.path.exists(base_data_dir):
        for item in os.listdir(base_data_dir):
            item_path = os.path.join(base_data_dir, item)
            test_file = os.path.join(item_path, 'test.blind.cupt')
            if os.path.isdir(item_path) and os.path.exists(test_file):
                all_langs.append(item)
    return sorted(all_langs)


def get_languages_from_model_name(model_path):
    """Extract language codes from model directory name"""
    model_dir = Path(model_path).parent.name
    if 'multilingual' in model_dir:
        # Extract languages from name like "multilingual_PL+FR+EL+PT+RO"
        parts = model_dir.split('_')
        if len(parts) > 1:
            langs = parts[1].split('+')
            return langs
    return []


def generate_predictions(model_path, languages, base_data_dir, output_dir):
    """
    Generate predictions for all specified languages
    
    Args:
        model_path: Path to model checkpoint
        languages: List of language codes (e.g., ['PL', 'FR', 'EL'])
        base_data_dir: Base directory containing language subdirectories
        output_dir: Directory to save predictions
    """
    print(f"=" * 80)
    print(f"Generating predictions for {len(languages)} languages")
    print(f"Model: {model_path}")
    print(f"Languages: {', '.join(languages)}")
    print(f"=" * 80)
    
    predictions_generated = []
    
    for lang in languages:
        input_file = os.path.join(base_data_dir, lang, 'test.blind.cupt')
        output_file = os.path.join(output_dir, lang, 'test.system.cupt')
        
        # Check if input file exists
        if not os.path.exists(input_file):
            print(f"\n❌ WARNING: Input file not found for {lang}: {input_file}")
            continue
        
        # Create output directory
        os.makedirs(os.path.join(output_dir, lang), exist_ok=True)
        
        print(f"\n{'=' * 60}")
        print(f"Processing {lang}...")
        print(f"{'=' * 60}")
        
        # Run prediction
        cmd = [
            'python', 'src/predict.py',
            '--model', model_path,
            '--input', input_file,
            '--output', output_file
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(result.stdout)
            
            if os.path.exists(output_file):
                print(f"✅ Predictions saved to: {output_file}")
                predictions_generated.append((lang, output_file))
            else:
                print(f"❌ ERROR: Output file not created for {lang}")
                
        except subprocess.CalledProcessError as e:
            print(f"❌ ERROR generating predictions for {lang}:")
            print(e.stderr)
            continue
    
    return predictions_generated


def create_submission_zip(predictions, output_zip, model_name):
    """
    Create submission zip file containing all predictions
    
    Args:
        predictions: List of (lang, filepath) tuples
        output_zip: Path to output zip file
        model_name: Name of the model for metadata
    """
    print(f"\n{'=' * 80}")
    print(f"Creating submission zip file...")
    print(f"{'=' * 80}")
    
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for lang, filepath in predictions:
            # Add file with structure: LANG/test.system.cupt
            arcname = f"{lang}/test.system.cupt"
            zipf.write(filepath, arcname)
            print(f"✅ Added {lang}/test.system.cupt")
    
    print(f"\n{'=' * 80}")
    print(f"✅ Submission created: {output_zip}")
    print(f"   Total files: {len(predictions)} languages")
    print(f"{'=' * 80}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate predictions and create submission zip file'
    )
    parser.add_argument(
        '--model',
        required=True,
        help='Path to model checkpoint (e.g., models/multilingual_PL+FR+EL/best_model.pt)'
    )
    parser.add_argument(
        '--lang',
        nargs='+',
        help='Language codes to process (e.g., PL FR EL) or "all" for all available languages. If not specified, extracts from model name.'
    )
    parser.add_argument(
        '--data-dir',
        default='2.0/subtask1',
        help='Base directory containing language subdirectories (default: 2.0/subtask1)'
    )
    parser.add_argument(
        '--output-dir',
        default='predictions',
        help='Directory to save predictions (default: predictions)'
    )
    parser.add_argument(
        '--zip',
        default='submission.zip',
        help='Output zip file name (default: submission.zip)'
    )
    parser.add_argument(
        '--no-zip',
        action='store_true',
        help='Skip creating zip file (only generate predictions)'
    )
    
    args = parser.parse_args()
    
    # Validate model path
    if not os.path.exists(args.model):
        print(f"❌ ERROR: Model file not found: {args.model}")
        sys.exit(1)
    
    # Get languages
    if args.lang:
        # Check if user specified "all"
        if len(args.lang) == 1 and args.lang[0].lower() == 'all':
            languages = get_all_available_languages(args.data_dir)
            if not languages:
                print(f"❌ ERROR: No languages found in {args.data_dir}")
                sys.exit(1)
            print(f"Found {len(languages)} available languages: {', '.join(languages)}")
        else:
            # User specified specific languages
            languages = args.lang
    else:
        # Auto-detect from model name
        languages = get_languages_from_model_name(args.model)
        if not languages:
            print("❌ ERROR: Could not extract languages from model name.")
            print("   Please specify languages with --lang PL FR EL or --lang all")
            sys.exit(1)
    
    # Get model name for metadata
    model_name = Path(args.model).parent.name
    
    # Generate predictions
    predictions = generate_predictions(
        args.model,
        languages,
        args.data_dir,
        args.output_dir
    )
    
    if not predictions:
        print("\n❌ ERROR: No predictions were generated!")
        sys.exit(1)
    
    # Create submission zip
    if not args.no_zip:
        output_zip = args.zip
        if not output_zip.endswith('.zip'):
            output_zip += '.zip'
        
        create_submission_zip(predictions, output_zip, model_name)
    
    print("\n" + "=" * 80)
    print("✅ ALL DONE!")
    print("=" * 80)
    print(f"\nGenerated predictions for {len(predictions)} languages:")
    for lang, filepath in predictions:
        print(f"  - {lang}: {filepath}")
    
    if not args.no_zip:
        print(f"\n📦 Submission file: {output_zip}")
        print(f"   Ready to submit!")


if __name__ == '__main__':
    main()
