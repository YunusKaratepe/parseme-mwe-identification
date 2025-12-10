"""
Ensemble workflow for PARSEME 2.0 MWE identification
Convenient interface for ensemble prediction and evaluation
"""
import os
import sys
import argparse


def print_banner():
    print("\n" + "=" * 80)
    print(" " * 25 + "ENSEMBLE WORKFLOW")
    print(" " * 20 + "Cross-Entropy + Focal Loss")
    print("=" * 80 + "\n")


def predict_single_language(language: str, ce_model_dir: str, focal_model_dir: str, output_base: str = "predictions"):
    """Generate ensemble predictions for a single language"""
    print(f"\n🔮 Generating ensemble predictions for {language}...\n")
    
    ce_model = os.path.join(ce_model_dir, "best_model.pt")
    focal_model = os.path.join(focal_model_dir, "best_model.pt")
    
    input_file = f"2.0/subtask1/{language}/test.blind.cupt"
    output_dir = f"{output_base}/{language}"
    output_file = f"{output_dir}/test_ensemble.cupt"
    
    if not os.path.exists(input_file):
        print(f"❌ Error: Test file not found: {input_file}")
        return False
    
    if not os.path.exists(ce_model):
        print(f"❌ Error: CE model not found: {ce_model}")
        return False
    
    if not os.path.exists(focal_model):
        print(f"❌ Error: Focal model not found: {focal_model}")
        return False
    
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = f"python src/ensemble_predict.py --ce_model {ce_model} --focal_model {focal_model} --input {input_file} --output {output_file}"
    result = os.system(cmd)
    
    if result == 0:
        print(f"\n✅ Ensemble predictions saved to: {output_file}")
        return True
    else:
        print(f"\n❌ Ensemble prediction failed")
        return False


def evaluate_ensemble(languages: list, ce_model_dir: str, focal_model_dir: str, output_dir: str = "ensemble"):
    """Evaluate ensemble on validation/test sets"""
    print(f"\n📊 Evaluating ensemble on {len(languages)} languages...\n")
    
    if not os.path.exists(ce_model_dir):
        print(f"❌ Error: CE model directory not found: {ce_model_dir}")
        return False
    
    if not os.path.exists(focal_model_dir):
        print(f"❌ Error: Focal model directory not found: {focal_model_dir}")
        return False
    
    langs_str = ' '.join(languages)
    cmd = f"python src/ensemble_evaluate.py --ce_model_dir {ce_model_dir} --focal_model_dir {focal_model_dir} --languages {langs_str} --output {output_dir}"
    result = os.system(cmd)
    
    if result == 0:
        print(f"\n✅ Evaluation results saved to: {output_dir}/evaluation_results.json")
        return True
    else:
        print(f"\n❌ Evaluation failed")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Ensemble Workflow for PARSEME 2.0 MWE Identification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate ensemble on all languages
  python ensemble_workflow.py evaluate --languages FR PL EL PT RO SL SR SV UK NL EGY KA JA HE LV FA
  
  # Generate ensemble predictions for French
  python ensemble_workflow.py predict FR
  
  # Generate predictions for multiple languages
  python ensemble_workflow.py predict FR PL EL
  
  # Use custom model directories
  python ensemble_workflow.py evaluate --languages FR PL --ce_model ensemble/ce/multilingual_XXX --focal_model ensemble/focal/multilingual_XXX

Default setup:
  CE Model:    ensemble/ce/multilingual_FR+PL+EL+PT+SL+SR+SV+UK+NL+EGY+KA+JA+HE+LV+FA+RO
  Focal Model: ensemble/focal/multilingual_FR+PL+EL+PT+SL+SR+SV+UK+NL+EGY+KA+JA+HE+LV+FA+RO
        """
    )
    
    parser.add_argument('command', choices=['predict', 'evaluate'],
                       help='Command to execute')
    parser.add_argument('languages', nargs='*', 
                       help='Language code(s) for prediction (or use --languages for evaluate)')
    parser.add_argument('--languages', type=str, nargs='+', dest='eval_languages',
                       help='Language codes for evaluation')
    parser.add_argument('--ce_model', type=str, 
                       default='ensemble/ce/multilingual_FR+PL+EL+PT+SL+SR+SV+UK+NL+EGY+KA+JA+HE+LV+FA+RO',
                       help='CE model directory')
    parser.add_argument('--focal_model', type=str,
                       default='ensemble/focal/multilingual_FR+PL+EL+PT+SL+SR+SV+UK+NL+EGY+KA+JA+HE+LV+FA+RO',
                       help='Focal model directory')
    parser.add_argument('--output', type=str, default='ensemble',
                       help='Output directory (default: ensemble)')
    
    args = parser.parse_args()
    
    print_banner()
    
    if args.command == 'predict':
        if not args.languages:
            print("❌ Error: Please specify at least one language for prediction")
            print("   Example: python ensemble_workflow.py predict FR")
            return
        
        success_count = 0
        for lang in args.languages:
            if predict_single_language(lang, args.ce_model, args.focal_model, 'predictions'):
                success_count += 1
        
        print(f"\n{'='*80}")
        print(f"Prediction Summary: {success_count}/{len(args.languages)} languages completed")
        print(f"{'='*80}\n")
    
    elif args.command == 'evaluate':
        languages = args.eval_languages if args.eval_languages else args.languages
        
        if not languages:
            print("❌ Error: Please specify languages for evaluation")
            print("   Example: python ensemble_workflow.py evaluate --languages FR PL EL")
            return
        
        evaluate_ensemble(languages, args.ce_model, args.focal_model, args.output)


if __name__ == '__main__':
    main()
