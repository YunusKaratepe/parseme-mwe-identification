"""
Validate all prediction files for PARSEME 2.0 submission
Checks all test.system.cupt files in the predictions directory
"""
import os
import sys
import subprocess
from pathlib import Path


def get_available_predictions(predictions_dir='predictions'):
    """Get all languages that have prediction files"""
    predictions = []
    if not os.path.exists(predictions_dir):
        return predictions
    
    for item in os.listdir(predictions_dir):
        item_path = os.path.join(predictions_dir, item)
        test_file = os.path.join(item_path, 'test.system.cupt')
        if os.path.isdir(item_path) and os.path.exists(test_file):
            predictions.append((item, test_file))
    
    return sorted(predictions)


def validate_prediction(lang, filepath, validator_script='2.0/subtask1/tools/parseme_validate.py'):
    """
    Validate a single prediction file using parseme_validate.py
    
    Args:
        lang: Language code
        filepath: Path to test.system.cupt file
        validator_script: Path to parseme_validate.py
    
    Returns:
        tuple: (passed: bool, output: str)
    """
    if not os.path.exists(validator_script):
        return False, f"ERROR: Validator script not found: {validator_script}"
    
    try:
        result = subprocess.run(
            ['python', validator_script, '--lang', lang, filepath],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        output = result.stdout + result.stderr
        # Check if PASSED appears in the output (ignore return code due to UD validation warning)
        passed = '*** PASSED ***' in output
        
        return passed, output
        
    except subprocess.TimeoutExpired:
        return False, "ERROR: Validation timed out (30s)"
    except Exception as e:
        return False, f"ERROR: {str(e)}"


def main():
    """Main validation function"""
    print("=" * 80)
    print(" " * 20 + "PARSEME 2.0 - Submission Validation")
    print("=" * 80)
    
    # Find all predictions
    predictions = get_available_predictions()
    
    if not predictions:
        print("\n❌ ERROR: No predictions found in 'predictions/' directory")
        print("   Please generate predictions first using generate_submission.py")
        sys.exit(1)
    
    print(f"\nFound {len(predictions)} prediction files to validate:")
    for lang, filepath in predictions:
        print(f"  - {lang}: {filepath}")
    
    print("\n" + "=" * 80)
    print("Starting validation...")
    print("=" * 80)
    
    # Validate each prediction
    passed_count = 0
    failed_count = 0
    results = []
    
    for lang, filepath in predictions:
        print(f"\n{'=' * 60}")
        print(f"Validating {lang}...")
        print(f"{'=' * 60}")
        
        passed, output = validate_prediction(lang, filepath)
        
        if passed:
            passed_count += 1
            status = "✅ PASSED"
            print(status)
        else:
            failed_count += 1
            status = "❌ FAILED"
            print(status)
            print("\nValidation output:")
            print(output)
        
        results.append((lang, status, passed))
    
    # Print summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    for lang, status, _ in results:
        print(f"  {status} {lang}")
    
    print("\n" + "=" * 80)
    print(f"Total: {len(predictions)} files")
    print(f"Passed: {passed_count}")
    print(f"Failed: {failed_count}")
    print("=" * 80)
    
    if failed_count > 0:
        print("\n⚠️  Some validations failed. Please fix the errors before submission.")
        sys.exit(1)
    else:
        print("\n🎉 All predictions passed validation!")
        print("   Your submission is ready!")
        sys.exit(0)


if __name__ == '__main__':
    main()
