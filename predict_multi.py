import subprocess
import sys
import os

# --- CONFIGURATION ---
languages = ["FR", "PL", "EL", "PT", "SL", "SR", "SV", "UK", "NL", "EGY", "KA", "JA", "HE", "LV", "FA", "RO", "GRC"] 

# The path to your trained model
model_path = "models/251210_multilingual_FR+PL+EL+PT+SL+SR+SV+UK+NL+EGY+KA+JA+HE+LV+FA+RO/best_model.pt"

# Script location
predict_script = "src/predict.py"

def run_predictions():
    for lang in languages:
        # Construct paths
        # NOTE: Using 'dev.cupt' as input and 'test.system.cupt' as output per your request.
        input_path = f"2.0/subtask1/{lang}/dev.cupt"
        output_path = f"predictions/{lang}/dev.system.cupt"
        
        # 1. Ensure the output directory exists, otherwise the script might fail
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # 2. Construct the command
        command = [
            sys.executable, predict_script,
            "--model", model_path,
            "--input", input_path,
            "--output", output_path
        ]
        
        print(f"Running prediction for {lang}...")
        
        try:
            # 3. Run the command
            result = subprocess.run(
                command,
                capture_output=True,
                text=True
            )
            
            # Check if the command was successful
            if result.returncode == 0:
                print(f" -> Success! Output saved to: {output_path}")
            else:
                print(f" -> Error processing {lang}!")
                print(result.stderr) # Print error details to console
                
        except Exception as e:
            print(f" -> An unexpected error occurred: {e}")

    print("\nAll predictions finished.")

if __name__ == "__main__":
    run_predictions()