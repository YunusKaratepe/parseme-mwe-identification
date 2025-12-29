import subprocess
import sys

# --- CONFIGURATION ---
# List of languages to evaluate
languages = ["FR", "PL", "EL", "PT", "SL", "SR", "SV", "UK", "NL", "EGY", "KA", "JA", "HE", "LV", "FA", "RO", "GRC"] 

# The output text file where results will be appended
output_filename = "evaluation_results.txt"

# Path to the evaluation script
eval_script = "2.0/subtask1/tools/parseme_evaluate.py"

def run_evaluation():
    # Open the file in write mode (clears previous run) 
    # Change to "a" if you want to keep history across multiple runs of this script
    with open(output_filename, "w", encoding="utf-8") as f:
        
        for lang in languages:
            # Construct the specific paths for this language
            gold_path = f"2.0/subtask1/{lang}/dev.cupt"
            pred_path = f"predictions/{lang}/dev.system.cupt"
            
            # Construct the command list for subprocess
            # We use sys.executable to ensure we use the same Python interpreter running this script
            command = [
                sys.executable, eval_script,
                "--gold", gold_path,
                "--pred", pred_path
            ]
            
            # Create a header for readability in the text file
            header = f"\n{'='*30}\nRESULTS FOR LANGUAGE: {lang}\n{'='*30}\n"
            print(f"Evaluating {lang}...") # Print to console so you know progress
            
            f.write(header)
            f.flush()
            
            try:
                # Execute the command
                result = subprocess.run(
                    command,
                    capture_output=True, # Captures both stdout and stderr
                    text=True,           # Decodes bytes to string automatically
                    check=False          # Set to True if you want the script to crash on error
                )
                
                # Write the standard output (the actual scores)
                f.write(result.stdout)
                
                # If there were errors/warnings in stderr, write them too
                if result.stderr:
                    f.write("\n--- STDERR (Warnings/Errors) ---\n")
                    f.write(result.stderr)
                    
            except FileNotFoundError:
                error_msg = f"Error: Could not find script or files for {lang}.\n"
                print(error_msg)
                f.write(error_msg)
            except Exception as e:
                error_msg = f"An unexpected error occurred for {lang}: {e}\n"
                print(error_msg)
                f.write(error_msg)
                
            f.write("\n") # Add spacing between languages

    print(f"Done! All results saved to {output_filename}")

if __name__ == "__main__":
    run_evaluation()