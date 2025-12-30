"""
Check all prediction files for discontinuous MWE patterns
"""
import os
from collections import defaultdict

predictions_dir = "predictions"

# Get all language directories
languages = [d for d in os.listdir(predictions_dir) 
             if os.path.isdir(os.path.join(predictions_dir, d))]

print("=" * 80)
print("CHECKING ALL PREDICTIONS FOR DISCONTINUOUS MWE PATTERNS")
print("=" * 80)
print()

total_continuous = 0
total_discontinuous = 0
results_by_lang = {}

for lang in sorted(languages):
    pred_file = os.path.join(predictions_dir, lang, "dev.system.cupt")
    
    if not os.path.exists(pred_file):
        print(f"{lang}: No prediction file found")
        continue
    
    # Read and analyze predictions
    continuous_count = 0
    discontinuous_count = 0
    discontinuous_examples = []
    
    with open(pred_file, 'r', encoding='utf-8') as f:
        sentence_mwes = []
        sentence_tokens = []
        
        for line in f:
            if line.startswith('#') or not line.strip():
                # Analyze sentence
                if sentence_mwes:
                    # Track MWE IDs and their positions
                    mwe_positions = {}
                    for idx, mwe_val in enumerate(sentence_mwes):
                        if mwe_val != '*':
                            # Extract MWE ID (number before : or just number)
                            if ':' in mwe_val:
                                mwe_id = mwe_val.split(':')[0]
                            else:
                                mwe_id = mwe_val.split(';')[0] if ';' in mwe_val else mwe_val
                            
                            # Handle multiple MWEs (e.g., "1;2")
                            for mid in mwe_id.split(';'):
                                if mid not in mwe_positions:
                                    mwe_positions[mid] = []
                                mwe_positions[mid].append(idx)
                    
                    # Check each MWE for continuity
                    for mwe_id, positions in mwe_positions.items():
                        if len(positions) > 1:
                            # Check if continuous
                            is_continuous = all(positions[i] + 1 == positions[i+1] 
                                              for i in range(len(positions) - 1))
                            if is_continuous:
                                continuous_count += 1
                            else:
                                discontinuous_count += 1
                                # Save example for first 3
                                if len(discontinuous_examples) < 3:
                                    tokens_str = ' '.join([sentence_tokens[p] for p in positions])
                                    discontinuous_examples.append(
                                        f"MWE {mwe_id} at positions {positions}: {tokens_str}"
                                    )
                
                sentence_mwes = []
                sentence_tokens = []
                continue
            
            parts = line.strip().split('\t')
            if len(parts) >= 11 and not ('-' in parts[0] or '.' in parts[0]):
                sentence_mwes.append(parts[10])
                sentence_tokens.append(parts[1])  # FORM
    
    # Store results
    results_by_lang[lang] = {
        'continuous': continuous_count,
        'discontinuous': discontinuous_count,
        'examples': discontinuous_examples
    }
    
    total_continuous += continuous_count
    total_discontinuous += discontinuous_count
    
    # Print results
    status = "✓" if discontinuous_count > 0 else "✗"
    print(f"{status} {lang:6s}: Continuous={continuous_count:4d}, Discontinuous={discontinuous_count:4d}")
    
    if discontinuous_count > 0 and discontinuous_examples:
        for example in discontinuous_examples:
            print(f"         └─ {example}")

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Total continuous MWEs:    {total_continuous}")
print(f"Total discontinuous MWEs: {total_discontinuous}")
print()

if total_discontinuous == 0:
    print("⚠️  CRITICAL FINDING: ZERO DISCONTINUOUS MWEs IN ALL PREDICTIONS!")
    print()
    print("This confirms that:")
    print("  1. The model is NOT generating discontinuous patterns")
    print("  2. Postprocessing is either:")
    print("     a) Not being applied, OR")
    print("     b) Not finding any B-MWE...O...I-MWE patterns to stitch")
    print()
    print("Root cause: Token-level classification cannot predict")
    print("            B-MWE followed by O(s) followed by I-MWE with same category")
    print()
    print("Why? Because the model learns that:")
    print("  - After B-MWE comes I-MWE (continuous pattern)")
    print("  - If O comes after I-MWE, the MWE has ended")
    print("  - So it never predicts I-MWE after a gap of O tokens")
else:
    print(f"✓ Found {total_discontinuous} discontinuous MWEs across {sum(1 for r in results_by_lang.values() if r['discontinuous'] > 0)} languages")
    print()
    print("Languages with discontinuous predictions:")
    for lang, results in sorted(results_by_lang.items(), key=lambda x: x[1]['discontinuous'], reverse=True):
        if results['discontinuous'] > 0:
            print(f"  {lang}: {results['discontinuous']}")
