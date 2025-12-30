"""
Test script to verify discontinuous MWE prediction and formatting
"""
import sys
sys.path.insert(0, 'src')

from predict import bio_tags_to_mwe_column
from postprocess_discontinuous import fix_discontinuous_mwes

print("=" * 70)
print("TEST 1: Verify bio_tags_to_mwe_column handles discontinuous correctly")
print("=" * 70)

# Simulate a discontinuous pattern AFTER postprocessing
bio_tags = ['B-MWE', 'I-MWE', 'I-MWE', 'O', 'B-MWE', 'I-MWE']
categories = ['VID', 'VID', 'VID', 'O', 'NID', 'NID']

mwe_column = bio_tags_to_mwe_column(bio_tags, categories)
print(f"BIO tags:  {bio_tags}")
print(f"Categories: {categories}")
print(f"MWE column: {mwe_column}")
print(f"Expected:   ['1:VID', '1', '1', '*', '2:NID', '2']")
print()

print("=" * 70)
print("TEST 2: Check if postprocessing creates discontinuous patterns")
print("=" * 70)

# Model outputs: separate MWE pieces with gaps
model_bio = ['B-MWE', 'I-MWE', 'O', 'O', 'I-MWE', 'O']
model_cats = ['VID', 'VID', 'O', 'O', 'VID', 'O']

print(f"BEFORE postprocessing:")
print(f"  BIO: {model_bio}")
print(f"  Cats: {model_cats}")

# Apply postprocessing
fixed = fix_discontinuous_mwes([(model_bio, model_cats)])
fixed_bio, fixed_cats = fixed[0]

print(f"AFTER postprocessing:")
print(f"  BIO: {fixed_bio}")
print(f"  Cats: {fixed_cats}")

# Convert to CUPT format
mwe_column = bio_tags_to_mwe_column(fixed_bio, fixed_cats)
print(f"MWE column: {mwe_column}")
print(f"Expected:   ['1:VID', '1', '1', '1', '1', '*']")
print()

print("=" * 70)
print("TEST 3: Check actual prediction file format")
print("=" * 70)

# Read a few lines from actual prediction
with open('predictions/FR/dev.system.cupt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find first sentence with MWEs
mwe_found = False
sent_lines = []
for line in lines:
    if line.startswith('#') or not line.strip():
        if sent_lines and mwe_found:
            break
        sent_lines = []
        mwe_found = False
        continue
    
    parts = line.strip().split('\t')
    if len(parts) >= 11:
        sent_lines.append(parts)
        if parts[10] != '*':
            mwe_found = True

if sent_lines:
    print("Sample sentence with MWEs:")
    for parts in sent_lines[:10]:
        token_id = parts[0]
        form = parts[1]
        mwe_col = parts[10] if len(parts) > 10 else '*'
        print(f"  {token_id:4s} {form:15s} {mwe_col}")
print()

print("=" * 70)
print("TEST 4: Count discontinuous patterns in predictions")
print("=" * 70)

# Read all predictions and check for discontinuous patterns
discontinuous_count = 0
continuous_count = 0

with open('predictions/FR/dev.system.cupt', 'r', encoding='utf-8') as f:
    sentence_mwes = []
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
                        is_continuous = all(positions[i] + 1 == positions[i+1] for i in range(len(positions) - 1))
                        if is_continuous:
                            continuous_count += 1
                        else:
                            discontinuous_count += 1
                            print(f"  Found discontinuous MWE {mwe_id} at positions: {positions}")
            
            sentence_mwes = []
            continue
        
        parts = line.strip().split('\t')
        if len(parts) >= 11 and not ('-' in parts[0] or '.' in parts[0]):
            sentence_mwes.append(parts[10])

print(f"\nContinuous MWEs found: {continuous_count}")
print(f"Discontinuous MWEs found: {discontinuous_count}")
print()

if discontinuous_count == 0:
    print("⚠️  WARNING: NO DISCONTINUOUS MWEs FOUND IN PREDICTIONS!")
    print("   This explains why the discontinuous F1 score is 0.0")
    print("   Either:")
    print("   1. Model is not predicting discontinuous patterns")
    print("   2. Postprocessing is not being applied")
    print("   3. Postprocessing is not creating discontinuous patterns")
else:
    print("✓ Discontinuous MWEs are present in predictions")
