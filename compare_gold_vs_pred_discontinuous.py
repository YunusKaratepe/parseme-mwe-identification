"""
Compare: How many discontinuous MWEs in gold vs predictions
Also test if the model CAN predict discontinuous patterns
"""
import os
import sys
sys.path.insert(0, 'src')

def analyze_file(filepath):
    """Analyze a CUPT file for continuous/discontinuous MWEs"""
    continuous = 0
    discontinuous = 0
    
    sentence_mwes = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                # Analyze sentence
                if sentence_mwes:
                    # Track MWE IDs and their positions
                    mwe_positions = {}
                    for idx, mwe_val in enumerate(sentence_mwes):
                        if mwe_val != '*':
                            # Extract MWE ID
                            if ':' in mwe_val:
                                mwe_ids = mwe_val.split(':')[0]
                            else:
                                mwe_ids = mwe_val
                            
                            # Handle multiple MWEs (e.g., "1;2")
                            for mwe_id in mwe_ids.split(';'):
                                if mwe_id not in mwe_positions:
                                    mwe_positions[mwe_id] = []
                                mwe_positions[mwe_id].append(idx)
                    
                    # Check each MWE for continuity
                    for mwe_id, positions in mwe_positions.items():
                        if len(positions) > 1:
                            is_continuous = all(positions[i] + 1 == positions[i+1] 
                                              for i in range(len(positions) - 1))
                            if is_continuous:
                                continuous += 1
                            else:
                                discontinuous += 1
                
                sentence_mwes = []
                continue
            
            parts = line.strip().split('\t')
            if len(parts) >= 11 and not ('-' in parts[0] or '.' in parts[0]):
                sentence_mwes.append(parts[10])
    
    return continuous, discontinuous


# Languages to check
languages = ["FR", "PL", "EL", "PT", "SL", "SR", "SV", "UK", "NL", "EGY", "KA", "JA", "HE", "LV", "FA", "RO"]

print("=" * 90)
print("GOLD vs PREDICTIONS: Continuous vs Discontinuous MWEs")
print("=" * 90)
print()
print(f"{'Lang':<6} | {'Gold Cont.':<10} | {'Gold Disc.':<10} | {'Pred Cont.':<10} | {'Pred Disc.':<10} | {'Disc %':<8}")
print("-" * 90)

total_gold_cont = 0
total_gold_disc = 0
total_pred_cont = 0
total_pred_disc = 0

for lang in languages:
    gold_file = f"2.0/subtask1/{lang}/dev.cupt"
    pred_file = f"predictions/{lang}/dev.system.cupt"
    
    if not os.path.exists(gold_file):
        continue
    
    gold_cont, gold_disc = analyze_file(gold_file)
    
    if os.path.exists(pred_file):
        pred_cont, pred_disc = analyze_file(pred_file)
    else:
        pred_cont, pred_disc = 0, 0
    
    # Calculate discontinuous percentage in gold
    total_gold = gold_cont + gold_disc
    disc_pct = (gold_disc / total_gold * 100) if total_gold > 0 else 0
    
    print(f"{lang:<6} | {gold_cont:<10} | {gold_disc:<10} | {pred_cont:<10} | {pred_disc:<10} | {disc_pct:>6.1f}%")
    
    total_gold_cont += gold_cont
    total_gold_disc += gold_disc
    total_pred_cont += pred_cont
    total_pred_disc += pred_disc

print("-" * 90)
total_gold = total_gold_cont + total_gold_disc
total_pred = total_pred_cont + total_pred_disc
gold_disc_pct = (total_gold_disc / total_gold * 100) if total_gold > 0 else 0
pred_disc_pct = (total_pred_disc / total_pred * 100) if total_pred > 0 else 0

print(f"{'TOTAL':<6} | {total_gold_cont:<10} | {total_gold_disc:<10} | {total_pred_cont:<10} | {total_pred_disc:<10} | {gold_disc_pct:>6.1f}%")
print()
print("=" * 90)
print("ANALYSIS:")
print(f"  Gold data has {total_gold_disc} discontinuous MWEs ({gold_disc_pct:.1f}% of all MWEs)")
print(f"  Predictions have {total_pred_disc} discontinuous MWEs ({pred_disc_pct:.1f}% of all MWEs)")
print()

if total_pred_disc == 0 and total_gold_disc > 0:
    print("⚠️  CRITICAL: Model predicts ZERO discontinuous MWEs while gold has thousands!")
    print()
    print("Impact on F1 score:")
    print(f"  - Discontinuous Recall: 0/{total_gold_disc} = 0.00")
    print(f"  - Discontinuous Precision: 0/0 = 0.00")
    print(f"  - Discontinuous F1: 0.00")
    print()
    print("This explains the 0.0000 discontinuous F1 in evaluation reports.")
    print()
    print("Why this happens:")
    print("  Token-level BERT classification learns:")
    print("    Pattern: B-MWE → I-MWE → I-MWE (continuous)")
    print("    Sequence: token1=B-MWE, token2=I-MWE, token3=O → MWE ended")
    print()
    print("  It CANNOT learn:")
    print("    Pattern: B-MWE → O → O → I-MWE (discontinuous)")
    print("    Because after seeing O, it assumes MWE ended")
    print()
    print("Solutions:")
    print("  1. Add CRF layer → learns sequence constraints")
    print("  2. Span-based prediction → predicts MWE spans, not token tags")
    print("  3. Graph-based model → explicitly models token relationships")
    print("  4. Multi-stage approach → first predict, then link")
