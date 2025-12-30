"""
Debug script to check what the model actually predicts before postprocessing
"""
import sys
import os
sys.path.insert(0, 'src')
import torch

from data_loader import CUPTDataLoader
from model import MWEIdentificationModel, MWETokenizer, predict_mwe_tags

# Load model
model_path = "models/focalLoss-langTokens/best_model.pt"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Loading model from {model_path}...")
checkpoint = torch.load(model_path, map_location=device)

model_name = checkpoint['model_name']
label_to_id = checkpoint['label_to_id']
category_to_id = checkpoint['category_to_id']
pos_to_id = checkpoint.get('pos_to_id', None)
use_pos = checkpoint.get('use_pos_features', False)
use_lang_tokens = checkpoint.get('use_language_tokens', False)

id_to_label = {v: k for k, v in label_to_id.items()}
id_to_category = {v: k for k, v in category_to_id.items()}

# Load tokenizer from saved directory instead of model_name
tokenizer_path = os.path.join(os.path.dirname(model_path), 'tokenizer')
tokenizer = MWETokenizer(tokenizer_path)

model = MWEIdentificationModel(
    model_name=model_name,
    num_labels=len(label_to_id),
    num_categories=len(category_to_id),
    use_pos_features=use_pos,
    pos_vocab_size=len(pos_to_id) if pos_to_id else 0,
    use_language_tokens=use_lang_tokens
)

model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

print(f"Model loaded. Use POS: {use_pos}, Use Lang Tokens: {use_lang_tokens}")
print()

# Load a few sentences from FR dev
input_file = "2.0/subtask1/FR/dev.cupt"

sentences_with_gold = []
current_tokens = []
current_gold_mwe = []

with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        
        if not line or line.startswith('#'):
            if current_tokens:
                sentences_with_gold.append((current_tokens, current_gold_mwe))
                current_tokens = []
                current_gold_mwe = []
                
                if len(sentences_with_gold) >= 10:  # Only check first 10 sentences
                    break
            continue
        
        parts = line.split('\t')
        if len(parts) < 11:
            continue
        
        token_id = parts[0]
        if '-' in token_id or '.' in token_id:
            continue
        
        form = parts[1]
        pos_tag = parts[3]
        gold_mwe = parts[10]
        
        current_tokens.append(form)
        current_gold_mwe.append(gold_mwe)

print(f"Loaded {len(sentences_with_gold)} sentences for testing")
print()

# Check for discontinuous patterns in model predictions
discontinuous_in_model = 0
discontinuous_in_gold = 0

for sent_idx, (tokens, gold_mwe) in enumerate(sentences_with_gold):
    # Get model prediction
    pos_tags = ['NOUN'] * len(tokens)  # Dummy POS
    pred_tags, pred_cats = predict_mwe_tags(
        model, tokenizer, tokens, id_to_label, id_to_category, 
        device, pos_tags, pos_to_id, use_pos, language='FR'
    )
    
    # Check for B-MWE ... O ... I-MWE pattern in model output
    has_discontinuous_pattern = False
    i = 0
    while i < len(pred_tags):
        if pred_tags[i] == 'B-MWE':
            # Look ahead for O followed by I-MWE with same category
            cat = pred_cats[i]
            j = i + 1
            found_o = False
            
            while j < len(pred_tags):
                if pred_tags[j] == 'O':
                    found_o = True
                    j += 1
                elif pred_tags[j] == 'I-MWE' and found_o and pred_cats[j] == cat:
                    has_discontinuous_pattern = True
                    print(f"Sentence {sent_idx+1}: Model DOES predict discontinuous pattern!")
                    print(f"  Tokens: {tokens}")
                    print(f"  BIO: {pred_tags}")
                    print(f"  Cats: {pred_cats}")
                    print()
                    break
                elif pred_tags[j] == 'B-MWE':
                    break
                else:
                    j += 1
            
            if has_discontinuous_pattern:
                discontinuous_in_model += 1
                break
        i += 1
    
    # Check gold for discontinuous
    mwe_positions = {}
    for idx, mwe_val in enumerate(gold_mwe):
        if mwe_val != '*':
            if ':' in mwe_val:
                mwe_ids = mwe_val.split(':')[0]
            else:
                mwe_ids = mwe_val
            
            for mwe_id in mwe_ids.split(';'):
                if mwe_id not in mwe_positions:
                    mwe_positions[mwe_id] = []
                mwe_positions[mwe_id].append(idx)
    
    for mwe_id, positions in mwe_positions.items():
        if len(positions) > 1:
            is_continuous = all(positions[i] + 1 == positions[i+1] for i in range(len(positions) - 1))
            if not is_continuous:
                discontinuous_in_gold += 1
                if not has_discontinuous_pattern:
                    print(f"Sentence {sent_idx+1}: Gold HAS discontinuous, model does NOT")
                    print(f"  Tokens: {tokens}")
                    print(f"  Gold MWE: {gold_mwe}")
                    print(f"  Model BIO: {pred_tags}")
                    print(f"  Model Cats: {pred_cats}")
                    print(f"  MWE {mwe_id} positions: {positions}")
                    print()
                break

print("=" * 70)
print(f"Discontinuous patterns in gold (first 10 sent): {discontinuous_in_gold}")
print(f"Discontinuous patterns in model output: {discontinuous_in_model}")
print("=" * 70)

if discontinuous_in_model == 0:
    print("\n⚠️  MODEL IS NOT PREDICTING DISCONTINUOUS PATTERNS")
    print("   This is the root cause of 0.0 F1 for discontinuous MWEs")
    print("\n   The issue is NOT in:")
    print("   - Postprocessing code (it works correctly)")
    print("   - Format conversion (bio_tags_to_mwe_column works)")
    print("\n   The issue IS in:")
    print("   - Model architecture cannot learn long-range dependencies")
    print("   - Token-level classification treats each token independently")
