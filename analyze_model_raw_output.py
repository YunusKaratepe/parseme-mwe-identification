"""
Directly check model raw outputs BEFORE postprocessing
Load model and predict on a few sentences, check if any B-MWE...O...I-MWE patterns exist
"""
import sys
import os
sys.path.insert(0, 'src')
import torch

from model import MWEIdentificationModel, MWETokenizer, predict_mwe_tags

# Load model
model_path = "models/focalLoss-langTokens/best_model.pt"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Loading model...")
checkpoint = torch.load(model_path, map_location=device, weights_only=False)

label_to_id = checkpoint['label_to_id']
category_to_id = checkpoint['category_to_id']
pos_to_id = checkpoint.get('pos_to_id', None)
use_pos = checkpoint.get('use_pos_features', False)
use_lang_tokens = checkpoint.get('use_language_tokens', False)

id_to_label = {v: k for k, v in label_to_id.items()}
id_to_category = {v: k for k, v in category_to_id.items()}

# Load tokenizer from saved directory
tokenizer_path = os.path.join(os.path.dirname(model_path), 'tokenizer')
tokenizer = MWETokenizer(tokenizer_path)

# Load model
model = MWEIdentificationModel(
    model_name=tokenizer_path,
    num_labels=len(label_to_id),
    num_categories=len(category_to_id),
    num_pos_tags=len(pos_to_id) if pos_to_id else 18,
    use_pos=use_pos,
    loss_type=checkpoint.get('loss_type', 'ce')
)

model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

print(f"Model loaded. Device: {device}")
print()

# Load sentences from FR dev
input_file = "2.0/subtask1/FR/dev.cupt"

sentences = []
current_tokens = []
current_pos = []

print("Loading test sentences...")
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        
        if not line or line.startswith('#'):
            if current_tokens:
                sentences.append((current_tokens, current_pos))
                current_tokens = []
                current_pos = []
                
                if len(sentences) >= 100:  # Check first 100 sentences
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
        
        current_tokens.append(form)
        current_pos.append(pos_tag)

print(f"Loaded {len(sentences)} sentences")
print()

# Check for B-MWE...O...I-MWE patterns
print("Analyzing model predictions...")
print("=" * 80)

patterns_found = 0
total_sentences_checked = 0

for sent_idx, (tokens, pos_tags) in enumerate(sentences):
    # Get model prediction
    pred_tags, pred_cats = predict_mwe_tags(
        model, tokenizer, tokens, id_to_label, id_to_category, 
        device, pos_tags, pos_to_id, use_pos, language='FR'
    )
    
    total_sentences_checked += 1
    
    # Check for B-MWE ... O ... I-MWE pattern
    i = 0
    while i < len(pred_tags):
        if pred_tags[i] == 'B-MWE':
            cat = pred_cats[i]
            j = i + 1
            found_o = False
            
            # Look for O followed by I-MWE with same category
            while j < len(pred_tags):
                if pred_tags[j] == 'O':
                    found_o = True
                    j += 1
                elif pred_tags[j] == 'I-MWE' and found_o and pred_cats[j] == cat:
                    # FOUND IT!
                    patterns_found += 1
                    print(f"\n✓ FOUND discontinuous pattern in sentence {sent_idx+1}!")
                    print(f"  Tokens: {' '.join(tokens[:20])}{'...' if len(tokens) > 20 else ''}")
                    print(f"  BIO tags: {pred_tags}")
                    print(f"  Categories: {pred_cats}")
                    
                    # Highlight the pattern
                    pattern_tokens = []
                    for k in range(i, min(j+1, len(pred_tags))):
                        if pred_tags[k] != 'O':
                            pattern_tokens.append(f"{tokens[k]}({pred_tags[k]})")
                        else:
                            pattern_tokens.append(tokens[k])
                    print(f"  Pattern: {' '.join(pattern_tokens)}")
                    break
                elif pred_tags[j] == 'B-MWE':
                    break
                else:
                    j += 1
        i += 1

print()
print("=" * 80)
print(f"Checked {total_sentences_checked} sentences")
print(f"Found {patterns_found} B-MWE...O...I-MWE patterns in model raw output")
print("=" * 80)
print()

if patterns_found == 0:
    print("⚠️  MODEL RAW OUTPUT: NO discontinuous patterns!")
    print()
    print("This proves:")
    print("  1. The model itself does NOT predict B-MWE...O...I-MWE patterns")
    print("  2. Postprocessing has nothing to work with")
    print("  3. This is a fundamental limitation of token-level classification")
    print()
    print("Why does this happen?")
    print("  - Model treats each token independently (with context, but still)")
    print("  - Learns: B-MWE → I-MWE (continuous)")
    print("  - Learns: I-MWE → O means MWE ended")
    print("  - Never learns: after O comes I-MWE of same MWE")
    print()
    print("Solution options:")
    print("  1. Add CRF layer (sequence constraints)")
    print("  2. Use span-based approach (predict spans, not tokens)")
    print("  3. Add explicit discontinuity signals during training")
    print("  4. Use graph-based approach")
else:
    print(f"✓ Model CAN predict discontinuous patterns!")
    print(f"  But postprocessing might not be applied during prediction")
