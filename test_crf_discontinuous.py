"""
Test if CRF-enabled model can actually predict discontinuous MWE patterns
"""
import sys
sys.path.insert(0, 'src')
import torch
from model import MWEIdentificationModel, MWETokenizer
import os

# Load model
model_path = "models/multilingual_FR/best_model.pt"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Loading model from {model_path}...")
checkpoint = torch.load(model_path, map_location=device)

model_name = checkpoint['model_name']
label_to_id = checkpoint['label_to_id']
category_to_id = checkpoint['category_to_id']
pos_to_id = checkpoint.get('pos_to_id', None)
use_pos = checkpoint.get('use_pos', False)
use_lang_tokens = checkpoint.get('use_lang_tokens', False)
use_crf = checkpoint.get('use_crf', False)
loss_type = checkpoint.get('loss_type', 'ce')

id_to_label = {v: k for k, v in label_to_id.items()}
id_to_category = {v: k for k, v in category_to_id.items()}

print(f"\n=== Model Configuration ===")
print(f"use_crf: {use_crf}")
print(f"use_pos: {use_pos}")
print(f"use_lang_tokens: {use_lang_tokens}")

# Handle invalid model paths
if not os.path.exists(model_name) and '/' in model_name:
    print(f"WARNING: Model path '{model_name}' not found, using bert-base-multilingual-cased")
    model_name = 'bert-base-multilingual-cased'

# Initialize tokenizer
tokenizer_path = os.path.join(os.path.dirname(model_path), 'tokenizer')
if os.path.exists(tokenizer_path):
    tokenizer = MWETokenizer(tokenizer_path, use_lang_tokens=use_lang_tokens)
else:
    tokenizer = MWETokenizer(model_name, use_lang_tokens=use_lang_tokens)

# Initialize model
model = MWEIdentificationModel(
    model_name, 
    num_labels=len(label_to_id), 
    num_categories=len(category_to_id),
    num_pos_tags=len(pos_to_id) if pos_to_id else 18,
    use_pos=use_pos,
    loss_type=loss_type,
    use_crf=use_crf
)

# Load state dict
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

print(f"\n=== Model Loaded ===")
print(f"Model has CRF: {hasattr(model, 'crf')}")
print(f"Model.use_crf: {model.use_crf}")

if model.use_crf:
    print(f"\n=== CRF Transition Matrix ===")
    print(f"Shape: {model.crf.transitions.shape}")
    print(f"\nTransition probabilities:")
    print(f"O -> O: {model.crf.transitions[0, 0].item():.4f}")
    print(f"O -> B-MWE: {model.crf.transitions[1, 0].item():.4f}")
    print(f"O -> I-MWE: {model.crf.transitions[2, 0].item():.4f}  <-- KEY for discontinuous!")
    print(f"B-MWE -> I-MWE: {model.crf.transitions[2, 1].item():.4f}")
    print(f"I-MWE -> I-MWE: {model.crf.transitions[2, 2].item():.4f}")
    print(f"I-MWE -> O: {model.crf.transitions[0, 2].item():.4f}")
    
    print(f"\n=== Analysis ===")
    o_to_i = model.crf.transitions[2, 0].item()
    if o_to_i < -10:
        print(f"[!] O -> I-MWE transition is HEAVILY PENALIZED ({o_to_i:.2f})")
        print(f"    This means CRF will NEVER predict discontinuous patterns!")
        print(f"    Expected: value >= -5.0 for discontinuous MWEs to be possible")
    elif o_to_i < -5:
        print(f"[!] O -> I-MWE transition is moderately penalized ({o_to_i:.2f})")
        print(f"    Discontinuous MWEs are possible but unlikely")
    else:
        print(f"[OK] O -> I-MWE transition is allowed ({o_to_i:.2f})")
        print(f"     Discontinuous MWEs should be possible")
else:
    print("\n[!] CRF is NOT enabled!")

# Test on a sample sentence from gold data with known discontinuous MWE
print(f"\n=== Testing Prediction ===")
from data_loader import CUPTDataLoader

loader = CUPTDataLoader()
test_file = "2.0/subtask1/FR/dev.cupt"
print(f"Loading: {test_file}")
sentences = loader.read_cupt_file(test_file)

# Find first sentence with discontinuous MWE
found_count = 0
for sent_idx, sent in enumerate(sentences):
    if sent_idx > 1000:  # Check more sentences
        break
    mwe_raw = sent.get('mwe_raw', [])
    # Check if has discontinuous pattern (MWE IDs with semicolons)
    has_disc = any(';' in str(mwe) for mwe in mwe_raw if mwe != '*')
    
    if has_disc:
        found_count += 1
        if found_count == 1:  # Only analyze first one
            print(f"\nFound discontinuous MWE in sentence {sent_idx + 1}")
            print(f"Sentence: {sent.get('text', ' '.join(sent['tokens']))}")
            print(f"\nGold MWE annotations:")
            for i, (token, mwe) in enumerate(zip(sent['tokens'], mwe_raw)):
                if mwe != '*':
                    print(f"  {i+1}: {token:<15} {mwe}")
            
            # Predict
            tokens = sent['tokens']
            pos_tags = sent['pos_tags'] if use_pos else None
        
        from model import predict_mwe_tags
        bio_tags, categories = predict_mwe_tags(
            model, tokenizer, tokens, id_to_label, id_to_category,
            device, pos_tags, pos_to_id, use_pos, language='FR' if use_lang_tokens else None
        )
        
        print(f"\nPredicted BIO tags:")
        for i, (token, bio, cat) in enumerate(zip(tokens, bio_tags, categories)):
            if bio != 'O':
                print(f"  {i+1}: {token:<15} {bio:<7} {cat}")
        
        # Check if predicted any discontinuous pattern
        has_b = False
        has_gap = False
        for bio in bio_tags:
            if bio == 'B-MWE':
                has_b = True
                has_gap = False
            elif bio == 'O' and has_b:
                has_gap = True
            elif bio == 'I-MWE' and has_b and has_gap:
                print(f"\n[OK] DISCONTINUOUS PATTERN DETECTED IN PREDICTIONS!")
                break
        else:
            print(f"\n[X] No discontinuous pattern in predictions (continuous or no MWE)")
            
            break

print(f"\nTotal discontinuous MWEs found in dev set: {found_count}")
if found_count == 0:
    print("[!] No discontinuous MWE found in first 1000 sentences")
