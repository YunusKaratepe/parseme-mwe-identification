"""
Analyze model predictions vs gold to understand discontinuous MWE issue
"""
import sys
sys.path.insert(0, 'src')
import torch
from model import MWEIdentificationModel, MWETokenizer, predict_mwe_tags
from data_loader import CUPTDataLoader
import os

# Load model
model_path = "models/multilingual_FR/best_model.pt"
device = torch.device('cpu')

print("Loading model...")
checkpoint = torch.load(model_path, map_location=device)

model_name = checkpoint['model_name']
if not os.path.exists(model_name) and '/' in model_name:
    model_name = 'bert-base-multilingual-cased'

label_to_id = checkpoint['label_to_id']
category_to_id = checkpoint['category_to_id']
pos_to_id = checkpoint.get('pos_to_id', None)
use_pos = checkpoint.get('use_pos', False)
use_crf = checkpoint.get('use_crf', False)
loss_type = checkpoint.get('loss_type', 'ce')

id_to_label = {v: k for k, v in label_to_id.items()}
id_to_category = {v: k for k, v in category_to_id.items()}

tokenizer_path = os.path.join(os.path.dirname(model_path), 'tokenizer')
tokenizer = MWETokenizer(tokenizer_path if os.path.exists(tokenizer_path) else model_name)

model = MWEIdentificationModel(
    model_name, num_labels=len(label_to_id), num_categories=len(category_to_id),
    num_pos_tags=len(pos_to_id) if pos_to_id else 18,
    use_pos=use_pos, loss_type=loss_type, use_crf=use_crf
)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

print(f"Model loaded. CRF: {model.use_crf}")

# Load gold data
loader = CUPTDataLoader()
sentences = loader.read_cupt_file('2.0/subtask1/FR/dev.cupt')

print(f"\nAnalyzing {len(sentences)} sentences...")
print("=" * 80)

# Count statistics
disc_in_gold = 0
disc_in_pred = 0
cont_in_gold = 0
cont_in_pred = 0

for sent_idx, sent in enumerate(sentences):
    if sent_idx >= 100:  # Just analyze first 100
        break
    
    tokens = sent['tokens']
    gold_tags = sent['mwe_tags']
    gold_cats = sent['mwe_categories']
    pos_tags = sent['pos_tags'] if use_pos else None
    
    # Check if gold has discontinuous
    has_disc_gold = False
    in_mwe = False
    mwe_ended = False
    for tag in gold_tags:
        if tag == 'B-MWE':
            in_mwe = True
            mwe_ended = False
        elif tag == 'O' and in_mwe:
            mwe_ended = True
        elif tag == 'I-MWE' and mwe_ended:
            has_disc_gold = True
            break
    
    # Predict
    bio_tags, categories = predict_mwe_tags(
        model, tokenizer, tokens, id_to_label, id_to_category,
        device, pos_tags, pos_to_id, use_pos, language=None
    )
    
    # Check if prediction has discontinuous
    has_disc_pred = False
    in_mwe = False
    mwe_ended = False
    for tag in bio_tags:
        if tag == 'B-MWE':
            in_mwe = True
            mwe_ended = False
        elif tag == 'O' and in_mwe:
            mwe_ended = True
        elif tag == 'I-MWE' and mwe_ended:
            has_disc_pred = True
            break
    
    if has_disc_gold:
        disc_in_gold += 1
        if has_disc_pred:
            disc_in_pred += 1
            print(f"\n[MATCH] Sentence {sent_idx + 1} has discontinuous in both:")
            print(f"  Gold: {' '.join([f'{t}:{g}' if g != 'O' else t for t, g in zip(tokens, gold_tags)])}")
            print(f"  Pred: {' '.join([f'{t}:{p}' if p != 'O' else t for t, p in zip(tokens, bio_tags)])}")
        else:
            print(f"\n[MISS] Sentence {sent_idx + 1} has discontinuous in gold only:")
            print(f"  Text: {sent.get('text', '')[:100]}")
            print(f"  Gold: {' '.join([f'{t}:{g}' if g != 'O' else t for t, g in zip(tokens, gold_tags)])}")
            print(f"  Pred: {' '.join([f'{t}:{p}' if p != 'O' else t for t, p in zip(tokens, bio_tags)])}")
    
    # Check for continuous MWEs
    has_mwe_gold = any(t != 'O' for t in gold_tags)
    has_mwe_pred = any(t != 'O' for t in bio_tags)
    if has_mwe_gold and not has_disc_gold:
        cont_in_gold += 1
        if has_mwe_pred:
            cont_in_pred += 1

print(f"\n" + "=" * 80)
print(f"STATISTICS (first 100 sentences):")
print(f"  Gold discontinuous MWEs: {disc_in_gold}")
print(f"  Pred discontinuous MWEs: {disc_in_pred}")
print(f"  Gold continuous MWEs: {cont_in_gold}")
print(f"  Pred continuous MWEs: {cont_in_pred}")
print(f"\nDiscontinuous recall: {disc_in_pred}/{disc_in_gold} = {disc_in_pred/disc_in_gold*100 if disc_in_gold > 0 else 0:.1f}%")
