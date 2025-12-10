"""
Inference script for PARSEME 2.0 MWE identification
Generates predictions in CUPT format
"""
import os
import sys
import torch
from typing import List, Dict
from collections import defaultdict

from data_loader import CUPTDataLoader
from model import MWEIdentificationModel, MWETokenizer, predict_mwe_tags


def bio_tags_to_mwe_column(bio_tags: List[str], mwe_categories: List[str] = None) -> List[str]:
    """
    Convert BIO tags back to CUPT MWE column format
    
    Args:
        bio_tags: List of BIO tags
        mwe_categories: List of MWE categories (optional)
    
    Returns:
        List of MWE column values (e.g., '*', '1', '1:VID', etc.)
    """
    # Valid PARSEME categories (generic "MWE" is not valid)
    VALID_CATEGORIES = {
        'IAV', 'IRV', 'LVC.cause', 'LVC.full', 'MVC', 'VID',
        'IVPC.full', 'IVPC.semi', 'AdjID', 'AdpID', 'AdvID', 
        'ConjID', 'DetID', 'IntjID', 'NID', 'PronID',
        'AV.LVC.full', 'AV.LVC.cause', 'AV.VID', 'AV.IRV',
        'AV.IVPC.full', 'AV.IVPC.semi', 'AV.MVC', 'AV.IAV',
        'NV.LVC.full', 'NV.LVC.cause', 'NV.VID', 'NV.IRV',
        'NV.IVPC.full', 'NV.IVPC.semi', 'NV.MVC', 'NV.IAV'
    }
    
    mwe_column = ['*'] * len(bio_tags)
    mwe_id_counter = 1
    current_mwe_id = None
    current_category = None
    
    for idx, tag in enumerate(bio_tags):
        if tag == 'B-MWE':
            # Start new MWE - first token gets ID:CATEGORY
            current_mwe_id = mwe_id_counter
            
            if mwe_categories and idx < len(mwe_categories) and mwe_categories[idx] != 'O':
                cat = mwe_categories[idx]
                # Map invalid "MWE" to "VID" (most common verbal idiom)
                if cat == 'MWE' or cat not in VALID_CATEGORIES:
                    cat = 'VID'
                current_category = cat
                mwe_column[idx] = f"{current_mwe_id}:{cat}"
            else:
                # No category provided, use VID as default
                current_category = 'VID'
                mwe_column[idx] = f"{current_mwe_id}:VID"
            mwe_id_counter += 1
            
        elif tag == 'I-MWE' and current_mwe_id is not None:
            # Continue current MWE - subsequent tokens just get ID (no category)
            mwe_column[idx] = str(current_mwe_id)
        else:
            # Outside MWE or error
            current_mwe_id = None
            current_category = None
    
    return mwe_column


def predict_cupt_file(
    model_path: str,
    input_file: str,
    output_file: str,
    device: torch.device = None,
    fix_discontinuous: bool = True
):
    """
    Predict MWE annotations for a CUPT file
    
    Args:
        model_path: Path to saved model checkpoint
        input_file: Path to input .cupt file (with empty MWE column)
        output_file: Path to output .cupt file (with predicted MWE annotations)
        device: torch device
        fix_discontinuous: Apply heuristic stitching for discontinuous MWEs (default: True)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Load model checkpoint
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    model_name = checkpoint['model_name']
    label_to_id = checkpoint['label_to_id']
    category_to_id = checkpoint['category_to_id']
    pos_to_id = checkpoint.get('pos_to_id', None)
    use_pos = checkpoint.get('use_pos', False)
    use_lang_tokens = checkpoint.get('use_lang_tokens', False)
    id_to_label = {v: k for k, v in label_to_id.items()}
    id_to_category = {v: k for k, v in category_to_id.items()}
    
    # Handle invalid model paths (e.g., Colab paths that don't exist locally)
    if not os.path.exists(model_name) and '/' in model_name and not model_name.startswith('bert-'):
        print(f"WARNING: Model path '{model_name}' not found locally.")
        print(f"         Falling back to 'bert-base-multilingual-cased'")
        model_name = 'bert-base-multilingual-cased'
    
    # Initialize model
    model = MWEIdentificationModel(
        model_name, 
        num_labels=len(label_to_id), 
        num_categories=len(category_to_id),
        num_pos_tags=len(pos_to_id) if pos_to_id else 18,
        use_pos=use_pos
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Resize token embeddings if language tokens were used during training
    if use_lang_tokens:
        tokenizer_path = os.path.join(os.path.dirname(model_path), 'tokenizer')
        if os.path.exists(tokenizer_path):
            tokenizer = MWETokenizer(tokenizer_path, use_lang_tokens=True)
        else:
            tokenizer = MWETokenizer(model_name, use_lang_tokens=True)
        model.transformer.resize_token_embeddings(len(tokenizer.get_tokenizer()))
    else:
        tokenizer_path = os.path.join(os.path.dirname(model_path), 'tokenizer')
        if os.path.exists(tokenizer_path):
            tokenizer = MWETokenizer(tokenizer_path)
        else:
            tokenizer = MWETokenizer(model_name)
    
    model.to(device)
    model.eval()
    
    print(f"POS feature injection: {'ENABLED' if use_pos else 'DISABLED'}")
    print(f"Language-conditioned inputs: {'ENABLED' if use_lang_tokens else 'DISABLED'}")
    
    print(f"Model loaded. Best F1: {checkpoint.get('best_f1', 'N/A')}")
    print(f"Categories: {len(category_to_id)} types")
    
    # Extract language from input file path
    language = None
    if use_lang_tokens:
        # Try to extract from path like "2.0/subtask1/FR/test.blind.cupt"
        parts = input_file.replace('\\', '/').split('/')
        for lang in ['EGY', 'EL', 'FA', 'FR', 'GRC', 'HE', 'JA', 'KA', 'LV', 'NL', 'PL', 'PT', 'RO', 'SL', 'SR', 'SV', 'UK']:
            if lang in parts:
                language = lang
                print(f"Detected language: {language}")
                break
        if language is None:
            print("WARNING: Could not detect language from file path, language tokens will not be used")
    
    # Read input file
    print(f"\nReading input file: {input_file}")
    
    predictions_by_sentence = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        current_tokens = []
        current_pos_tags = []
        
        for line in f:
            line = line.rstrip('\n')
            
            # Skip comments
            if line.startswith('#'):
                continue
            
            # Empty line = end of sentence
            if not line.strip():
                if current_tokens:
                    # Predict for this sentence (returns bio_tags and categories)
                    pred_tags, pred_cats = predict_mwe_tags(model, tokenizer, current_tokens, id_to_label, id_to_category, device, current_pos_tags, pos_to_id, use_pos, language)
                    predictions_by_sentence.append((pred_tags, pred_cats))
                    current_tokens = []
                    current_pos_tags = []
                else:
                    # Empty sentence (comment-only)
                    predictions_by_sentence.append(([], []))
                continue
            
            # Parse token line
            parts = line.split('\t')
            if len(parts) < 11:
                continue
            
            token_id = parts[0]
            
            # Skip multi-word tokens
            if '-' in token_id or '.' in token_id:
                continue
            
            form = parts[1]
            pos_tag = parts[3]  # UPOS column
            current_tokens.append(form)
            current_pos_tags.append(pos_tag)
        
        # Don't forget last sentence
        if current_tokens:
            pred_tags, pred_cats = predict_mwe_tags(model, tokenizer, current_tokens, id_to_label, id_to_category, device, current_pos_tags, pos_to_id, use_pos, language)
            predictions_by_sentence.append((pred_tags, pred_cats))
    
    print(f"Predicted MWEs for {len(predictions_by_sentence)} sentences")
    
    # Apply discontinuous MWE fixing if enabled
    if fix_discontinuous:
        print("Applying discontinuous MWE post-processing...")
        from postprocess_discontinuous import fix_discontinuous_mwes
        predictions_by_sentence = fix_discontinuous_mwes(predictions_by_sentence)
        print("✓ Discontinuous MWE patterns fixed")
    
    # Convert predictions to MWE column format
    print("Converting predictions to CUPT format...")
    mwe_columns_by_sentence = []
    for pred_tags, pred_cats in predictions_by_sentence:
        mwe_col = bio_tags_to_mwe_column(pred_tags, pred_cats)
        mwe_columns_by_sentence.append(mwe_col)
    
    # Write output file
    print(f"\nWriting predictions to: {output_file}")
    
    with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
        sentence_idx = -1
        token_idx = -1
        
        for line in fin:
            line = line.rstrip('\n')
            
            # Copy comments
            if line.startswith('#'):
                fout.write(line + '\n')
                continue
            
            # Empty line = end of sentence
            if not line.strip():
                fout.write('\n')
                sentence_idx += 1
                token_idx = -1
                continue
            
            # Parse and update token line
            parts = line.split('\t')
            
            if len(parts) < 11:
                fout.write(line + '\n')
                continue
            
            token_id = parts[0]
            
            # Multi-word tokens: replace '_' with '*' for predictions
            if '-' in token_id or '.' in token_id:
                parts[10] = '*'
                fout.write('\t'.join(parts) + '\n')
                continue
            
            # Regular token: add prediction
            token_idx += 1
            
            # sentence_idx points to last completed sentence, so current sentence is at sentence_idx + 1
            current_sent_idx = sentence_idx + 1
            if current_sent_idx < len(mwe_columns_by_sentence) and token_idx < len(mwe_columns_by_sentence[current_sent_idx]):
                mwe_annotation = mwe_columns_by_sentence[current_sent_idx][token_idx]
                parts[10] = mwe_annotation
            else:
                parts[10] = '*'
            
            fout.write('\t'.join(parts) + '\n')
    
    print("\nPrediction completed!")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict MWE annotations for CUPT file')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Path to input .cupt file')
    parser.add_argument('--output', type=str, required=True, help='Path to output .cupt file')
    parser.add_argument('--no_fix_discontinuous', action='store_true',
                       help='Disable discontinuous MWE post-processing (enabled by default)')
    
    args = parser.parse_args()
    
    predict_cupt_file(args.model, args.input, args.output, fix_discontinuous=not args.no_fix_discontinuous)
