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


def bio_tags_to_mwe_column(tokens: List[str], bio_tags: List[str], mwe_categories: List[str] = None) -> List[str]:
    """
    Convert BIO tags back to CUPT MWE column format
    
    Args:
        tokens: List of words
        bio_tags: List of BIO tags
        mwe_categories: List of MWE categories (optional)
    
    Returns:
        List of MWE column values (e.g., '*', '1', '1:VID', etc.)
    """
    mwe_column = ['*'] * len(tokens)
    mwe_id_counter = 1
    current_mwe_id = None
    
    for idx, tag in enumerate(bio_tags):
        if tag == 'B-MWE':
            # Start new MWE
            current_mwe_id = mwe_id_counter
            if mwe_categories and mwe_categories[idx] != 'O':
                mwe_column[idx] = f"{current_mwe_id}:{mwe_categories[idx]}"
            else:
                mwe_column[idx] = str(current_mwe_id)
            mwe_id_counter += 1
        elif tag == 'I-MWE' and current_mwe_id is not None:
            # Continue current MWE
            if mwe_categories and mwe_categories[idx] != 'O':
                mwe_column[idx] = f"{current_mwe_id}:{mwe_categories[idx]}"
            else:
                mwe_column[idx] = str(current_mwe_id)
        else:
            # Outside MWE or error
            current_mwe_id = None
    
    return mwe_column


def predict_cupt_file(
    model_path: str,
    input_file: str,
    output_file: str,
    device: torch.device = None
):
    """
    Predict MWE annotations for a CUPT file
    
    Args:
        model_path: Path to saved model checkpoint
        input_file: Path to input .cupt file (with empty MWE column)
        output_file: Path to output .cupt file (with predicted MWE annotations)
        device: torch device
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Load model checkpoint
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    model_name = checkpoint['model_name']
    label_to_id = checkpoint['label_to_id']
    id_to_label = {v: k for k, v in label_to_id.items()}
    
    # Initialize model
    model = MWEIdentificationModel(model_name, num_labels=len(label_to_id))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Initialize tokenizer
    tokenizer_path = os.path.join(os.path.dirname(model_path), 'tokenizer')
    if os.path.exists(tokenizer_path):
        tokenizer = MWETokenizer(tokenizer_path)
    else:
        tokenizer = MWETokenizer(model_name)
    
    print(f"Model loaded. Best F1: {checkpoint.get('best_f1', 'N/A')}")
    
    # Read input file
    print(f"\nReading input file: {input_file}")
    
    predictions_by_sentence = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        current_tokens = []
        
        for line in f:
            line = line.rstrip('\n')
            
            # Skip comments
            if line.startswith('#'):
                continue
            
            # Empty line = end of sentence
            if not line.strip():
                if current_tokens:
                    # Predict for this sentence
                    pred_tags = predict_mwe_tags(model, tokenizer, current_tokens, id_to_label, device)
                    predictions_by_sentence.append(pred_tags)
                    current_tokens = []
                else:
                    predictions_by_sentence.append([])
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
            current_tokens.append(form)
        
        # Don't forget last sentence
        if current_tokens:
            pred_tags = predict_mwe_tags(model, tokenizer, current_tokens, id_to_label, device)
            predictions_by_sentence.append(pred_tags)
    
    print(f"Predicted MWEs for {len(predictions_by_sentence)} sentences")
    
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
            
            # Multi-word tokens: keep original
            if '-' in token_id or '.' in token_id:
                fout.write(line + '\n')
                continue
            
            # Regular token: add prediction
            token_idx += 1
            
            if sentence_idx + 1 < len(predictions_by_sentence) and token_idx < len(predictions_by_sentence[sentence_idx + 1]):
                bio_tag = predictions_by_sentence[sentence_idx + 1][token_idx]
                
                # Convert BIO tag to MWE column format
                # For now, we just use simple numbering (could be improved)
                if bio_tag == 'O':
                    parts[10] = '*'
                else:
                    # This is simplified - a proper implementation would track MWE IDs
                    # We'll do a second pass for this
                    parts[10] = bio_tag
            else:
                parts[10] = '*'
            
            fout.write('\t'.join(parts) + '\n')
    
    # Second pass: convert BIO tags to proper MWE numbering
    print("Converting BIO tags to MWE format...")
    _convert_bio_to_mwe_format(output_file, predictions_by_sentence)
    
    print("\nPrediction completed!")


def _convert_bio_to_mwe_format(cupt_file: str, predictions: List[List[str]]):
    """
    Convert BIO tags in CUPT file to proper MWE column format
    This function does a second pass to properly number MWEs
    """
    # Read file with BIO tags
    with open(cupt_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Convert predictions to MWE column format
    sentence_idx = -1
    token_idx = -1
    output_lines = []
    
    for line in lines:
        line = line.rstrip('\n')
        
        # Copy comments and empty lines
        if line.startswith('#') or not line.strip():
            output_lines.append(line + '\n')
            if not line.strip():
                sentence_idx += 1
                token_idx = -1
            continue
        
        # Parse token line
        parts = line.split('\t')
        
        if len(parts) < 11:
            output_lines.append(line + '\n')
            continue
        
        token_id = parts[0]
        
        # Multi-word tokens: keep original
        if '-' in token_id or '.' in token_id:
            output_lines.append(line + '\n')
            continue
        
        # Regular token
        token_idx += 1
        
        # Get BIO tag and convert
        if sentence_idx < len(predictions) and token_idx < len(predictions[sentence_idx]):
            bio_tag = predictions[sentence_idx][token_idx]
            
            # Keep the tag for now (will be converted in next step)
            parts[10] = bio_tag
        else:
            parts[10] = '*'
        
        output_lines.append('\t'.join(parts) + '\n')
    
    # Now convert BIO tags to MWE IDs
    final_lines = []
    sentence_idx = -1
    
    for line in output_lines:
        line = line.rstrip('\n')
        
        if line.startswith('#') or not line.strip():
            final_lines.append(line + '\n')
            if not line.strip():
                sentence_idx += 1
            continue
        
        parts = line.split('\t')
        
        if len(parts) < 11:
            final_lines.append(line + '\n')
            continue
        
        token_id = parts[0]
        
        if '-' in token_id or '.' in token_id:
            final_lines.append(line + '\n')
            continue
        
        # Convert BIO to MWE format
        bio_tag = parts[10]
        if bio_tag in ['O', '*']:
            parts[10] = '*'
        # Will be properly converted below
        
        final_lines.append('\t'.join(parts) + '\n')
    
    # Third pass: actually convert BIO sequences to MWE IDs
    final_output = []
    sentence_bio_tags = []
    sentence_lines = []
    
    for line in final_lines:
        line = line.rstrip('\n')
        
        if not line.strip():
            # Process accumulated sentence
            if sentence_bio_tags:
                mwe_column = bio_tags_to_mwe_column([''] * len(sentence_bio_tags), sentence_bio_tags)
                
                for i, sent_line in enumerate(sentence_lines):
                    if sent_line.startswith('#') or not sent_line.strip():
                        final_output.append(sent_line + '\n')
                        continue
                    
                    parts = sent_line.split('\t')
                    if len(parts) >= 11 and '-' not in parts[0] and '.' not in parts[0]:
                        if i < len(mwe_column):
                            parts[10] = mwe_column[i]
                    
                    final_output.append('\t'.join(parts) + '\n')
                
                sentence_bio_tags = []
                sentence_lines = []
            
            final_output.append('\n')
            continue
        
        if line.startswith('#'):
            final_output.append(line + '\n')
            continue
        
        parts = line.split('\t')
        
        if '-' in parts[0] or '.' in parts[0]:
            sentence_lines.append(line)
            continue
        
        if len(parts) >= 11:
            sentence_bio_tags.append(parts[10])
            sentence_lines.append(line)
    
    # Write final output
    with open(cupt_file, 'w', encoding='utf-8') as f:
        f.writelines(final_output)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict MWE annotations for CUPT file')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Path to input .cupt file')
    parser.add_argument('--output', type=str, required=True, help='Path to output .cupt file')
    
    args = parser.parse_args()
    
    predict_cupt_file(args.model, args.input, args.output)
