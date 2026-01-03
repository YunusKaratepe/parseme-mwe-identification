"""
Ensemble prediction script for PARSEME 2.0 MWE identification
Combines predictions from Cross-Entropy and Focal Loss models
Uses probability averaging to leverage strengths of both models
"""
import os
import sys
import torch
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict

from data_loader import CUPTDataLoader
from model import MWEIdentificationModel, MWETokenizer


def load_ensemble_model(model_path: str, device: torch.device) -> Tuple:
    """
    Load a single model for ensemble
    
    Returns:
        Tuple of (model, tokenizer, label_to_id, category_to_id, pos_to_id, use_pos, use_lang_tokens, model_name)
    """
    print(f"Loading model: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    model_name = checkpoint.get('model_name', 'bert-base-multilingual-cased')
    label_to_id = checkpoint['label_to_id']
    category_to_id = checkpoint['category_to_id']
    pos_to_id = checkpoint.get('pos_to_id', None)
    use_pos = checkpoint.get('use_pos', False)
    use_lang_tokens = checkpoint.get('use_lang_tokens', False)
    use_crf = checkpoint.get('use_crf', False)
    loss_type = checkpoint.get('loss_type', 'ce')

    # If the checkpoint was trained with CRF, we must instantiate a CRF-enabled model.
    # If CRF dependencies are missing, loading would silently drop CRF parameters and
    # produce inconsistent decoding, so we fail fast.
    try:
        from model import CRF_AVAILABLE
    except Exception:
        CRF_AVAILABLE = False
    if use_crf and not CRF_AVAILABLE:
        raise RuntimeError(
            "This checkpoint was trained with CRF (use_crf=True), but pytorch-crf is not available "
            "in the current environment. Install it with: pip install pytorch-crf"
        )
    
    # Handle invalid model paths
    if not os.path.exists(model_name) and '/' in model_name and not model_name.startswith('bert-'):
        print(f"  WARNING: Model path '{model_name}' not found, using 'bert-base-multilingual-cased'")
        model_name = 'bert-base-multilingual-cased'
    
    # Initialize tokenizer first (needed to get correct vocab size before loading state)
    tokenizer_path = os.path.join(os.path.dirname(model_path), 'tokenizer')
    if use_lang_tokens:
        if os.path.exists(tokenizer_path):
            tokenizer = MWETokenizer(tokenizer_path, use_lang_tokens=True)
        else:
            tokenizer = MWETokenizer(model_name, use_lang_tokens=True)
    else:
        if os.path.exists(tokenizer_path):
            tokenizer = MWETokenizer(tokenizer_path)
        else:
            tokenizer = MWETokenizer(model_name)

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

    # Resize token embeddings BEFORE loading state dict if needed.
    # This is critical for language-token checkpoints where vocab size differs from the base model.
    state_dict = checkpoint['model_state_dict']
    expected_vocab_size = None
    emb_key = 'transformer.embeddings.word_embeddings.weight'
    if emb_key in state_dict and hasattr(state_dict[emb_key], 'shape'):
        expected_vocab_size = int(state_dict[emb_key].shape[0])

    tokenizer_vocab_size = None
    try:
        tokenizer_vocab_size = len(tokenizer.get_tokenizer())
    except Exception:
        tokenizer_vocab_size = None

    if expected_vocab_size is not None:
        # If tokenizer folder is missing but vocab is non-standard, fail fast to avoid silent mismatches.
        if (not os.path.exists(tokenizer_path)) and (tokenizer_vocab_size is not None) and (expected_vocab_size != tokenizer_vocab_size):
            raise RuntimeError(
                "Checkpoint vocab size does not match the loaded tokenizer. "
                f"expected_vocab_size={expected_vocab_size}, tokenizer_vocab_size={tokenizer_vocab_size}. "
                "Make sure the checkpoint directory contains the matching 'tokenizer/' folder."
            )
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'resize_token_embeddings'):
            model.transformer.resize_token_embeddings(expected_vocab_size)

    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    
    print(f"  Loss type: {loss_type.upper()}, Best F1: {checkpoint.get('best_f1', 'N/A'):.4f}")
    print(f"  POS: {'ON' if use_pos else 'OFF'}, Lang tokens: {'ON' if use_lang_tokens else 'OFF'}")
    
    return model, tokenizer, label_to_id, category_to_id, pos_to_id, use_pos, use_lang_tokens, model_name


def predict_with_probabilities(
    model: MWEIdentificationModel,
    tokenizer: MWETokenizer,
    tokens: List[str],
    label_to_id: Dict[str, int],
    category_to_id: Dict[str, int],
    device: torch.device,
    pos_tags: List[str] = None,
    pos_to_id: Dict[str, int] = None,
    use_pos: bool = False,
    language: str = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict MWE tags and return probability distributions
    
    Returns:
        Tuple of (bio_probs, category_probs) as numpy arrays of shape [seq_len, num_classes]
    """
    model.eval()
    
    # Create dummy labels for tokenization
    dummy_labels = ['O'] * len(tokens)
    dummy_categories = ['VID'] * len(tokens)
    
    # Tokenize
    tokenized = tokenizer.tokenize_and_align_labels(
        tokens, dummy_labels, label_to_id,
        dummy_categories, category_to_id,
        pos_tags, pos_to_id, 512, language
    )
    
    input_ids = tokenized['input_ids'].to(device)
    attention_mask = tokenized['attention_mask'].to(device)
    pos_ids = tokenized['pos_ids'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, pos_ids=pos_ids)

        # BIO probabilities:
        # - If CRF is enabled, mimic src/predict.py behavior by decoding with CRF.
        #   CRF does not provide straightforward per-token marginals here, so we convert
        #   the decoded sequence into one-hot "probabilities" (ensembling becomes voting).
        # - Otherwise, use softmax over emissions.
        if getattr(model, 'use_crf', False) and hasattr(model, 'crf'):
            mask = attention_mask.bool()
            decoded = model.crf.decode(outputs['bio_logits'], mask=mask)
            # decoded is a list (batch) of label-id lists with length == seq_len (for batch_first=True)
            decoded_ids = decoded[0]
            bio_probs = torch.zeros_like(outputs['bio_logits'])
            for t, lab in enumerate(decoded_ids):
                if t < bio_probs.size(1):
                    bio_probs[0, t, lab] = 1.0
        else:
            bio_probs = torch.softmax(outputs['bio_logits'], dim=-1)  # [1, seq_len, num_labels]

        # Category probabilities (always softmax)
        category_probs = torch.softmax(outputs['category_logits'], dim=-1)  # [1, seq_len, num_categories]
    
    # Extract word-level probabilities (skip special tokens and subwords)
    # NOTE: MWETokenizer prepends a language token as an extra "word" when use_lang_tokens=True.
    # For ensembling, we must return probabilities aligned to the *original* `tokens` list.
    word_ids = tokenized.word_ids(batch_index=0)
    word_bio_probs = []
    word_category_probs = []
    previous_word_idx = None

    for idx, word_idx in enumerate(word_ids):
        if word_idx is not None and word_idx != previous_word_idx:
            word_bio_probs.append(bio_probs[0, idx].cpu().numpy())
            word_category_probs.append(category_probs[0, idx].cpu().numpy())
            previous_word_idx = word_idx

    # Drop the prepended language token "word" if present.
    if getattr(tokenizer, 'use_lang_tokens', False) and language:
        # With language tokens, MWETokenizer shifts word indices by +1 and adds one extra word.
        if len(word_bio_probs) == len(tokens) + 1:
            word_bio_probs = word_bio_probs[1:]
            word_category_probs = word_category_probs[1:]
        elif len(word_bio_probs) > len(tokens):
            # Conservative fallback: drop first and then clip.
            word_bio_probs = word_bio_probs[1:len(tokens) + 1]
            word_category_probs = word_category_probs[1:len(tokens) + 1]

    # Final safety: enforce equal length across models.
    if len(word_bio_probs) != len(tokens):
        clip_len = min(len(word_bio_probs), len(tokens))
        word_bio_probs = word_bio_probs[:clip_len]
        word_category_probs = word_category_probs[:clip_len]

    return np.array(word_bio_probs), np.array(word_category_probs)


def ensemble_predict(
    models: List[MWEIdentificationModel],
    tokenizers: List[MWETokenizer],
    tokens: List[str],
    label_to_id: Dict[str, int],
    category_to_id: Dict[str, int],
    id_to_label: Dict[int, str],
    id_to_category: Dict[int, str],
    device: torch.device,
    pos_tags_list: List[List[str]],
    pos_to_id_list: List[Dict[str, int]],
    use_pos_list: List[bool],
    language: str = None
) -> Tuple[List[str], List[str]]:
    """
    Ensemble prediction by averaging probabilities from multiple models
    
    Args:
        models: List of models
        tokenizers: List of tokenizers (one per model)
        tokens: List of words
        label_to_id: BIO label mapping
        category_to_id: Category mapping
        id_to_label: Reverse BIO label mapping
        id_to_category: Reverse category mapping
        device: torch device
        pos_tags_list: List of POS tags for each model
        pos_to_id_list: List of POS mappings for each model
        use_pos_list: List of use_pos flags for each model
        language: Language code
    
    Returns:
        Tuple of (bio_tags, categories)
    """
    all_bio_probs = []
    all_category_probs = []
    
    # Get predictions from each model
    for i, (model, tokenizer) in enumerate(zip(models, tokenizers)):
        bio_probs, cat_probs = predict_with_probabilities(
            model, tokenizer, tokens, label_to_id, category_to_id, device,
            pos_tags_list[i], pos_to_id_list[i], use_pos_list[i], language
        )
        all_bio_probs.append(bio_probs)
        all_category_probs.append(cat_probs)
    
    # Average probabilities across models
    avg_bio_probs = np.mean(all_bio_probs, axis=0)  # [seq_len, num_labels]
    avg_category_probs = np.mean(all_category_probs, axis=0)  # [seq_len, num_categories]
    
    # Get predictions from averaged probabilities
    bio_predictions = np.argmax(avg_bio_probs, axis=-1)
    category_predictions = np.argmax(avg_category_probs, axis=-1)
    
    # Convert to labels
    bio_tags = [id_to_label[pred_id] for pred_id in bio_predictions]
    categories = [id_to_category[pred_id] for pred_id in category_predictions]
    
    return bio_tags, categories


def bio_tags_to_mwe_column(bio_tags: List[str], mwe_categories: List[str] = None) -> List[str]:
    """Convert BIO tags back to CUPT MWE column format.

    Keep this aligned with src/predict.py so ensemble and single-model
    predictions follow the same conversion rules.
    """
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
    last_mwe_id = None  # Track last MWE ID for discontinuous patterns
    
    for idx, tag in enumerate(bio_tags):
        if tag == 'B-MWE':
            current_mwe_id = mwe_id_counter
            last_mwe_id = current_mwe_id  # Remember for potential discontinuous continuation
            
            if mwe_categories and idx < len(mwe_categories) and mwe_categories[idx] != 'O':
                cat = mwe_categories[idx]
                if cat == 'MWE' or cat not in VALID_CATEGORIES:
                    cat = 'VID'
                current_category = cat
            else:
                current_category = 'VID'
            
            mwe_column[idx] = f'{current_mwe_id}:{current_category}'
            mwe_id_counter += 1
            
        elif tag == 'I-MWE':
            if current_mwe_id is not None:
                # Continuous: B-MWE ... I-MWE
                mwe_column[idx] = str(current_mwe_id)
            elif last_mwe_id is not None:
                # Discontinuous: B-MWE ... O ... I-MWE
                mwe_column[idx] = str(last_mwe_id)
                current_mwe_id = last_mwe_id
        else:
            # Outside MWE: reset current but keep last_mwe_id for discontinuous continuation
            current_mwe_id = None
            current_category = None
    
    return mwe_column


def ensemble_predict_file(
    ce_model_path: str,
    focal_model_path: str,
    input_file: str,
    output_file: str,
    fix_discontinuous: bool = False
):
    """
    Generate ensemble predictions for a CUPT file
    
    Args:
        ce_model_path: Path to Cross-Entropy model
        focal_model_path: Path to Focal Loss model
        input_file: Input CUPT file
        output_file: Output CUPT file
        fix_discontinuous: Apply discontinuous MWE post-processing
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load both models
    print("="*80)
    print("LOADING ENSEMBLE MODELS")
    print("="*80)
    
    ce_model, ce_tokenizer, label_to_id, category_to_id, ce_pos_to_id, ce_use_pos, ce_use_lang_tokens, ce_model_name = load_ensemble_model(ce_model_path, device)
    focal_model, focal_tokenizer, _, _, focal_pos_to_id, focal_use_pos, focal_use_lang_tokens, focal_model_name = load_ensemble_model(focal_model_path, device)
    
    id_to_label = {v: k for k, v in label_to_id.items()}
    id_to_category = {v: k for k, v in category_to_id.items()}
    
    # Extract language from input file
    language = None
    if ce_use_lang_tokens or focal_use_lang_tokens:
        parts = input_file.replace('\\', '/').split('/')
        for lang in ['EGY', 'EL', 'FA', 'FR', 'GRC', 'HE', 'JA', 'KA', 'LV', 'NL', 'PL', 'PT', 'RO', 'SL', 'SR', 'SV', 'UK']:
            if lang in parts:
                language = lang
                print(f"\nDetected language: {language}")
                break
    
    # Read input file and generate predictions
    print(f"\n{'='*80}")
    print(f"GENERATING ENSEMBLE PREDICTIONS")
    print(f"{'='*80}")
    print(f"Input: {input_file}")
    
    predictions_by_sentence = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        current_tokens = []
        current_pos_tags = []
        
        for line in f:
            line = line.rstrip('\n')
            
            if line.startswith('#'):
                continue
            
            if not line.strip():
                if current_tokens:
                    # Ensemble prediction
                    pred_tags, pred_cats = ensemble_predict(
                        [ce_model, focal_model],
                        [ce_tokenizer, focal_tokenizer],
                        current_tokens,
                        label_to_id,
                        category_to_id,
                        id_to_label,
                        id_to_category,
                        device,
                        [current_pos_tags, current_pos_tags],
                        [ce_pos_to_id, focal_pos_to_id],
                        [ce_use_pos, focal_use_pos],
                        language
                    )
                    predictions_by_sentence.append((pred_tags, pred_cats))
                    current_tokens = []
                    current_pos_tags = []
                else:
                    predictions_by_sentence.append(([], []))
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
            current_pos_tags.append(pos_tag)
        
        if current_tokens:
            pred_tags, pred_cats = ensemble_predict(
                [ce_model, focal_model],
                [ce_tokenizer, focal_tokenizer],
                current_tokens,
                label_to_id,
                category_to_id,
                id_to_label,
                id_to_category,
                device,
                [current_pos_tags, current_pos_tags],
                [ce_pos_to_id, focal_pos_to_id],
                [ce_use_pos, focal_use_pos],
                language
            )
            predictions_by_sentence.append((pred_tags, pred_cats))
    
    print(f"Predicted MWEs for {len(predictions_by_sentence)} sentences")
    
    # Apply discontinuous MWE fixing (optional)
    if fix_discontinuous:
        print("Applying discontinuous MWE post-processing...")
        from postprocess_discontinuous import fix_discontinuous_mwes
        predictions_by_sentence = fix_discontinuous_mwes(predictions_by_sentence)
        print("✓ Discontinuous MWE patterns fixed")
    
    # Convert to CUPT format
    print("Converting predictions to CUPT format...")
    mwe_columns_by_sentence = []
    
    for pred_tags, pred_cats in predictions_by_sentence:
        if len(pred_tags) == 0:
            mwe_columns_by_sentence.append([])
        else:
            mwe_column = bio_tags_to_mwe_column(pred_tags, pred_cats)
            mwe_columns_by_sentence.append(mwe_column)
    
    # Write output file (mirror src/predict.py behavior)
    print(f"Writing predictions to: {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        sentence_idx = -1
        token_idx = -1
        
        for line in f_in:
            line = line.rstrip('\n')
            
            if line.startswith('#'):
                f_out.write(line + '\n')
                continue
            
            if not line.strip():
                f_out.write('\n')
                sentence_idx += 1
                token_idx = -1
                continue
            
            parts = line.split('\t')
            if len(parts) < 11:
                f_out.write(line + '\n')
                continue
            
            token_id = parts[0]
            
            # Multi-word tokens: always output '*'
            if '-' in token_id or '.' in token_id:
                parts[10] = '*'
                f_out.write('\t'.join(parts) + '\n')
                continue
            
            # Regular token
            token_idx += 1
            current_sent_idx = sentence_idx + 1
            if current_sent_idx < len(mwe_columns_by_sentence) and token_idx < len(mwe_columns_by_sentence[current_sent_idx]):
                parts[10] = mwe_columns_by_sentence[current_sent_idx][token_idx]
            else:
                parts[10] = '*'
            
            f_out.write('\t'.join(parts) + '\n')
    
    print(f"\n✓ Ensemble predictions saved to: {output_file}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Ensemble prediction for MWE identification')
    parser.add_argument('--ce_model', type=str, required=True, help='Path to Cross-Entropy model (best_model.pt)')
    parser.add_argument('--focal_model', type=str, required=True, help='Path to Focal Loss model (best_model.pt)')
    parser.add_argument('--input', type=str, required=True, help='Input CUPT file')
    parser.add_argument('--output', type=str, required=True, help='Output CUPT file')
    parser.add_argument('--fix_discontinuous', action='store_true', help='Enable discontinuous MWE post-processing (NOT recommended)')
    # Backward-compatibility: older scripts used this flag; default is now OFF.
    parser.add_argument('--no_fix_discontinuous', action='store_true', help='(Deprecated) Disable discontinuous MWE post-processing')
    
    args = parser.parse_args()
    
    ensemble_predict_file(
        args.ce_model,
        args.focal_model,
        args.input,
        args.output,
        fix_discontinuous=(args.fix_discontinuous and not args.no_fix_discontinuous)
    )
