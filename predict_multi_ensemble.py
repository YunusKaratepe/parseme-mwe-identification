import os
import sys
from typing import List

# --- CONFIGURATION ---
# languages = [
#     "FR", "PL", "EL", "PT", "SL", "SR", "SV", "UK", "NL",
#     "EGY", "KA", "JA", "HE", "LV", "FA", "RO", "GRC"
# ]
languages = ["FR"]

# Paths to your trained models (checkpoints)
# Add as many checkpoints as you want; the script will average probabilities.
model_paths = [
    "models/FR_crf_focal/multilingual_FR/best_model.pt",
    "models/FR_crf/multilingual_FR/best_model.pt",
]

# Input/output convention (match predict_multi.py defaults)
input_filename = "dev.cupt"  # e.g., dev.cupt or test.blind.cupt
output_filename = "dev.system.cupt"  # written under predictions/<LANG>/

# Discontinuous heuristic post-processing
# NOTE: We generally keep this OFF (no stitching) for consistency.
fix_discontinuous = False


def _load_ensemble_module():
    """Import src/ensemble_predict.py as a module (PowerShell/Windows-safe)."""
    repo_root = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(repo_root, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    import ensemble_predict as ep  # src/ensemble_predict.py
    return ep


def ensemble_predict_file(
    model_paths: List[str],
    input_file: str,
    output_file: str,
    fix_discontinuous: bool,
):
    ep = _load_ensemble_module()
    import torch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if len(model_paths) < 2:
        raise ValueError("model_paths must contain at least 2 checkpoints for ensembling")

    # Load models
    models = []
    tokenizers = []
    pos_to_id_list = []
    use_pos_list = []
    use_lang_tokens_any = False

    label_to_id = None
    category_to_id = None

    print("=" * 80)
    print(f"LOADING {len(model_paths)} ENSEMBLE MODELS")
    print("=" * 80)

    for idx, mp in enumerate(model_paths):
        model, tokenizer, l2i, c2i, pos_to_id, use_pos, use_lang_tokens, _model_name = ep.load_ensemble_model(mp, device)
        if idx == 0:
            label_to_id = l2i
            category_to_id = c2i
        else:
            # Basic safety: require identical label/category mappings
            if l2i != label_to_id:
                raise ValueError(f"Label mapping mismatch between model[0] and model[{idx}]: {mp}")
            if c2i != category_to_id:
                raise ValueError(f"Category mapping mismatch between model[0] and model[{idx}]: {mp}")

        models.append(model)
        tokenizers.append(tokenizer)
        pos_to_id_list.append(pos_to_id)
        use_pos_list.append(use_pos)
        use_lang_tokens_any = use_lang_tokens_any or use_lang_tokens

    id_to_label = {v: k for k, v in label_to_id.items()}
    id_to_category = {v: k for k, v in category_to_id.items()}

    # Extract language from input path if any model uses language tokens
    language = None
    if use_lang_tokens_any:
        parts = input_file.replace('\\', '/').split('/')
        for lang in ['EGY', 'EL', 'FA', 'FR', 'GRC', 'HE', 'JA', 'KA', 'LV', 'NL', 'PL', 'PT', 'RO', 'SL', 'SR', 'SV', 'UK']:
            if lang in parts:
                language = lang
                print(f"Detected language: {language}")
                break

    print("=" * 80)
    print("GENERATING ENSEMBLE PREDICTIONS")
    print("=" * 80)
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
                    pred_tags, pred_cats = ep.ensemble_predict(
                        models,
                        tokenizers,
                        current_tokens,
                        label_to_id,
                        category_to_id,
                        id_to_label,
                        id_to_category,
                        device,
                        [current_pos_tags for _ in models],
                        pos_to_id_list,
                        use_pos_list,
                        language,
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
            pred_tags, pred_cats = ep.ensemble_predict(
                models,
                tokenizers,
                current_tokens,
                label_to_id,
                category_to_id,
                id_to_label,
                id_to_category,
                device,
                [current_pos_tags for _ in models],
                pos_to_id_list,
                use_pos_list,
                language,
            )
            predictions_by_sentence.append((pred_tags, pred_cats))

    print(f"Predicted MWEs for {len(predictions_by_sentence)} sentences")

    if fix_discontinuous:
        print("Applying discontinuous MWE post-processing...")
        from postprocess_discontinuous import fix_discontinuous_mwes
        predictions_by_sentence = fix_discontinuous_mwes(predictions_by_sentence)
        print("✓ Discontinuous MWE patterns fixed")

    mwe_columns_by_sentence = []
    for pred_tags, pred_cats in predictions_by_sentence:
        if len(pred_tags) == 0:
            mwe_columns_by_sentence.append([])
        else:
            mwe_columns_by_sentence.append(ep.bio_tags_to_mwe_column(pred_tags, pred_cats))

    print(f"Writing predictions to: {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        sentence_idx = 0
        token_idx_in_sentence = 0

        for line in f_in:
            line = line.rstrip('\n')

            if line.startswith('#'):
                f_out.write(line + '\n')
                continue

            if not line.strip():
                f_out.write('\n')
                sentence_idx += 1
                token_idx_in_sentence = 0
                continue

            parts = line.split('\t')
            if len(parts) < 11:
                f_out.write(line + '\n')
                continue

            token_id = parts[0]
            if '-' in token_id or '.' in token_id:
                f_out.write(line + '\n')
                continue

            if sentence_idx < len(mwe_columns_by_sentence):
                mwe_column = mwe_columns_by_sentence[sentence_idx]
                if token_idx_in_sentence < len(mwe_column):
                    parts[10] = mwe_column[token_idx_in_sentence]
                    token_idx_in_sentence += 1

            f_out.write('\t'.join(parts) + '\n')

    print(f"✓ Ensemble predictions saved to: {output_file}")


def run_predictions():
    for lang in languages:
        input_path = f"2.0/subtask1/{lang}/{input_filename}"
        output_path = f"predictions/{lang}/{output_filename}"

        print(f"Running ensemble prediction for {lang}...")
        ensemble_predict_file(model_paths, input_path, output_path, fix_discontinuous)

    print("\nAll ensemble predictions finished.")


if __name__ == "__main__":
    run_predictions()
