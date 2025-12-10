"""
Ensemble evaluation script for PARSEME 2.0 MWE identification
Evaluates ensemble predictions on validation and test sets
"""
import os
import sys
import json
from typing import Dict, List, Tuple
from collections import defaultdict

from data_loader import CUPTDataLoader


def parse_mwe_column(mwe_value: str) -> Tuple[int, str]:
    """
    Parse MWE column value to extract MWE ID and category
    
    Args:
        mwe_value: MWE column value (e.g., '1:VID', '1', '*')
    
    Returns:
        Tuple of (mwe_id, category) or (None, None) for non-MWE tokens
    """
    if mwe_value == '*':
        return None, None
    
    if ':' in mwe_value:
        parts = mwe_value.split(':')
        mwe_id = int(parts[0])
        category = parts[1]
        return mwe_id, category
    else:
        return int(mwe_value), None


def read_cupt_mwes(file_path: str) -> List[Dict]:
    """
    Read CUPT file and extract MWE information
    
    Returns:
        List of sentences, each containing tokens with MWE annotations
    """
    sentences = []
    current_sentence = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            
            if line.startswith('#'):
                continue
            
            if not line.strip():
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
                continue
            
            parts = line.split('\t')
            if len(parts) < 11:
                continue
            
            token_id = parts[0]
            if '-' in token_id or '.' in token_id:
                continue
            
            form = parts[1]
            mwe_column = parts[10]
            
            current_sentence.append({
                'form': form,
                'mwe': mwe_column
            })
    
    if current_sentence:
        sentences.append(current_sentence)
    
    return sentences


def extract_mwes_from_sentence(sentence: List[Dict]) -> Dict[int, Dict]:
    """
    Extract MWEs from a sentence
    
    Returns:
        Dictionary mapping MWE ID to MWE info (tokens, category)
    """
    mwes = defaultdict(lambda: {'tokens': [], 'indices': [], 'category': None})
    
    for idx, token in enumerate(sentence):
        mwe_id, category = parse_mwe_column(token['mwe'])
        
        if mwe_id is not None:
            mwes[mwe_id]['tokens'].append(token['form'])
            mwes[mwe_id]['indices'].append(idx)
            
            if category is not None:
                mwes[mwe_id]['category'] = category
    
    return dict(mwes)


def is_discontinuous_mwe(mwe_info: Dict) -> bool:
    """Check if MWE is discontinuous (has gaps)"""
    indices = mwe_info['indices']
    if len(indices) <= 1:
        return False
    
    return any(indices[i+1] - indices[i] > 1 for i in range(len(indices) - 1))


def compute_metrics(pred_file: str, gold_file: str) -> Dict:
    """
    Compute MWE identification metrics
    
    Returns:
        Dictionary with precision, recall, F1 for overall and discontinuous MWEs
    """
    pred_sentences = read_cupt_mwes(pred_file)
    gold_sentences = read_cupt_mwes(gold_file)
    
    if len(pred_sentences) != len(gold_sentences):
        print(f"WARNING: Number of sentences mismatch: pred={len(pred_sentences)}, gold={len(gold_sentences)}")
    
    # Overall metrics
    tp = 0  # True positives (exact MWE match)
    fp = 0  # False positives (predicted but not in gold)
    fn = 0  # False negatives (in gold but not predicted)
    
    # Discontinuous MWE metrics
    disc_tp = 0
    disc_fp = 0
    disc_fn = 0
    
    # Category metrics
    category_correct = 0
    category_total = 0
    category_breakdown = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    
    for pred_sent, gold_sent in zip(pred_sentences, gold_sentences):
        pred_mwes = extract_mwes_from_sentence(pred_sent)
        gold_mwes = extract_mwes_from_sentence(gold_sent)
        
        # Create MWE signatures (sorted token indices)
        pred_signatures = {tuple(mwe['indices']): mwe for mwe in pred_mwes.values()}
        gold_signatures = {tuple(mwe['indices']): mwe for mwe in gold_mwes.values()}
        
        # Count matches
        for sig in pred_signatures:
            pred_mwe = pred_signatures[sig]
            is_disc = is_discontinuous_mwe(pred_mwe)
            
            if sig in gold_signatures:
                # True positive
                tp += 1
                if is_disc:
                    disc_tp += 1
                
                # Check category accuracy
                gold_mwe = gold_signatures[sig]
                category_total += 1
                if pred_mwe['category'] == gold_mwe['category']:
                    category_correct += 1
                
                # Category breakdown
                if gold_mwe['category']:
                    category_breakdown[gold_mwe['category']]['tp'] += 1
            else:
                # False positive
                fp += 1
                if is_disc:
                    disc_fp += 1
                
                # Category breakdown
                if pred_mwe['category']:
                    category_breakdown[pred_mwe['category']]['fp'] += 1
        
        # Count false negatives
        for sig in gold_signatures:
            if sig not in pred_signatures:
                gold_mwe = gold_signatures[sig]
                fn += 1
                
                if is_discontinuous_mwe(gold_mwe):
                    disc_fn += 1
                
                # Category breakdown
                if gold_mwe['category']:
                    category_breakdown[gold_mwe['category']]['fn'] += 1
    
    # Compute overall metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Compute discontinuous metrics
    disc_precision = disc_tp / (disc_tp + disc_fp) if (disc_tp + disc_fp) > 0 else 0.0
    disc_recall = disc_tp / (disc_tp + disc_fn) if (disc_tp + disc_fn) > 0 else 0.0
    disc_f1 = 2 * disc_precision * disc_recall / (disc_precision + disc_recall) if (disc_precision + disc_recall) > 0 else 0.0
    
    # Category accuracy
    category_acc = category_correct / category_total if category_total > 0 else 0.0
    
    # Category F1 scores
    category_f1_scores = {}
    for cat, counts in category_breakdown.items():
        cat_tp = counts['tp']
        cat_fp = counts['fp']
        cat_fn = counts['fn']
        
        cat_precision = cat_tp / (cat_tp + cat_fp) if (cat_tp + cat_fp) > 0 else 0.0
        cat_recall = cat_tp / (cat_tp + cat_fn) if (cat_tp + cat_fn) > 0 else 0.0
        cat_f1 = 2 * cat_precision * cat_recall / (cat_precision + cat_recall) if (cat_precision + cat_recall) > 0 else 0.0
        
        category_f1_scores[cat] = {
            'precision': cat_precision,
            'recall': cat_recall,
            'f1': cat_f1,
            'tp': cat_tp,
            'fp': cat_fp,
            'fn': cat_fn,
            'support': cat_tp + cat_fn
        }
    
    return {
        'overall': {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        },
        'discontinuous': {
            'precision': disc_precision,
            'recall': disc_recall,
            'f1': disc_f1,
            'tp': disc_tp,
            'fp': disc_fp,
            'fn': disc_fn
        },
        'category_accuracy': category_acc,
        'category_f1': category_f1_scores
    }


def evaluate_ensemble(
    ce_model_dir: str,
    focal_model_dir: str,
    languages: List[str],
    output_dir: str = 'ensemble'
):
    """
    Evaluate ensemble on validation and test sets
    
    Args:
        ce_model_dir: Directory containing CE model (e.g., ensemble/ce/multilingual_XXX)
        focal_model_dir: Directory containing Focal model (e.g., ensemble/focal/multilingual_XXX)
        languages: List of language codes to evaluate
        output_dir: Output directory for results
    """
    from ensemble_predict import ensemble_predict_file
    
    print("="*80)
    print("ENSEMBLE EVALUATION")
    print("="*80)
    print(f"CE Model: {ce_model_dir}")
    print(f"Focal Model: {focal_model_dir}")
    print(f"Languages: {', '.join(languages)}")
    print("="*80 + "\n")
    
    ce_model_path = os.path.join(ce_model_dir, 'best_model.pt')
    focal_model_path = os.path.join(focal_model_dir, 'best_model.pt')
    
    if not os.path.exists(ce_model_path):
        print(f"ERROR: CE model not found: {ce_model_path}")
        return
    
    if not os.path.exists(focal_model_path):
        print(f"ERROR: Focal model not found: {focal_model_path}")
        return
    
    all_results = {}
    
    for language in languages:
        print(f"\n{'='*80}")
        print(f"EVALUATING: {language}")
        print(f"{'='*80}")
        
        dev_file = f"2.0/subtask1/{language}/dev.cupt"
        
        if not os.path.exists(dev_file):
            print(f"WARNING: Dev file not found: {dev_file}")
            continue
        
        # Generate ensemble predictions
        pred_file = f"{output_dir}/predictions/{language}_ensemble.cupt"
        
        print(f"\nGenerating ensemble predictions...")
        ensemble_predict_file(ce_model_path, focal_model_path, dev_file, pred_file)
        
        # Evaluate predictions
        print(f"\nEvaluating predictions...")
        metrics = compute_metrics(pred_file, dev_file)
        
        # Print results
        print(f"\n{'='*80}")
        print(f"RESULTS: {language}")
        print(f"{'='*80}")
        print(f"\nOverall Performance:")
        print(f"  Precision: {metrics['overall']['precision']:.4f}")
        print(f"  Recall:    {metrics['overall']['recall']:.4f}")
        print(f"  F1 Score:  {metrics['overall']['f1']:.4f}")
        print(f"  TP: {metrics['overall']['tp']}, FP: {metrics['overall']['fp']}, FN: {metrics['overall']['fn']}")
        
        print(f"\nDiscontinuous MWEs:")
        print(f"  Precision: {metrics['discontinuous']['precision']:.4f}")
        print(f"  Recall:    {metrics['discontinuous']['recall']:.4f}")
        print(f"  F1 Score:  {metrics['discontinuous']['f1']:.4f}")
        print(f"  TP: {metrics['discontinuous']['tp']}, FP: {metrics['discontinuous']['fp']}, FN: {metrics['discontinuous']['fn']}")
        
        print(f"\nCategory Accuracy: {metrics['category_accuracy']:.4f}")
        
        # Top 10 categories by support
        sorted_cats = sorted(metrics['category_f1'].items(), key=lambda x: x[1]['support'], reverse=True)[:10]
        if sorted_cats:
            print(f"\nTop 10 Categories by Support:")
            for cat, scores in sorted_cats:
                print(f"  {cat:15s} - F1: {scores['f1']:.4f}, P: {scores['precision']:.4f}, R: {scores['recall']:.4f}, Support: {scores['support']}")
        
        all_results[language] = metrics
    
    # Save results to JSON
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, 'evaluation_results.json')
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {results_file}")
    print(f"{'='*80}\n")
    
    # Compute and print average metrics
    if all_results:
        print(f"{'='*80}")
        print("AVERAGE METRICS ACROSS ALL LANGUAGES")
        print(f"{'='*80}")
        
        avg_precision = sum(r['overall']['precision'] for r in all_results.values()) / len(all_results)
        avg_recall = sum(r['overall']['recall'] for r in all_results.values()) / len(all_results)
        avg_f1 = sum(r['overall']['f1'] for r in all_results.values()) / len(all_results)
        
        avg_disc_f1 = sum(r['discontinuous']['f1'] for r in all_results.values()) / len(all_results)
        avg_cat_acc = sum(r['category_accuracy'] for r in all_results.values()) / len(all_results)
        
        print(f"\nOverall:")
        print(f"  Precision: {avg_precision:.4f}")
        print(f"  Recall:    {avg_recall:.4f}")
        print(f"  F1 Score:  {avg_f1:.4f}")
        
        print(f"\nDiscontinuous F1: {avg_disc_f1:.4f}")
        print(f"Category Accuracy: {avg_cat_acc:.4f}")
        print(f"{'='*80}\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate ensemble predictions')
    parser.add_argument('--ce_model_dir', type=str, required=True, 
                       help='Directory containing CE model (e.g., ensemble/ce/multilingual_XXX)')
    parser.add_argument('--focal_model_dir', type=str, required=True,
                       help='Directory containing Focal model (e.g., ensemble/focal/multilingual_XXX)')
    parser.add_argument('--languages', type=str, nargs='+', required=True,
                       help='Language codes to evaluate (e.g., FR PL EL)')
    parser.add_argument('--output', type=str, default='ensemble',
                       help='Output directory for results (default: ensemble)')
    
    args = parser.parse_args()
    
    evaluate_ensemble(
        args.ce_model_dir,
        args.focal_model_dir,
        args.languages,
        args.output
    )
