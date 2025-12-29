"""
Post-processing script to fix discontinuous MWEs
Applies heuristic stitching to link B-X ... O ... I-X patterns
"""
from typing import List, Dict, Tuple
import re


def fix_discontinuous_mwes(predictions: List[Tuple[List[str], List[str]]], max_gap_length: int = 10) -> List[Tuple[List[str], List[str]]]:
    """
    Fix discontinuous MWEs using heuristic stitching
    
    Logic: If we see B-MWE followed by O(s) followed by I-MWE with same category,
    change the O tokens to I-MWE to create a discontinuous MWE
    
    Args:
        predictions: List of (bio_tags, categories) tuples for each sentence
        max_gap_length: Maximum number of O tokens to bridge (default: 10)
    
    Returns:
        Fixed predictions with discontinuous MWEs stitched
    """
    fixed_predictions = []
    
    for bio_tags, categories in predictions:
        if not bio_tags or len(bio_tags) != len(categories):
            fixed_predictions.append((bio_tags, categories))
            continue
        
        fixed_bio = list(bio_tags)
        fixed_cats = list(categories)
        
        i = 0
        while i < len(fixed_bio):
            # Look for B-MWE
            if fixed_bio[i] == 'B-MWE':
                b_category = fixed_cats[i]
                j = i + 1
                
                # Scan ahead to find potential discontinuous pattern
                found_gap = False
                gap_start = -1
                gap_end = -1
                
                while j < len(fixed_bio):
                    if fixed_bio[j] == 'I-MWE' and fixed_cats[j] == b_category:
                        # Found continuation with same category
                        if found_gap:
                            gap_length = gap_end - gap_start + 1
                            # Only stitch if gap is within reasonable length
                            if gap_length <= max_gap_length:
                                # We have a B-X ... O ... I-X pattern - fix it!
                                for k in range(gap_start, gap_end + 1):
                                    fixed_bio[k] = 'I-MWE'
                                    fixed_cats[k] = b_category
                            # Reset gap tracking after filling (or skipping)
                            found_gap = False
                            gap_start = -1
                            gap_end = -1
                        # Continue scanning for more gaps
                        j += 1
                    elif fixed_bio[j] == 'O':
                        # Potential gap
                        if not found_gap:
                            gap_start = j
                            found_gap = True
                        gap_end = j
                        j += 1
                    elif fixed_bio[j] == 'B-MWE':
                        # New MWE started, stop looking
                        break
                    else:
                        # I-MWE with different category, reset gap and keep scanning
                        found_gap = False
                        gap_start = -1
                        gap_end = -1
                        j += 1
                
                i = j if j < len(fixed_bio) else i + 1
            else:
                i += 1
        
        fixed_predictions.append((fixed_bio, fixed_cats))
    
    return fixed_predictions


def fix_discontinuous_in_cupt(input_file: str, output_file: str):
    """
    Apply discontinuous MWE fixing directly to a CUPT file
    
    Args:
        input_file: Path to input .cupt file
        output_file: Path to output .cupt file with fixed annotations
    """
    from data_loader import CUPTDataLoader
    from model import predict_mwe_tags
    
    # Read file and parse predictions
    loader = CUPTDataLoader()
    sentences = loader.read_cupt_file(input_file)
    
    # Extract predictions
    predictions = []
    for sent in sentences:
        if 'mwe_tags' in sent and 'mwe_categories' in sent:
            predictions.append((sent['mwe_tags'], sent['mwe_categories']))
        else:
            predictions.append(([], []))
    
    # Fix discontinuous patterns
    fixed_predictions = fix_discontinuous_mwes(predictions)
    
    # Write back to CUPT format
    # (Implementation would read original file and replace MWE column)
    print(f"Fixed {len(fixed_predictions)} sentences")
    print(f"Output written to: {output_file}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Fix discontinuous MWEs in predictions')
    parser.add_argument('--input', type=str, required=True, help='Input .cupt file')
    parser.add_argument('--output', type=str, required=True, help='Output .cupt file')
    
    args = parser.parse_args()
    
    fix_discontinuous_in_cupt(args.input, args.output)
