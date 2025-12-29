"""
CUPT data loader for PARSEME 2.0 Subtask 1
Reads .cupt files and converts MWE annotations to BIO tagging scheme
"""
import re
from typing import List, Dict, Tuple
from collections import defaultdict


class CUPTDataLoader:
    """Load and parse CUPT format files for MWE identification"""
    
    def __init__(self):
        self.mwe_categories = set()
        
    def read_cupt_file(self, filepath: str, language: str = None) -> List[Dict]:
        """
        Read a .cupt file and return list of sentences with tokens and MWE annotations
        
        Args:
            filepath: Path to .cupt file
            language: Language code (e.g., 'FR', 'PL') for language-conditioned inputs
        
        Returns:
            List of sentences, each sentence is a dict with:
            - 'tokens': list of word forms
            - 'lemmas': list of lemmas
            - 'pos_tags': list of POS tags
            - 'mwe_tags': list of BIO tags for MWEs
            - 'mwe_categories': list of MWE category labels
            - 'language': language code (if provided)
        """
        # Extract language from filepath if not provided
        if language is None:
            # Try to extract from path like "2.0/subtask1/FR/train.cupt"
            import os
            parts = filepath.split(os.sep)
            for i, part in enumerate(parts):
                if part in ['EGY', 'EL', 'FA', 'FR', 'GRC', 'HE', 'JA', 'KA', 'LV', 'NL', 'PL', 'PT', 'RO', 'SL', 'SR', 'SV', 'UK']:
                    language = part
                    break
        
        sentences = []
        current_sentence = {
            'tokens': [],
            'lemmas': [],
            'pos_tags': [],
            'mwe_raw': [],  # Raw MWE column
            'text': '',
            'language': language
        }
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n')
                
                # Skip comments but extract text
                if line.startswith('#'):
                    if 'text =' in line:
                        current_sentence['text'] = line.split('text =', 1)[1].strip()
                    continue
                
                # Empty line = end of sentence
                if not line.strip():
                    if current_sentence['tokens']:
                        # Convert MWE annotations to BIO tags
                        self._process_mwe_annotations(current_sentence)
                        sentences.append(current_sentence)
                        current_sentence = {
                            'tokens': [],
                            'lemmas': [],
                            'pos_tags': [],
                            'mwe_raw': [],
                            'text': '',
                            'language': language
                        }
                    continue
                
                # Parse token line
                parts = line.split('\t')
                if len(parts) < 11:
                    continue
                
                token_id = parts[0]
                
                # Skip multi-word tokens (e.g., "9-10")
                if '-' in token_id or '.' in token_id:
                    continue
                
                form = parts[1]
                lemma = parts[2]
                upos = parts[3]
                mwe_col = parts[10] if len(parts) > 10 else '*'
                
                current_sentence['tokens'].append(form)
                current_sentence['lemmas'].append(lemma)
                current_sentence['pos_tags'].append(upos)
                current_sentence['mwe_raw'].append(mwe_col)
        
        # Don't forget last sentence
        if current_sentence['tokens']:
            self._process_mwe_annotations(current_sentence)
            sentences.append(current_sentence)
        
        return sentences
    
    def _process_mwe_annotations(self, sentence: Dict):
        """
        Convert raw MWE column annotations to BIO tags
        
        MWE column format examples:
        - '*' = no MWE
        - '1' = part of MWE #1
        - '1:VID' = part of MWE #1 with category VID
        - '1;2:NID' = part of both MWE #1 and MWE #2
        
        For overlapping MWEs (e.g., 1;2), we select the PRIMARY MWE at each token:
        - Prefer the one with explicit category at THIS token (e.g., '2:NID' over '1')
        - This allows us to pick the "most specific" MWE at overlapping points
        """
        num_tokens = len(sentence['tokens'])
        mwe_tags = ['O'] * num_tokens
        mwe_categories = ['O'] * num_tokens
        
        # First pass: collect all MWE spans and their categories
        # mwe_id -> {category: str, tokens: list of token_idx}
        mwe_info = defaultdict(lambda: {'category': None, 'tokens': []})
        
        # Track which MWEs are present at each token, with priority info
        # token_idx -> list of (mwe_id, category, has_explicit_cat)
        token_mwes = defaultdict(list)
        
        for idx, mwe_annotation in enumerate(sentence['mwe_raw']):
            if mwe_annotation == '*':
                continue
            
            # Parse multiple MWE annotations (separated by ;)
            mwe_parts = mwe_annotation.split(';')
            
            for mwe_part in mwe_parts:
                mwe_part = mwe_part.strip()
                if not mwe_part or mwe_part == '*':
                    continue
                
                # Parse MWE ID and category
                has_explicit_category = ':' in mwe_part
                if has_explicit_category:
                    mwe_id, category = mwe_part.split(':', 1)
                    self.mwe_categories.add(category)
                    # Update MWE category if we found an explicit one
                    if mwe_info[mwe_id]['category'] is None:
                        mwe_info[mwe_id]['category'] = category
                else:
                    mwe_id = mwe_part
                    category = None
                
                mwe_info[mwe_id]['tokens'].append(idx)
                token_mwes[idx].append((mwe_id, category, has_explicit_category))
        
        # Second pass: for each token, decide which MWE it belongs to (for overlaps)
        # token_idx -> (mwe_id, category)
        token_assignment = {}
        
        for idx in range(num_tokens):
            if idx not in token_mwes or not token_mwes[idx]:
                continue
            
            candidates = token_mwes[idx]
            
            # For overlapping MWEs at this token, prefer:
            # 1. One with explicit category at THIS token
            # 2. If tie, prefer numerically first MWE ID
            candidates.sort(key=lambda x: (not x[2], x[0]))
            selected_mwe_id, selected_category, _ = candidates[0]
            
            # If no category at this token, use MWE's overall category
            if not selected_category:
                selected_category = mwe_info[selected_mwe_id]['category']
            
            # Still no category? Use default
            if not selected_category:
                selected_category = 'VID'
                self.mwe_categories.add(selected_category)
            
            token_assignment[idx] = (selected_mwe_id, selected_category)
        
        # Third pass: convert to BIO tags based on token assignments
        # Track which MWEs we've seen to determine B- vs I- tags
        # For discontinuous MWEs: first occurrence = B-MWE, later occurrences = I-MWE (even with gaps)
        seen_mwes = set()
        
        for idx in range(num_tokens):
            if idx not in token_assignment:
                mwe_tags[idx] = 'O'
                mwe_categories[idx] = 'O'
                continue
            
            mwe_id, category = token_assignment[idx]
            
            # B-MWE only for the FIRST token of each MWE (by ID)
            # All subsequent tokens (even after gaps) get I-MWE
            if mwe_id not in seen_mwes:
                mwe_tags[idx] = 'B-MWE'
                seen_mwes.add(mwe_id)
            else:
                mwe_tags[idx] = 'I-MWE'
            
            mwe_categories[idx] = category
        
        sentence['mwe_tags'] = mwe_tags
        sentence['mwe_categories'] = mwe_categories
        
        # Remove raw column
        del sentence['mwe_raw']
    
    def get_label_mapping(self):
        """Get mapping of BIO tags to indices"""
        labels = ['O', 'B-MWE', 'I-MWE']
        return {label: idx for idx, label in enumerate(labels)}
    
    def get_category_mapping(self):
        """Get mapping of MWE categories to indices"""
        categories = ['O'] + sorted(list(self.mwe_categories))
        return {cat: idx for idx, cat in enumerate(categories)}
    
    def get_pos_mapping(self, sentences: List[Dict] = None):
        """Get mapping of POS tags to indices (Universal POS tags + padding)"""
        # Universal POS tags from UD
        universal_pos = [
            '<PAD>',  # 0 for padding/unknown
            'ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN',
            'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 
            'VERB', 'X', '_'  # _ for missing
        ]
        pos_mapping = {pos: idx for idx, pos in enumerate(universal_pos)}
        
        # Add any additional POS tags found in data
        if sentences:
            for sent in sentences:
                for pos in sent.get('pos_tags', []):
                    if pos not in pos_mapping:
                        pos_mapping[pos] = len(pos_mapping)
        
        return pos_mapping


def create_bio_tags_from_cupt(cupt_file: str) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Convenience function to load CUPT file and extract tokens and BIO tags
    
    Returns:
        tokens_list: List of token lists (one per sentence)
        tags_list: List of BIO tag lists (one per sentence)
    """
    loader = CUPTDataLoader()
    sentences = loader.read_cupt_file(cupt_file)
    
    tokens_list = [sent['tokens'] for sent in sentences]
    tags_list = [sent['mwe_tags'] for sent in sentences]
    
    return tokens_list, tags_list


if __name__ == '__main__':
    # Test the loader
    import sys
    
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = r'c:\ITU\phd\NLP\project\sharedtask-data-master\2.0\subtask1\FR\train.cupt'
    
    loader = CUPTDataLoader()
    sentences = loader.read_cupt_file(filepath)
    
    print(f"Loaded {len(sentences)} sentences")
    print(f"MWE categories found: {sorted(loader.mwe_categories)}")
    
    # Show first sentence
    if sentences:
        sent = sentences[0]
        print(f"\nFirst sentence ({len(sent['tokens'])} tokens):")
        print(f"Text: {sent['text']}")
        print("\nTokens and tags:")
        for token, tag, cat in zip(sent['tokens'], sent['mwe_tags'], sent['mwe_categories']):
            if tag != 'O':
                print(f"  {token:20s} {tag:10s} {cat}")
