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
        
    def read_cupt_file(self, filepath: str) -> List[Dict]:
        """
        Read a .cupt file and return list of sentences with tokens and MWE annotations
        
        Returns:
            List of sentences, each sentence is a dict with:
            - 'tokens': list of word forms
            - 'lemmas': list of lemmas
            - 'pos_tags': list of POS tags
            - 'mwe_tags': list of BIO tags for MWEs
            - 'mwe_categories': list of MWE category labels
        """
        sentences = []
        current_sentence = {
            'tokens': [],
            'lemmas': [],
            'pos_tags': [],
            'mwe_raw': [],  # Raw MWE column
            'text': ''
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
                            'text': ''
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
        """
        num_tokens = len(sentence['tokens'])
        mwe_tags = ['O'] * num_tokens
        mwe_categories = ['O'] * num_tokens
        
        # Track MWE spans: mwe_id -> list of (token_idx, category)
        mwe_spans = defaultdict(list)
        
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
                if ':' in mwe_part:
                    mwe_id, category = mwe_part.split(':', 1)
                else:
                    mwe_id = mwe_part
                    category = 'MWE'  # Default category
                
                self.mwe_categories.add(category)
                mwe_spans[mwe_id].append((idx, category))
        
        # Convert to BIO tags
        for mwe_id, spans in mwe_spans.items():
            if not spans:
                continue
            
            # Sort by token index
            spans.sort(key=lambda x: x[0])
            
            # First token gets B- tag
            first_idx, category = spans[0]
            mwe_tags[first_idx] = f'B-MWE'
            mwe_categories[first_idx] = category
            
            # Rest get I- tag
            for idx, cat in spans[1:]:
                mwe_tags[idx] = f'I-MWE'
                mwe_categories[idx] = cat
        
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
