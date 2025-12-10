"""
Quick test script to verify language token logic
"""
import sys
sys.path.insert(0, 'src')

from model import MWETokenizer

# Test 1: Without language tokens
print("=" * 60)
print("Test 1: Without language tokens")
print("=" * 60)
tokenizer = MWETokenizer('bert-base-multilingual-cased', use_lang_tokens=False)
tokens = ['Les', 'droits', 'de', "l'", 'homme']
labels = ['O', 'B-MWE', 'I-MWE', 'I-MWE', 'I-MWE']
categories = ['O', 'VID', 'VID', 'VID', 'VID']
label_to_id = {'O': 0, 'B-MWE': 1, 'I-MWE': 2}
category_to_id = {'O': 0, 'VID': 1}

result = tokenizer.tokenize_and_align_labels(
    tokens, labels, label_to_id, categories, category_to_id
)
print(f"Input tokens: {tokens}")
print(f"Input IDs shape: {result['input_ids'].shape}")
print(f"Input IDs: {result['input_ids'][0][:10]}")
print(f"Labels: {result['labels'][0][:10]}")
print(f"Category labels: {result['category_labels'][0][:10]}")
print()

# Test 2: With language tokens
print("=" * 60)
print("Test 2: With language tokens (FR)")
print("=" * 60)
tokenizer2 = MWETokenizer('bert-base-multilingual-cased', use_lang_tokens=True)
result2 = tokenizer2.tokenize_and_align_labels(
    tokens, labels, label_to_id, categories, category_to_id, 
    language='FR'
)
print(f"Input tokens: {tokens} (with [FR] prepended)")
print(f"Input IDs shape: {result2['input_ids'].shape}")
print(f"Input IDs: {result2['input_ids'][0][:10]}")
print(f"Labels: {result2['labels'][0][:10]}")
print(f"Category labels: {result2['category_labels'][0][:10]}")
print()

# Test 3: Verify language token ID
print("=" * 60)
print("Test 3: Verify language token is in vocabulary")
print("=" * 60)
fr_token_id = tokenizer2.get_tokenizer().convert_tokens_to_ids('[FR]')
print(f"[FR] token ID: {fr_token_id}")
print(f"Vocab size: {len(tokenizer2.get_tokenizer())}")
print()

print("✓ All tests completed!")
