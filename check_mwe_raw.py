import sys
sys.path.insert(0, 'src')
from data_loader import CUPTDataLoader

loader = CUPTDataLoader()
sents = loader.read_cupt_file('2.0/subtask1/FR/dev.cupt')

print(f"Total sentences: {len(sents)}")
print(f"\nFirst sentence keys: {list(sents[0].keys())}")
print(f"\nFirst 3 sentences:")
for i, s in enumerate(sents[:3]):
    print(f"\nSentence {i+1}: {s.get('text', 'NO TEXT')[:80]}")
    print(f"  Tokens: {s['tokens'][:10]}")
    print(f"  MWE tags: {s.get('mwe_tags', [])[:10]}")
    print(f"  Categories: {s.get('mwe_categories', [])[:10]}")
