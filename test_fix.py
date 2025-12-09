from src.data_loader import CUPTDataLoader

loader = CUPTDataLoader()
data = loader.read_cupt_file('2.0/subtask1/FR/dev.cupt')
sent = data[0]

print('Sentence:', ' '.join(sent['tokens']))
print()
print('Token | MWE Tag | Category')
print('-' * 50)
for i, (tok, tag, cat) in enumerate(zip(sent['tokens'], sent['mwe_tags'], sent['mwe_categories']), 1):
    print(f'{i:2d}. {tok:15s} {tag:8s} {cat}')

print()
print('Expected (for overlapping MWEs, we pick primary - MWE #2 with NID):')
print('8. reçoit: B-MWE LVC.full (MWE #1)')
print('13. coup: I-MWE NID (continues MWE #1, overlaps with start of MWE #2 - pick #2)')
print('14-15. de fil: I-MWE NID (continue MWE #2)')
