"""
Visualize MWE predictions - show examples of detected MWEs
"""
import sys


def visualize_predictions(cupt_file='predictions/FR/test.cupt', num_examples=10):
    """Display examples of MWE predictions with highlighted MWEs"""
    
    print("=" * 80)
    print("MWE IDENTIFICATION - PREDICTION EXAMPLES")
    print("=" * 80)
    print()
    
    sentences = []
    current_sentence = {'tokens': [], 'mwe_tags': [], 'text': ''}
    
    with open(cupt_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            
            # Extract text from comments
            if line.startswith('#'):
                if 'text =' in line:
                    current_sentence['text'] = line.split('text =', 1)[1].strip()
                continue
            
            # Empty line = end of sentence
            if not line.strip():
                if current_sentence['tokens']:
                    # Check if sentence has MWEs
                    has_mwe = any(tag != '*' for tag in current_sentence['mwe_tags'])
                    if has_mwe:
                        sentences.append(current_sentence)
                    current_sentence = {'tokens': [], 'mwe_tags': [], 'text': ''}
                continue
            
            # Parse token line
            parts = line.split('\t')
            if len(parts) < 11:
                continue
            
            token_id = parts[0]
            if '-' in token_id or '.' in token_id:
                continue
            
            form = parts[1]
            mwe_tag = parts[10] if len(parts) > 10 else '*'
            
            current_sentence['tokens'].append(form)
            current_sentence['mwe_tags'].append(mwe_tag)
    
    # Don't forget last sentence
    if current_sentence['tokens']:
        has_mwe = any(tag != '*' for tag in current_sentence['mwe_tags'])
        if has_mwe:
            sentences.append(current_sentence)
    
    print(f"Found {len(sentences)} sentences with MWEs\n")
    print("=" * 80)
    
    # Display examples
    for idx, sent in enumerate(sentences[:num_examples]):
        print(f"\nExample {idx + 1}:")
        print("-" * 80)
        
        # Show original text
        if sent['text']:
            print(f"Text: {sent['text']}")
            print()
        
        # Parse MWEs
        mwes = {}  # mwe_id -> list of (token, position)
        for i, (token, tag) in enumerate(zip(sent['tokens'], sent['mwe_tags'])):
            if tag != '*':
                # Parse MWE ID (e.g., "1", "1:VID", "2:NID")
                mwe_id = tag.split(':')[0]
                category = tag.split(':')[1] if ':' in tag else 'MWE'
                
                if mwe_id not in mwes:
                    mwes[mwe_id] = {'tokens': [], 'category': category}
                mwes[mwe_id]['tokens'].append(token)
        
        # Display identified MWEs
        print("Identified MWEs:")
        for mwe_id, info in sorted(mwes.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0):
            mwe_text = ' '.join(info['tokens'])
            category = info['category']
            print(f"  [{category}] {mwe_text}")
        
        # Show token-by-token breakdown
        print("\nToken breakdown:")
        for i, (token, tag) in enumerate(zip(sent['tokens'], sent['mwe_tags'])):
            if tag != '*':
                print(f"  {i+1:2d}. {token:20s} → {tag}")
            else:
                print(f"  {i+1:2d}. {token:20s}")
        
        print()
    
    print("=" * 80)
    print(f"\nDisplayed {min(num_examples, len(sentences))} examples")
    print(f"Total sentences with MWEs: {len(sentences)}")
    print()
    
    # Statistics
    print("=" * 80)
    print("STATISTICS")
    print("=" * 80)
    print()
    
    # Count MWE types
    category_counts = {}
    total_mwes = 0
    
    for sent in sentences:
        mwes_in_sent = set()
        for tag in sent['mwe_tags']:
            if tag != '*':
                mwe_id = tag.split(':')[0]
                category = tag.split(':')[1] if ':' in tag else 'MWE'
                
                # Count unique MWEs
                if mwe_id not in mwes_in_sent:
                    mwes_in_sent.add(mwe_id)
                    total_mwes += 1
                    category_counts[category] = category_counts.get(category, 0) + 1
    
    print(f"Total MWEs detected: {total_mwes}")
    print(f"\nMWE Categories:")
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_mwes * 100) if total_mwes > 0 else 0
        print(f"  {category:15s}: {count:4d} ({percentage:5.1f}%)")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    cupt_file = 'predictions/FR/test.cupt'
    num_examples = 15
    
    if len(sys.argv) > 1:
        cupt_file = sys.argv[1]
    if len(sys.argv) > 2:
        num_examples = int(sys.argv[2])
    
    visualize_predictions(cupt_file, num_examples)
