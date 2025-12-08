"""
Analyze MWE types and discontinuity in PARSEME 2.0 data
"""
import os
import sys
from collections import defaultdict, Counter


def read_cupt_with_raw(filepath):
    """Read CUPT file and keep raw MWE annotations"""
    sentences = []
    current_sentence = {
        'tokens': [],
        'mwe_raw': [],
        'text': ''
    }
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            
            if line.startswith('#'):
                if 'text =' in line:
                    current_sentence['text'] = line.split('text =', 1)[1].strip()
                continue
            
            if not line.strip():
                if current_sentence['tokens']:
                    sentences.append(current_sentence)
                    current_sentence = {
                        'tokens': [],
                        'mwe_raw': [],
                        'text': ''
                    }
                continue
            
            parts = line.split('\t')
            if len(parts) < 11:
                continue
            
            token_id = parts[0]
            if '-' in token_id or '.' in token_id:
                continue
            
            form = parts[1]
            mwe_col = parts[10] if len(parts) > 10 else '*'
            
            current_sentence['tokens'].append(form)
            current_sentence['mwe_raw'].append(mwe_col)
    
    if current_sentence['tokens']:
        sentences.append(current_sentence)
    
    return sentences


def analyze_mwe_discontinuity(sentences):
    """Analyze whether MWEs are continuous or discontinuous"""
    continuous_mwes = 0
    discontinuous_mwes = 0
    mwe_details = []
    
    for sent in sentences:
        # Extract MWE information
        mwe_dict = defaultdict(list)  # mwe_id -> list of (position, token)
        
        for i, mwe_annotation in enumerate(sent['mwe_raw']):
            if mwe_annotation != '*':
                # Parse annotations like "1:VID" or "1;2:LVC"
                for annotation in mwe_annotation.split(';'):
                    if ':' in annotation:
                        mwe_id, mwe_type = annotation.split(':', 1)
                    else:
                        mwe_id = annotation
                        mwe_type = 'MWE'
                    mwe_dict[(mwe_id, mwe_type)].append((i, sent['tokens'][i]))
        
        # Check continuity for each MWE
        for (mwe_id, mwe_type), positions in mwe_dict.items():
            token_positions = [pos for pos, _ in positions]
            token_positions.sort()
            
            # Check if positions are continuous
            is_continuous = all(
                token_positions[i+1] - token_positions[i] == 1 
                for i in range(len(token_positions) - 1)
            )
            
            if is_continuous:
                continuous_mwes += 1
            else:
                discontinuous_mwes += 1
                # Record discontinuous MWE details
                tokens = [tok for _, tok in sorted(positions, key=lambda x: x[0])]
                mwe_details.append({
                    'type': mwe_type,
                    'tokens': tokens,
                    'positions': token_positions,
                    'gaps': [token_positions[i+1] - token_positions[i] - 1 
                            for i in range(len(token_positions) - 1) if token_positions[i+1] - token_positions[i] > 1]
                })
    
    return continuous_mwes, discontinuous_mwes, mwe_details


def analyze_mwe_categories(sentences):
    """Count MWE types/categories"""
    category_counter = Counter()
    category_examples = defaultdict(list)
    
    for sent in sentences:
        for i, mwe_annotation in enumerate(sent['mwe_raw']):
            if mwe_annotation != '*':
                # Parse annotations like "1:VID" or "1;2:LVC"
                for annotation in mwe_annotation.split(';'):
                    if ':' in annotation:
                        _, mwe_type = annotation.split(':', 1)
                    else:
                        mwe_type = 'MWE'
                    
                    category_counter[mwe_type] += 1
                    
                    # Store example (limit to 3 per category)
                    if len(category_examples[mwe_type]) < 3:
                        # Get full MWE context
                        category_examples[mwe_type].append(' '.join(sent['tokens']))
    
    return category_counter, category_examples


def analyze_mwe_length(sentences):
    """Analyze MWE length distribution"""
    length_counter = Counter()
    
    for sent in sentences:
        mwe_dict = defaultdict(list)
        
        for i, mwe_annotation in enumerate(sent['mwe_raw']):
            if mwe_annotation != '*':
                for annotation in mwe_annotation.split(';'):
                    if ':' in annotation:
                        mwe_id, _ = annotation.split(':', 1)
                    else:
                        mwe_id = annotation
                    mwe_dict[mwe_id].append(i)
        
        for mwe_id, positions in mwe_dict.items():
            length_counter[len(positions)] += 1
    
    return length_counter


def analyze_language(language_code, train_file, dev_file):
    """Analyze MWE statistics for a single language"""
    print(f"\nAnalyzing {language_code}...")
    
    results = {
        'language': language_code,
        'train': {},
        'dev': {}
    }
    
    for dataset_name, filepath in [('train', train_file), ('dev', dev_file)]:
        if not os.path.exists(filepath):
            print(f"  {dataset_name} file not found: {filepath}")
            continue
        
        sentences = read_cupt_with_raw(filepath)
        
        # Basic stats
        num_sentences = len(sentences)
        num_tokens = sum(len(s['tokens']) for s in sentences)
        num_mwe_tokens = sum(1 for s in sentences for mwe in s['mwe_raw'] if mwe != '*')
        
        # MWE categories
        categories, examples = analyze_mwe_categories(sentences)
        
        # Discontinuity analysis
        continuous, discontinuous, disc_details = analyze_mwe_discontinuity(sentences)
        
        # Length analysis
        lengths = analyze_mwe_length(sentences)
        
        results[dataset_name] = {
            'sentences': num_sentences,
            'tokens': num_tokens,
            'mwe_tokens': num_mwe_tokens,
            'mwe_density': num_mwe_tokens / num_tokens if num_tokens > 0 else 0,
            'categories': dict(categories),
            'examples': dict(examples),
            'continuous_mwes': continuous,
            'discontinuous_mwes': discontinuous,
            'discontinuity_rate': discontinuous / (continuous + discontinuous) if (continuous + discontinuous) > 0 else 0,
            'discontinuous_details': disc_details[:5],  # Store first 5 examples
            'length_distribution': dict(lengths)
        }
        
        print(f"  {dataset_name}: {num_sentences} sentences, {continuous + discontinuous} MWEs ({discontinuous} discontinuous)")
    
    return results


def main():
    # Languages to analyze
    languages = ['PL', 'FR', 'EL', 'PT', 'RO', 'SL', 'SR', 'SV', 'UK', 'NL', 'EGY', 'KA', 'GRC', 'JA', 'HE', 'LV', 'FA']
    
    all_results = []
    
    for lang in languages:
        train_file = f"2.0/subtask1/{lang}/train.cupt"
        dev_file = f"2.0/subtask1/{lang}/dev.cupt"
        
        if not os.path.exists(train_file):
            print(f"Skipping {lang} (files not found)")
            continue
        
        results = analyze_language(lang, train_file, dev_file)
        all_results.append(results)
    
    # Generate report
    generate_report(all_results)


def generate_report(all_results):
    """Generate markdown report"""
    
    report = []
    report.append("# MWE Type Analysis Report\n")
    report.append(f"**Generated**: December 9, 2025\n")
    report.append(f"**Dataset**: PARSEME 2.0 Subtask 1\n")
    report.append(f"**Languages Analyzed**: {len(all_results)}\n\n")
    
    # Overall statistics
    report.append("## 📊 Overall Statistics\n\n")
    
    total_continuous = sum(r['train'].get('continuous_mwes', 0) + r['dev'].get('continuous_mwes', 0) for r in all_results)
    total_discontinuous = sum(r['train'].get('discontinuous_mwes', 0) + r['dev'].get('discontinuous_mwes', 0) for r in all_results)
    total_mwes = total_continuous + total_discontinuous
    
    report.append(f"**Total MWEs Across All Languages**: {total_mwes:,}\n")
    report.append(f"- **Continuous MWEs**: {total_continuous:,} ({total_continuous/total_mwes*100:.1f}%)\n")
    report.append(f"- **Discontinuous MWEs**: {total_discontinuous:,} ({total_discontinuous/total_mwes*100:.1f}%)\n\n")
    
    if total_discontinuous > 0:
        report.append(f"✅ **Discontinuous MWEs DO EXIST** in the data ({total_discontinuous/total_mwes*100:.1f}% of all MWEs)\n\n")
    else:
        report.append(f"❌ **No discontinuous MWEs found** in the data\n\n")
    
    # Per-language discontinuity
    report.append("## 🌍 Discontinuity by Language\n\n")
    report.append("| Language | Train MWEs | Discontinuous | Rate | Dev MWEs | Discontinuous | Rate |\n")
    report.append("|----------|------------|---------------|------|----------|---------------|------|\n")
    
    for r in sorted(all_results, key=lambda x: x['train'].get('discontinuity_rate', 0), reverse=True):
        lang = r['language']
        train_total = r['train'].get('continuous_mwes', 0) + r['train'].get('discontinuous_mwes', 0)
        train_disc = r['train'].get('discontinuous_mwes', 0)
        train_rate = r['train'].get('discontinuity_rate', 0) * 100
        
        dev_total = r['dev'].get('continuous_mwes', 0) + r['dev'].get('discontinuous_mwes', 0)
        dev_disc = r['dev'].get('discontinuous_mwes', 0)
        dev_rate = r['dev'].get('discontinuity_rate', 0) * 100
        
        report.append(f"| **{lang}** | {train_total:,} | {train_disc:,} | {train_rate:.1f}% | {dev_total:,} | {dev_disc:,} | {dev_rate:.1f}% |\n")
    
    # MWE Categories
    report.append("\n## 📝 MWE Categories Distribution\n\n")
    
    # Aggregate categories across all languages
    all_categories = Counter()
    for r in all_results:
        for cat, count in r['train'].get('categories', {}).items():
            all_categories[cat] += count
        for cat, count in r['dev'].get('categories', {}).items():
            all_categories[cat] += count
    
    report.append("| Category | Count | Percentage |\n")
    report.append("|----------|-------|------------|\n")
    
    total_cat_tokens = sum(all_categories.values())
    for cat, count in all_categories.most_common():
        pct = count / total_cat_tokens * 100
        report.append(f"| **{cat}** | {count:,} | {pct:.2f}% |\n")
    
    # MWE Length Distribution
    report.append("\n## 📏 MWE Length Distribution\n\n")
    
    all_lengths = Counter()
    for r in all_results:
        for length, count in r['train'].get('length_distribution', {}).items():
            all_lengths[length] += count
        for length, count in r['dev'].get('length_distribution', {}).items():
            all_lengths[length] += count
    
    report.append("| MWE Length (tokens) | Count | Percentage |\n")
    report.append("|---------------------|-------|------------|\n")
    
    total_length = sum(all_lengths.values())
    for length in sorted(all_lengths.keys()):
        count = all_lengths[length]
        pct = count / total_length * 100
        report.append(f"| {length} | {count:,} | {pct:.2f}% |\n")
    
    # Discontinuous MWE Examples
    report.append("\n## 🔍 Discontinuous MWE Examples\n\n")
    
    example_count = 0
    for r in all_results:
        lang = r['language']
        for dataset in ['train', 'dev']:
            disc_details = r[dataset].get('discontinuous_details', [])
            if disc_details and example_count < 10:
                for detail in disc_details[:2]:
                    if example_count >= 10:
                        break
                    report.append(f"**{lang} ({dataset})** - Type: `{detail['type']}`\n")
                    report.append(f"- Tokens: {' ... '.join(detail['tokens'])}\n")
                    report.append(f"- Positions: {detail['positions']}\n")
                    report.append(f"- Gaps: {detail['gaps']} token(s) between components\n\n")
                    example_count += 1
    
    # Detailed per-language breakdown
    report.append("\n## 📋 Detailed Language Statistics\n\n")
    
    for r in all_results:
        lang = r['language']
        report.append(f"### {lang}\n\n")
        
        # Train stats
        if 'sentences' in r['train']:
            train = r['train']
            report.append(f"**Training Set**:\n")
            report.append(f"- Sentences: {train['sentences']:,}\n")
            report.append(f"- Tokens: {train['tokens']:,}\n")
            report.append(f"- MWE tokens: {train['mwe_tokens']:,} ({train['mwe_density']*100:.2f}% density)\n")
            report.append(f"- Continuous MWEs: {train['continuous_mwes']:,}\n")
            report.append(f"- Discontinuous MWEs: {train['discontinuous_mwes']:,} ({train['discontinuity_rate']*100:.1f}%)\n\n")
            
            if train.get('categories'):
                report.append(f"**Top MWE Categories**:\n")
                for cat, count in Counter(train['categories']).most_common(5):
                    report.append(f"- {cat}: {count:,}\n")
                report.append("\n")
        
        # Dev stats
        if 'sentences' in r['dev']:
            dev = r['dev']
            report.append(f"**Dev Set**:\n")
            report.append(f"- Sentences: {dev['sentences']:,}\n")
            report.append(f"- MWE tokens: {dev['mwe_tokens']:,}\n")
            report.append(f"- Discontinuous MWEs: {dev['discontinuous_mwes']:,} ({dev['discontinuity_rate']*100:.1f}%)\n\n")
    
    # Key insights
    report.append("\n## 💡 Key Insights\n\n")
    
    avg_discontinuity = total_discontinuous / total_mwes * 100
    
    report.append(f"1. **Discontinuity is relatively rare** ({avg_discontinuity:.1f}% overall)\n")
    report.append(f"2. **Most MWEs are short**: {all_lengths.get(2, 0):,} are 2-word MWEs\n")
    report.append(f"3. **Category distribution varies by language**\n")
    
    if avg_discontinuity > 5:
        report.append(f"4. ⚠️ **Discontinuous MWEs are significant enough** to warrant special handling\n")
    else:
        report.append(f"4. ℹ️ **Discontinuous MWEs are rare** - current BIO tagging may be sufficient\n")
    
    report.append(f"\n---\n\n")
    report.append(f"**Conclusion**: {'Special handling for discontinuous MWEs could improve results' if avg_discontinuity > 5 else 'Current BIO tagging approach is likely adequate for most cases'}\n")
    
    # Write report
    with open('MWE_ANALYSIS_REPORT.md', 'w', encoding='utf-8') as f:
        f.writelines(report)
    
    print("\n" + "="*80)
    print("Report generated: MWE_ANALYSIS_REPORT.md")
    print("="*80)


if __name__ == '__main__':
    main()
