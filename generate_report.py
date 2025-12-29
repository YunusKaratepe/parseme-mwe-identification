import re
import collections

# --- CONFIGURATION ---
input_filename = "results/evaluation_results.txt"
output_filename = "results/detailed_evaluation_report.txt"

def parse_and_report():
    with open(input_filename, "r", encoding="utf-8") as f:
        content = f.read()

    # Data structures to hold aggregated data
    # Structure: [{'lang': 'FR', 'mwe_f': 0.5, 'tok_f': 0.6}, ...]
    global_stats = []
    
    # Structure: {'VID': {'MWE': [{'lang': 'FR', 'f': 0.5}], 'Tok': [...]}}
    category_stats = collections.defaultdict(lambda: {'MWE': [], 'Tok': []})
    
    # Structure: {'Continuous': {'MWE': [...], 'Tok': [...]}}
    continuity_stats = collections.defaultdict(lambda: {'MWE': [], 'Tok': []})

    # Split by language blocks
    blocks = content.split("RESULTS FOR LANGUAGE:")

    for block in blocks[1:]:
        # 1. Get Language Name
        lang_match = re.search(r"^\s*(\w+)", block)
        if not lang_match: continue
        lang = lang_match.group(1)

        # 2. Get Global Scores (MWE & Token)
        g_mwe_match = re.search(r"\* MWE-based: .*? F=([\d\.]+)", block)
        g_tok_match = re.search(r"\* Tok-based: .*? F=([\d\.]+)", block)
        
        f_mwe = float(g_mwe_match.group(1)) if g_mwe_match else 0.0
        f_tok = float(g_tok_match.group(1)) if g_tok_match else 0.0
        
        global_stats.append({'lang': lang, 'mwe_f': f_mwe, 'tok_f': f_tok})

        # 3. Get Per-Category Scores (MWE & Token)
        # Matches lines like: * VID: MWE-based: ... F=0.2308
        # We capture the Category Name, the Type (MWE or Tok), and the F-score
        cat_matches = re.findall(r"\* ([A-Za-z0-9\._]+): (MWE|Tok)-based: .*? F=([\d\.]+)", block)
        
        for cat_name, eval_type, cat_f in cat_matches:
            # Filter out the Continuity/Token count headers if they appear here
            if cat_name in ["Continuous", "Discontinuous", "Multi-token", "Single-token"]:
                continue
            category_stats[cat_name][eval_type].append({'lang': lang, 'f': float(cat_f)})

        # 4. Get Continuity & Token Count Scores
        # These are often listed as "* Continuous: MWE-based..."
        special_matches = re.findall(r"\* (Continuous|Discontinuous|Multi-token|Single-token): (MWE|Tok)-based: .*? F=([\d\.]+)", block)
        for type_name, eval_type, val_f in special_matches:
            continuity_stats[type_name][eval_type].append({'lang': lang, 'f': float(val_f)})

    # --- WRITING THE REPORT ---
    with open(output_filename, "w", encoding="utf-8") as out:
        
        # HEADER
        out.write("============================================================\n")
        out.write("          DETAILED MULTILINGUAL EVALUATION REPORT           \n")
        out.write("============================================================\n\n")

        # SECTION 1: GLOBAL RANKING
        out.write("1. GLOBAL RANKING (Sorted by MWE F1)\n")
        out.write("-" * 55 + "\n")
        out.write(f"{'LANGUAGE':<10} | {'MWE F1':<10} | {'TOK F1':<10}\n")
        out.write("-" * 55 + "\n")
        
        # Sort languages by MWE F1 score descending
        sorted_global = sorted(global_stats, key=lambda x: x['mwe_f'], reverse=True)
        for item in sorted_global:
            out.write(f"{item['lang']:<10} | {item['mwe_f']:.4f}     | {item['tok_f']:.4f}\n")
        
        if global_stats:
            avg_mwe = sum(x['mwe_f'] for x in global_stats) / len(global_stats)
            avg_tok = sum(x['tok_f'] for x in global_stats) / len(global_stats)
            out.write("-" * 55 + "\n")
            out.write(f"{'MACRO AVG':<10} | {avg_mwe:.4f}     | {avg_tok:.4f}\n\n")


        # SECTION 2: CATEGORY DIFFICULTY ANALYSIS
        out.write("2. CATEGORY DIFFICULTY (Averages across all languages)\n")
        out.write("-" * 80 + "\n")
        out.write(f"{'CATEGORY':<15} | {'MWE AVG':<8} | {'TOK AVG':<8} | {'BEST LANG (MWE)'}\n")
        out.write("-" * 80 + "\n")

        cat_summary = []
        for cat, types in category_stats.items():
            mwe_list = types['MWE']
            tok_list = types['Tok']
            
            avg_mwe = sum(s['f'] for s in mwe_list) / len(mwe_list) if mwe_list else 0
            avg_tok = sum(s['f'] for s in tok_list) / len(tok_list) if tok_list else 0
            
            # Find best language for this category (MWE based)
            best_lang = "N/A"
            if mwe_list:
                best = max(mwe_list, key=lambda x: x['f'])
                best_lang = f"{best['lang']} ({best['f']:.2f})"

            cat_summary.append({
                'cat': cat,
                'mwe_avg': avg_mwe,
                'tok_avg': avg_tok,
                'best': best_lang
            })
        
        # Sort by difficulty (Lowest MWE Avg F1 first)
        for item in sorted(cat_summary, key=lambda x: x['mwe_avg']):
            out.write(f"{item['cat']:<15} | {item['mwe_avg']:.4f}   | {item['tok_avg']:.4f}   | {item['best']}\n")
        out.write("\n")


        # SECTION 3: CONTINUITY & TOKEN COUNT ANALYSIS
        out.write("3. CONTINUITY & TOKEN COUNT STATISTICS\n")
        out.write("-" * 50 + "\n")
        out.write(f"{'TYPE':<20} | {'MWE AVG':<10} | {'TOK AVG':<10}\n")
        out.write("-" * 50 + "\n")
        
        for name, types in continuity_stats.items():
            mwe_list = types['MWE']
            tok_list = types['Tok']
            avg_mwe = sum(s['f'] for s in mwe_list) / len(mwe_list) if mwe_list else 0
            avg_tok = sum(s['f'] for s in tok_list) / len(tok_list) if tok_list else 0
            
            out.write(f"{name:<20} | {avg_mwe:.4f}     | {avg_tok:.4f}\n")
        out.write("\n")


        # SECTION 4: FULL BREAKDOWN BY LANGUAGE
        out.write("4. DETAILED BREAKDOWN PER LANGUAGE\n")
        out.write("============================================================\n")
        
        for block in blocks[1:]:
            lang_match = re.search(r"^\s*(\w+)", block)
            if not lang_match: continue
            lang = lang_match.group(1)
            
            out.write(f"\n--- {lang} ANALYSIS ---\n")
            
            # Extract Global F1 again for this section
            g_mwe = re.search(r"\* MWE-based: .*? F=([\d\.]+)", block)
            g_tok = re.search(r"\* Tok-based: .*? F=([\d\.]+)", block)
            
            if g_mwe: out.write(f"Global MWE F1: {g_mwe.group(1)}\n")
            if g_tok: out.write(f"Global Tok F1: {g_tok.group(1)}\n")

            out.write("Top & Bottom Categories (Based on MWE Scores):\n")
            
            # Extract categories for this specific block
            local_cats = re.findall(r"\* ([A-Za-z0-9\._]+): MWE-based: .*? F=([\d\.]+)", block)
            
            # Filter out non-categories like "Continuous" just in case regex caught them
            filtered_cats = []
            for name, score in local_cats:
                if name not in ["Continuous", "Discontinuous", "Multi-token", "Single-token"]:
                    filtered_cats.append((name, float(score)))
            
            # Sort local categories by score
            sorted_local = sorted(filtered_cats, key=lambda x: x[1], reverse=True)
            
            # Print top 3 and bottom 3
            if sorted_local:
                out.write(f"  Top 3:    {', '.join([f'{k}={v}' for k,v in sorted_local[:3]])}\n")
                out.write(f"  Bottom 3: {', '.join([f'{k}={v}' for k,v in sorted_local[-3:]])}\n")
            else:
                out.write("  No category data available.\n")

    print(f"Detailed analysis saved to: {output_filename}")

if __name__ == "__main__":
    parse_and_report()