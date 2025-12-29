import re
import os

# --- CONFIGURATION ---
INPUT_FILE = "results/evaluation_results.txt"
OUTPUT_FILE = "results/report.html"

# --- CSS STYLING ---
CSS_STYLE = """
<style>
.styled-table {
	border-collapse: collapse;
	margin: 14px 0;
	margin-left: 60px;
	font-size: 0.9em;
	font-family: sans-serif;
	min-width: 400px;
	box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
}
.styled-table thead tr {
	background-color: #34495e;
	border: 1px solid #ffffff;
	color: #ffffff;
	text-align: center;
}
.styled-table th,
.styled-table td {
	padding: 12px 15px;
	text-align: center;
	min-width: 45px;
}
.styled-table tbody tr {
	border-bottom: 1px solid #dddddd;
}
.styled-table tbody tr:nth-of-type(even) {
	background-color: #f3f3f3;
}
.styled-table tbody tr:last-of-type {
	border-bottom: 2px solid #34495e;
}
</style>
"""

def parse_results(filename):
    """Parses the specific format of the evaluation_results.txt"""
    if not os.path.exists(filename):
        print(f"Error: {filename} not found.")
        return {}

    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()

    data = {}
    
    # Split by language sections
    lang_blocks = content.split("RESULTS FOR LANGUAGE:")
    
    for block in lang_blocks[1:]:
        # Get Language Name
        lang_match = re.search(r"^\s*([A-Z]+)", block)
        if not lang_match: continue
        lang = lang_match.group(1)
        
        lang_data = {
            "global_mwe": {}, "global_tok": {},
            "continuous": {}, "discontinuous": {},
            "multi": {}, "single": {},
            "categories": {}
        }

        # 1. Global Scores
        g_mwe = re.search(r"\* MWE-based: P=[\d/]+=([\d\.]+) R=[\d/]+=([\d\.]+) F=([\d\.]+)", block)
        g_tok = re.search(r"\* Tok-based: P=[\d/]+=([\d\.]+) R=[\d/]+=([\d\.]+) F=([\d\.]+)", block)
        
        if g_mwe: lang_data["global_mwe"] = {"P": float(g_mwe.group(1)), "R": float(g_mwe.group(2)), "F": float(g_mwe.group(3))}
        if g_tok: lang_data["global_tok"] = {"P": float(g_tok.group(1)), "R": float(g_tok.group(2)), "F": float(g_tok.group(3))}

        # 2. Continuity
        cont = re.search(r"\* Continuous: MWE-based: P=[\d/]+=([\d\.]+) R=[\d/]+=([\d\.]+) F=([\d\.]+)", block)
        discont = re.search(r"\* Discontinuous: MWE-based: P=[\d/]+=([\d\.]+) R=[\d/]+=([\d\.]+) F=([\d\.]+)", block)
        
        if cont: lang_data["continuous"] = {"P": float(cont.group(1)), "R": float(cont.group(2)), "F": float(cont.group(3))}
        if discont: lang_data["discontinuous"] = {"P": float(discont.group(1)), "R": float(discont.group(2)), "F": float(discont.group(3))}

        # 3. Tokens
        multi = re.search(r"\* Multi-token: MWE-based: P=[\d/]+=([\d\.]+) R=[\d/]+=([\d\.]+) F=([\d\.]+)", block)
        single = re.search(r"\* Single-token: MWE-based: P=[\d/]+=([\d\.]+) R=[\d/]+=([\d\.]+) F=([\d\.]+)", block)
        
        if multi: lang_data["multi"] = {"P": float(multi.group(1)), "R": float(multi.group(2)), "F": float(multi.group(3))}
        if single: lang_data["single"] = {"P": float(single.group(1)), "R": float(single.group(2)), "F": float(single.group(3))}

        # 4. Categories
        cat_regex = re.compile(r"\* ([a-zA-Z0-9\.]+): (MWE|Tok)-based: P=[\d/]+=([\d\.]+) R=[\d/]+=([\d\.]+) F=([\d\.]+)")
        for match in cat_regex.finditer(block):
            cat_name = match.group(1)
            eval_type = match.group(2) # MWE or Tok
            scores = {"P": float(match.group(3)), "R": float(match.group(4)), "F": float(match.group(5))}
            
            if cat_name not in lang_data["categories"]:
                lang_data["categories"][cat_name] = {"MWE": {}, "Tok": {}}
            
            lang_data["categories"][cat_name][eval_type] = scores

        data[lang] = lang_data

    return data

def calc_avg(data, key, subkey=None):
    """Calculates macro average P, R, F for a specific data key across all languages"""
    metrics = ["P", "R", "F"]
    sums = {m: 0.0 for m in metrics}
    counts = {m: 0 for m in metrics}
    
    for lang in data:
        target = data[lang].get(key, {})
        if subkey and target:
            target = target.get(subkey, {})
        
        # Only count if the dict is not empty (i.e., language actually had this metric)
        if target:
            for m in metrics:
                sums[m] += target.get(m, 0.0)
                counts[m] += 1
    
    avgs = {}
    for m in metrics:
        # Avoid division by zero
        avgs[m] = (sums[m] / counts[m] * 100) if counts[m] > 0 else 0.0
    
    count_str = f"{counts['F']}/{len(data)}"
    return avgs, count_str

def generate_html(data):
    html = ["<!DOCTYPE html>", "<html>", "<head>", CSS_STYLE, "</head>", "<body>"]
    
    # --- 1. GENERAL RANKING TABLE ---
    avg_glob_mwe, count_mwe = calc_avg(data, "global_mwe")
    avg_glob_tok, count_tok = calc_avg(data, "global_tok")
    
    html.append('<h1 style="color: steelblue;">Cross-lingual macro-averages</h1>')
    html.append('<p style="line-height: 65%;"> <br> </p>')
    html.append('<h2>General ranking:</h2>')
    html.append('<table class="styled-table"><thead><tr>')
    html.append('<th rowspan="2">#Langs</th><th colspan="3">Global MWE-based</th><th colspan="3">Global Token-based</th></tr>')
    html.append('<tr><th>P</th><th>R</th><th>F</th><th>P</th><th>R</th><th>F</th></tr></thead><tbody>')
    html.append(f'<tr><td>{count_mwe}</td>')
    html.append(f'<td>{avg_glob_mwe["P"]:.2f}</td><td>{avg_glob_mwe["R"]:.2f}</td><td>{avg_glob_mwe["F"]:.2f}</td>')
    html.append(f'<td>{avg_glob_tok["P"]:.2f}</td><td>{avg_glob_tok["R"]:.2f}</td><td>{avg_glob_tok["F"]:.2f}</td></tr>')
    html.append('</tbody></table><br>')

    # --- 2. CONTINUITY TABLE ---
    avg_cont, count_cont = calc_avg(data, "continuous")
    avg_discont, count_discont = calc_avg(data, "discontinuous")
    
    html.append('<h2>Discontinuous vs Continuous VMWEs:</h2>')
    html.append('<table class="styled-table"><thead><tr>')
    html.append('<th colspan="4">Discontinuous MWE-based</th><th colspan="4">Continuous MWE-based</th></tr>')
    html.append('<tr><th>#Langs</th><th>P</th><th>R</th><th>F</th><th>#Langs</th><th>P</th><th>R</th><th>F</th></tr></thead><tbody>')
    html.append(f'<tr><td>{count_discont}</td>')
    html.append(f'<td>{avg_discont["P"]:.2f}</td><td>{avg_discont["R"]:.2f}</td><td>{avg_discont["F"]:.2f}</td>')
    html.append(f'<td>{count_cont}</td>')
    html.append(f'<td>{avg_cont["P"]:.2f}</td><td>{avg_cont["R"]:.2f}</td><td>{avg_cont["F"]:.2f}</td></tr>')
    html.append('</tbody></table><br>')
    
    # --- 3. TOKENS TABLE ---
    avg_multi, count_multi = calc_avg(data, "multi")
    avg_single, count_single = calc_avg(data, "single")

    html.append('<h2>Single-token vs Multi-token VMWEs:</h2>')
    html.append('<table class="styled-table"><thead><tr>')
    html.append('<th colspan="4">Single-token MWE-based</th><th colspan="4">Multi-token MWE-based</th></tr>')
    html.append('<tr><th>#Langs</th><th>P</th><th>R</th><th>F</th><th>#Langs</th><th>P</th><th>R</th><th>F</th></tr></thead><tbody>')
    html.append(f'<tr><td>{count_single}</td>')
    html.append(f'<td>{avg_single["P"]:.2f}</td><td>{avg_single["R"]:.2f}</td><td>{avg_single["F"]:.2f}</td>')
    html.append(f'<td>{count_multi}</td>')
    html.append(f'<td>{avg_multi["P"]:.2f}</td><td>{avg_multi["R"]:.2f}</td><td>{avg_multi["F"]:.2f}</td></tr>')
    html.append('</tbody></table><br><hr>')

    # --- 4. LANGUAGE SPECIFIC RANKING ---
    html.append('<h1 style="color: steelblue;">Language-specific system ranking</h1><br>')
    html.append('<table class="styled-table"><thead><tr>')
    html.append('<th rowspan="2">Language</th><th colspan="3">Diversity</th><th colspan="3">Unseen MWE-based</th>')
    html.append('<th colspan="3">Global MWE-based</th><th colspan="3">Global Token-based</th></tr>')
    html.append('<tr><th>Index Richness</th><th>Shannon-Weaver Entropy</th><th>Shannon Evenness Index</th>')
    html.append('<th>P</th><th>R</th><th>F</th><th>P</th><th>R</th><th>F</th><th>P</th><th>R</th><th>F</th></tr></thead><tbody>')
    
    sorted_langs = sorted(data.keys())
    for lang in sorted_langs:
        # FIX IS HERE: use "or" to fallback if dictionary is empty
        gm = data[lang].get("global_mwe") or {"P":0, "R":0, "F":0}
        gt = data[lang].get("global_tok") or {"P":0, "R":0, "F":0}
        
        html.append(f'<tr><td>{lang}</td>')
        html.append('<td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td>')
        html.append(f'<td>{gm["P"]*100:.2f}</td><td>{gm["R"]*100:.2f}</td><td>{gm["F"]*100:.2f}</td>')
        html.append(f'<td>{gt["P"]*100:.2f}</td><td>{gt["R"]*100:.2f}</td><td>{gt["F"]*100:.2f}</td></tr>')
    html.append('</tbody></table><br><hr>')

    # --- 5. PER CATEGORY TABLES ---
    html.append('<h1 style="color: steelblue;">Results per VMWE category</h1><br>')
    
    for lang in sorted_langs:
        html.append(f'<h2>{lang}</h2>')
        html.append('<table class="styled-table"><thead><tr>')
        html.append('<th rowspan="2">VMWEs</th><th colspan="3">MWE-based</th><th colspan="3">Token-based</th></tr>')
        html.append('<tr><th>P</th><th>R</th><th>F</th><th>P</th><th>R</th><th>F</th></tr></thead><tbody>')
        
        cats = data[lang]["categories"]
        if not cats:
            html.append('<tr><td colspan="7">No results available (Evaluation Failed)</td></tr>')
        else:
            sorted_cats = sorted(cats.keys())
            for cat in sorted_cats:
                mwe = cats[cat].get("MWE") or {"P":0, "R":0, "F":0}
                tok = cats[cat].get("Tok") or {"P":0, "R":0, "F":0}
                
                html.append(f'<tr><td>{cat}</td>')
                html.append(f'<td>{mwe["P"]*100:.2f}</td><td>{mwe["R"]*100:.2f}</td><td>{mwe["F"]*100:.2f}</td>')
                html.append(f'<td>{tok["P"]*100:.2f}</td><td>{tok["R"]*100:.2f}</td><td>{tok["F"]*100:.2f}</td></tr>')
        
        html.append('</tbody></table><br>')

    html.append('</body></html>')
    return "\n".join(html)

def main():
    data = parse_results(INPUT_FILE)
    if not data:
        print("No data parsed. Please check if evaluation_results.txt exists.")
        return

    html_content = generate_html(data)
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"Successfully generated {OUTPUT_FILE} based on {len(data)} languages.")

if __name__ == "__main__":
    main()