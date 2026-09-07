import sys
sys.path.append(r'C:\Users\jutur\AppData\Roaming\Python\Python312\site-packages')
import os, json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def generate_plots():
    json_path = r"C:\Users\jutur\.gemini\antigravity\brain\1373ee1f-d55e-4622-9d0f-c8f0aa82dd58\scratch\audit_results.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    out_dir = r"C:\Users\jutur\Downloads\Flame_Ai\flamai-assignment\partA"
    os.makedirs(out_dir, exist_ok=True)
    
    # Set aesthetics
    sns.set_theme(style="whitegrid")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    
    # Organize data by tokenizer and language
    toks_sent = {}
    available_tokenizers = set()
    for d in data:
        t = d['tokenizer']
        l = d['lang'].upper()
        available_tokenizers.add(t)
        if t not in toks_sent:
            toks_sent[t] = {}
        toks_sent[t][l] = d['micro_tok_per_sentence']
        
    languages = ['ENG', 'HIN', 'KAN', 'TEL']
    tokenizers = sorted(list(available_tokenizers))
    
    # Plot 1: Micro Tokens per Parallel Sentence
    ax1 = axes[0]
    x = np.arange(len(languages))
    width = 0.8 / len(tokenizers)
    
    colors = ['#e74c3c', '#2ecc71', '#3498db', '#9b59b6']
    
    for i, tok in enumerate(tokenizers):
        vals = [toks_sent[tok][l] for l in languages]
        rects = ax1.bar(x + i*width, vals, width, label=tok, color=colors[i % len(colors)])
        for rect in rects:
            height = rect.get_height()
            ax1.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=7, rotation=0)
            
    ax1.set_title("Tokens per Parallel Sentence (Semantic Payload Constant)", fontsize=11, fontweight='bold', pad=12)
    ax1.set_ylabel("Average Tokens / Sentence", fontsize=10)
    ax1.set_xticks(x + (len(tokenizers)-1)*width/2)
    ax1.set_xticklabels(languages, fontsize=10, fontweight='bold')
    ax1.legend(title="Tokenizer", frameon=True, fontsize=8)
    
    # Plot 2: Relative Expansion Ratio vs English
    ax2 = axes[1]
    x2 = np.arange(len(languages))
    
    for i, tok in enumerate(tokenizers):
        eng_base = toks_sent[tok]['ENG']
        vals = [toks_sent[tok][l] / eng_base for l in languages]
        rects = ax2.bar(x2 + i*width, vals, width, label=f"{tok} (vs ENG)", color=colors[i % len(colors)])
        for rect in rects:
            height = rect.get_height()
            ax2.annotate(f'{height:.2f}x',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=7)
            
    ax2.axhline(1.0, color='black', linestyle='--', linewidth=1, label="English Baseline (1.0x)")
    ax2.set_title("Relative Token Expansion Ratio (vs. English Baseline)", fontsize=11, fontweight='bold', pad=12)
    ax2.set_ylabel("Expansion Ratio (Lang / English)", fontsize=10)
    ax2.set_xticks(x2 + (len(tokenizers)-1)*width/2)
    ax2.set_xticklabels(languages, fontsize=10, fontweight='bold')
    ax2.legend(frameon=True, fontsize=8)
    
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "tokenizer_audit.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"Generated plot saved to: {plot_path}")

if __name__ == "__main__":
    generate_plots()
