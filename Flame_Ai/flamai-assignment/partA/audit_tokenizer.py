#!/usr/bin/env python3
"""
audit_tokenizer.py -- Tokenizer Fertility & Compression Audit (Part A)

Audits fertility.py on parallel evaluation corpora and compares:
1. Tokenizers: gpt2 (tiktoken), xlm-roberta-base (HuggingFace), qwen2.5-7b / mbert
2. Denominators: whitespace words, grapheme clusters, UTF-8 bytes, parallel sentences
3. Averaging: Macro-average (average of line ratios) vs Micro-average (ratio of total sums)

Usage:
    python audit_tokenizer.py --corpus_dir corpus/
"""

import argparse
import os
import sys
import unicodedata
import json
import tiktoken
import grapheme
from transformers import AutoTokenizer

def load_corpus(file_path: str):
    lines = []
    with open(file_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line:
                lines.append(unicodedata.normalize("NFC", line))
    return lines

def analyze_corpus(lines, encode_fn):
    total_tokens = 0
    total_words_naive = 0   # split(" ")
    total_words_proper = 0  # split()
    total_chars_python = 0  # len(line)
    total_graphemes = 0     # grapheme.length(line)
    total_bytes = 0         # len(line.encode('utf-8'))
    n_sentences = len(lines)
    
    macro_fert_naive = []
    macro_fert_proper = []
    macro_tpc = []
    
    for line in lines:
        toks = encode_fn(line)
        n_toks = len(toks)
        
        w_naive = len(line.lower().split(" "))
        w_proper = len(line.split())
        c_py = len(line)
        g_graph = grapheme.length(line)
        b_bytes = len(line.encode("utf-8"))
        
        total_tokens += n_toks
        total_words_naive += w_naive
        total_words_proper += w_proper
        total_chars_python += c_py
        total_graphemes += g_graph
        total_bytes += b_bytes
        
        macro_fert_naive.append(n_toks / max(1, w_naive))
        macro_fert_proper.append(n_toks / max(1, w_proper))
        macro_tpc.append(n_toks / max(1, c_py))
        
    return {
        "sentences": n_sentences,
        "total_tokens": total_tokens,
        "total_words_proper": total_words_proper,
        "total_graphemes": total_graphemes,
        "total_bytes": total_bytes,
        "macro_fert_naive": sum(macro_fert_naive) / n_sentences,
        "macro_fert_proper": sum(macro_fert_proper) / n_sentences,
        "micro_fert_proper": total_tokens / total_words_proper,
        "macro_tpc": sum(macro_tpc) / n_sentences,
        "micro_tok_per_grapheme": total_tokens / total_graphemes,
        "micro_tok_per_byte": total_tokens / total_bytes,
        "micro_tok_per_sentence": total_tokens / n_sentences,
    }

def main():
    ap = argparse.ArgumentParser(description="Multilingual Tokenizer Audit (Part A)")
    ap.add_argument("--corpus_dir", default=os.path.join(os.path.dirname(__file__), "corpus"))
    args = ap.parse_args()
    
    languages = ["eng", "hin", "kan", "tel"]
    corpora = {}
    for lang in languages:
        path = os.path.join(args.corpus_dir, f"{lang}.txt")
        if os.path.exists(path):
            corpora[lang] = load_corpus(path)
        else:
            print(f"Warning: {path} not found.")
            
    if not corpora:
        print("No evaluation corpora found. Run build_corpus.py first.")
        sys.exit(1)
        
    print("Loading tokenizers for benchmark...")
    tokenizers = {}
    
    # 1. GPT-2 (Tiktoken)
    gpt2_enc = tiktoken.get_encoding("gpt2")
    tokenizers["gpt2"] = lambda s: gpt2_enc.encode(s)
    
    # 2. XLM-RoBERTa (Indic-aware BPE/WordPiece)
    try:
        xlm_tok = AutoTokenizer.from_pretrained("xlm-roberta-base")
        tokenizers["xlm-roberta-base"] = lambda s: xlm_tok.encode(s, add_special_tokens=False)
    except Exception as e:
        print("Error loading xlm-roberta-base:", e)
        
    # 3. mBERT
    try:
        mbert_tok = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        tokenizers["bert-base-multilingual-cased"] = lambda s: mbert_tok.encode(s, add_special_tokens=False)
    except Exception as e:
        print("Error loading mbert:", e)
        
    print("\n" + "="*80)
    print(f"{'tokenizer':<25}{'lang':<8}{'micro tok/sent':<18}{'micro tok/word':<18}{'tok/grapheme':<15}{'tok/byte':<10}")
    print("="*80)
    
    audit_data = []
    for tok_name, encode_fn in tokenizers.items():
        for lang in languages:
            if lang not in corpora:
                continue
            metrics = analyze_corpus(corpora[lang], encode_fn)
            metrics["tokenizer"] = tok_name
            metrics["lang"] = lang
            audit_data.append(metrics)
            
            print(f"{tok_name:<25}{lang:<8}{metrics['micro_tok_per_sentence']:<18.2f}{metrics['micro_fert_proper']:<18.2f}{metrics['micro_tok_per_grapheme']:<15.3f}{metrics['micro_tok_per_byte']:<10.3f}")
            
    out_file = os.path.join(os.path.dirname(__file__), "audit_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)
    print(f"\nSaved audit results to {out_file}")

if __name__ == "__main__":
    main()
