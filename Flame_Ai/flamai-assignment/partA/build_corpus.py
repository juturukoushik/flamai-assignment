#!/usr/bin/env python3
"""
build_corpus.py -- Multilingual Parallel Evaluation Corpus Builder

Fetches 200 parallel sentences from FLORES-101 devtest across 4 languages:
- English (eng)
- Hindi (hin)
- Kannada (kan)
- Telugu (tel)

Usage:
    python build_corpus.py
"""

import sys
import os
import unicodedata
import requests

def build_eval_corpus():
    out_dir = os.path.join(os.path.dirname(__file__), "corpus")
    os.makedirs(out_dir, exist_ok=True)
    
    configs = {
        'eng': 'eng',
        'hin': 'hin',
        'kan': 'kan',
        'tel': 'tel'
    }
    
    corpus_data = {lang: [] for lang in configs}
    
    for lang, cfg in configs.items():
        print(f"Fetching FLORES-101 devtest sentences for {lang} ({cfg})...")
        for offset in [0, 100]:
            url = f"https://datasets-server.huggingface.co/rows?dataset=gsarti/flores_101&config={cfg}&split=devtest&offset={offset}&length=100"
            r = requests.get(url)
            if r.status_code == 200:
                data = r.json()
                if 'rows' in data:
                    for row in data['rows']:
                        sent = row['row']['sentence'].strip().replace("\n", " ")
                        sent = unicodedata.normalize("NFC", sent)
                        corpus_data[lang].append(sent)
            else:
                print(f"Error fetching {cfg} at offset {offset}: status {r.status_code}")
                
    min_len = min(len(corpus_data[l]) for l in corpus_data)
    print(f"\nParallel sentences aligned across all {len(configs)} languages: {min_len}")
    
    for lang in corpus_data:
        file_path = os.path.join(out_dir, f"{lang}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            for line in corpus_data[lang][:min_len]:
                f.write(line + "\n")
        print(f"Saved {file_path} ({min_len} lines).")

if __name__ == '__main__':
    build_eval_corpus()
