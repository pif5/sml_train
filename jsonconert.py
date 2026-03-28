import pandas as pd
import json
import math

df = pd.read_excel("kichwa_corpus.xlsx")  # or read_csv if it's a CSV

def safe(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return str(x).strip()

records = []
grouped = df.groupby(["Source File", "Phrase #"])

for (source, phrase_num), group in grouped:
    kichwa_phrase = safe(group["Kichwa Phrase"].iloc[0])
    phrase_en = safe(group["Phrase Translation (EN)"].iloc[0])
    
    if not kichwa_phrase or not phrase_en:
        continue

    # Build word-level gloss string
    gloss_lines = []
    for _, row in group.iterrows():
        word = safe(row["Kichwa Word"])
        pos = safe(row["POS"])
        gloss = safe(row["English Gloss (Word)"])
        if word:
            gloss_lines.append(f"{word} | POS: {pos} | GLOSS: {gloss}")
    
    gloss_block = "\n".join(gloss_lines)

    text = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "You are a helpful translation assistant for the Kichwa language.\n"
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"Translate and gloss the following Kichwa phrase into English.\n\n"
        f"Kichwa: {kichwa_phrase}\n"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        f"English: {phrase_en}\n\n"
        f"Word glosses:\n{gloss_block}\n"
        "<|eot_id|>"
    )
    
    records.append({"text": text})

with open("train.jsonl", "w", encoding="utf-8") as f:
    for r in records:
        json.dump(r, f, ensure_ascii=False)
        f.write("\n")

print(f"Wrote {len(records)} examples to train.jsonl")
