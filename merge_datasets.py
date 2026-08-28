import pandas as pd
import os

# ── Load ILDC splits ───────────────────────────────
ildc_train = pd.read_csv("data/ildc_multi_train.csv")
ildc_dev   = pd.read_csv("data/ildc_single_dev.csv")
ildc_test  = pd.read_csv("data/ildc_test.csv")

# Combine all ILDC splits
ildc_all   = pd.concat([ildc_train, ildc_dev, ildc_test], ignore_index=True)

# Keep only text and label
ildc_clean = pd.DataFrame({
    "text"  : ildc_all["text"],
    "label" : ildc_all["label"]   # 0=rejected, 1=accepted
})

print(f"ILDC total     : {len(ildc_clean):,} rows")
print(f"ILDC labels    : {ildc_clean['label'].value_counts().to_dict()}")

# ── Load your original dataset ─────────────────────
original = pd.read_csv("data/legal_cases_dataset.csv")

# Drop neutral (label=0), map -1→0 and 1→1
original_clean = original[original["label"] != 0].copy()
original_clean["label"] = original_clean["label"].map({-1: 0, 1: 1})

original_clean = pd.DataFrame({
    "text"  : original_clean["text"],
    "label" : original_clean["label"]
})

print(f"\nOriginal total : {len(original_clean):,} rows")
print(f"Original labels: {original_clean['label'].value_counts().to_dict()}")

# ── Merge both ─────────────────────────────────────
merged = pd.concat([ildc_clean, original_clean], ignore_index=True)
merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nMerged total   : {len(merged):,} rows")
print(f"Merged labels  : {merged['label'].value_counts().to_dict()}")
print(f"Columns        : {merged.columns.tolist()}")

# ── Save ───────────────────────────────────────────
merged.to_csv("data/final_dataset.csv", index=False)
print("\nSaved → data/final_dataset.csv")
print("All done! Ready to build the project.")