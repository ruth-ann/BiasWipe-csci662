import pandas as pd

# Load CSV
df = pd.read_csv("/home/exouser/data/unintended-ml-bias-analysis/sentence_templates/en_sentence_templates.csv")

# Extract keyword (last word in the phrase)
df["keyword"] = df["phrase"].apply(lambda x: x.split()[-1])

# Comment is simply the full phrase
df["comment"] = df["phrase"]

# Convert toxicity into binary label
df["is_toxic"] = df["toxicity"].apply(lambda x: 1 if x == "toxic" else 0)

# Select and order columns
df_out = df[["comment", "keyword", "is_toxic"]]

# Save to new CSV
df_out.to_csv("/home/exouser/data/unintended-ml-bias-analysis/english_entities.csv", index=False)
