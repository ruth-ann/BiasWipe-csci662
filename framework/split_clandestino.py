import pandas as pd
from sklearn.model_selection import train_test_split

# Load the TSV
df = pd.read_csv("clandestino_cleaned.tsv", sep="\t")

# Stratified split: train 60%, temp 40%
train_df, temp_df = train_test_split(
    df, test_size=0.4, stratify=df['is_toxic'], random_state=42
)

# Split temp into dev 20% and test 20% (half of temp each)
dev_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df['is_toxic'], random_state=42
)

# Save to TSV with header
train_df.to_csv("train_es.tsv", sep="\t", index=False)
dev_df.to_csv("dev_es.tsv", sep="\t", index=False)
test_df.to_csv("test_es.tsv", sep="\t", index=False)
