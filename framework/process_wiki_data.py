import pandas as pd
import os

data_dir = "/home/exouser/data/wikipedia_talks_labels"

# Load the files
comments = pd.read_csv(os.path.join(data_dir, "toxicity_annotated_comments.tsv"), sep='\t')
annotations = pd.read_csv(os.path.join(data_dir, "toxicity_annotations.tsv"), sep='\t')

# Aggregate the toxicity labels per comment
toxicity_labels = annotations.groupby("rev_id")["toxicity"].mean().round().astype(int).reset_index()
toxicity_scores = annotations.groupby("rev_id")["toxicity_score"].mean().reset_index()

# Merge the labels/scores with the comment text
comments_with_labels = comments.merge(toxicity_labels, on="rev_id", how="left")
comments_with_scores = comments.merge(toxicity_scores, on="rev_id", how="left")

# Merge both into one file if you like
comments_full = comments.merge(toxicity_labels, on="rev_id").merge(toxicity_scores, on="rev_id")
comments_full['is_toxic'] = comments_full['toxicity']

# Save the merged files
comments_with_labels.to_csv(os.path.join(data_dir, "comments_with_labels.tsv"), sep='\t', index=False)
comments_with_scores.to_csv(os.path.join(data_dir, "comments_with_scores.tsv"), sep='\t', index=False)
comments_full.to_csv(os.path.join(data_dir, "comments_full.tsv"), sep='\t', index=False)

print("Files saved:")
print(os.path.join(data_dir, "comments_with_labels.tsv"))
print(os.path.join(data_dir, "comments_with_scores.tsv"))
print(os.path.join(data_dir, "comments_full.tsv"))

