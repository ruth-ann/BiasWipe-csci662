import pandas as pd
import os
import argparse

def main(comments_file, annotations_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Load the files
    comments = pd.read_csv(comments_file, sep='\t')
    annotations = pd.read_csv(annotations_file, sep='\t')

    # Aggregate the toxicity labels per comment
    toxicity_labels = annotations.groupby("rev_id")["toxicity"].mean().round().astype(int).reset_index()
    toxicity_scores = annotations.groupby("rev_id")["toxicity_score"].mean().reset_index()

    # Merge both into one file
    comments_full = comments.merge(toxicity_labels, on="rev_id").merge(toxicity_scores, on="rev_id")
    comments_full['is_toxic'] = comments_full['toxicity']
    comments_full['is_toxic'] = comments_full['is_toxic'].apply(lambda x: "FALSE" if x == 0 else "TRUE")

    # Save the merged files
    out_labels = os.path.join(output_dir, "wikitalks_dataset.tsv")
    comments_full.to_csv(out_labels, sep='\t', index=False)
    
    print("File saved:")
    print(out_labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate toxicity labels from comments and annotations.")
    parser.add_argument("--comments_file", required=True, help="Path to the comments TSV file")
    parser.add_argument("--annotations_file", required=True, help="Path to the annotations TSV file")
    parser.add_argument("--output_dir", required=True, help="Directory to save the merged output files")

    args = parser.parse_args()
    main(args.comments_file, args.annotations_file, args.output_dir)
