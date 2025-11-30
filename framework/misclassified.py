import pandas as pd
import argparse
import os

# Set up argument parser
parser = argparse.ArgumentParser(description="Filter biased entities from CSV based on keywords.")
parser.add_argument('-i', '--input', required=True, help="Path to the input CSV file")
parser.add_argument('-k', '--keywords', required=True, help="Path to a .txt file with keywords (one per line)")
parser.add_argument('-o', '--output', required=True, help="Path to a .txt file to save filtered examples")
args = parser.parse_args()

# Determine delimiter based on file extension
file_ext = os.path.splitext(args.input)[1].lower()
if file_ext == '.tsv':
    delimiter = '\t'
elif file_ext == '.csv':
    delimiter = ','
else:
    raise ValueError("Input file must be .csv or .tsv")

# Read CSV
df = pd.read_csv(args.input, sep=delimiter)

# Read keywords from txt file
with open(args.keywords, 'r') as f:
    keywords = [line.strip() for line in f if line.strip()]

# Print columns for debugging
print("Columns in CSV:", df.columns)

# Create an empty list to store the results
result_data = []

# Iterate over the DataFrame
for index, row in df.iterrows():
    text = row['text']
    actual_label = row['actual_label']
    predicted_label = row['predicted_labels']
    
    for keyword in keywords:
        if keyword.lower() in text.lower():  # case-insensitive match
            if actual_label != predicted_label:
                # Identify incorrect classifications
                if actual_label and not predicted_label:
                    classification = "FALSE"
                elif not actual_label and predicted_label:
                    classification = "TRUE"
                
                result_data.append({
                    'comment': text,
                    'is_toxic': actual_label,
                    'prev_predlabel': predicted_label,
                    'keyword': keyword
                })
                break  # Stop after first matching keyword

# Create a DataFrame from the results
result_df = pd.DataFrame(result_data)

# Save the DataFrame to a CSV file
result_df.to_csv(args.output, index=False)
print(f"Filtered results saved to {args.output}")

