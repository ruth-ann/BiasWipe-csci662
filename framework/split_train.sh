#!/bin/bash

INPUT="comments_with_labels.tsv"

TRAIN_OUT="train.tsv"
DEV_OUT="dev.tsv"
TEST_OUT="test.tsv"

# Step 1: Extract header
header=$(head -n 1 "$INPUT")

# Step 2: Find column positions for 'split' and 'is_toxic'
split_col=$(echo "$header" | awk -F'\t' '{for(i=1;i<=NF;i++){if($i=="split"){print i; exit}}}')
label_col=$(echo "$header" | awk -F'\t' '{for(i=1;i<=NF;i++){if($i=="is_toxic"){print i; exit}}}')

if [ -z "$split_col" ] || [ -z "$label_col" ]; then
    echo "Error: 'split' or 'is_toxic' column not found!"
    exit 1
fi

echo "Split column: $split_col, Label column: $label_col"

# Step 3: Function to process each split
process_split() {
    split_name=$1
    output_file=$2
    awk -F'\t' -v split_col="$split_col" -v label_col="$label_col" -v split_name="$split_name" 'BEGIN{OFS="\t"} 
    NR>1 && $split_col==split_name {
        if($label_col=="0") $label_col="FALSE"
        else if($label_col=="1") $label_col="TRUE"
        print
    }' "$INPUT" | {
        echo "$header"
        cat
    } > "$output_file"
    echo "Created $output_file"
}

# Step 4: Create train, dev, test files
process_split "train" "$TRAIN_OUT"
process_split "dev" "$DEV_OUT"
process_split "test" "$TEST_OUT"

echo "All done!"
