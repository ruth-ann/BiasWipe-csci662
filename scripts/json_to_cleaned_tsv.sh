#!/bin/bash

input_json="$1"
output_tsv="$2"

# Step 1: Convert JSON to TSV
tmp_tsv=$(mktemp)

# Write header to temp TSV
echo -e "is_toxic\tcomment" > "$tmp_tsv"

# Process each line of NDJSON with jq
jq -r '
  select(type=="object") |

  .Sentence as $comment |

  # Collect Q5 values
  [ .Annotators[]
      | ."Question 5: In your opinion, how does the text refer to the targeted group?"
      | select(. != null and . != "")
  ] as $labels |

  # Majority vote
  (
    if ($labels | length == 0) then "Unknown"
    else
      $labels
      | map({key: ., value: 1})
      | group_by(.key)
      | map({key: .[0].key, count: map(.value)|add})
      | max_by(.count).key
    end
  ) as $majority |

  "\($majority)\t\($comment)"
' "$input_json" >> "$tmp_tsv"

# Step 2: Clean labels (Toxic -> TRUE, else FALSE)
awk -F'\t' 'BEGIN{OFS="\t"} NR==1{print $1,$2; next} { $1 = ($1 ~ /Toxic/) ? "TRUE" : "FALSE"; print $1,$2 }' "$tmp_tsv" > "$output_tsv"

# Remove temp file
rm "$tmp_tsv"

echo "Saved cleaned TSV to $output_tsv"
