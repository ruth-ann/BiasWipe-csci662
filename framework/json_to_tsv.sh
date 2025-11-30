#!/bin/bash

input="$1"
output="$2"

# Write header to output
echo -e "is_toxic\tcomment" > "$output"

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
' "$input" >> "$output"
