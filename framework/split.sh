#!/bin/bash
header=$(head -n 1 toxicity_annotated_comments.tsv)

# Randomize everything except header
tail -n +2 toxicity_annotated_comments.tsv | shuf > shuffled.tsv

total=$(wc -l < shuffled.tsv)
train=$(( total * 80 / 100 ))
dev=$(( total * 10 / 100 ))
test=$(( total - train - dev ))

# Split
head -n $train shuffled.tsv > train_body.tsv
tail -n +$((train+1)) shuffled.tsv | head -n $dev > dev_body.tsv
tail -n $test shuffled.tsv > test_body.tsv

# Re-add header
{ echo "$header"; cat train_body.tsv; } > train.tsv
{ echo "$header"; cat dev_body.tsv; } > dev.tsv
{ echo "$header"; cat test_body.tsv; } > test.tsv

rm shuffled.tsv train_body.tsv dev_body.tsv test_body.tsv

