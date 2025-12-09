import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


## CHANGE THESE
output_file = "/home/exouser/data/outputs/figures/test.png"

files = {
    "OLD BERT": "/home/exouser/data/outputs/evaluations/old_bert_bias_eval.txt",
    "Lesbian": "/home/exouser/data/outputs/evaluations/old_bert_lesbian_150_bias_eval.txt",
}



# --------------------------
# 1. Parse a single .txt output file
# --------------------------
def parse_eval_file(path, model_name):
    entities = []
    fprs = []

    with open(path, "r") as f:
        lines = f.readlines()

    cur_entity = None

    for line in lines:
        line = line.strip()

        if line.startswith("ENTITY TERM:"):
            cur_entity = line.split(":")[1].strip()

        elif line.startswith("False Positive Rate:"):
            fpr = float(line.split(":")[1].replace("%", "").strip())
            entities.append(cur_entity)
            fprs.append(fpr)

    return pd.DataFrame({
        "entity": entities,
        "fpr": fprs,
        "model": model_name
    })


# --------------------------
# 2. Load all your model result files
# --------------------------

all_dfs = []

for model, filename in files.items():
    df = parse_eval_file(filename, model)
    all_dfs.append(df)

df_all = pd.concat(all_dfs)


# --------------------------
# 3. Plot the grouped bar chart
# --------------------------
plt.rcParams.update({
    "font.size": 22,          # base font size
    "axes.titlesize": 28,     # title
    "axes.labelsize": 24,     # x/y labels
    "xtick.labelsize": 20,    # x tick labels
    "ytick.labelsize": 20,    # y tick labels
    "legend.fontsize": 22,    # legend
})

plt.figure(figsize=(20, 6))

entities = sorted(df_all["entity"].unique())
models = df_all["model"].unique()

x = np.arange(len(entities))
width = 0.2

for i, model in enumerate(models):
    model_df = df_all[df_all["model"] == model].set_index("entity")
    plt.bar(x + i*width, model_df.loc[entities]["fpr"], width, label=model)

plt.xticks(x + width * 1.5, entities, rotation=60, ha="right")
plt.ylabel("False Positive Rate (%)")
plt.xlabel("Entity")
plt.title("False Positive Rate Across Entity Terms")
plt.legend()

# Add horizontal lines at each y-tick
ax = plt.gca()
for y in ax.get_yticks():
    plt.axhline(y=y, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)


plt.tight_layout()
plt.savefig(output_file, dpi=300)
