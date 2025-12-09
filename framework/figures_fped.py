import re
import pandas as pd
import matplotlib.pyplot as plt
import glob


## CHANGE THESE
output_file = "/home/exouser/data/outputs/figures/hehe.png"

files_by_model = {
    "LESBIAN": [
        "/home/exouser/data/outputs/evaluations/old_bert_lesbian_150_accuracy_eval.txt",
        "/home/exouser/data/outputs/evaluations/old_bert_lesbian_200_accuracy_eval.txt",
    ],
    "FAKE": [
        "/home/exouser/data/outputs/evaluations/silly_150_bias_eval.txt",
        "/home/exouser/data/outputs/evaluations/silly_100_bias_eval.txt",
    ]
}
data = []

for model_name, file_list in files_by_model.items():
    for file_path in file_list:
        with open(file_path, "r") as f:
            content = f.read()

        # Extract weight pruning from filename
        match = re.search(r"_(\d+)_", file_path)
        weight_pruning = int(match.group(1)) if match else None

        # Extract metrics
        fpr_match = re.search(r"False Positive Rate:\s*([\d.]+)%", content)
        fnr_match = re.search(r"False Negative Rate:\s*([\d.]+)%", content)

        if fpr_match and fnr_match:
            fpr = float(fpr_match.group(1))
            fnr = float(fnr_match.group(1))
            data.append({
                "Model": model_name,
                "Weight Pruning": weight_pruning,
                "FPR": fpr,
                "FNR": fnr
            })

# Convert to DataFrame
df = pd.DataFrame(data).sort_values(["Model", "Weight Pruning"])

# Plot
plt.figure(figsize=(8,6))
for model_name in df["Model"].unique():
    model_df = df[df["Model"] == model_name]
    plt.plot(model_df["Weight Pruning"], model_df["FPR"], marker='o', label=f"{model_name} FPED")
    plt.plot(model_df["Weight Pruning"], model_df["FNR"], marker='o', label=f"{model_name} FNED")

plt.xlabel("Weight Pruning")
plt.ylabel("Bias (%)")
plt.title("Effect of Weight Pruning on Models")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

plt.tight_layout()
plt.savefig(output_file, dpi=300)