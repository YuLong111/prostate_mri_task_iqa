import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("finetune_parsed_completed.csv")

plt.figure(figsize=(8, 5))

for ratio in sorted(df["ratio"].unique()):
    sub = df[df["ratio"] == ratio].sort_values("epoch")
    plt.plot(sub["epoch"], sub["val_auc"], marker="o", linewidth=2, label=ratio)

plt.xlabel("Epoch")
plt.ylabel("Validation AUC")
plt.title("Validation AUC for Different Input Ratios")
plt.legend(title="Input ratio")
plt.grid(True)
plt.tight_layout()
plt.show()