import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("finetune_parsed_completed.csv")

plt.figure(figsize=(8, 5))

for ratio in sorted(df["ratio"].astype(str).unique()):
    sub = df[df["ratio"].astype(str) == ratio].sort_values("integrated_epoch")
    plt.plot(
        sub["integrated_epoch"],
        sub["val_auc"],
        marker="o",
        linewidth=2,
        label=ratio
    )

plt.xlabel("Epoch")
plt.ylabel("Validation AUC")
plt.title("Validation AUC for Different Input Ratios")
plt.legend(title="Input ratio")
plt.grid(True)
plt.tight_layout()
plt.show()