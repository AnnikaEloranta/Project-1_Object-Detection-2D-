import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# Small script to generate a plot of the validation and training error over epochs
# =============================================================================

df = pd.read_csv("final_80_20_run/results.csv")

df["train_loss"] = df["train/box_loss"] + df["train/cls_loss"] + df["train/dfl_loss"]
df["val_loss"] = df["val/box_loss"] + df["val/cls_loss"] + df["val/dfl_loss"]

plt.figure()
plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
plt.plot(df["epoch"], df["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig("plot.png", dpi=300, bbox_inches="tight")

plt.show()