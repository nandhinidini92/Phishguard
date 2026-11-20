import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
boxes = [("URL stream /\nDataset", 0.2), ("Feature\nExtraction\n(18 lexical)", 2.6), ("Drift Monitor\n(PSI + KS)", 5.0),
         ("Classifier\n(XGBoost)", 7.4), ("SHAP\nExplainer", 9.8), ("Decision +\nReasons", 12.2)]
fig, ax = plt.subplots(figsize=(13, 3.2)); ax.axis("off"); ax.set_xlim(0, 14.5); ax.set_ylim(0, 3)
for t, x in boxes:
    ax.add_patch(FancyBboxPatch((x, 0.9), 2.0, 1.2, boxstyle="round,pad=0.05", fc="#dbe9f6", ec="#1f4e79"))
    ax.text(x + 1, 1.5, t, ha="center", va="center", fontsize=9)
for i in range(len(boxes) - 1):
    ax.annotate("", xy=(boxes[i+1][1], 1.5), xytext=(boxes[i][1] + 2.0, 1.5), arrowprops=dict(arrowstyle="->", color="#1f4e79"))
ax.annotate("", xy=(8.4, 0.9), xytext=(6.0, 0.9), arrowprops=dict(arrowstyle="->", color="#c00000", connectionstyle="arc3,rad=0.5"))
ax.text(7.2, 0.15, "drift alarm -> weighted retraining on recent window", color="#c00000", ha="center", fontsize=9)
plt.tight_layout(); plt.savefig("outputs/fig_architecture.png", dpi=170)
