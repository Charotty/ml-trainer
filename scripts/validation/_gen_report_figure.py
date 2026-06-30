#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
report = json.loads(
    (ROOT / "results/validation_runs/clinical_honest_20260630/metrics/clinical_honest_report.json").read_text(
        encoding="utf-8"
    )
)
m = report["groupkfold_oof_87"]
targets = list(m["per_target_mae_mm"].keys())
vals = [m["per_target_mae_mm"][t] for t in targets]
labels = [t.replace("kidney_", "").replace("_delta_", " ") for t in targets]
colors = ["#4C78A8" if "_x" in t else "#F58518" if "_y" in t else "#E45756" for t in targets]
out = ROOT / "docs/figures/production_mae_per_target.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(9, 4.5))
bars = ax.bar(labels, vals, color=colors)
ax.axhline(m["avg_mae_mm"], color="gray", ls="--", label=f"avg {m['avg_mae_mm']:.2f} mm")
ax.set_ylabel("MAE (mm)")
ax.set_title("Production model: GKF-5 OOF per target (87 patients)")
ax.legend()
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.2, f"{v:.1f}", ha="center", fontsize=9)
fig.tight_layout()
fig.savefig(out, dpi=160)
plt.close()
print(out)
