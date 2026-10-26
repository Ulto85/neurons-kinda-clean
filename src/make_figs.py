"""Static figures for the README (figs/*.png) from results/summary.json."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
S = json.load(open(f"{ROOT}/results/summary.json"))
os.makedirs(f"{ROOT}/figs", exist_ok=True)
INK, INK2, GRID, BG = "#14171c", "#4a505b", "#e8ebf0", "#ffffff"
RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": BG, "axes.facecolor": BG, "savefig.dpi": 200})


def pctfmt(ax, axis="y"):
    f = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0%}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(f)


# 1. E1: predictions unchanged vs weak activations removed
fig, ax = plt.subplots(figsize=(7, 3.8))
qs = [0.5, 0.75, 0.9, 0.95, 0.99]
xs = range(len(qs))
for c, m in zip(RAMP, ["Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B"]):
    pts = S["e1"][m]["curves"]["weak_q"]
    ax.plot(xs, [p["agree"] for p in pts], "-o", color=c, lw=2, ms=5, mec="white", mew=1.2, label=m)
ax.plot([1], [0.9], "D", mfc="white", mec=INK2, mew=1.4, ms=8, label="ConvNeXt (Cammarata), image top-1")
ax.set_xticks(list(xs), [f"{q:.0%}" for q in qs])
ax.set_xlabel("Weakest activations removed from every neuron (per pole)")
ax.set_ylabel("Top predictions unchanged")
ax.set_ylim(0, 1.02); pctfmt(ax); ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=9, loc="upper right")
ax.set_title("Weak activations matter in language models", loc="left", color=INK, fontsize=11.5, fontweight="bold")
fig.tight_layout(); fig.savefig(f"{ROOT}/figs/e1_weak_ablation.png"); plt.close(fig)

# 2. E3: damage avoided per neuron
e3 = S["e3"]
fig, ax = plt.subplots(figsize=(7, 3.4))
rows = [("strong50", "True activations, ≥50% of max", S3), ("strong20", "True activations, ≥20% of max", S2),
        ("sim_cal", "Simulation from explanation", S1)]
rng = np.random.default_rng(0)
for i, (k, name, c) in enumerate(rows):
    v = np.clip(np.array(e3["rec_values"][k]), -1, 1)
    ax.scatter(v, i + rng.uniform(-0.25, 0.25, len(v)), s=12, color=c, alpha=0.5, lw=0)
    med = np.median(e3["rec_values"][k])
    ax.plot([med, med], [i - 0.36, i + 0.36], color=INK, lw=2.5, solid_capstyle="round")
    ax.text(med + 0.03, i + 0.3, f"median {med:.0%}", fontsize=8.5, color=INK)
ax.axvline(0, color="#9aa1ab", lw=1)
ax.set_yticks(range(3), [r[1] for r in rows]); ax.set_xlim(-1.05, 1.05); pctfmt(ax, "x")
ax.set_xlabel("Damage avoided vs deleting the neuron, on ordinary web text")
fig.suptitle(f"Can the explanation stand in for the neuron? (n={e3['n']} Qwen3-4B neurons)", x=0.01, ha="left", color=INK, fontsize=11.5, fontweight="bold")
ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(f"{ROOT}/figs/e3_damage_avoided.png"); plt.close(fig)

# 3. E3: by monosemanticity rating
fig, ax = plt.subplots(figsize=(7, 3.4))
mono = e3["by_mono"]; ks = sorted(mono, key=int); x = np.arange(len(ks)); w = 0.36
a = [mono[k]["sim_cal"] for k in ks]; b = [mono[k]["sim_cal_active"] for k in ks]
ax.bar(x - w / 2 - 0.01, a, w, color=RAMP[0], label="On ordinary text")
ax.bar(x + w / 2 + 0.01, b, w, color=RAMP[2], label="Where the neuron fires")
for xi, va, vb in zip(x, a, b):
    lab = lambda v: f"{v:.0%}" if abs(v) >= 0.005 else "0%"
    ax.text(xi - w / 2, max(va, 0) + 0.02, lab(va), ha="center", fontsize=8)
    ax.text(xi + w / 2, max(vb, 0) + 0.02, lab(vb), ha="center", fontsize=8)
ax.set_xticks(x, [f"{k}/5\nn={mono[k]['n']}" for k in ks])
ax.set_xlabel("How monosemantic the neuron looked to the explainer")
ax.set_ylabel("Damage avoided (median)"); pctfmt(ax); ax.axhline(0, color="#9aa1ab", lw=1)
ax.set_ylim(-0.1, 1.0); ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.set_title("Looking clean predicts being explained, but only at the top", loc="left", color=INK, fontsize=11.5, fontweight="bold")
fig.tight_layout(); fig.savefig(f"{ROOT}/figs/e3_by_rating.png"); plt.close(fig)
print("figs written")
