"""Sanity check: is the simulator aligned with the right token?
Correlation between simulated and true activations on held-out active windows,
at shifts -1 / 0 / +1. Aligned output peaks at shift 0.
usage: python align_check.py <sim file tag, e.g. '' or '_v2pilot'>"""
import json, sys
import numpy as np
from common import OUT
from evalsets import window_acts

tag = sys.argv[1] if len(sys.argv) > 1 else ""
ES = json.load(open(f"{OUT}/e3_evalsets.json"))
sim = np.load(f"{OUT}/e3_sim{tag}.npz")
ev = np.load(f"{OUT}/e3_acts_eval.npy")
mine = np.load(f"{OUT}/e3_acts_mine.npy", mmap_mode="r")
res, hit = {-1: [], 0: [], 1: []}, []
for k in sim.files:
    s = sim[k]
    if np.isnan(s).any():
        continue
    a = window_acts(ES[k], int(k), mine, ev)[128:]
    s = s[128:]
    pairs = {0: (s, a), 1: (s[:, 1:], a[:, :-1]), -1: (s[:, :-1], a[:, 1:])}
    for sh, (x, y) in pairs.items():
        if x.std() > 0:
            res[sh].append(np.corrcoef(x.ravel(), y.ravel())[0, 1])
    hit += [ws[np.abs(wa).argmax()] != 0 for wa, ws in zip(a, s)]
for sh in (-1, 0, 1):
    print(f"shift {sh:+d}: median corr on active windows {np.median(res[sh]):.3f}  (n={len(res[sh])})")
print(f"active-window peaks where simulator predicted nonzero: {np.mean(hit):.3f}")
