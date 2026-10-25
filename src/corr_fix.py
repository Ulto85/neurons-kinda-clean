"""Recompute the stratum-weighted correlation between calibrated simulation and
true activations (score.py v1 normalized token weights to sum to T instead of 1,
which inflated that one statistic toward 1; KL scores are unaffected)."""
import json
import numpy as np
from common import OUT
from evalsets import window_acts, weights

src = open("score.py").read()
exec("def calibrate" + src.split("def calibrate")[1].split("CONDS")[0])

ES = json.load(open(f"{OUT}/e3_evalsets.json"))
sim = np.load(f"{OUT}/e3_sim.npz")
ev = np.load(f"{OUT}/e3_acts_eval.npy")
mine = np.load(f"{OUT}/e3_acts_mine.npy", mmap_mode="r")
out = {}
for k in sim.files:
    s = sim[k]
    if np.isnan(s).any():
        continue
    es = ES[k]
    a = window_acts(es, int(k), mine, ev)
    w = weights(es)
    tw = np.repeat(w, a.shape[1]) / a.shape[1]
    x, y = calibrate(s, a).ravel(), a.ravel()
    mx, my = (tw * x).sum(), (tw * y).sum()
    vx, vy = (tw * (x - mx) ** 2).sum(), (tw * (y - my) ** 2).sum()
    cov = (tw * (x - mx) * (y - my)).sum()
    # also: weighted fraction of activation variance explained by the calibrated sim
    out[k] = dict(wcorr=float(cov / np.sqrt(vx * vy)) if vx > 0 and vy > 0 else 0.0,
                  wfve=float(1 - (tw * (x - y) ** 2).sum() / vy) if vy > 0 else 0.0)
json.dump(out, open(f"{OUT}/e3_corr.json", "w"))
print(len(out), "median wcorr", np.median([v["wcorr"] for v in out.values()]))
