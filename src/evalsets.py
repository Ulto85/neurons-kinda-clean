"""Per-neuron stratified eval sets.

Every neuron is scored on 160 windows: the 128 shared random windows (e3 split)
plus 32 held-out mining windows (odd indices; the explainer saw only even ones)
where the neuron is active: 16 with window peak |a| >= 0.4 M ("hi") and 16 with
0.1 M <= peak < 0.4 M ("mid"). Strata are defined by window peak |a| / M, and
their population frequencies come from the held-out mining windows, so a
weighted average over strata is an unbiased estimate of the random-text average
with far more weight of evidence on the rare windows where the neuron fires.
"""
import json
import numpy as np
from common import OUT, load_ids

EDGES = (0.1, 0.4)
N_PER = 16


def stratum(peak_rel):
    return np.digitize(peak_rel, EDGES)  # 0=low, 1=mid, 2=hi


def build():
    mine = np.load(f"{OUT}/e3_acts_mine.npy", mmap_mode="r")
    ev = np.load(f"{OUT}/e3_acts_eval.npy")
    neurons = json.load(open(f"{OUT}/e3_neurons.json"))
    odd = np.arange(1, mine.shape[0], 2)
    rng = np.random.default_rng(3)
    out = {}
    for k in range(len(neurons)):
        col = np.asarray(mine[:, :, k], dtype=np.float32)
        M = float(np.abs(col).max())
        pk = np.abs(col[odd]).max(1) / M
        st = stratum(pk)
        freq = [float(np.mean(st == s)) for s in range(3)]
        chosen = []
        for s in (2, 1):
            cand = odd[st == s]
            chosen += rng.choice(cand, min(N_PER, len(cand)), replace=False).tolist()
        rand_st = stratum(np.abs(ev[:, :, k]).max(1) / M).tolist()
        mine_st = stratum(np.abs(col[chosen]).max(1) / M).tolist()
        out[k] = dict(M=M, mean=float(col.mean()), freq=freq, mine_windows=chosen,
                      strata=rand_st + mine_st)  # strata for the 160 windows: 128 random then mined
    json.dump(out, open(f"{OUT}/e3_evalsets.json", "w"))
    return out


def window_ids(es):
    """token ids [160, T] for one neuron's eval set"""
    return np.concatenate([load_ids("e3"), load_ids("mine")[es["mine_windows"]]])


def window_acts(es, k, mine, ev):
    return np.concatenate([ev[:, :, k], np.asarray(mine[es["mine_windows"], :, k])]).astype(np.float32)


def weights(es):
    """per-window weights summing to 1: stratum frequency / windows in that stratum"""
    st = np.array(es["strata"])
    w = np.zeros(len(st))
    for s in range(3):
        n = (st == s).sum()
        if n:
            w[st == s] = es["freq"][s] / n
    return w / w.sum()


if __name__ == "__main__":
    o = build()
    f = np.array([v["freq"] for v in o.values()])
    print("mean stratum freq (low, mid, hi):", f.mean(0))
    print("neurons with <16 hi windows:", sum(len(v["mine_windows"]) < 32 for v in o.values()))
