"""Export token-level examples (true vs simulated activation) for chosen neurons.
usage: python export_examples.py k1 k2 ...  -> results/examples.json"""
import json, sys
import numpy as np
from transformers import AutoTokenizer
from common import OUT
from evalsets import window_ids, window_acts

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
ES = json.load(open(f"{OUT}/e3_evalsets.json"))
sim = np.load(f"{OUT}/e3_sim.npz")
expl = json.load(open(f"{OUT}/e3_explanations.json"))
ev = np.load(f"{OUT}/e3_acts_eval.npy")
mine = np.load(f"{OUT}/e3_acts_mine.npy", mmap_mode="r")
out = []
for k in sys.argv[1:]:
    es = ES[k]; M = es["M"]
    ids, a, s = window_ids(es), window_acts(es, int(k), mine, ev), sim[k]
    act = list(range(128, len(ids)))
    # held-out active window with the largest true peak, plus one random window where the neuron is quiet
    w_hi = max(act, key=lambda w: np.abs(a[w]).max())
    w_rand = 7  # a fixed random window, same for every neuron
    wins = []
    for w in (w_hi, w_rand):
        s_ = slice(max(0, int(np.abs(a[w]).argmax()) - 30), 64) if w == w_hi else slice(20, 64)
        wins.append(dict(kind="held-out active" if w == w_hi else "random",
                         tokens=[tok.decode([int(t)]) for t in ids[w][s_]],
                         true=[round(float(10 * v / M), 2) for v in a[w][s_]],
                         sim=[float(v) for v in s[w][s_]]))
    e = expl[k]["expl"]
    out.append(dict(k=int(k), layer=expl[k]["layer"], neuron=expl[k]["neuron"], label=e["label"],
                    positive_pole=e["positive_pole"], negative_pole=e["negative_pole"], mono=e["looks_monosemantic"],
                    windows=wins))
json.dump(out, open(f"{OUT}/examples.json", "w"), indent=1)
print("exported", len(out))
