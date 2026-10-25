"""E1: do weak activations matter in an LM?

Cammarata reports (ConvNeXt, top-1 accuracy) that zeroing the weakest 75% of
activations barely matters (72% -> 71%). Gao guesses that in LMs ablating small
activations substantially degrades the model. We zero, for every MLP neuron in
every layer, activations below a per-neuron, per-pole threshold and measure KL to
the clean model, CE loss, top-1 agreement, and next-token accuracy.
"""
import json, sys, time
import torch
from common import *

name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-4B"
torch.backends.cuda.matmul.allow_tf32 = False
tok, model = load_model(name)
L = len(model.model.layers)
P = NeuronPatcher(model)

# ---- per-neuron, per-pole thresholds from 512 mining windows (32k tokens)
mine = load_ids("mine")[:512]
for i in range(L):
    P.capture[i] = slice(None)
with torch.no_grad():
    for b in range(0, len(mine), 64):
        model(torch.as_tensor(mine[b:b + 64]).cuda())
QS = [0.5, 0.75, 0.9, 0.95, 0.99]
thr = {}  # layer -> dict(pos=[Q,d], neg=[Q,d], pmax=[d], nmax=[d])
for i in range(L):
    a = torch.cat(P.captured[i]).flatten(0, 1).cuda()  # [N, d]
    N = a.shape[0]
    out = {}
    for pole, v in (("pos", a.clamp(min=0)), ("neg", (-a).clamp(min=0))):
        s = v.sort(0).values
        n = (v > 0).sum(0)  # positives per neuron
        idx = torch.stack([(N - n + (q * n).long()).clamp(max=N - 1) for q in QS])
        out[pole] = s.gather(0, idx)
        out[pole[0] + "max"] = s[-1].clone()  # clone: a view would keep all of s alive
        del s, v
    thr[i] = out
    del a
    torch.cuda.empty_cache()
P.clear()

ids = load_ids("e1")
clean_lp = logprobs(model, ids)
base = compare(model, ids, clean_lp)  # clean vs itself: gives clean CE and next-token accuracy
print(name, "clean", base, flush=True)

mass = {}


def make_fn(i, kind, arg):
    t = thr[i]

    def fn(a):
        if kind == "weak_q":      # remove weakest q of each pole
            q = QS.index(arg)
            keep = (a >= t["pos"][q]) | (a <= -t["neg"][q])
        elif kind == "strong_q":  # remove the strongest (1-q) of each pole
            q = QS.index(arg)
            keep = ~((a >= t["pos"][q]) | (a <= -t["neg"][q]))
        elif kind == "weak_f":    # remove |a| < f * pole max  (f=0.5 ~ "strong half of range")
            keep = (a >= arg * t["pmax"]) | (a <= -arg * t["nmax"])
        elif kind == "rand":
            keep = torch.rand_like(a) >= arg
        m = mass.setdefault("x", [0.0, 0.0, 0.0])
        a2 = a.float() ** 2
        m[0] += a2.sum().item(); m[1] += (a2 * (~keep)).sum().item(); m[2] += (~keep).float().mean().item()
        m.append(1)
        return a * keep
    return fn


def run(kind, arg, layers):
    P.fns = {i: make_fn(i, kind, arg) for i in layers}
    mass.clear()
    r = compare(model, ids, clean_lp)
    m = mass["x"]
    r.update(kind=kind, arg=arg, layers=layers if len(layers) < L else "all",
             frac_entries_removed=m[2] / (len(m) - 3), frac_sqmass_removed=m[1] / m[0])
    P.fns = {}
    return r


res = []
t0 = time.time()
for q in QS:
    res.append(run("weak_q", q, list(range(L))))
    res.append(run("strong_q", q, list(range(L))))
    res.append(run("rand", q, list(range(L))))
for f in [0.05, 0.1, 0.2, 0.5]:
    res.append(run("weak_f", f, list(range(L))))
for i in range(L):
    res.append(run("weak_q", 0.9, [i]))
    res.append(run("weak_f", 0.2, [i]))
for r in res:
    print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items() if k != "layers" or r["layers"] == "all"}, flush=True)
print("time", time.time() - t0)
os.makedirs(OUT, exist_ok=True)
json.dump(dict(model=name, base=base, results=res), open(f"{OUT}/e1_{name.split('/')[-1]}.json", "w"), indent=1)
