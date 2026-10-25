"""E3 step 4: simulation-KL scoring (Gao's proposal).

For each neuron, replace its activation on its 160-window eval set with:
  zero, mean (mining-set mean), strong-only (keep |a| >= f*M, f in {.2,.5}),
  sim_raw (simulator value/10 * M), sim_cal (linear calibration of simulator
  values, cross-fit: fit on even windows, apply to odd and vice versa),
and measure KL(clean || patched). Per-window KL is averaged with stratum
weights (evalsets.py), giving an estimate of the KL on ordinary web text.
Headline score:
  recovered = 1 - KL(sim_cal) / KL(mean)   (1 = perfect stand-in, 0 = no better than mean-ablation)
Also: the old-style correlation score, and many-neurons-at-once patches on the
shared random windows.

usage: python score.py [--pilot]
"""
import argparse, json
import numpy as np
import torch
import torch.nn.functional as F
from common import *
from evalsets import window_ids, window_acts, weights

ap = argparse.ArgumentParser()
ap.add_argument("--pilot", action="store_true")
ap.add_argument("--tag", default="", help="score an alternate simulation file e3_sim<tag>.npz")
args = ap.parse_args()
sfx = "_pilot" if args.pilot else ""

torch.backends.cuda.matmul.allow_tf32 = False
tok, model = load_model("Qwen/Qwen3-4B")
neurons = json.load(open(f"{OUT}/e3_neurons.json"))
expl = json.load(open(f"{OUT}/e3_explanations{sfx}.json"))
ES = json.load(open(f"{OUT}/e3_evalsets.json"))
sim = np.load(f"{OUT}/e3_sim{sfx}{args.tag}.npz")
ev = np.load(f"{OUT}/e3_acts_eval.npy")
mine = np.load(f"{OUT}/e3_acts_mine.npy", mmap_mode="r")
P = NeuronPatcher(model)


BS = 40  # windows per forward chunk (keeps [W,T,V] fp32 log-probs small)


@torch.no_grad()
def run(X, patches=()):
    """patches: list of (layer, neuron, R[W,T]). Returns list of log-prob chunks [BS,T,V]."""
    by = {}
    for l, n, R in patches:
        by.setdefault(l, []).append((n, torch.as_tensor(R, dtype=torch.float32).cuda()))
    out = []
    for b in range(0, len(X), BS):
        for l, lst in by.items():
            idx = torch.tensor([n for n, _ in lst]).cuda()
            vals = torch.stack([R[b:b + BS] for _, R in lst], -1)

            def fn(a, idx=idx, vals=vals):
                a = a.clone(); a[..., idx] = vals.to(a.dtype); return a
            P.fns[l] = fn
        out.append(F.log_softmax(model(X[b:b + BS]).logits.float(), -1))
        P.fns = {}
    return out


@torch.no_grad()
def per_window(clean, lps, X):
    kls, ces = [], []
    for i, (c, lp) in enumerate(zip(clean, lps)):
        x = X[i * BS:(i + 1) * BS]
        kls.append((c.exp() * (c - lp)).sum(-1).mean(-1))
        ces.append(-lp[:, :-1].gather(-1, x[:, 1:, None]).squeeze(-1).mean(-1))
    return torch.cat(kls).cpu().numpy(), torch.cat(ces).cpu().numpy()


def calibrate(s, a):
    out = np.zeros_like(a)
    par = np.arange(len(a)) % 2 == 0
    for tr in (par, ~par):
        Z = np.stack([s[tr].ravel(), np.ones(tr.sum() * a.shape[1])], 1)
        coef = np.linalg.lstsq(Z, a[tr].ravel(), rcond=None)[0]
        out[~tr] = coef[0] * s[~tr] + coef[1]
    return out


CONDS = ("zero", "mean", "strong20", "strong50", "sim_raw", "sim_cal")
rows, rand_vals = [], {}
NR = ev.shape[0]
for k in sorted(sim.files, key=int):
    s = sim[k]
    if np.isnan(s).any():
        print("skip (sim failure)", k); continue
    ki, es = int(k), ES[k]
    l, n = neurons[ki]
    M, mu = es["M"], es["mean"]
    X = torch.as_tensor(window_ids(es)).cuda()
    a = window_acts(es, ki, mine, ev)
    w = weights(es)
    conds = dict(zero=np.zeros_like(a), mean=np.full_like(a, mu),
                 strong20=a * (np.abs(a) >= 0.2 * M), strong50=a * (np.abs(a) >= 0.5 * M),
                 sim_raw=s / 10 * M, sim_cal=calibrate(s, a))
    rand_vals[ki] = {c: v[:NR] for c, v in conds.items()}
    clean = run(X)
    ce0 = per_window(clean, clean, X)[1]
    e = expl[k]["expl"]
    r = dict(k=ki, layer=l, neuron=n, M=M, mean=mu, label=e["label"], mono=e.get("looks_monosemantic"),
             freq=es["freq"], n_windows=len(a),
             corr=float(np.corrcoef(s.ravel(), a.ravel())[0, 1]) if s.std() > 0 else 0.0,
             corr_cal_weighted=None)
    # weighted correlation of calibrated sim vs actual (token weights = window weights)
    tw = np.repeat(w, a.shape[1]) / a.shape[1]; x, y = conds["sim_cal"].ravel(), a.ravel()
    mx, my = (tw * x).sum(), (tw * y).sum()
    cov = (tw * (x - mx) * (y - my)).sum(); vx, vy = (tw * (x - mx) ** 2).sum(), (tw * (y - my) ** 2).sum()
    r["corr_cal_weighted"] = float(cov / np.sqrt(vx * vy)) if vx > 0 and vy > 0 else 0.0
    for c in CONDS:
        kl, ce = per_window(clean, run(X, [(l, n, conds[c])]), X)
        r[f"{c}_kl"] = float((w * kl).sum())                 # stratified estimate on web text
        r[f"{c}_kl_random"] = float(kl[:NR].mean())          # plain random windows only
        r[f"{c}_kl_active"] = float(kl[NR:].mean()) if len(kl) > NR else None  # held-out active windows
        r[f"{c}_dce"] = float((w * (ce - ce0)).sum())
    for c in ("zero", "strong20", "strong50", "sim_raw", "sim_cal"):
        r[f"{c}_rec"] = 1 - r[f"{c}_kl"] / r["mean_kl"] if r["mean_kl"] > 0 else float("nan")
        if r["mean_kl_active"]:
            r[f"{c}_rec_active"] = 1 - r[f"{c}_kl_active"] / r["mean_kl_active"]
    rows.append(r)
    print(f"L{l:2d} n{n:5d} mono={r['mono']} corr={r['corr']:+.2f} wcorr={r['corr_cal_weighted']:+.2f} "
          f"rec(sim)={r['sim_cal_rec']:+.2f} rec(simraw)={r['sim_raw_rec']:+.2f} rec(strong20)={r['strong20_rec']:+.2f} "
          f"meanKL={r['mean_kl']:.1e} | {r['label'][:50]}", flush=True)
    del clean
    json.dump(dict(neurons=rows), open(f"{OUT}/e3_scores{sfx}{args.tag}.json", "w"), indent=1)

# many neurons at once on the shared random windows
X = torch.as_tensor(load_ids("e3")).cuda()
clean = run(X)
group = []
layers = sorted({neurons[k][0] for k in rand_vals})
for name, ks in [(f"L{l}", [k for k in rand_vals if neurons[k][0] == l]) for l in layers] + [("all", list(rand_vals))]:
    g = dict(group=name, n=len(ks))
    for c in ("zero", "mean", "strong20", "sim_cal"):
        kl, _ = per_window(clean, run(X, [(*neurons[k], rand_vals[k][c]) for k in ks]), X)
        g[f"{c}_kl"] = float(kl.mean())
    g["sim_cal_rec"] = 1 - g["sim_cal_kl"] / g["mean_kl"]
    g["strong20_rec"] = 1 - g["strong20_kl"] / g["mean_kl"]
    group.append(g)
    print(g, flush=True)
json.dump(dict(neurons=rows, groups=group), open(f"{OUT}/e3_scores{sfx}{args.tag}.json", "w"), indent=1)
print("saved")
