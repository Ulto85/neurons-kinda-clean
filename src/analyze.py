"""Summarize results/*.json into results/summary.json (feeds the write-up page)."""
import json, glob, os
import numpy as np

R = os.environ.get("OUT_DIR", os.path.join(os.path.dirname(__file__), "..", "results"))
rng = np.random.default_rng(0)


def boot_median(v, n=2000):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    b = [np.median(rng.choice(v, len(v))) for _ in range(n)]
    return dict(median=float(np.median(v)), lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5)), n=len(v))


out = {"e1": {}, "e3": {}}

# ---- E1
for f in sorted(glob.glob(f"{R}/e1_Qwen3-*.json")):
    try:
        d = json.load(open(f))
    except json.JSONDecodeError:
        continue
    name = d["model"].split("/")[-1]
    allL = [r for r in d["results"] if r["layers"] == "all"]
    curves = {}
    for kind in ("weak_q", "strong_q", "rand", "weak_f"):
        curves[kind] = [dict(arg=r["arg"], kl=r["kl"], ce=r["ce"], agree=r["agree"], acc=r["acc"],
                             entries=r["frac_entries_removed"]) for r in allL if r["kind"] == kind]
    per_layer = {k: [r["kl"] for r in d["results"] if r["layers"] != "all" and r["kind"] == k] for k in ("weak_q", "weak_f")}
    out["e1"][name] = dict(clean=d["base"], curves=curves, per_layer=per_layer)

# ---- E3
sc = json.load(open(f"{R}/e3_scores.json"))
rows = sc["neurons"]
corr = json.load(open(f"{R}/e3_corr.json")) if os.path.exists(f"{R}/e3_corr.json") else {}
for r in rows:
    c = corr.get(str(r["k"]), {})
    r["wcorr"] = c.get("wcorr"); r["wfve"] = c.get("wfve")
conds = ["zero", "strong50", "strong20", "sim_raw", "sim_cal"]
e3 = out["e3"]
e3["n"] = len(rows)
e3["rec"] = {c: boot_median([r[f"{c}_rec"] for r in rows]) for c in conds}
e3["rec_active"] = {c: boot_median([r.get(f"{c}_rec_active", np.nan) for r in rows]) for c in conds}
e3["rec_values"] = {c: [r[f"{c}_rec"] for r in rows] for c in ("strong50", "strong20", "sim_cal")}
e3["frac_above"] = {c: {t: float(np.mean([r[f"{c}_rec"] > t for r in rows])) for t in (0.25, 0.5, 0.8)} for c in conds}
e3["corr"] = dict(unweighted_160=boot_median([r["corr"] for r in rows]),
                  weighted=boot_median([r["wcorr"] for r in rows if r["wcorr"] is not None]))
by_mono = {}
for m in range(1, 6):
    sel = [r for r in rows if r["mono"] == m]
    if sel:
        by_mono[m] = dict(n=len(sel), sim_cal=float(np.median([r["sim_cal_rec"] for r in sel])),
                          sim_cal_active=float(np.nanmedian([r.get("sim_cal_rec_active", np.nan) for r in sel])),
                          strong20=float(np.median([r["strong20_rec"] for r in sel])),
                          corr=float(np.median([r["corr"] for r in sel])))
e3["by_mono"] = by_mono
from scipy.stats import spearmanr  # noqa: E402
e3["spearman_mono_vs_simrec"] = float(spearmanr([r["mono"] for r in rows], [r["sim_cal_rec"] for r in rows]).correlation)
e3["spearman_mono_vs_simrec_active"] = float(spearmanr([r["mono"] for r in rows],
                                                       [r.get("sim_cal_rec_active", np.nan) for r in rows], nan_policy="omit").correlation)
by_layer = {}
for l in sorted({r["layer"] for r in rows}):
    sel = [r for r in rows if r["layer"] == l]
    by_layer[l] = {c: float(np.median([r[f"{c}_rec"] for r in sel])) for c in ("strong20", "strong50", "sim_cal")}
e3["by_layer"] = by_layer
e3["groups"] = sc.get("groups", [])
e3["mono_hist"] = {m: sum(r["mono"] == m for r in rows) for m in range(1, 6)}
# best / typical neurons by simulation score
srt = sorted(rows, key=lambda r: -r["sim_cal_rec"])
keep = ("k", "layer", "neuron", "label", "mono", "sim_cal_rec", "sim_cal_rec_active", "strong20_rec", "strong50_rec", "corr", "wcorr")
e3["top"] = [{k: r.get(k) for k in keep} for r in srt[:12]]
e3["clean_but_unexplained"] = [{k: r.get(k) for k in keep} for r in srt if r["mono"] >= 4][-8:]
e3["table"] = [{k: r.get(k) for k in keep} for r in rows]

# simulator-strength check: Opus vs Sonnet on the same neurons
if os.path.exists(f"{R}/e3_scores_opus.json"):
    op = {r["k"]: r for r in json.load(open(f"{R}/e3_scores_opus.json"))["neurons"]}
    pairs = [(rows_k, op[rows_k["k"]]) for rows_k in rows if rows_k["k"] in op]
    e3["opus_check"] = dict(
        n=len(pairs),
        sonnet=boot_median([a["sim_cal_rec"] for a, _ in pairs]), opus=boot_median([b["sim_cal_rec"] for _, b in pairs]),
        sonnet_active=boot_median([a.get("sim_cal_rec_active", np.nan) for a, _ in pairs]),
        opus_active=boot_median([b.get("sim_cal_rec_active", np.nan) for _, b in pairs]),
        sonnet_corr=float(np.median([a["corr"] for a, _ in pairs])), opus_corr=float(np.median([b["corr"] for _, b in pairs])),
        strong20=float(np.median([a["strong20_rec"] for a, _ in pairs])),
        pairs=[dict(k=a["k"], sonnet=a["sim_cal_rec"], opus=b["sim_cal_rec"]) for a, b in pairs])

json.dump(out, open(f"{R}/summary.json", "w"), indent=1)
print(json.dumps({k: v for k, v in e3.items() if k in ("n", "rec", "rec_active", "frac_above", "corr", "by_mono",
                                                        "spearman_mono_vs_simrec", "spearman_mono_vs_simrec_active")}, indent=1))
if "opus_check" in e3:
    print({k: v for k, v in e3["opus_check"].items() if k != "pairs"})
print("groups:", [(g["group"], round(g["sim_cal_rec"], 3), round(g["strong20_rec"], 3)) for g in e3["groups"]])
