"""Build the write-up page (writeup/index.html) from results/summary.json + examples.json."""
import json, os

ROOT = os.path.join(os.path.dirname(__file__), "..")
S = json.load(open(f"{ROOT}/results/summary.json"))
EX = {e["k"]: e for e in json.load(open(f"{ROOT}/results/examples.json"))}
e1, e3 = S["e1"], S["e3"]
rows = {r["k"]: r for r in e3["table"]}
HIDE = {103}  # label not shown on the page

data = dict(
    e1={m: [dict(q=c["arg"], agree=c["agree"], kl=c["kl"], acc=c["acc"]) for c in e1[m]["curves"]["weak_q"]]
        for m in ("Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B") if m in e1},
    e1clean={m: e1[m]["clean"]["acc"] for m in e1},
    strip={c: e3["rec_values"][c] for c in ("sim_cal", "strong20", "strong50")},
    labels=[r["label"] if r["k"] not in HIDE else f"Layer {r['layer']} neuron" for r in e3["table"]],
    layers=[r["layer"] for r in e3["table"]],
    mono={m: v for m, v in e3["by_mono"].items()},
    examples=[dict(**EX[k], web=rows[k]["sim_cal_rec"], active=rows[k]["sim_cal_rec_active"], s20=rows[k]["strong20_rec"])
              for k in (26, 23, 236) if k in EX],
)


def pct(x):
    return f"{100 * x:.0f}%".replace("-", "−")


R, RA, OC = e3["rec"], e3["rec_active"], e3.get("opus_check", {})
G = {g["group"]: g for g in e3["groups"]}
w4 = {c["arg"]: c for c in e1["Qwen3-4B"]["curves"]["weak_q"]}
w4f = {c["arg"]: c for c in e1["Qwen3-4B"]["curves"]["weak_f"]}
fill = dict(
    N=e3["n"],
    SIM_WEB=pct(R["sim_cal"]["median"]), SIM_ACT=pct(RA["sim_cal"]["median"]),
    S20_WEB=pct(R["strong20"]["median"]), S20_ACT=pct(RA["strong20"]["median"]),
    S50_WEB=pct(R["strong50"]["median"]), S50_ACT=pct(RA["strong50"]["median"]),
    SIM_GT50=pct(e3["frac_above"]["sim_cal"]["0.5"]), S20_GT50=pct(e3["frac_above"]["strong20"]["0.5"]),
    CORR=f"{e3['corr']['unweighted_160']['median']:.2f}", WCORR=f"{e3['corr']['weighted']['median']:.2f}",
    AG75={m: pct(1 - [c for c in e1[m]["curves"]["weak_q"] if c["arg"] == 0.75][0]["agree"]) for m in e1},
    KL75=f"{w4[0.75]['kl']:.2f}", AG75_4B=pct(w4[0.75]["agree"]), ACC_CLEAN=pct(e1["Qwen3-4B"]["clean"]["acc"]),
    ACC75=pct(w4[0.75]["acc"]), AG_F05=pct(w4f[0.05]["agree"]), AG_F50=pct(w4f[0.5]["agree"]),
    OPUS_WEB=pct(OC.get("opus", {}).get("median", 0)), SON_WEB=pct(OC.get("sonnet", {}).get("median", 0)),
    OPUS_ACT=pct(OC.get("opus_active", {}).get("median", 0)), SON_ACT=pct(OC.get("sonnet_active", {}).get("median", 0)),
    OPUS_N=OC.get("n", 0),
    ALL_SIM=pct(G["all"]["sim_cal_rec"]), ALL_S20=pct(G["all"]["strong20_rec"]),
    M5=e3["by_mono"]["5"], M4=e3["by_mono"]["4"], M3=e3["by_mono"]["3"],
    RHO=f"{e3['spearman_mono_vs_simrec_active']:.2f}",
    SIMRAW=pct(R["sim_raw"]["median"]),
)
fill["AG75_LIST"] = " → ".join(fill["AG75"][m] for m in ("Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B"))

html = open(f"{ROOT}/src/page_template.html").read()
html = html.replace("/*__DATA__*/null", json.dumps(data))
for k, v in fill.items():
    if isinstance(v, dict) and "sim_cal" in v:
        html = html.replace(f"{{{{{k}_N}}}}", str(v["n"])).replace(f"{{{{{k}_WEB}}}}", pct(v["sim_cal"])).replace(f"{{{{{k}_ACT}}}}", pct(v["sim_cal_active"]))
    elif not isinstance(v, dict):
        html = html.replace(f"{{{{{k}}}}}", str(v))
assert "{{" not in html, html[html.index("{{"):html.index("{{") + 40]
os.makedirs(f"{ROOT}/writeup", exist_ok=True)
open(f"{ROOT}/writeup/index.html", "w").write(html)
print("wrote writeup/index.html", len(html))
