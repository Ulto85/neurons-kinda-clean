"""Choose example neurons for the write-up: best simulated, clean-looking but unexplained, typical."""
import json, subprocess, sys
from common import OUT
rows = json.load(open(f"{OUT}/e3_scores.json"))["neurons"]
srt = sorted(rows, key=lambda r: -r["sim_cal_rec"])
best = [r["k"] for r in srt[:2]]
clean_bad = [r["k"] for r in sorted([r for r in rows if r["mono"] >= 4], key=lambda r: r["sim_cal_rec"])[:2]]
typical = [srt[len(srt) // 2]["k"]]
ks = [str(k) for k in dict.fromkeys(best + clean_bad + typical)]
subprocess.run([sys.executable, "export_examples.py", *ks], check=True)
