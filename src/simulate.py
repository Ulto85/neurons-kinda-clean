"""E3 step 3: Sonnet predicts every token's activation on the shared random eval
set (128 windows x 64 tokens of FineWeb), using only the explanation.

usage: python simulate.py [--pilot] [--batch]
"""
import argparse, json
import numpy as np
from transformers import AutoTokenizer
from common import OUT, load_ids
from llm import call_many, call_batch, parse_json, spent

ap = argparse.ArgumentParser()
ap.add_argument("--pilot", action="store_true")
ap.add_argument("--batch", action="store_true")
ap.add_argument("--resume", default=None)
ap.add_argument("--model", default="claude-sonnet-5-5")
ap.add_argument("--effort", default="low")
ap.add_argument("--subset", type=int, default=0, help="random subset of neurons (simulator-strength check)")
ap.add_argument("--tag", default="")
args = ap.parse_args()

MODEL = args.model
CHUNK = 32  # windows per request
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
from evalsets import window_ids
ES = json.load(open(f"{OUT}/e3_evalsets.json"))
sfx = "_pilot" if args.pilot else ""
expl = json.load(open(f"{OUT}/e3_explanations{sfx}.json"))
if args.subset:
    keep = set(np.random.default_rng(7).choice(sorted(expl, key=int), args.subset, replace=False).tolist())
    expl = {k: v for k, v in expl.items() if k in keep}
sfx += args.tag


def show(t):
    return t.replace("\n", "↵").replace("\t", "⇥")


def fmt(w, wi):
    # one token per line: index, TAB, the token as a JSON string (whitespace explicit).
    # (v1 used inline "⟦i⟧tok" markers; the simulator often attached index i to the
    # PREVIOUS token, shifting predictions one token late.)
    lines = [f"{i}\t{json.dumps(tok.decode([int(t)]), ensure_ascii=False)}" for i, t in enumerate(w)]
    return f"<window id={wi}>\n" + "\n".join(lines) + "\n</window>"


SYS = """You simulate one neuron of a language model. You are given an explanation of the neuron written by a researcher, then text windows split into tokens. Each line of a window is one token: its index, a tab, then the token itself as a quoted string (so leading spaces and newlines are visible). The model reads top to bottom, so a token's activation can depend only on that token and earlier tokens.

For every token, predict the neuron's activation on that exact token, on a scale from -10 to 10 (10 = the neuron's strongest positive firing, -10 = strongest negative; near-zero is the typical background). Base predictions only on the explanation. Use the index printed on the same line as the token.

Output only JSON mapping window id to a list of [token_index, value] pairs for tokens whose predicted value is nonzero (integers; omit zeros), e.g. {"0": [[5, 3], [6, 8]], "1": []}. Include every window id."""


def params(e, wins, local):
    d = {k: e[k] for k in ("label", "positive_pole", "negative_pole", "rubric")}
    body = "\n".join(fmt(wins[w], j) for j, w in enumerate(local))  # ids = position in chunk
    return dict(model=MODEL, max_tokens=24000, output_config={"effort": args.effort},
                system=SYS,
                messages=[{"role": "user", "content": f"Explanation:\n{json.dumps(d, indent=1)}\n\n{body}"}])


# each neuron: its 160-window eval set (128 random + 32 held-out active), shuffled so
# the simulator cannot tell which windows were selected for activity
reqs, shapes = [], {}
for k, r in expl.items():
    if r["expl"] is None:
        continue
    wins = window_ids(ES[k])
    shapes[k] = wins.shape
    order = np.random.default_rng(int(k)).permutation(len(wins)).tolist()
    for c in range(0, len(order), CHUNK):
        reqs.append((f"n{k}_c{c}", params(r["expl"], wins, order[c:c + CHUNK])))
print(f"{len(reqs)} requests; spent so far ${spent():.2f}", flush=True)
if args.batch:
    outs = call_batch(reqs, resume_id=args.resume)
    bad = [c for c, p in reqs if outs.get(c, "__ERROR__").startswith("__ERROR__") or parse_json(outs[c]) is None]
    print(f"retrying {len(bad)} failed/unparseable", flush=True)
    pm = dict(reqs)
    for c, t in zip(bad, call_many([pm[c] for c in bad], workers=12)):
        outs[c] = t
else:
    outs = dict(zip([c for c, _ in reqs], call_many([p for _, p in reqs], workers=12)))

sim = {k: np.zeros(shapes[k], np.float32) for k in shapes}
fails = {}
for cid, t in outs.items():
    k, c = cid[1:].split("_c")
    j = parse_json(t) if not t.startswith("__ERROR__") else None
    order = np.random.default_rng(int(k)).permutation(shapes[k][0])[int(c):int(c) + CHUNK]
    if j is None:
        fails[cid] = t[:300]
        sim[k][order] = np.nan
        continue
    try:
        for pos, pairs in j.items():
            for i, v in pairs:
                if 0 <= int(pos) < len(order) and 0 <= int(i) < shapes[k][1]:
                    sim[k][order[int(pos)], int(i)] = float(v)
    except (TypeError, ValueError) as e:
        fails[cid] = f"parse {e!r}"; sim[k][order] = np.nan
np.savez(f"{OUT}/e3_sim{sfx}.npz", **sim)
json.dump(fails, open(f"{OUT}/e3_sim_fails{sfx}.json", "w"), indent=1)
print(f"saved; failures {len(fails)}/{len(outs)}; spent ${spent():.2f}")
