"""E3 step 2: Opus writes a Cammarata-style explanation + simulation rubric for
each neuron from activating examples (top / mid / weak, both poles, random).

usage: python explain.py [--pilot N] [--batch]
"""
import argparse, json
import numpy as np
from transformers import AutoTokenizer
from common import OUT, load_ids
from llm import call_many, call_batch, parse_json, spent

ap = argparse.ArgumentParser()
ap.add_argument("--pilot", type=int, default=0)
ap.add_argument("--batch", action="store_true")
ap.add_argument("--resume", default=None, help="batch id to re-download instead of resubmitting")
args = ap.parse_args()

MODEL = "claude-opus-5-5"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
# explainer sees only EVEN mining windows; odd ones are held out for scoring
ids = load_ids("mine")[0::2]
acts = np.load(f"{OUT}/e3_acts_mine.npy")[0::2]
neurons = json.load(open(f"{OUT}/e3_neurons.json"))
rng = np.random.default_rng(2)


def show(t):
    return t.replace("\n", "↵").replace("\t", "⇥")


def fmt_window(w, a, M, center, before=40, after=4):
    s, e = max(0, center - before), min(len(w), center + after + 1)
    parts = []
    for i in range(s, e):
        t = show(tok.decode([int(w[i])]))
        v = 10 * a[i] / M
        parts.append(f"«{t}|{v:+.0f}»" if abs(v) >= 0.5 else t)
    return "".join(parts)


def build_prompt(k):
    A = np.asarray(acts[:, :, k], dtype=np.float32)  # [W, T]
    M = float(np.abs(A).max()) + 1e-8
    peak_pos, peak_neg = A.max(1), A.min(1)
    flat = np.abs(A).ravel() / M * 10
    sections = []

    def pick(order, n, sign, title):
        ex = []
        for w in order[:n]:
            c = int(A[w].argmax() if sign > 0 else A[w].argmin())
            ex.append(fmt_window(ids[w], A[w], M, c))
        if ex:
            sections.append(f"## {title}\n" + "\n".join(f"- {x}" for x in ex))

    def band(peaks, lo, hi, n, sign, title):
        p = sign * peaks / M
        cand = np.where((p >= lo) & (p < hi))[0]
        pick(rng.permutation(cand), n, sign, title)

    pick(np.argsort(-peak_pos), 20, +1, "Top positive activations")
    band(peak_pos, 0.25, 0.5, 8, +1, "Mid positive activations (peak 2.5-5)")
    band(peak_pos, 0.08, 0.25, 6, +1, "Weak positive activations (peak 0.8-2.5)")
    pick(np.argsort(peak_neg), 12, -1, "Top negative activations")
    band(peak_neg, 0.25, 0.5, 6, -1, "Mid negative activations")
    rnd = rng.choice(len(ids), 6, replace=False)
    sections.append("## Random windows (activations shown wherever |value| >= 0.5)\n" + "\n".join(
        f"- {fmt_window(ids[w], A[w], M, 63, before=63, after=0)}" for w in rnd))
    stats = (f"Over {flat.size:,} tokens of web text: fraction with |value|>=1: {np.mean(flat >= 1):.4f}, "
             f">=3: {np.mean(flat >= 3):.5f}, >=5: {np.mean(flat >= 5):.5f}. "
             f"Positive peak = +10; most negative value = {10 * A.min() / M:+.1f}.")
    return M, stats, "\n\n".join(sections)


INSTR = """You are an expert interpretability researcher studying one MLP neuron in the language model Qwen3-4B (a SwiGLU neuron, so it has both a positive and a negative pole).

Below are text windows from web text. Tokens where the neuron is active are shown as «token|value», with values scaled so the neuron's largest-magnitude activation is ±10. Unmarked tokens have |value| < 0.5. Each window ends a few tokens after its peak (the model is causal, so only left context can matter).

{stats}

{examples}

Your job: figure out what this neuron does, in the spirit of careful neuron-by-neuron science: look for the simplest coherent explanation, including how a core concept may generalize outward to related cases, and do not dismiss weak activations.

Your explanation will be handed to another model that must predict this neuron's activation on EVERY token of fresh, random web text, using only your explanation. So it must be predictive, not just a label: say what fires the neuron, which token in a phrase carries the activation, what controls magnitude, what the negative pole responds to, and what does NOT fire it despite looking similar.

Return only JSON:
{{"label": "<=12 word summary",
  "positive_pole": "<what drives positive activations>",
  "negative_pole": "<what drives negative activations, or 'none/negligible'>",
  "rubric": "<concrete rules for assigning values from -10 to 10 to each token, with example magnitudes; mention typical background level>",
  "looks_monosemantic": <1-5, how cleanly this looks like one coherent job from the examples (5 = clearly one job)>}}"""


def params(k):
    M, stats, ex = build_prompt(k)
    return M, dict(model=MODEL, max_tokens=16000, output_config={"effort": "medium"},
                   messages=[{"role": "user", "content": INSTR.format(stats=stats, examples=ex)}])


ks = list(range(len(neurons)))
if args.pilot:
    ks = sorted(rng.choice(ks, args.pilot, replace=False).tolist())
built = {k: params(k) for k in ks}
print(f"{len(ks)} prompts built; spent so far ${spent():.2f}", flush=True)
if args.batch:
    outs = call_batch([(f"n{k}", built[k][1]) for k in ks], resume_id=args.resume)
    texts = [outs.get(f"n{k}", "__ERROR__ missing") for k in ks]
    # retry anything that failed or didn't parse, once, on the standard API
    bad = [i for i, t in enumerate(texts) if t.startswith("__ERROR__") or parse_json(t) is None]
    print(f"retrying {len(bad)} failed/unparseable", flush=True)
    for i, t in zip(bad, call_many([built[ks[i]][1] for i in bad])):
        texts[i] = t
else:
    texts = call_many([built[k][1] for k in ks])

path = f"{OUT}/e3_explanations{'_pilot' if args.pilot else ''}.json"
res = {}
for k, t in zip(ks, texts):
    j = parse_json(t) if not t.startswith("__ERROR__") else None
    res[k] = dict(layer=neurons[k][0], neuron=neurons[k][1], M=built[k][0], expl=j, raw=None if j else t)
json.dump(res, open(path, "w"), indent=1)
print(f"saved {path}; ok={sum(r['expl'] is not None for r in res.values())}/{len(res)}; spent ${spent():.2f}")
