# Neural networks are kinda clean

> **[My analysis upcoming]**

---

## AI summary

*The rest of this README was written by Claude (Opus 5.5), which designed and ran the experiments with me. Every number comes from the files in `results/`. Treat it as a lab notebook, not a paper.*

### The question

Two recent posts disagree about whether we can read neural networks neuron by neuron:

- **Nick Cammarata, [Neural networks are already clean](https://already-clean.goodfire.com/) (Goodfire).** Look carefully and most neurons have one main job. Polysemanticity was mostly weak tools and bad explanations. Weak activations "usually don't matter": in ConvNeXt, removing the weakest 75% costs only 72% → 71% accuracy.
- **Leo Gao, [Neural networks are not already clean](https://www.lesswrong.com/posts/Dkh7pXKjoNWkkxQDz/neural-networks-are-not-already-clean) (LessWrong).** Clean-looking activating examples can mislead you in two ways:
  - a neuron may fire on only part of what the explanation describes (precision is not recall);
  - the weak activations may matter.

  The real test is to replace the neuron with a simulation built from its explanation, and check that the model doesn't break, measured by **downstream KL on a large dataset**.

This repo runs Leo's test on Qwen3-4B (the same family as Nick's Qwen3-32B audit), and the weak-activation ablation that Leo said is where his results and Nick's diverge.

### What we found

**1. Some neurons really are clean.** A handful of neurons are fully explained by one sentence. Swapping in a simulation from the explanation recovers almost everything. They are nearly all single-word detectors in the first layers or the last one: `info`, `site`, `afford`, `cert`, `rental`, the ` un` prefix. 11 of 278 neurons pass 50% damage avoided. The explainer's "looks monosemantic" rating does predict this: 5/5 neurons score a median 83% where they fire.

**2. Weak activations matter in language models.** Zeroing each neuron's weakest 75% of activations (Nick's ConvNeXt setting) changes the top next-token prediction on **59% → 43% → 33% → 30%** of tokens across Qwen3 0.6B → 8B, against about 10% in Nick's ConvNeXt. Keeping only each neuron's "strong half" destroys every model.

![Weak-activation ablation](figs/e1_weak_ablation.png)

**3. For most neurons, the explanation doesn't work as a stand-in.**
- **Explanation-based simulation:** a median **−1%** of the damage avoided on ordinary text, which is no better than deleting the neuron. On windows where the neuron fires, it reaches 15%.
- **Perfect knowledge of the strong firings:** keeping the neuron's *real* values wherever |a| ≥ 20% of max recovers only **28%** on ordinary text.
- **Stronger simulator:** using Opus instead of Sonnet does no better.

![Damage avoided per neuron](figs/e3_damage_avoided.png)

![By monosemanticity rating](figs/e3_by_rating.png)

**In one line:** neurons look clean from their top examples, and a few really are, but for most of them that clean story leaves out what the model uses the neuron for.

### Two neurons that show the difference

**The `info` neuron (layer 0) is clean all the way down.** It fires on the token "info" ("more info", "phpinfo"), and the explanation says exactly that. The simulation recovers **86%** of the damage on ordinary text and **96%** where it fires, as good as the oracle that uses the neuron's true values. A layer-0 neuron mostly sees the raw token, so "fires on *info*" really is the whole story.

**The `PCR` neuron (layer 0) looks clean but isn't.** It fires on uppercase acronyms like PCR, PCB, CPR and DNA, and the explainer rated it 4/5 clean.
- **On its strongest firings:** the simulation is nearly perfect. It recovers 92% of the damage, and its correlation with the real neuron is 0.94. An old correlation-style score would call this neuron solved.
- **On ordinary text:** it scores **−57%**, worse than deleting the neuron. Two things go wrong:
  - *Background:* even the neuron's real strong firings recover only 14% on ordinary text, so most of its everyday effect is small activations the explanation doesn't describe. This is Leo's magnitude point.
  - *Over-generalizing:* a negative score means the simulation is putting wrong values into the model, not just missing some. The likely cause is that "acronyms starting with C/P/D" is broader than what the neuron responds to. This is Leo's precision-versus-recall point. It is inferred from the score and has not been checked token by token.

### How it works

1. **Neurons:** 278 random MLP neurons of Qwen3-4B, about 28 in each of layers 0, 4, …, 32 and 35. A neuron is one coordinate of the SwiGLU hidden state, so it has a positive and a negative pole.
2. **Activations:** recorded on 2.1M tokens of FineWeb in 64-token windows.
3. **Explainer (Claude Opus 5.5):** sees about 60 example windows (top, mid and weak examples of both poles, plus random ones) and writes an explanation with a scoring rubric, Cammarata-style. It also rates how monosemantic the neuron looks, from 1 to 5.
4. **Simulator (Claude Sonnet 5.5):** sees *only the explanation* and predicts every token's activation on 160 windows. These are 128 random windows plus 32 held-out windows where the neuron fires, shuffled together; none was shown to the explainer.
5. **Scoring:** substitute values into the neuron and measure KL(clean ‖ patched). Windows are reweighted to their frequency in real text. **Damage avoided = 1 − KL(substitute) / KL(mean-ablation).** 100% is a perfect stand-in; 0% is no better than deleting the neuron.

| Substitution | What replaces the neuron |
|---|---|
| `mean` | its average activation (the reference) |
| `strong20` / `strong50` | the true activation where \|a\| ≥ 20% / 50% of max, else 0. An oracle for "explain only the strong firings perfectly" |
| `sim_cal` | simulator output with a per-neuron linear calibration, cross-fit on held-out halves |

The weak-activation ablation (`src/e1_weak_ablation.py`) zeroes, for every neuron in every layer at once, activations below a per-neuron, per-pole quantile. It then measures KL, loss, top-1 agreement and accuracy on held-out FineWeb.

### Caveats

- **Automated explanations.** About 60 examples per neuron, not the long hand investigation Nick's audit used. Better explanations could score higher; the oracle rows bound how much is available from the strong firings alone.
- **Small scale.** One model for the simulation test, 278 neurons, 64-token windows.
- **Not a clean replication of the ConvNeXt result.** The comparison crosses tasks (image classification vs next-token prediction over 151k tokens) and metrics.
- **Tiny per-neuron effects.** KL from one neuron is about 1e-6 to 1e-5 nats, so all scores are relative to the same neuron's mean ablation. Everything runs in fp32 with TF32 off.
- **A bug caught along the way.** The first simulator prompt marked tokens inline (`⟦i⟧token`), and the model often attached each index to the previous token, which shifted predictions one token late. The run was redone with one token per line, and `src/align_check.py` now confirms alignment peaks at shift 0. The buggy run isn't used anywhere.

### Repo

```
src/prepare_data.py      FineWeb -> 64-token windows
src/collect.py           sample 280 neurons, record activations
src/e1_weak_ablation.py  weak-activation ablation (any Qwen3 size)
src/evalsets.py          per-neuron stratified eval sets
src/explain.py           Opus explanations (Batch API)
src/simulate.py          Sonnet / Opus simulations (Batch API)
src/score.py             KL substitution scoring
src/align_check.py       simulator token-alignment check
src/analyze.py           -> results/summary.json
src/make_figs.py         -> figs/*.png
writeup/index.html       interactive version of this write-up
results/                 explanations, scores, eval sets, summaries
```

Reproduce on one H100 (80GB). Total cost was about 3 GPU-hours and $70 of Claude API.
```
python src/prepare_data.py && python src/collect.py
python src/e1_weak_ablation.py Qwen/Qwen3-4B
python src/evalsets.py
python src/explain.py --batch && python src/simulate.py --batch
python src/align_check.py && python src/score.py && python src/corr_fix.py
python src/analyze.py && python src/make_figs.py
```
Environment variables: `DATA_DIR`, `OUT_DIR`, `ANTHROPIC_API_KEY`, and `CLAUDE_BUDGET_USD` (a hard spend stop, default 120).
