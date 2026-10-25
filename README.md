# sim-kl: testing "neural networks are already clean" with simulation KL

A small experiment that takes the two sides of a recent exchange and runs the test one of them proposed.

- Nick Cammarata, [Neural networks are already clean](https://already-clean.goodfire.com/) (Goodfire): look carefully and most neurons have one main job; polysemanticity was mostly weak tools; weak activations "usually don't matter."
- Leo Gao, [Neural networks are not already clean](https://www.lesswrong.com/posts/Dkh7pXKjoNWkkxQDz/neural-networks-are-not-already-clean) (LessWrong): activating-example vibes can mislead (precision is not recall, magnitudes matter). The test is whether you can **simulate** the neuron, substitute the simulation into the model, and not degrade it, measured by **downstream KL on a large dataset**.

This repo runs that test on Qwen3-4B (same family as Nick's Qwen3-32B audit), plus the weak-activation ablation that Leo flagged as the place their results diverge.

## Experiments

**E1: weak-activation ablation** (`src/e1_weak_ablation.py`). For every MLP neuron in every layer, zero activations below a per-neuron, per-pole threshold (weakest q of each pole, or below f x the pole max). Measure KL to the clean model, CE, top-1 agreement, next-token accuracy on held-out FineWeb. Qwen3 0.6B / 1.7B / 4B (8B pending). Nick's analogous ConvNeXt result: removing the weakest 75% changed 72% -> 71% top-1, about 1 in 10 answers.

**E3: simulation KL** (`src/collect.py, evalsets.py, explain.py, simulate.py, score.py`).
1. 280 random neurons (28 in each of layers 0, 4, ..., 32, 35) of Qwen3-4B. A neuron is one coordinate of the SwiGLU hidden state (input to `down_proj`); it is signed, so it has two poles.
2. Activations recorded on 2.1M tokens of FineWeb (64-token windows).
3. **Explainer**: Claude Opus 5.5 sees top / mid / weak examples of both poles plus random windows (even-indexed windows only) and writes a Cammarata-style explanation with a simulation rubric, and rates how monosemantic the neuron looks (1-5).
4. **Simulator**: Claude Sonnet 5.5 sees only the explanation and predicts every token's activation (-10..10) on 160 windows per neuron: 128 shared random windows + 32 held-out (odd-indexed) windows where the neuron is active, shuffled together.
5. **Scoring**: substitute values into the neuron and measure KL(clean || patched) per window; windows are reweighted by stratum (peak |a| < 0.1M, 0.1-0.4M, >= 0.4M) to their frequency in web text, giving an unbiased estimate of KL on ordinary text with enough rare-firing windows to measure.

Substitutions compared, per neuron:

| name | what replaces the neuron |
|---|---|
| `mean` | its mean activation (reference: total loss of the neuron) |
| `zero` | 0 |
| `strong20` / `strong50` | the **true** activation where \|a\| >= 0.2 / 0.5 x max, else 0 (an oracle for "explain only the strong firings perfectly") |
| `sim_raw` | simulator value / 10 x max |
| `sim_cal` | simulator value, linearly calibrated (fit on even windows, applied to odd and vice versa) |

Headline: **recovered = 1 - KL(x) / KL(mean)**. 1 means a perfect stand-in; 0 means no better than deleting the neuron.

Simulator-strength check: a random 40-neuron subset re-simulated by Opus 5.5 (medium effort) instead of Sonnet (low effort).

## Reproduce

1x H100 80GB, ~3-4 h total; Claude API ~$55 (Batch API).
```
python src/prepare_data.py && python src/collect.py
python src/e1_weak_ablation.py Qwen/Qwen3-4B
python src/evalsets.py
python src/explain.py --batch && python src/simulate.py --batch && python src/score.py
python src/corr_fix.py
```
Env: `DATA_DIR`, `OUT_DIR`, `ANTHROPIC_API_KEY`, `CLAUDE_BUDGET_USD` (hard stop, default 120).

## Known limitations

- One explainer and one simulator. A weak simulator understates explanation quality; the Opus subset bounds this.
- Explanations are automated (Opus, about 60 examples), not the hours of human or agent investigation Nick's audit used. This tests explanations of the kind auto-interp produces, which are better than the 2023 baselines but not Nick's best.
- 64-token windows, so long-context features are under-represented.
- Single-neuron KL effects are tiny in absolute terms (about 1e-6 to 1e-5 nats); everything runs in fp32 with TF32 off, and comparisons are always relative to the same neuron's mean ablation.
- E1 ablates every layer at once; errors compound across 36 layers.
