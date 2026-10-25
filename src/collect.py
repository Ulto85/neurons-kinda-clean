"""E3 step 1: sample neurons and record their activations on the mining set
(for building explanations) and on the shared e3 eval set (for scoring)."""
import json
import numpy as np
import torch
from common import *

name = "Qwen/Qwen3-4B"
torch.backends.cuda.matmul.allow_tf32 = False
tok, model = load_model(name)
L, d = len(model.model.layers), model.config.intermediate_size
LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 32, 35]
PER = 28
rng = np.random.default_rng(1)
neurons = [(l, int(n)) for l in LAYERS for n in sorted(rng.choice(d, PER, replace=False))]
P = NeuronPatcher(model)


def record(ids, bs=128):
    P.clear()
    for l in LAYERS:
        P.capture[l] = torch.tensor([n for (ll, n) in neurons if ll == l]).cuda()
    with torch.no_grad():
        for b in range(0, len(ids), bs):
            model(torch.as_tensor(ids[b:b + bs]).cuda())
    acts = torch.cat([torch.cat(P.captured[l]) for l in LAYERS], -1)  # [W, T, K], same order as neurons
    P.clear()
    return acts.numpy()  # float32: some Qwen neurons are huge


os.makedirs(OUT, exist_ok=True)
np.save(f"{OUT}/e3_acts_eval.npy", record(load_ids("e3")))
np.save(f"{OUT}/e3_acts_mine.npy", record(load_ids("mine")))
json.dump(neurons, open(f"{OUT}/e3_neurons.json", "w"))
print("done", len(neurons))
