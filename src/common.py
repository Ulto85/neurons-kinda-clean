"""Shared utilities: model loading, data, MLP-neuron hooks, KL evaluation.

A "neuron" is one coordinate of the SwiGLU MLP hidden state, i.e. the input to
down_proj: act(gate(x)) * up(x). It is signed, so each neuron has a positive and
a negative pole (following Cammarata's framing).
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA = os.environ.get("DATA_DIR", "/workspace/data")
OUT = os.environ.get("OUT_DIR", "/workspace/results")
SEQ = 64


def load_model(name, dtype=torch.float32):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).cuda().eval()
    return tok, model


def mlp_layers(model):
    return [layer.mlp for layer in model.model.layers]


class NeuronPatcher:
    """Pre-hooks on down_proj. `fns[layer] = f(a, batch_slice) -> a'` edits the
    hidden activation a: [B, T, d_mlp]. Optional `capture[layer] = idx` stores
    a[..., idx] on every forward."""

    def __init__(self, model):
        self.mlps = mlp_layers(model)
        self.fns, self.capture, self.captured = {}, {}, {}
        self.handles = [m.down_proj.register_forward_pre_hook(self._mk(i)) for i, m in enumerate(self.mlps)]

    def _mk(self, i):
        def hook(mod, args):
            a = args[0]
            if i in self.capture:
                self.captured.setdefault(i, []).append(a[..., self.capture[i]].detach().float().cpu())
            if i in self.fns:
                return (self.fns[i](a),)
        return hook

    def clear(self):
        self.fns, self.capture, self.captured = {}, {}, {}

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def logprobs(model, ids, bs=64):
    out = []
    for i in range(0, len(ids), bs):
        x = torch.as_tensor(ids[i:i + bs]).cuda()
        out.append(F.log_softmax(model(x).logits.float(), -1))
    return torch.cat(out)


@torch.no_grad()
def compare(model, ids, clean_lp, bs=64):
    """Run model (with whatever hooks are active) and compare to clean log-probs.
    Returns mean KL(clean || patched), CE delta on next token, top-1 agreement,
    next-token accuracy (patched), all averaged over positions."""
    kls, ce_p, agree, acc = [], [], [], []
    for i in range(0, len(ids), bs):
        x = torch.as_tensor(ids[i:i + bs]).cuda()
        lp = F.log_softmax(model(x).logits.float(), -1)
        c = clean_lp[i:i + bs]
        kls.append((c.exp() * (c - lp)).sum(-1).flatten())
        tgt = x[:, 1:]
        ce_p.append(-lp[:, :-1].gather(-1, tgt[..., None]).squeeze(-1).flatten())
        agree.append((lp.argmax(-1) == c.argmax(-1)).float().flatten())
        acc.append((lp[:, :-1].argmax(-1) == tgt).float().flatten())
    return dict(kl=torch.cat(kls).mean().item(), ce=torch.cat(ce_p).mean().item(),
                agree=torch.cat(agree).mean().item(), acc=torch.cat(acc).mean().item())


def clean_ce(ids, clean_lp):
    x = torch.as_tensor(ids).cuda()
    return -clean_lp[:, :-1].gather(-1, x[:, 1:, None]).mean().item()


def load_ids(split):
    return np.load(f"{DATA}/{split}.npy")
