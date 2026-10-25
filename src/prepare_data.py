"""Tokenize FineWeb into 64-token windows (one random window per document).
Splits (disjoint documents): mine (activation mining / explanation examples),
e1 (ablation eval), e3 (simulation eval, shared by all neurons)."""
import os
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from common import DATA, SEQ

N = dict(mine=32768, e1=256, e3=128)
rng = np.random.default_rng(0)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
path = hf_hub_download("HuggingFaceFW/fineweb", "sample/10BT/000_00000.parquet", repo_type="dataset")
texts = pq.read_table(path, columns=["text"]).slice(0, 60000).column("text").to_pylist()
rng.shuffle(texts)

wins = []
for t in texts:
    ids = tok(t[:4000]).input_ids
    if len(ids) < SEQ:
        continue
    s = rng.integers(0, len(ids) - SEQ + 1)
    wins.append(ids[s:s + SEQ])
    if len(wins) == sum(N.values()):
        break
wins = np.array(wins, dtype=np.int32)
os.makedirs(DATA, exist_ok=True)
o = 0
for k, n in N.items():
    np.save(f"{DATA}/{k}.npy", wins[o:o + n]); o += n
    print(k, wins[o - n:o].shape)
