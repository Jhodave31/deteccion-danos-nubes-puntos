"""Prueba rápida de velocidad y memoria de PTv3 (no forma parte del protocolo)."""
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from train_ptv3 import BlockDataset, Segmentor, load_clouds, make_batch, model_defaults

name = sys.argv[1]
max_pts = int(sys.argv[2]) if len(sys.argv) > 2 else 16000
bs = int(sys.argv[3]) if len(sys.argv) > 3 else 4
scale, grid = model_defaults(name)
clouds = load_clouds("../data/processed", {"Area_7"})
ds = BlockDataset(clouds, 12.0, max_pts, 100)
model = Segmentor(name, 5).cuda()
print(name, "params", sum(p.numel() for p in model.parameters()) / 1e6, "M", flush=True)
opt = torch.optim.AdamW(model.parameters(), 1e-4)
for it in range(int(sys.argv[4]) if len(sys.argv) > 4 else 8):
    items = [ds[i] for i in range(bs)]
    torch.cuda.synchronize()
    t = time.time()
    point, keep = make_batch([(a, b, c) for a, b, c, _ in items], scale, grid, "cuda")
    lab = torch.cat([torch.from_numpy(l).cuda()[first] for (_, _, _, l), (first, _) in zip(items, keep)])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(point)
    loss = F.cross_entropy(logits.float(), lab)
    opt.zero_grad()
    loss.backward()
    opt.step()
    torch.cuda.synchronize()
    print(f"it {it}: {time.time() - t:.2f}s pts={len(lab)} loss={loss.item():.3f}", flush=True)
print("mem GB", torch.cuda.max_memory_allocated() / 1e9)
