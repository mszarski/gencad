"""Robustness test sets: re-render rich/real test parts with inputs that differ from our training renders
(scripts/render_views.py: 4 fixed orthographic cameras, grey Lambert shading with a camera-fixed light, dark B-rep
edges, white background, 448 px).

Variants (each a copy of the part list with its own renders, same GT programs):
  cam15 / cam30  every camera direction rotated by a random angle of up to 15 / 30 degrees (seeded per part)
  light          light from another side, darker shading range, light-grey background
  color          tinted (steel-blue) shading on an off-white background, edges a darker tint
  noedge         no B-rep edge lines (shading only)
  lowres         rendered at 448 px, then down to 160 px and back up (blurry)

  python scripts/render_variants.py --src data/rich/real --out data/rich/robust --variants cam15 cam30 light color noedge lowres
Writes <out>/<variant>/<shard>/{parts.parquet, renders/<uid>_{0..3}.png}; score with predict_vllm.py --rich and
rich_iou.py (own IoU), paired against the same parts rendered normally.
"""
import argparse
import os
import shutil
import sys
import zlib

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mm.sysutil import usable_cpus  # noqa: E402

VARIANTS = ["cam15", "cam30", "light", "color", "noedge", "lowres"]


def rotate_dir(d, max_deg, rng):
    """Rotate direction d by a random angle up to max_deg about a random axis perpendicular to it."""
    d = d / np.linalg.norm(d)
    a = rng.normal(size=3)
    a -= a.dot(d) * d
    a /= np.linalg.norm(a)
    t = np.radians(rng.uniform(0.5, 1.0) * max_deg)
    return d * np.cos(t) + np.cross(a, d) * np.sin(t)


def render(tris, fid, d, size=448, margin=0.06, light=(0.4, 0.7), shade=(0.45, 0.5), bg=255, edges=True, tint=None):
    """scripts/render_views.render with knobs: light offset (u, v), shade (ambient, diffuse), background, edges, tint."""
    from scripts.render_views import rasterize
    d = d / np.linalg.norm(d)
    up = np.array([0, 0, 1.0]) if abs(d[2]) < 0.99 else np.array([0, 1.0, 0])
    u = np.cross(up, d)
    u /= np.linalg.norm(u)
    v = np.cross(d, u)
    c = (tris.reshape(-1, 3).max(0) + tris.reshape(-1, 3).min(0)) / 2
    radius = np.linalg.norm(tris.reshape(-1, 3) - c, axis=1).max()
    T = tris - c
    s = size * (1 - 2 * margin) / (2 * radius)
    P = np.stack([T @ u * s + size / 2, -(T @ v) * s + size / 2], -1)
    tid = rasterize(P, T @ d, size)
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    n[(n @ d) < 0] *= -1
    lt = d + light[0] * u + light[1] * v
    lt /= np.linalg.norm(lt)
    shade_v = np.clip(shade[0] + shade[1] * np.clip(n @ lt, 0, 1), 0, 1)
    g = np.where(tid >= 0, shade_v[tid], 0.0)
    f = np.where(tid >= 0, fid[tid] + 1, 0)
    edge = np.zeros_like(f, bool)
    edge[:, 1:] |= f[:, 1:] != f[:, :-1]
    edge[1:, :] |= f[1:, :] != f[:-1, :]
    col = np.array(tint if tint else (230, 230, 230), float)
    img = np.where((tid >= 0)[..., None], g[..., None] * col, np.array(bg if isinstance(bg, tuple) else (bg,) * 3,
                                                                                float))
    if edges:
        img[edge] = (np.array(tint) * 0.2) if tint else 40
    return Image.fromarray(img.clip(0, 255).astype(np.uint8))


def work(args):
    step, stems, size = args
    import cadquery as cq
    from scripts.render_views import DIRS, tessellate_cq
    try:
        shape = cq.importers.importStep(step).solids().vals()[0]
        tris, fid = tessellate_cq(shape)
    except Exception as e:  # noqa: BLE001
        return f"fail {type(e).__name__}"
    rng = np.random.default_rng(zlib.crc32(step.encode()))
    for var, stem in stems.items():
        os.makedirs(os.path.dirname(stem), exist_ok=True)
        for k, d0 in enumerate(DIRS):
            d = d0.copy()
            kw = {}
            if var in ("cam15", "cam30"):
                d = rotate_dir(d0, 15 if var == "cam15" else 30, rng)
            elif var == "light":
                kw = {"light": (-0.8, -0.3), "shade": (0.25, 0.65), "bg": 235}
            elif var == "color":
                kw = {"tint": (110, 150, 205), "bg": (248, 246, 240)}
            elif var == "noedge":
                kw = {"edges": False}
            im = render(tris, fid, d, size=size, **kw)
            if var == "lowres":
                im = im.resize((160, 160), Image.BILINEAR).resize((size, size), Image.BILINEAR)
            im.save(f"{stem}_{k}.png", optimize=True)
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/rich/real")
    ap.add_argument("--out", default="data/rich/robust")
    ap.add_argument("--variants", nargs="*", default=VARIANTS)
    ap.add_argument("--size", type=int, default=448)
    ap.add_argument("--workers", type=int, default=usable_cpus())
    a = ap.parse_args()
    import re
    import pandas as pd
    from pebble import ProcessPool
    jobs = []
    for sh in sorted(os.listdir(a.src)):
        pq = os.path.join(a.src, sh, "parts.parquet")
        if not os.path.exists(pq):
            continue
        df = pd.read_parquet(pq)
        for var in a.variants:
            os.makedirs(os.path.join(a.out, var, sh), exist_ok=True)
            shutil.copy(pq, os.path.join(a.out, var, sh, "parts.parquet"))
        for u, code in zip(df.uid, df.code):
            step = re.search(r'importStep\("([^"]+)"\)', code).group(1)
            jobs.append((step, {v: os.path.join(a.out, v, sh, "renders", u) for v in a.variants}, a.size))
    print(f"[variants] {len(jobs)} parts x {a.variants}", flush=True)
    res = []
    with ProcessPool(max_workers=a.workers) as pool:
        futs = [pool.schedule(work, args=(j,), timeout=600) for j in jobs]
        for fu in futs:
            try:
                res.append(fu.result())
            except Exception as e:  # noqa: BLE001
                res.append(f"fail {type(e).__name__}")
    print(f"[variants] done: {sum(r == 'ok' for r in res)} ok, {sum(r != 'ok' for r in res)} failed", flush=True)


if __name__ == "__main__":
    main()
