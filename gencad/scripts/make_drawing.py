"""Dimensioned engineering drawings of our parts with SolidCraft (https://github.com/tangency-labs/solidcraft), with the
list of every dimension placed (its kind and true value), for drawing -> CAD training data and dimension-reading tests.

Per part: the STEP is inserted into a SolidCraft design (`file.insert_step`, coordinates kept), a drawing with the
standard views (front, top, right, isometric) is created, then dimensions chosen from the B-rep are added with
`drawing.dimension.add` by 3D points: overall X / Y / Z sizes, each distinct hole / boss diameter on the view that sees
it face on and its centre's position from the part's corner, and a few arc radii. A dimension SolidCraft refuses is
dropped and the rest retried. The sheet is exported as PNG.

  python scripts/make_drawing.py --steps data/rich/real/0/step --limit 40 --cli <solidcraft-cli> --out data/dimprobe
writes <out>/<name>.png and <out>/dims.json ({name: [{kind, value, view}, ...]}).
"""
import argparse
import json
import os
import random
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mm.sysutil import usable_cpus  # noqa: E402

# view label -> (axis it looks along, horizontal axis, vertical axis) in SolidCraft's standard views
VIEWS = {"Front": (1, 0, 2), "Above Front": (2, 0, 1), "Right of Front": (0, 1, 2)}


def _run(cli, cmds, tmp, tag, extra=()):
    js, out = os.path.join(tmp, f"{tag}.json"), os.path.join(tmp, f"{tag}.out")
    json.dump({"commands": cmds}, open(js, "w"))
    p = subprocess.run([cli, "run", js, "--results", "--quiet", "--json-out", out, *extra],
                       capture_output=True, text=True, timeout=300)
    try:
        o = json.load(open(out))
    except Exception:  # noqa: BLE001
        o = {"error": {"step": -1, "error": (p.stderr or p.stdout)[-300:]}}
    return o


def choose_dims(shape, rng, max_circles=8, max_arcs=3):
    """-> list of (view label, kind params, record) chosen from the B-rep of a cadquery shape."""
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.GeomAbs import GeomAbs_Circle
    verts = np.array([v.toTuple() for v in shape.Vertices()])
    if len(verts) < 2:
        return []
    lo, hi = verts.min(0), verts.max(0)
    out = []

    def corner(axis, side):                       # a vertex at the min / max of one axis (nearest the box corner)
        sel = verts[np.isclose(verts[:, axis], (lo if side == 0 else hi)[axis], atol=1e-6)]
        return sel[np.argmin(np.linalg.norm(sel - lo, axis=1))]

    for axis, (view, d) in {0: ("Front", "horizontal"), 2: ("Front", "vertical"), 1: ("Right of Front", "horizontal")
                            }.items():
        if hi[axis] - lo[axis] > 1e-3:
            out.append((view, {"linear": {"a": corner(axis, 0).tolist(), "b": corner(axis, 1).tolist(),
                                          "direction": d}},
                        {"kind": "overall", "axis": "xyz"[axis], "value": float(hi[axis] - lo[axis])}))
    circles, arcs, seen = [], [], set()
    for e in shape.Edges():
        ad = BRepAdaptor_Curve(e.wrapped)
        if ad.GetType() != GeomAbs_Circle:
            continue
        c = ad.Circle()
        ax = np.array(c.Axis().Direction().Coord())
        k = int(np.argmax(np.abs(ax)))
        if abs(abs(ax[k]) - 1) > 1e-6:
            continue                                     # not parallel to a view direction
        cen = np.array(c.Location().Coord())
        view = next(v for v, (look, _, _) in VIEWS.items() if look == k)
        pt = e.positionAt(0.5).toTuple()
        key = (view, tuple(np.round(np.delete(cen, k), 1)))   # one diameter per centre: concentric labels overlap
        if key in seen:
            continue
        seen.add(key)
        (circles if e.IsClosed() else arcs).append((view, k, cen, c.Radius(), pt))
    rng.shuffle(circles)
    rng.shuffle(arcs)
    for view, k, cen, r, pt in circles[:max_circles]:
        out.append((view, {"radial": {"edge": list(pt), "diameter": True}},
                    {"kind": "diameter", "value": 2 * r}))
        _, h, v = VIEWS[view]
        for ax, d in ((h, "horizontal"), (v, "vertical")):
            a = corner(ax, 0)
            if abs(cen[ax] - a[ax]) > 0.5:
                out.append((view, {"linear": {"a": a.tolist(), "b": {"centre": list(pt)}, "direction": d}},
                            {"kind": "hole_position", "axis": "xyz"[ax], "value": float(abs(cen[ax] - a[ax]))}))
    for view, k, cen, r, pt in arcs[:max_arcs]:
        out.append((view, {"radial": {"edge": list(pt)}}, {"kind": "radius", "value": r}))
    return out


def make(args):
    step, name, cli, out_dir, seed, dpi = args
    import cadquery as cq
    rng = random.Random(seed)
    shape = cq.importers.importStep(step).val()
    dims = choose_dims(shape, rng)
    with tempfile.TemporaryDirectory() as tmp:
        sc = os.path.join(tmp, "d.sc")
        o = _run(cli, [{"command": "file.insert_step", "params": {"path": os.path.abspath(step)}},
                       {"command": "drawing.create", "params": {"standard_views": True, "units": "mm",
                                                                "projection": rng.choice(["first", "third"])}}],
                 tmp, "p1", ("--save", sc))
        if isinstance(o, dict):
            return {"name": name, "error": str(o.get("error"))[:200]}
        dr = o[-1]
        sheet = dr["sheets"][0]
        vid = {}                                    # our view names, whatever the projection angle calls them
        for v in sheet["views"]:
            if v["kind"] == "base" and not v["label"].lower().startswith("iso"):
                vid["Front"] = v["id"]
            elif v.get("side") in ("above", "below"):
                vid["Above Front"] = v["id"]
            elif v.get("side") in ("left", "right"):
                vid["Right of Front"] = v["id"]
        keep = [d for d in dims if d[0] in vid]
        png = os.path.abspath(os.path.join(out_dir, f"{name}.png"))
        for _ in range(12):                          # drop refused dimensions one at a time
            cmds = [{"command": "drawing.open", "params": {"drawing": dr["id"]}}]
            cmds += [{"command": "drawing.dimension.add", "params": {"view": vid[v], "kind": kind, "decimals": 1}}
                     for v, kind, _ in keep]
            cmds += [{"command": "drawing.export", "params": {"format": "png", "path": png, "dpi": dpi,
                                                              "sheet": sheet["name"]}}]
            o = _run(cli, cmds, tmp, "p2", ("--in", sc))
            if not isinstance(o, dict):
                break
            st = o.get("error", {}).get("step", -1)       # 1-based; step 1 is drawing.open
            if not (2 <= st <= len(keep) + 1):
                return {"name": name, "error": str(o.get("error"))[:200]}
            keep.pop(st - 2)
        else:
            return {"name": name, "error": "too many refused dimensions"}
    return {"name": name, "dims": [{**rec, "view": v} for v, _, rec in keep], "n_tried": len(dims)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", required=True, help="folder of STEP files")
    ap.add_argument("--cli", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dpi", type=int, default=188, help="A3 sheet: 188 dpi ~ 3100 px wide")
    ap.add_argument("--workers", type=int, default=usable_cpus())
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from pebble import ProcessPool
    os.makedirs(a.out, exist_ok=True)
    files = sorted(f for f in os.listdir(a.steps) if f.lower().endswith((".step", ".stp")))
    random.Random(a.seed).shuffle(files)
    files = files[:a.limit]
    res = {}
    with ProcessPool(max_workers=a.workers) as pool:
        futs = [(f, pool.schedule(make, args=((os.path.join(a.steps, f), os.path.splitext(f)[0], a.cli, a.out,
                                                a.seed + i, a.dpi),), timeout=900)) for i, f in enumerate(files)]
        for f, fu in futs:
            try:
                r = fu.result()
            except Exception as e:  # noqa: BLE001
                r = {"name": f, "error": str(e)[:200]}
            res[r["name"]] = r
    ok = {k: v["dims"] for k, v in res.items() if "dims" in v}
    json.dump(ok, open(os.path.join(a.out, "dims.json"), "w"), indent=0)
    n = [len(v) for v in ok.values()]
    print(f"[draw] {len(ok)}/{len(res)} drawings, dims per drawing median {np.median(n) if n else 0}; errors: "
          f"{[v['error'][:80] for v in res.values() if 'error' in v][:5]}", flush=True)


if __name__ == "__main__":
    main()
