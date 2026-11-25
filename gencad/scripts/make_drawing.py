"""Dimensioned engineering drawings of our parts with SolidCraft (https://github.com/tangency-labs/solidcraft), with the
list of every dimension placed (its kind and true value), for drawing -> CAD training data and dimension-reading tests.

Per part: the STEP is inserted into a SolidCraft design (`file.insert_step`, coordinates kept), a drawing with the
standard views (front, top, right, isometric) is created, then dimensions chosen from the B-rep are added with
`drawing.dimension.add` by 3D points: overall X / Y / Z sizes between the part's true extreme points, levels of flat
faces, every distinct concentric diameter on the view that sees it face on and each centre's position from the part's
edge, and a few arc radii. A dimension SolidCraft refuses is
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


def _edge_points(shape, n=24):
    """Points sampled along every edge (B-rep vertices alone miss the extremes of round parts)."""
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.GeomAbs import GeomAbs_Circle
    pts = []
    for e in shape.Edges():
        try:
            ad = BRepAdaptor_Curve(e.wrapped)
            if ad.GetType() == GeomAbs_Circle:
                c = ad.Circle()
                cen, r = np.array(c.Location().Coord()), c.Radius()
                nrm = np.array(c.Axis().Direction().Coord())
                if e.IsClosed():                     # exact extremes of a full circle along each axis
                    for j in range(3):
                        u = np.eye(3)[j] - nrm[j] * nrm
                        if np.linalg.norm(u) > 1e-9:
                            u = u / np.linalg.norm(u)
                            pts += [tuple(cen + r * u), tuple(cen - r * u)]
                    pts += [e.positionAt(t).toTuple() for t in np.linspace(0, 1, n)]
                else:
                    pts += [e.positionAt(t).toTuple() for t in np.linspace(0, 1, 4 * n)]
            else:
                pts += [e.positionAt(t).toTuple() for t in np.linspace(0, 1, n)]
        except Exception:  # noqa: BLE001
            pass
    return np.array(pts)


def choose_dims(shape, rng, max_circles=8, max_arcs=3, max_levels=2, max_per_centre=3):
    """-> list of (view label, kind params, record) chosen from the B-rep of a cadquery shape.

    Overall sizes are measured between the part's true extreme points (sampled along the edges, so round parts get
    their real width), levels come from flat faces square to an axis, and every concentric diameter (outer, step, bore;
    up to max_per_centre) is dimensioned."""
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.GeomAbs import GeomAbs_Circle
    pts = _edge_points(shape)
    verts = np.array([v.toTuple() for v in shape.Vertices()])
    if len(pts) < 2 or len(verts) < 1:
        return []
    lo, hi = pts.min(0), pts.max(0)                  # the part's true extent (edges sampled, circle extremes exact)
    span = hi - lo
    tol = max(1e-4 * float(span.max()), 1e-6)
    out = []

    def corner(axis, side):                       # a B-rep vertex at the true min / max of one axis (SolidCraft's
        sel = verts[np.abs(verts[:, axis] - (lo if side == 0 else hi)[axis]) <= tol]    # linear dims need corners)
        return sel[np.argmin(np.linalg.norm(sel - lo, axis=1))] if len(sel) else None

    big = {}                                       # per axis: a full circle seen edge-on whose diameter is the span
    for e in shape.Edges():
        ad = BRepAdaptor_Curve(e.wrapped)
        if ad.GetType() != GeomAbs_Circle or not e.IsClosed():
            continue
        c = ad.Circle()
        nrm = np.array(c.Axis().Direction().Coord())
        k = int(np.argmax(np.abs(nrm)))
        if abs(abs(nrm[k]) - 1) > 1e-6:
            continue
        for axis in range(3):
            if axis != k and abs(2 * c.Radius() - span[axis]) <= 2 * tol:
                big.setdefault(axis, (k, e.positionAt(0.5).toTuple()))

    cyl = {}                                       # per axis, per side: a cylindrical face whose outline is the extreme
    for f in shape.Faces():
        if f.geomType() != "CYLINDER":
            continue
        try:
            from OCP.BRepAdaptor import BRepAdaptor_Surface
            cy = BRepAdaptor_Surface(f.wrapped).Cylinder()
            nrm = np.array(cy.Axis().Direction().Coord())
            cen, r = np.array(cy.Location().Coord()), cy.Radius()
            fp = f.positionAt(0.5, 0.5).toTuple()
        except Exception:  # noqa: BLE001
            continue
        k = int(np.argmax(np.abs(nrm)))
        if abs(abs(nrm[k]) - 1) > 1e-6:
            continue
        for axis in range(3):
            if axis == k:
                continue
            if abs(cen[axis] - r - lo[axis]) <= 2 * tol:
                cyl.setdefault((axis, 0), (k, fp))
            if abs(cen[axis] + r - hi[axis]) <= 2 * tol:
                cyl.setdefault((axis, 1), (k, fp))

    axes = {0: ("Front", "horizontal"), 2: ("Front", "vertical"), 1: ("Right of Front", "horizontal")}
    for axis, (view, d) in axes.items():
        if span[axis] <= 1e-3:
            continue
        a, b = corner(axis, 0), corner(axis, 1)
        rec = {"kind": "overall", "axis": "xyz"[axis], "value": float(span[axis])}
        if a is not None and b is not None:
            out.append((view, {"linear": {"a": a.tolist(), "b": b.tolist(), "direction": d}}, rec))
        elif axis in big:                          # a round part: the edge-on circle's line, in a view that sees it so
            k, pt = big[axis]
            v2 = next((v for v, (look, h, vv) in VIEWS.items() if look != k and look != axis and axis in (h, vv)), None)
            if v2:
                dd = "horizontal" if VIEWS[v2][1] == axis else "vertical"
                out.append((v2, {"length": {"edge": list(pt), "direction": dd}}, rec))
        elif (axis, 0) in cyl and (axis, 1) in cyl and cyl[(axis, 0)][0] == cyl[(axis, 1)][0]:
            k = cyl[(axis, 0)][0]                  # rounded ends: between the two end cylinders' far outlines
            v2 = next((v for v, (look, h, vv) in VIEWS.items() if look != k and look != axis and axis in (h, vv)), None)
            if v2 and cyl[(axis, 0)][1] != cyl[(axis, 1)][1]:
                dd = "horizontal" if VIEWS[v2][1] == axis else "vertical"
                out.append((v2, {"silhouettes": {"a": {"face": list(cyl[(axis, 0)][1]), "side": "far"},
                                                 "b": {"face": list(cyl[(axis, 1)][1]), "side": "far"},
                                                 "direction": dd}}, rec))
    levels = {0: set(), 1: set(), 2: set()}           # flat faces square to an axis, strictly inside the part
    for f in shape.Faces():
        if f.geomType() != "PLANE":
            continue
        try:
            nrm = np.array(f.normalAt().toTuple())
            c = np.array(f.Center().toTuple())
        except Exception:  # noqa: BLE001
            continue
        k = int(np.argmax(np.abs(nrm)))
        if abs(abs(nrm[k]) - 1) < 1e-6 and min(c[k] - lo[k], hi[k] - c[k]) > 0.02 * span[k]:
            levels[k].add(round(float(c[k]), 3))
    for axis, (view, d) in axes.items():
        L_ = sorted(levels[axis])
        rng.shuffle(L_)
        for L in L_[:max_levels]:
            on = verts[np.abs(verts[:, axis] - L) <= max(tol, 1e-3)]
            a = corner(axis, 0)
            if not len(on) or a is None:
                continue
            b = on[np.argmin(np.linalg.norm(on - lo, axis=1))]
            out.append((view, {"linear": {"a": a.tolist(), "b": b.tolist(), "direction": d}},
                        {"kind": "level", "axis": "xyz"[axis], "value": float(L - lo[axis])}))
    circles, arcs, seen = [], [], {}
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
        key = (view, tuple(np.round(np.delete(cen, k), 1)))
        radii = seen.setdefault(key, set())
        r = round(c.Radius(), 2)
        if r in radii or len(radii) >= max_per_centre:   # each distinct diameter at a centre, a few per centre
            continue
        radii.add(r)
        (circles if e.IsClosed() else arcs).append((view, k, cen, c.Radius(), pt, len(radii) == 1))
    main = {}                                      # per view: the largest full circle (datum for round parts)
    for view, k, cen, r, pt, first in circles:
        if view not in main or r > main[view][1]:
            main[view] = (cen, r, pt)
    rng.shuffle(circles)
    rng.shuffle(arcs)
    groups = {}
    for view, k, cen, r, pt, first in circles:
        groups.setdefault((view, round(r, 2)), []).append(cen)
    called, called_n = set(), {}
    for view, k, cen, r, pt, first in circles[:max_circles]:
        n_same = len(groups[(view, round(r, 2))])
        if n_same > 1:                             # repeated holes: one "4X Ø6.0" callout, positions of two
            if (view, round(r, 2)) in called and called_n.get((view, round(r, 2)), 0) >= 2:
                continue
            if (view, round(r, 2)) not in called:
                called.add((view, round(r, 2)))
                out.append((view, {"radial": {"edge": list(pt), "diameter": True}},
                            {"kind": "diameter", "value": 2 * r, "count": n_same}))
            called_n[(view, round(r, 2))] = called_n.get((view, round(r, 2)), 0) + 1
        else:
            out.append((view, {"radial": {"edge": list(pt), "diameter": True}},
                        {"kind": "diameter", "value": 2 * r}))
        if not first:
            continue                                     # position once per centre
        _, h, v = VIEWS[view]
        for ax, d in ((h, "horizontal"), (v, "vertical")):
            a = corner(ax, 0)
            if a is not None:
                ref, aa, kind = a[ax], a.tolist(), "hole_position"
            elif view in main and not np.allclose(main[view][0], cen):
                ref, aa, kind = main[view][0][ax], {"centre": list(main[view][2])}, "hole_centre_distance"
            else:
                continue
            if abs(cen[ax] - ref) > 0.5:
                out.append((view, {"linear": {"a": aa, "b": {"centre": list(pt)}, "direction": d}},
                            {"kind": kind, "axis": "xyz"[ax], "value": float(abs(cen[ax] - ref))}))
    radii = [round(r, 1) for *_, r, _, _ in arcs]
    common = max(set(radii), key=radii.count) if radii else None
    if common is not None and radii.count(common) >= 3 and rng.random() < 0.7:
        out.append(("NOTE", f"ALL RADII {common:g} UNLESS OTHERWISE MARKED", {"kind": "note_radius", "value": common}))
        arcs = [a for a in arcs if round(a[3], 1) != common]
    for view, k, cen, r, pt, _ in arcs[:max_arcs]:
        out.append((view, {"radial": {"edge": list(pt)}}, {"kind": "radius", "value": r}))
    return out


def _slots(keep):
    """-> per dimension its index among the same-view dims of its family (linear: per direction; radial)."""
    seen, out = {}, []
    for v, kind, _ in keep:
        fam = (v, "radial") if "radial" in kind else (v, next(iter(kind.values())).get("direction", "aligned"))
        out.append(seen.get(fam, 0))
        seen[fam] = seen.get(fam, 0) + 1
    return out


def _place(view_id, kind, rec, i, decimals):
    """dimension.add params with staggered placement so labels do not collide, and "NX" callouts."""
    p = {"view": view_id, "kind": kind, "decimals": decimals}
    if "radial" in kind:                               # labels fanned out around the view, further out each time
        sx = 1 if i % 2 == 0 else -1
        p["text"] = {"at": [sx * (28 + 7 * (i // 2)), 22 + 9 * i]}
    else:
        p["offset_mm"] = 8 + 8 * i                     # parallel dims stacked outwards
    if rec.get("count", 1) > 1:
        p["override_text"] = f"{rec['count']}X \u00d8{rec['value']:.{decimals}f}"
    return p


def make(args):
    r = _make(args)
    if "collapsed curve" in str(r.get("error", "")):     # tiny features at the fitted scale: try the smallest sheet
        r = _make(args, paper="a4")
    return r


def _make(args, paper=None):
    src, name, cli, out_dir, seed, dpi = args
    import cadquery as cq
    rng = random.Random(seed)
    if src.lower().endswith((".step", ".stp")):
        step = src
        shape = cq.importers.importStep(step).val()
    else:                                             # CadQuery source: build it and write a STEP to draw from
        from mm.cadquery_io import run_cadquery
        shape = run_cadquery(src)
        if shape is None:
            return {"name": name, "error": "program does not build"}
        step = os.path.join(tempfile.gettempdir(), f"md_{os.getpid()}_{name}.step")
        cq.exporters.export(cq.Workplane().add(shape), step)
    dims = choose_dims(shape, rng)
    with tempfile.TemporaryDirectory() as tmp:
        sc = os.path.join(tmp, "d.sc")
        paper = paper or rng.choices(["a3", "a2", "a4"], weights=[0.5, 0.35, 0.15])[0]
        width_mm = {"a2": 594, "a3": 420, "a4": 297}[paper]
        dpi = dpi or int(round(2400 / (width_mm / 25.4)))            # ~2400 px wide whatever the sheet
        o = _run(cli, [{"command": "file.insert_step", "params": {"path": os.path.abspath(step)}},
                       {"command": "drawing.create", "params": {
                           "standard_views": True, "units": "mm", "projection": rng.choice(["first", "third"]),
                           "first_sheet_paper": {"size": {"standard": paper}, "orientation": "landscape"}}}],
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
        notes = [d for d in dims if d[0] == "NOTE"]
        vals = [rec["value"] for v, _, rec in dims if v != "NOTE"]
        decimals = 0 if all(abs(x - round(x)) < 0.01 for x in vals) else rng.choices([1, 2], [0.7, 0.3])[0]
        keep = [d for d in dims if d[0] in vid]
        png = os.path.abspath(os.path.join(out_dir, f"{name}.png"))
        for _ in range(12):                          # drop refused dimensions one at a time
            cmds = [{"command": "drawing.open", "params": {"drawing": dr["id"]}}]
            cmds += [{"command": "drawing.dimension.add", "params": _place(vid[v], kind, rec, slots, decimals)}
                     for (v, kind, rec), slots in zip(keep, _slots(keep))]
            cmds += [{"command": "drawing.note.add", "params": {
                "anchor": {"sheet": sheet["name"]}, "content": txt, "at": [25, 45 + 8 * j], "cap_mm": 3.5}}
                for j, (_, txt, _) in enumerate(notes)]
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
    return {"name": name, "dims": [{**rec, "view": v} for v, _, rec in keep + notes], "n_tried": len(dims),
            "paper": paper, "decimals": decimals}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default=None, help="folder of STEP files")
    ap.add_argument("--codes_parquet", default=None, help="or: parquet with uid, code (CadQuery programs)")
    ap.add_argument("--cli", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dpi", type=int, default=0, help="0: per sheet size, ~2400 px wide")
    ap.add_argument("--workers", type=int, default=usable_cpus())
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from pebble import ProcessPool
    os.makedirs(a.out, exist_ok=True)
    codes = {}
    if a.codes_parquet:
        import pandas as pd
        T = pd.read_parquet(a.codes_parquet)
        T = T.sample(frac=1.0, random_state=a.seed).head(a.limit) if a.limit else T
        items = [(str(u).replace("/", "_"), c) for u, c in zip(T.uid, T.code)]
        codes = dict(items)
    else:
        files = sorted(f for f in os.listdir(a.steps) if f.lower().endswith((".step", ".stp")))
        random.Random(a.seed).shuffle(files)
        items = [(os.path.splitext(f)[0], os.path.join(a.steps, f)) for f in files[:a.limit]]
    res = {}
    with ProcessPool(max_workers=a.workers, max_tasks=100) as pool:
        futs = [(n, pool.schedule(make, args=((src, n, a.cli, a.out, a.seed + i, a.dpi),), timeout=600))
                for i, (n, src) in enumerate(items)]
        for f, fu in futs:
            try:
                r = fu.result()
            except Exception as e:  # noqa: BLE001
                r = {"name": f, "error": str(e)[:200]}
            res[r["name"]] = r
    ok = {k: v["dims"] for k, v in res.items() if "dims" in v}
    dec = {k: v.get("decimals", 1) for k, v in res.items() if "dims" in v}
    json.dump(ok, open(os.path.join(a.out, "dims.json"), "w"), indent=0)
    if codes:                                         # training rows: drawing -> the program that built it
        import pandas as pd
        pd.DataFrame([{"uid": k, "code": codes[k], "png": f"{k}.png", "dims": json.dumps(v), "decimals": dec[k]}
                      for k, v in ok.items()]
                     ).to_parquet(os.path.join(a.out, "drawings.parquet"))
    n = [len(v) for v in ok.values()]
    print(f"[draw] {len(ok)}/{len(res)} drawings, dims per drawing median {np.median(n) if n else 0}; errors: "
          f"{[v['error'][:80] for v in res.values() if 'error' in v][:5]}", flush=True)


if __name__ == "__main__":
    main()
