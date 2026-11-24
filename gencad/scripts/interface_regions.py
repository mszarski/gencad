"""CADGenBench-style interface regions for our own parts, generated from the GT B-rep (CADGenBench's own regions are
hand-made per part and private).

Holes and bosses = full cylinders in the GT (faces on one axis line and radius whose angular spans add up to >= 324
deg, so cylinders split into halves still count): a concave one (material outside) is a hole -> keep-out region (KOR,
must be empty), a convex one is a boss / pin -> keep-in region (KIR, must be solid). The region is the cylinder over
the face's extent along its axis. Holes sharing an axis direction and diameter (within 2 %) form one mating group (a
bolt pattern), as in CADGenBench where a group scores as its worst feature; each boss is its own group. Files are
written as `jig_<group>__<index>__KOR|KIR.step` next to `ground_truth.step`, the layout
`cadgenbench.eval.evaluate.evaluate_result` reads.

  from scripts.interface_regions import write_regions; write_regions(gt_shape, fixture_dir)
"""
import os

import numpy as np


def cylinders(shape, min_span_deg=324.0, min_r=0.2):
    """-> [(kind 'hole'|'boss', axis point p0 at the low end, unit axis, height, radius)]."""
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Cylinder
    import cadquery as cq
    groups = {}
    for f in shape.Faces():
        ad = BRepAdaptor_Surface(f.wrapped)
        if ad.GetType() != GeomAbs_Cylinder:
            continue
        cyl = ad.Cylinder()
        r = cyl.Radius()
        if r < min_r:
            continue
        ax = np.array(cyl.Axis().Direction().Coord())
        ax = ax / np.linalg.norm(ax)
        if ax[np.argmax(np.abs(ax))] < 0:
            ax = -ax
        o = np.array(cyl.Axis().Location().Coord())
        o = o - np.dot(o, ax) * ax                       # canonical point on the axis line
        span = np.degrees(ad.LastUParameter() - ad.FirstUParameter())
        u0, u1, v0, v1 = f._uvBounds()
        p3 = ad.Value((u0 + u1) / 2, (v0 + v1) / 2)
        pt = np.array(p3.Coord())
        nrm = np.array(f.normalAt(cq.Vector(*pt)).toTuple())     # respects the face's orientation
        radial = pt - (np.dot(pt - o, ax) * ax + o)
        concave = np.dot(nrm, radial) < 0                # normal points at the axis: material outside: a hole
        ts = [np.dot(np.array(v.toTuple()) - o, ax) for v in f.Vertices()] or [np.dot(pt - o, ax)]
        key = (tuple(np.round(ax, 3)), tuple(np.round(o, 2)), round(r, 2), bool(concave))
        g = groups.setdefault(key, {"span": 0.0, "t": [], "ax": ax, "o": o, "r": r, "hole": concave})
        g["span"] += span
        g["t"] += ts
    out = []
    for g in groups.values():
        if g["span"] < min_span_deg or max(g["t"]) - min(g["t"]) < 1e-3:
            continue
        t0, t1 = min(g["t"]), max(g["t"])
        out.append(("hole" if g["hole"] else "boss", g["o"] + t0 * g["ax"], g["ax"], t1 - t0, g["r"]))
    return out


def write_regions(shape, fixture_dir, max_regions=40):
    """Write jig_*.step files for `shape` into fixture_dir; -> number of regions written."""
    import cadquery as cq
    feats = cylinders(shape)
    holes = [f for f in feats if f[0] == "hole"]
    bosses = [f for f in feats if f[0] == "boss"]
    groups = []
    for h in holes:                                      # bolt patterns: same axis direction and diameter
        for gr in groups:
            g = gr[0]
            if g[0] == "hole" and np.allclose(g[2], h[2], atol=1e-3) and abs(g[4] - h[4]) <= 0.02 * g[4]:
                gr.append(h)
                break
        else:
            groups.append([h])
    groups += [[b] for b in bosses]
    os.makedirs(fixture_dir, exist_ok=True)
    n = 0
    for gi, gr in enumerate(groups, 1):
        for fi, (kind, p0, ax, h, r) in enumerate(gr, 1):
            if n >= max_regions:
                return n
            solid = cq.Solid.makeCylinder(r, h, cq.Vector(*p0), cq.Vector(*ax))
            cq.exporters.export(cq.Workplane().add(solid),
                                os.path.join(fixture_dir, f"jig_{gi}__{fi}__{'KOR' if kind == 'hole' else 'KIR'}.step"))
            n += 1
    return n
