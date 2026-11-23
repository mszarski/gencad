"""Convert our CadQuery programs to SolidCraft command scripts (https://github.com/tangency-labs/solidcraft) and check
that SolidCraft builds the same solid.

How: the program is executed with hooks on CadQuery's shape layer, which record what each operation actually did in
world coordinates: extrude / revolve of a planar profile (its edges), cylinders and boxes, booleans, fillets and
chamfers (a point on each picked edge), shells, and rigid moves. That history is a tree of operations, re-emitted as
SolidCraft commands: `sketch.create` on an explicit plane, `sketch.line` / `sketch.arc.three_point` /
`sketch.circle.center` / `sketch.spline.fit_point`, `solid.extrude` / `solid.revolve` as new bodies,
`solid.combine` for booleans, `solid.fillet` / `solid.chamfer` / `solid.shell` with points, `solid.move` with a
matrix. Selectors (`faces('>Z')`, `edges(near(...))`) never need translating: the hooks see the edges they picked.

  python scripts/to_solidcraft.py --preds results/gbon_real_0/preds.pkl --method g --cli <solidcraft-cli> --out <dir>
Per program: converted (or why not), the SolidCraft run (ok / failing step + error), and the volume IoU between the
CadQuery solid and the STEP SolidCraft writes (same absolute frame, no refitting).
"""
import argparse
import json
import os
import pickle
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mm.sysutil import usable_cpus  # noqa: E402


class Unsupported(Exception):
    pass


# ------------------------------------------------------------------------------------------------ tracing
class Node:
    __slots__ = ("op", "args", "kids", "id")
    _n = [0]

    def __init__(self, op, args, kids=()):
        self.op, self.args, self.kids = op, args, list(kids)
        Node._n[0] += 1
        self.id = Node._n[0]


_REG = []      # (TopoDS_Shape, Node) of every shape produced by a traced op


def _register(shape, node):
    try:
        shape._sc = node
    except Exception:  # noqa: BLE001
        pass
    _REG.append((shape.wrapped, node))
    return shape


def _node(shape):
    n = getattr(shape, "_sc", None)
    if n is not None:
        return n
    from cadquery.occ_impl.shapes import Compound
    for w, nd in reversed(_REG):              # most recent first: re-wrapped shapes, or a solid inside a traced
        if w.IsSame(shape.wrapped):           # compound (e.g. .solids() of a multi-solid result)
            return nd
        if w.ShapeType() == 0 and any(s.wrapped.IsSame(shape.wrapped) for s in Compound(w).Solids()):
            return Node("part", {"shape": shape}, [nd])
    raise Unsupported(f"untracked shape ({type(shape).__name__})")


def _trsf_matrix(tr):
    return [[tr.Value(i, j) for j in range(1, 5)] for i in range(1, 4)]


def install_hooks():
    from cadquery.occ_impl import shapes as S
    from OCP.gp import gp_Trsf
    if getattr(S, "_sc_hooked", False):
        return
    S._sc_hooked = True
    Solid, Shape, Mixin3D, Compound, Face = S.Solid, S.Shape, S.Mixin3D, S.Compound, S.Face

    def wrap_cls(cls, name, make):
        orig = getattr(cls, name)                      # bound classmethod (a multimethod dispatcher)

        def f(klass, *a, **k):
            r = orig(*a, **k)
            return _register(r, make(*a, **k))
        setattr(cls, name, classmethod(f))

    def face_of(a):
        if isinstance(a[0], Face):
            return a[0], a[1:]
        return Face.makeFromWires(a[0], list(a[1])), a[2:]

    def mk_extrude(*a, **k):
        face, rest = face_of(a)
        vec = S.Vector(rest[0] if rest else k["vecNormal"])
        taper = (rest[1] if len(rest) > 1 else k.get("taper", 0)) or 0
        if abs(taper) > 1e-9:
            raise Unsupported("tapered extrude")
        return Node("extrude", {"face": face, "vec": vec})

    def mk_revolve(*a, **k):
        face, rest = face_of(a)
        ang, p0, p1 = rest[0], S.Vector(rest[1]), S.Vector(rest[2])
        return Node("revolve", {"face": face, "angle": float(ang), "p0": p0, "p1": p1})

    def mk_cyl(radius, height, pnt=S.Vector(0, 0, 0), dir=S.Vector(0, 0, 1), angleDegrees=360):  # noqa: A002
        if abs(angleDegrees - 360) > 1e-6:
            raise Unsupported("partial cylinder")
        return Node("cylinder", {"r": radius, "h": height, "p": S.Vector(pnt), "d": S.Vector(dir)})

    def mk_box(length, width, height, pnt=S.Vector(0, 0, 0), dir=S.Vector(0, 0, 1)):  # noqa: A002
        if (S.Vector(dir) - S.Vector(0, 0, 1)).Length > 1e-9:
            raise Unsupported("box with a non-Z direction")
        return Node("box", {"l": length, "w": width, "h": height, "p": S.Vector(pnt)})

    wrap_cls(Solid, "extrudeLinear", mk_extrude)
    wrap_cls(Solid, "revolve", mk_revolve)
    wrap_cls(Solid, "makeCylinder", mk_cyl)
    wrap_cls(Solid, "makeBox", mk_box)
    for nm in ("makeCone", "makeSphere", "makeTorus", "makeWedge", "makeLoft", "sweep", "extrudeLinearWithRotation"):
        if hasattr(Solid, nm):
            o = getattr(Solid, nm)

            def bad(klass, *a, _nm=nm, _o=o, **k):
                raise Unsupported(_nm)
            setattr(Solid, nm, classmethod(bad))

    oc = Compound.makeCompound

    def make_compound(klass, listOfShapes):
        L = list(listOfShapes)
        r = oc(L)
        try:
            kids = [_node(s) for s in L]
        except Unsupported:
            return r                                    # e.g. compounds of faces/wires, not solids
        return _register(r, Node("group", {}, kids))
    Compound.makeCompound = classmethod(make_compound)

    def wrap_bool(cls, name, op):
        orig = cls.__dict__[name]

        def f(self, *others, **k):
            r = orig(self, *others, **k)
            if not self.Solids():                        # 2D booleans inside sketches: not a body op
                return r
            return _register(r, Node(op, {}, [_node(self)] + [_node(o) for o in others]))
        setattr(cls, name, f)
    for cls in (Shape, Compound):                       # Compound overrides the booleans
        wrap_bool(cls, "fuse", "join")
        wrap_bool(cls, "cut", "cut")
        wrap_bool(cls, "intersect", "intersect")

    oclean = Shape.clean

    def clean(self):
        r = oclean(self)
        n = getattr(self, "_sc", None)
        return _register(r, n) if n is not None else r
    Shape.clean = clean

    def edge_pts(edgeList):
        return [tuple(e.positionAt(0.5).toTuple()) for e in edgeList]

    ofil, ocha, oshell = Mixin3D.fillet, Mixin3D.chamfer, Shape.shell

    def fillet(self, radius, edgeList):
        edgeList = list(edgeList)
        r = ofil(self, radius, edgeList)
        return _register(r, Node("fillet", {"r": radius, "pts": edge_pts(edgeList)}, [_node(self)]))

    def chamfer(self, length, length2, edgeList):
        edgeList = list(edgeList)
        r = ocha(self, length, length2, edgeList)
        return _register(r, Node("chamfer", {"d": length, "d2": length2, "pts": edge_pts(edgeList)}, [_node(self)]))

    def shell(self, faceList, thickness, kind="arc"):
        faceList = list(faceList)
        r = oshell(self, faceList, thickness, kind)
        pts = []
        for f in faceList:
            u0, u1, v0, v1 = f._uvBounds()
            pts.append(tuple(f.positionAt((u0 + u1) / 2, (v0 + v1) / 2, mode="parameter").toTuple()))
        return _register(r, Node("shell", {"t": thickness, "pts": pts}, [_node(self)]))
    Mixin3D.fillet, Mixin3D.chamfer, Shape.shell = fillet, chamfer, shell

    oapply = Shape._apply_transform

    def recorded(r, start):                          # an inner hook already recorded this result
        return any(w.IsSame(r.wrapped) for w, _ in _REG[start:])

    def apply_transform(self, Tr):
        start = len(_REG)
        r = oapply(self, Tr)
        if recorded(r, start):
            return r
        n = getattr(self, "_sc", None)
        if n is None:
            try:
                n = _node(self)
            except Unsupported:
                return r                                 # transforming wires/faces/points: not a body op
        if Tr.IsNegative():
            raise Unsupported("mirror")
        return _register(r, Node("move", {"m": _trsf_matrix(Tr)}, [n]))
    Shape._apply_transform = apply_transform

    def wrap_loc(name):
        orig = getattr(Shape, name)

        def f(self, *a, **k):
            start = len(_REG)
            t_in = self.wrapped.Location().Transformation()   # before: `move` changes self in place
            try:
                n = _node(self)
            except Unsupported:
                n = None
            r = orig(self, *a, **k)
            if n is None or recorded(r, start):
                return r
            t_out = r.wrapped.Location().Transformation()
            T = gp_Trsf()
            T.Multiply(t_out)
            T.Multiply(t_in.Inverted())
            if T.Form() == 0:                            # identity
                return _register(r, n)
            return _register(r, Node("move", {"m": _trsf_matrix(T)}, [n]))
        setattr(Shape, name, f)
    for nm in ("moved", "located", "move"):
        wrap_loc(nm)


# ------------------------------------------------------------------------------------------------ emission
def _f(x):
    return round(float(x), 6)


def _plane(face):
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Plane
    ad = BRepAdaptor_Surface(face.wrapped)
    if ad.GetType() != GeomAbs_Plane:
        raise Unsupported("non-planar profile")
    pl = ad.Plane()
    ax = pl.Position()
    o, x, y, n = pl.Location(), ax.XDirection(), ax.YDirection(), ax.Direction()
    v = lambda p: np.array([p.X(), p.Y(), p.Z()])  # noqa: E731
    return v(o), v(x), v(y), v(n)


def _sketch_cmds(face, name):
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.GeomAbs import GeomAbs_Line, GeomAbs_Circle
    o, x, y, n = _plane(face)
    uv = lambda p: [_f(np.dot(np.array(p) - o, x)), _f(np.dot(np.array(p) - o, y))]  # noqa: E731
    cmds = [{"command": "sketch.create", "params": {"name": name, "plane": {
        "origin": [_f(c) for c in o], "x_dir": [_f(c) for c in x], "y_dir": [_f(c) for c in y]}}}]
    for wire in [face.outerWire()] + face.innerWires():
        for e in wire.Edges():
            ad = BRepAdaptor_Curve(e.wrapped)
            t = ad.GetType()
            p0, p1 = e.startPoint().toTuple(), e.endPoint().toTuple()
            if t == GeomAbs_Line:
                cmds.append({"command": "sketch.line", "params": {"points": [uv(p0), uv(p1)]}})
            elif t == GeomAbs_Circle and e.IsClosed():
                c = ad.Circle()
                cen = c.Location()
                cmds.append({"command": "sketch.circle.center", "params": {
                    "center": uv((cen.X(), cen.Y(), cen.Z())), "radius": _f(c.Radius())}})
            elif t == GeomAbs_Circle:
                cmds.append({"command": "sketch.arc.three_point", "params": {
                    "start": uv(p0), "through": uv(e.positionAt(0.5).toTuple()), "end": uv(p1)}})
            else:                                       # splines, ellipses: fit points (approximate)
                pts = [uv(e.positionAt(s).toTuple()) for s in np.linspace(0, 1, 13)]
                cmds.append({"command": "sketch.spline.fit_point", "params": {"points": pts,
                                                                              "periodic": bool(e.IsClosed())}})
    cmds.append({"command": "sketch.finish", "params": {}})
    return cmds, n


def _moved_leaf(kid, m):
    """A primitive node with a rigid transform applied to its geometry."""
    from OCP.gp import gp_Trsf
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from cadquery.occ_impl.shapes import Face, Vector
    T = gp_Trsf()
    T.SetValues(*[float(v) for row in m for v in row])
    R = np.array(m)[:, :3]
    pt = lambda v: Vector(*(R @ np.array(v.toTuple()) + np.array(m)[:, 3]))  # noqa: E731
    dr = lambda v: Vector(*(R @ np.array(v.toTuple())))                       # noqa: E731
    a = dict(kid.args)
    if "face" in a:
        a["face"] = Face(BRepBuilderAPI_Transform(a["face"].wrapped, T, True).Shape())
    if kid.op == "extrude":
        a["vec"] = dr(a["vec"])
    elif kid.op == "revolve":
        a["p0"], a["p1"] = pt(a["p0"]), pt(a["p1"])
    else:
        a["p"], a["d"] = pt(a["p"]), dr(a["d"])
    return Node(kid.op, a)


def _uniq(xs):
    out = []
    for x in xs:
        if x not in out:
            out.append(x)
    return out


class Emitter:
    def __init__(self):
        self.cmds, self.k, self.done, self.moved = [], 0, {}, set()
        self.approx = 0

    def name(self, p="B"):
        self.k += 1
        return f"{p}{self.k}"

    def bodies(self, node):
        """Realise a node -> list of body names (emits commands once per node)."""
        if node.id in self.done:
            return self.done[node.id]
        op, a = node.op, node.args
        if op in ("extrude", "revolve"):
            b = self.name()
            sk, n = _sketch_cmds(a["face"], "S" + b)
            if any(c["command"] == "sketch.spline.fit_point" for c in sk):
                self.approx += 1
            self.cmds += sk
            if op == "extrude":
                v = np.array(a["vec"].toTuple())
                d = float(np.dot(v, n))
                if np.linalg.norm(v - d * n) > 1e-6 * max(1, abs(d)):
                    raise Unsupported("oblique extrude")
                self.cmds.append({"command": "solid.extrude", "params": {
                    "sketch": "S" + b, "distance": _f(abs(d)), "direction": "positive" if d > 0 else "negative",
                    "operation": "new", "body_name": b}})
            else:
                p0, p1 = np.array(a["p0"].toTuple()), np.array(a["p1"].toTuple())
                self.cmds.append({"command": "solid.revolve", "params": {
                    "sketch": "S" + b, "axis": {"origin": [_f(c) for c in p0], "dir": [_f(c) for c in p1 - p0]},
                    "angle": _f(a["angle"]), "operation": "new", "body_name": b}})
            out = [b]
        elif op == "cylinder":
            b = self.name()
            self.cmds.append({"command": "solid.cylinder", "params": {
                "base": [_f(c) for c in a["p"].toTuple()], "axis": [_f(c) for c in a["d"].toTuple()],
                "radius": _f(a["r"]), "height": _f(a["h"]), "operation": "new", "body_name": b}})
            out = [b]
        elif op == "box":
            b = self.name()
            self.cmds.append({"command": "solid.box", "params": {
                "corner": [_f(c) for c in a["p"].toTuple()], "length": _f(a["l"]), "width": _f(a["w"]),
                "height": _f(a["h"]), "operation": "new", "body_name": b}})
            out = [b]
        elif op == "group":
            out = _uniq(b for kid in node.kids for b in self.bodies(kid))
        elif op == "part":                               # one solid of a multi-solid result: SolidCraft keeps
            out = self.bodies(node.kids[0])              # the result as one body (touching: one_body)
        elif op in ("join", "cut", "intersect"):
            tgt = self.bodies(node.kids[0])
            tools = [b for b in _uniq(b for kid in node.kids[1:] for b in self.bodies(kid)) if b not in tgt]
            if len(tgt) > 1:                             # several target bodies: merge them first
                self.cmds.append({"command": "solid.combine", "params": {
                    "target": tgt[0], "tools": tgt[1:], "operation": "join", "touching": "one_body"}})
                tgt = tgt[:1]
            if tools:
                p = {"target": tgt[0], "tools": tools, "operation": op}
                if op == "join":
                    p["touching"] = "one_body"
                self.cmds.append({"command": "solid.combine", "params": p})
            out = tgt
        elif op in ("fillet", "chamfer", "shell"):
            src = self.bodies(node.kids[0])
            p = {"edges" if op != "shell" else "faces": [[_f(c) for c in q] for q in a["pts"]]}
            if len(src) == 1:
                p["body"] = src[0]
            if op == "fillet":
                p.update(radius=_f(a["r"]), tangent_chain=False)
            elif op == "chamfer":
                p.update(distance=_f(a["d"]), tangent_chain=False)
                if a["d2"] is not None and abs(a["d2"] - a["d"]) > 1e-9:
                    p["distance2"] = _f(a["d2"])
            else:
                p.update(thickness=_f(abs(a["t"])), direction="inside" if a["t"] < 0 else "outside")
                p.pop("body", None)
                p["body"] = src[0]
            self.cmds.append({"command": f"solid.{op}", "params": p})
            out = src
        elif op == "move":
            kid = node.kids[0]
            if kid.op in ("extrude", "revolve", "cylinder") and kid.id not in self.done:
                out = self.bodies(_moved_leaf(kid, a["m"]))  # fold the move into the feature: a fresh body
            else:                                            # per placement (e.g. one cylinder per hole)
                src = self.bodies(kid)
                p = {"bodies": src, "matrix": [[round(float(v), 12) for v in row] for row in a["m"]],
                     "tolerance": 1e-6}
                if kid.id in self.moved:                     # the same bodies placed again: move a copy
                    p["copy"] = True
                    p["body_names"] = [self.name() for _ in src]
                    out = p["body_names"]
                else:
                    out = src
                self.moved.add(kid.id)
                self.cmds.append({"command": "solid.move", "params": p})
        else:
            raise Unsupported(op)
        # a node used twice would have its bodies consumed by the first combine: not handled (rare)
        self.done[node.id] = out
        return out


def convert(code):
    """CadQuery source -> (cq shape, command list, info) ; raises Unsupported."""
    install_hooks()
    from mm.cad_helpers import namespace
    _REG.clear()
    ns = namespace()
    exec(code, ns)
    r = ns["r"]
    shape = r.val() if hasattr(r, "val") else r
    if shape is None or not shape.isValid():
        raise Unsupported("program does not build")
    node = _node(shape)
    em = Emitter()
    final = em.bodies(node)
    if len(final) > 1:
        em.cmds.append({"command": "solid.combine", "params": {"target": final[0], "tools": final[1:],
                                                               "operation": "join", "touching": "one_body"}})
    return shape, em.cmds, {"n_cmds": len(em.cmds), "approx_splines": em.approx}


def _sampled_iou(a, b, n=20000, seed=0):
    """Volume IoU from random points in the joint bounding box (an OCC boolean of two coincident solids built by
    different kernels often fails and reports ~0)."""
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    from OCP.BRepClass3d import BRepClass3d_SolidClassifier
    from OCP.TopAbs import TopAbs_IN, TopAbs_ON
    from OCP.gp import gp_Pnt
    bb = Bnd_Box()
    BRepBndLib.Add_s(a, bb)
    BRepBndLib.Add_s(b, bb)
    lo, hi = np.array(bb.CornerMin().Coord()), np.array(bb.CornerMax().Coord())
    P = np.random.default_rng(seed).uniform(lo, hi, size=(n, 3))
    ca, cb = BRepClass3d_SolidClassifier(a), BRepClass3d_SolidClassifier(b)

    def inside(c, p):
        c.Perform(gp_Pnt(*p), 1e-6)
        return c.State() in (TopAbs_IN, TopAbs_ON)
    ia = np.array([inside(ca, p) for p in P])
    ib = np.array([inside(cb, p) for p in P])
    return float((ia & ib).sum() / max((ia | ib).sum(), 1))


def work(args):
    code, cli, tmpdir, i = args
    rec = {"i": i}
    try:
        shape, cmds, info = convert(code)
    except Unsupported as e:
        return {**rec, "stage": "convert", "error": str(e)}
    except Exception as e:  # noqa: BLE001
        return {**rec, "stage": "convert", "error": f"{type(e).__name__}: {e}"[:200]}
    rec.update(info)
    rec["ops"] = sorted({c["command"] for c in cmds})
    js, st = os.path.join(tmpdir, f"p{i}.json"), os.path.join(tmpdir, f"p{i}.step")
    json.dump({"commands": cmds}, open(js, "w"))
    rec["json_chars"] = os.path.getsize(js)
    rec["cq_chars"] = len(code)
    try:
        p = subprocess.run([cli, "run", js, "--out", st, "--results", "--quiet", "--json-out", js + ".out"],
                           capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return {**rec, "stage": "run", "error": "timeout"}
    if p.returncode != 0 or not os.path.exists(st):
        err = ""
        try:
            o = json.load(open(js + ".out"))
            e = o.get("error", {}) if isinstance(o, dict) else {}
            err = f"step {e.get('step')} {e.get('command')}: {str(e.get('error'))[:300]}"
        except Exception:  # noqa: BLE001
            err = (p.stderr or p.stdout)[-300:]
        return {**rec, "stage": "run", "error": err}
    try:                                               # raw OCC here: the CadQuery layer is hooked in this process
        from OCP.STEPControl import STEPControl_Reader
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
        from OCP.GProp import GProp_GProps
        from OCP.BRepGProp import BRepGProp
        rd = STEPControl_Reader()
        rd.ReadFile(st)
        rd.TransferRoots()
        sc = rd.OneShape()

        def vol(x):
            g = GProp_GProps()
            BRepGProp.VolumeProperties_s(x, g)
            return g.Mass()
        va, vb = vol(shape.wrapped), vol(sc)
        rec["sc_volume"], rec["cq_volume"] = vb, va
        rec["iou"] = _sampled_iou(shape.wrapped, sc) if vb > 0 else 0.0
        rec["stage"] = "ok"
    except Exception as e:  # noqa: BLE001
        return {**rec, "stage": "compare", "error": f"{type(e).__name__}: {e}"[:200]}
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="pickle with preds[method] (list of CadQuery programs)")
    ap.add_argument("--method", default="g")
    ap.add_argument("--codes_parquet", default=None, help="alternatively a parquet with a `code` column")
    ap.add_argument("--cli", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=usable_cpus())
    ap.add_argument("--keep", default=None, help="folder to keep the generated scripts in")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.codes_parquet:
        import pandas as pd
        codes = list(pd.read_parquet(a.codes_parquet).code)
    else:
        codes = pickle.load(open(a.preds, "rb"))["preds"][a.method]
    codes = codes[:a.limit]
    tmp = a.keep or tempfile.mkdtemp()
    os.makedirs(tmp, exist_ok=True)
    from pebble import ProcessPool
    rows = []
    with ProcessPool(max_workers=a.workers, max_tasks=50) as pool:
        futs = [pool.schedule(work, args=((c, a.cli, tmp, i),), timeout=300) for i, c in enumerate(codes) if c]
        for fu in futs:
            try:
                rows.append(fu.result())
            except Exception as e:  # noqa: BLE001
                rows.append({"stage": "crash", "error": str(e)[:200]})
    pickle.dump(rows, open(a.out, "wb"))
    import collections
    st = collections.Counter(r["stage"] for r in rows)
    ok = [r for r in rows if r["stage"] == "ok"]
    print(f"[sc] {len(rows)} programs: {dict(st)}")
    if ok:
        io = np.array([r["iou"] for r in ok])
        print(f"[sc] built: mean IoU vs CadQuery {100 * io.mean():.1f}; >= 0.99: {(io >= 0.99).mean():.1%}; "
              f"< 0.9: {(io < 0.9).mean():.1%}; script size / CadQuery size "
              f"{np.median([r['json_chars'] / r['cq_chars'] for r in ok]):.1f}x (median)")
    for stage in ("convert", "run", "compare"):
        errs = collections.Counter(r["error"].split(":")[0][:60] if stage != "run" else
                                   " ".join(r["error"].split()[:3]) for r in rows if r["stage"] == stage)
        if errs:
            print(f"[sc] {stage} errors: {errs.most_common(12)}")


if __name__ == "__main__":
    main()
