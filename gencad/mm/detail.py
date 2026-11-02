"""Detail score: how well a predicted solid matches the fine features of the ground truth, which volume IoU barely
sees (a missing fillet, chamfer or small hole moves IoU by a percent or two).

Both solids are first fitted to their own unit bounding box (as for "own" IoU). Three parts, each in [0, 1]:
  surf   F-score of surface samples at a tight tolerance: precision = share of points sampled on the predicted surface
         within tau of the GT surface (exact point-to-mesh distance), recall = the reverse, harmonic mean; averaged over
         tau = 0.5 % and 1 % of the part size.
         Missing or extra features show up as surface that is far from the other solid.
  topo   through-hole agreement: genus from the Euler-Poincare formula on the B-rep (V - E + 2F - L = 2(S - G) for S
         shells, L wires); 1 / (1 + |G_pred - G_gt|).
  faces  face-type histogram overlap (plane, cylinder, cone, sphere, torus, other): sum of min counts / max of the
         totals. Fillets add cylinder/torus faces, chamfers planes, holes cylinders, so missing features cost here.
detail = 0.5 surf + 0.25 topo + 0.25 faces. Orientation matters (no rotation search), as for IoU.
"""
import numpy as np

TYPES = ("PLANE", "CYLINDER", "CONE", "SPHERE", "TORUS")
TAUS = (0.005, 0.01)


def _fit(shape):
    bb = shape.BoundingBox()
    c = ((bb.xmin + bb.xmax) / 2, (bb.ymin + bb.ymax) / 2, (bb.zmin + bb.zmax) / 2)
    return shape.translate(tuple(-v for v in c)).scale(1.0 / max(bb.xlen, bb.ylen, bb.zlen))


def mesh(shape):
    """Tessellated surface of a (unit-fitted) shape as a trimesh."""
    import trimesh
    V, F, off = [], [], 0
    for face in shape.Faces():
        verts, tri = face.tessellate(1e-3, 0.2)
        if tri:
            V.append(np.array([v.toTuple() for v in verts]))
            F.append(np.array(tri) + off)
            off += len(verts)
    return trimesh.Trimesh(np.concatenate(V), np.concatenate(F), process=False)


def _dist(points, m):
    """Exact distance from points to the surface of mesh m."""
    import trimesh
    _, d, _ = trimesh.proximity.closest_point(m, points)
    return d


def genus(shape):
    S = max(len(shape.Shells()), 1)
    V, E, F, L = len(shape.Vertices()), len(shape.Edges()), len(shape.Faces()), len(shape.Wires())
    return S - (V - E + 2 * F - L) / 2


def face_hist(shape):
    h = dict.fromkeys(TYPES + ("OTHER",), 0)
    for f in shape.Faces():
        t = f.geomType()
        h[t if t in TYPES else "OTHER"] += 1
    return h


def detail(gt, pred, n=2048):
    """gt, pred: CadQuery shapes (any frame). -> dict with surf, topo, faces, detail."""
    import trimesh
    a, b = _fit(gt), _fit(pred)
    ma, mb = mesh(a), mesh(b)
    pa = trimesh.sample.sample_surface(ma, n, seed=0)[0]
    pb = trimesh.sample.sample_surface(mb, n, seed=1)[0]
    da = _dist(pa, mb)                   # GT points -> predicted surface (recall)
    db = _dist(pb, ma)                   # predicted points -> GT surface (precision)
    fs = []
    for tau in TAUS:
        r, p = float(np.mean(da < tau)), float(np.mean(db < tau))
        fs.append(0.0 if r + p == 0 else 2 * r * p / (r + p))
    surf = float(np.mean(fs))
    topo = 1.0 / (1.0 + abs(genus(a) - genus(b)))
    ha, hb = face_hist(a), face_hist(b)
    faces = sum(min(ha[k], hb[k]) for k in ha) / max(sum(ha.values()), sum(hb.values()), 1)
    return {"surf": surf, "topo": topo, "faces": faces, "detail": 0.5 * surf + 0.25 * topo + 0.25 * faces}
