"""Feature sensing and defeaturing on B-reps, with quiddity (https://github.com/pzfreo/quiddity, Apache-2.0).

- `recognise(shape_or_step)`: quiddity's feature records with the OCC faces of each feature.
- `defeature(step, families)`: remove features (holes, fillets, chamfers, ...) with OCC's BRepAlgoAPI_Defeaturing,
  for edit-training pairs ("part without its holes" -> "part").
- `summary(feats)`: a short text description ("4 holes d6 through; 2 chamfers 1.0; ...") for the edit loop.

Face mapping: quiddity names faces by their STEP entity id (#n); OCC's STEP reader keeps no labels, but builds each
shell's faces in the order the CLOSED_SHELL / OPEN_SHELL lists them, so ids map to TopExp face order. The mapping is
checked against surface types (plane / cylinder / cone / sphere / torus); parts that disagree are refused (None).
"""
import json
import os
import re
import tempfile

SURF = {"PLANE": "PLANE", "CYLINDRICAL_SURFACE": "CYLINDER", "CONICAL_SURFACE": "CONE",
        "SPHERICAL_SURFACE": "SPHERE", "TOROIDAL_SURFACE": "TORUS"}


def _parse(step):
    """-> (face ids in shell order, {face id: surface kind or None})."""
    txt = open(step, errors="ignore").read()
    ents = dict(re.findall(r"#(\d+)\s*=\s*(.*?);\s*(?=#\d+\s*=|ENDSEC)", txt, re.S))
    order = []
    for k, v in ents.items():
        if re.match(r"(CLOSED_SHELL|OPEN_SHELL)\s*\(", v):
            order += [int(x) for x in re.findall(r"#(\d+)", v)]
    kind = {}
    for f in order:
        m = re.match(r"ADVANCED_FACE\s*\(\s*'[^']*'\s*,\s*\(.*?\)\s*,\s*#(\d+)", ents.get(str(f), ""), re.S)
        s = ents.get(m.group(1), "") if m else ""
        kind[f] = next((SURF[t] for t in SURF if s.startswith(t + "(") or s.startswith(t + " (")), None)
    return order, kind


def _occ_faces(shape):
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp
    from OCP.TopTools import TopTools_IndexedMapOfShape
    M = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, TopAbs_FACE, M)
    return [M.FindKey(i) for i in range(1, M.Extent() + 1)]


def _to_step(shape):
    import cadquery as cq
    d = tempfile.mkdtemp()
    f = os.path.join(d, "p.step")
    cq.exporters.export(cq.Workplane().add(shape), f)
    return f


def recognise(src):
    """src: STEP path or CadQuery shape. -> (OCC shape, [{family, record, faces: [OCC faces]}]) or None."""
    import cadquery as cq
    import quiddity
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.TopoDS import TopoDS
    step = src if isinstance(src, str) else _to_step(src)
    shape = cq.importers.importStep(step).val().wrapped
    E = quiddity.recognise_with_evidence(step)
    E = json.loads(E) if isinstance(E, str) else E
    ev = E.get("evidence") or {}
    src_ids = [s["entity"] for s in ev.get("face_sources", [])]
    order, kind = _parse(step)
    faces = _occ_faces(shape)
    if len(order) != len(faces) or sorted(order) != sorted(src_ids):
        return None
    names = {0: "PLANE", 1: "CYLINDER", 2: "CONE", 3: "SPHERE", 4: "TORUS"}
    pos = {f: i for i, f in enumerate(order)}
    for fid, face in zip(order, faces):        # check the order-based mapping against surface types
        k = kind.get(fid)
        if k is not None and names.get(int(BRepAdaptor_Surface(TopoDS.Face_s(face)).GetType())) != k:
            return None
    feats = [{"family": ft["family"], "record": ft.get("record", {}),
              "faces": [faces[pos[src_ids[i]]] for i in ft.get("constituent_faces", [])]}
             for ft in ev.get("features", [])]
    return shape, feats


def defeature(shape, feats):
    """Remove the faces of `feats` from `shape`; -> cq.Shape (valid single solid) or None."""
    import cadquery as cq
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Defeaturing
    from OCP.TopTools import TopTools_ListOfShape
    L = TopTools_ListOfShape()
    for ft in feats:
        for f in ft["faces"]:
            L.Append(f)
    d = BRepAlgoAPI_Defeaturing()
    d.SetShape(shape)
    d.AddFacesToRemove(L)
    d.SetRunParallel(False)
    d.Build()
    if not d.IsDone():
        return None
    out = cq.Shape.cast(d.Shape())
    sol = out.Solids()
    if len(sol) != 1 or not sol[0].isValid():
        return None
    return sol[0]


def _fmt(x):
    return f"{x:.3g}"


def summary(feats, scale=1.0):
    """Short text listing of features with sizes (sizes x scale)."""
    from collections import defaultdict
    groups = defaultdict(list)
    for ft in feats:
        r = ft["record"] or {}
        if ft["family"] == "holes":
            key = f"hole d{_fmt(r.get('diameter', 0) * scale)}" + (" through" if r.get("bottom") == "through" else "") \
                + (" cbore" if r.get("cbore") else "") + (" csink" if r.get("csink") else "")
        elif ft["family"] == "fillets":
            key = f"fillet r{_fmt(r.get('radius', 0) * scale)}"
        elif ft["family"] == "chamfers":
            key = f"chamfer {_fmt(r.get('leg1', 0) * scale)}"
        elif ft["family"] in ("slots", "pockets", "prismatic_pockets", "grooves", "bosses", "step_levels",
                              "turned_steps"):
            key = {"slots": "slot", "pockets": "pocket", "prismatic_pockets": "pocket", "grooves": "groove",
                   "bosses": "boss", "step_levels": "step level", "turned_steps": "turned step"}[ft["family"]]
        else:
            continue
        groups[key].append(1)
    return "; ".join(f"{len(v)} x {k}" for k, v in sorted(groups.items())) or "no holes, fillets or chamfers"
