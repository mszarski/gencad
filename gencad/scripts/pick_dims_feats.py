"""Offline picker experiment (Nov 25), step 1: per best-of-N candidate (results/<run>/bon_cands.pkl from
scripts/bon_drawings.py) its sorted bbox extents and cylinder diameters, plus tesseract OCR numbers of each drawing.
  python scripts/pick_dims_feats.py <dir with bon/bon_cands.pkl and dv/draw_val/>  -> bon/feats_ocr.pkl
Finding: OCR-dimension picking did not beat the consensus picker (see CLAUDE.md)."""
import sys, os, re, json, pickle, numpy as np, pandas as pd
sys.path.insert(0, '/home/user/gencad/gencad')
S = sys.argv[1]
def feats(code):
    from mm.cadquery_io import run_cadquery
    try:
        s = run_cadquery(code) if code else None
        if s is None: return None
        b = s.BoundingBox(); ext = sorted([b.xlen, b.ylen, b.zlen])
        diams = set()
        for f in s.Faces():
            if f.geomType() == 'CYLINDER':
                try: diams.add(round(2 * f._geomAdaptor().Cylinder().Radius(), 2))
                except Exception: pass
        return {'ext': ext, 'diam': sorted(diams)[:20]}
    except Exception:
        return None
def ocr(path):
    import pytesseract
    from PIL import Image
    im = Image.open(path).convert('L'); out = []
    for ang in (0, 90):
        t = pytesseract.image_to_string(im.rotate(ang, expand=True), config='--psm 11 -c tessedit_char_whitelist=0123456789.')
        out += [float(x) for x in re.findall(r'\d+\.\d+|\d+', t) if 0 < float(x) < 5000]
    return out
if __name__ == '__main__':
    from pebble import ProcessPool
    C = pickle.load(open(S + '/bon/bon_cands.pkl', 'rb'))
    items = [(u, p, c) for (u, p), c in zip(C['items'], C['cands']) if not u.startswith('x_')]
    with ProcessPool(max_workers=4) as pool:
        F = [[pool.schedule(feats, args=(c,), timeout=120) for c in cs] for _, _, cs in items]
        def res(f):
            try: return f.result()
            except Exception: return None
        out = {u: {'feats': [res(f) for f in fs]} for (u, _, _), fs in zip(items, F)}
    pickle.dump(out, open(S + '/bon/feats.pkl', 'wb')); print('feats done', flush=True)
    os.environ['OMP_THREAD_LIMIT'] = '1'
    for i, (u, p, _) in enumerate(items):
        try: out[u]['ocr'] = ocr(S + '/dv/' + p.replace('data/', ''))
        except Exception: out[u]['ocr'] = []
        if i % 100 == 0: print('ocr', i, flush=True)
    pickle.dump(out, open(S + '/bon/feats_ocr.pkl', 'wb')); print('done', len(out))
