"""Offline picker experiment, step 2: score pickers (consensus, OCR dims, printed dims, oracle) vs greedy.
  python scripts/pick_dims_score.py <dir> /bon/feats_ocr.pkl"""
import sys, json, pickle, numpy as np, pandas as pd
S = sys.argv[1]; F = pickle.load(open(S + sys.argv[2], 'rb'))
B = json.load(open(S + '/bon/bon.json'))['rows']
D = pd.read_parquet(S + '/dv/draw_val/drawings.parquet'); gtd = dict(zip(D.uid, D.dims))
def match(vals, nums, tol=0.006):
    nums = np.array(nums) if len(nums) else np.array([-1.0])
    return sum(np.any(np.abs(nums - v) <= max(tol * v, 0.051)) for v in vals)
def score(f, nums, w_diam=0.5):
    if f is None: return -1
    return match(f['ext'], nums) + w_diam * min(match(f['diam'], nums), 4)
res = {k: [] for k in ('greedy', 'consensus', 'ocr', 'ocr+cons', 'gtdims', 'gtdims+cons', 'oracle')}
by = {}
for u, r in B.items():
    if not r or 'gt_iou' not in r or u not in F: continue
    g = r['gt_iou']; ag = np.array(r['agree']); fs = F[u]['feats']
    ocr = F[u].get('ocr') or []
    gd = [d['value'] for d in json.loads(gtd[u])] if isinstance(gtd[u], str) else []
    def pick(nums, cons):
        s = np.array([score(f, nums) for f in fs], float)
        if cons: s = s + 0.99 * np.clip(ag, 0, 1)
        return int(np.argmax(s))   # ties -> lowest index (greedy first)
    picks = {'greedy': 0, 'consensus': r['pick'], 'ocr': pick(ocr, False), 'ocr+cons': pick(ocr, True),
             'gtdims': pick(gd, False), 'gtdims+cons': pick(gd, True), 'oracle': int(np.argmax(g))}
    src = 'gen' if u.startswith('gt') else 'recode'
    for k, i in picks.items():
        res[k].append(g[i]); by.setdefault((k, src), []).append(g[i])
n = len(res['greedy']); base = np.array(res['greedy'])
print('parts', n)
for k, v in res.items():
    d = np.array(v) - base
    print(f"{k:12s} {100*np.mean(v):5.1f}  vs greedy {100*d.mean():+.1f} +- {100*d.std()/np.sqrt(n):.1f}   recode {100*np.mean(by[(k,'recode')]):5.1f}  gen {100*np.mean(by[(k,'gen')]):5.1f}")
