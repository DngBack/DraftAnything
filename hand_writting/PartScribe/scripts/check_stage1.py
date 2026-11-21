"""Apply the frozen stage-1 criteria (docs/stage1_probe_criteria.json) to runs/probes/*.json."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / 'runs/probes'
crit = json.loads((ROOT / 'docs/stage1_probe_criteria.json').read_text())
r = {k: json.loads((P / f'{k}.json').read_text()) for k in ('slots', 'global', 'pixels')}
ex = {k: r[k]['p1']['test_heldout']['exact'] for k in r}
p2 = r['slots']['p2']['test_heldout']
p1_ok = ex['slots'] >= ex['global'] + 0.10 and ex['slots'] > ex['pixels']
p2_ok = p2['success'] >= 0.60 and p2['success'] > p2['majority_mark_rate']
print(f"P1 test_heldout.exact: slots {ex['slots']:.3f} | global {ex['global']:.3f} | pixels {ex['pixels']:.3f} -> "
      f"{'PASS' if p1_ok else 'FAIL'}")
print(f"P2 test_heldout: success {p2['success']:.3f} | majority {p2['majority_mark_rate']:.3f} -> "
      f"{'PASS' if p2_ok else 'FAIL'}")
print('criteria frozen at:', crit['frozen_at'])
print('STAGE 1:', 'GO' if p1_ok and p2_ok else 'NO-GO (see criteria if_P1_fails)')
