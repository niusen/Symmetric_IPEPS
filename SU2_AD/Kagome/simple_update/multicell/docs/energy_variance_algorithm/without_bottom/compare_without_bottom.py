"""Compare structural h0 dependence with/without the bottom warmup, not values."""
from pathlib import Path
import sys
import json

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parent / 'source'))
from trace_ctm_dependencies import Trace


def build(bottom):
    t = Trace(12)
    flags = [n['label'] in (('a', 6, 6), ('a', 7, 6)) for n in t.nodes]
    schedule = [(3, 7, 0)] + ([(4, 6, 0)] if bottom else [])
    for s in (1, 2):
        for d, js in ((1, range(5, 12)), (3, range(8, 1, -1)),
                      (2, range(5, 12)), (4, range(7, 1, -1))):
            schedule.extend((d, j, s) for j in js)
    snapshots, pending, expressions = [[]], [], []
    for d, j, s in schedule:
        before = t.grid.copy()
        start = len(t.nodes)
        t.move(j, d, s)
        for n in t.nodes[start:]:
            flags.append(any(flags[p] for p in n['parents']))
        def key(k):
            kind, x, y, index = k
            return f'{kind}{index},{x},{y}'
        marked = sorted(key(k) for k, n in t.grid.items() if flags[n])
        written = sorted(key(k) for k, n in t.grid.items()
                         if flags[n] and n != before[k])
        snapshots.append(marked)
        pending.append(written)
    return dict(sources=[[6, 6], [7, 6]], schedule=schedule,
                snapshots=snapshots, pending=pending)


old, new = build(True), build(False)
assert len(new['schedule']) == 55
assert old['snapshots'][1] == new['snapshots'][1]
new['description'] = 'Only right warmup; possible structural dependence, not numerical response'
(here / 'h0_without_bottom_expected.json').write_text(json.dumps(new), encoding='utf-8')

comparison = []
for ni, (d, j, s) in enumerate(new['schedule'], 1):
    oi = old['schedule'].index((d, j, s)) + 1
    a, b = set(old['snapshots'][oi]), set(new['snapshots'][ni])
    comparison.append(dict(direction=d, coordinate=j, sweep=s,
                           old_move=oi, new_move=ni,
                           old_count=len(a), new_count=len(b),
                           only_old=sorted(a-b), only_new=sorted(b-a)))
report = dict(interpretation='Boolean dependencies only; equality does not imply equal projectors or tensor values',
              old_moves=56, new_moves=55, aligned_moves=comparison)
(here / 'schedule_comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print('New/old schedules:', len(new['schedule']), len(old['schedule']))
for row in comparison:
    if row['new_move'] in (1, 2, 3, 8, 15, 16, 22, 28, 29, 55):
        print({k: v for k, v in row.items() if k not in ('only_old', 'only_new')},
              'different records:', len(row['only_old'])+len(row['only_new']))
print('First aligned stage with same dependency set after first left:',
      next((r for r in comparison[1:] if not r['only_old'] and not r['only_new']), None))
