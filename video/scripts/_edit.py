# tiny safe multi-replace helper: python scripts/_edit.py  (edits defined in a JSON on stdin)
import json, sys
for p, pairs in json.load(sys.stdin).items():
    s = open(p, encoding='utf-8').read()
    for a, b in pairs:
        assert a in s, (p, a[:70])
        s = s.replace(a, b)
    tmp = p + '.tmp'
    open(tmp, 'w', encoding='utf-8', newline='\n').write(s)
    import os; os.replace(tmp, p)
print('ok')
