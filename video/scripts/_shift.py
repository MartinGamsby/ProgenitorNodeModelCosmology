"""Shift beat indices in a scene file after inserting a narration beat.
   python scripts/_shift.py FILE FROM_LINE MIN_BEAT DELTA [TO_LINE]
Rewrites b.on/p/at/s/through(N...), from={N}, until={N} where N >= MIN_BEAT, on lines FROM..TO."""
import os, re, sys
path, lo, minb, d = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
hi = int(sys.argv[5]) if len(sys.argv) > 5 else 10**9
pat = re.compile(r"(b\.(?:on|p|at|s|through)\(|from=\{|until=\{)(\d+)")
lines = open(path, encoding="utf-8").read().split("\n")
n = 0
for i in range(lo - 1, min(hi, len(lines))):
    def rep(m):
        global n
        v = int(m.group(2))
        if v < minb:
            return m.group(0)
        n += 1
        return m.group(1) + str(v + d)
    lines[i] = pat.sub(rep, lines[i])
open(path + ".tmp", "w", encoding="utf-8", newline="\n").write("\n".join(lines)); os.replace(path + ".tmp", path)
print(f"shifted {n} refs")
