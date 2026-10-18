"""Sample aggregate RSS of browser descendants of a Playwright worker PID.

This is process-tree RSS (shared pages may be counted twice), not JS heap,
WASM-only memory, unique physical memory, or an exact high-water mark.
"""
import json
import sys
import psutil

rows=[]
for process in psutil.Process(int(sys.argv[1])).children(recursive=True):
    try:
        name=process.name()
        if any(token in name.lower() for token in ('chrome','headless','msedge')):
            rows.append(dict(pid=process.pid,name=name,rss=process.memory_info().rss))
    except (psutil.NoSuchProcess,psutil.AccessDenied): pass
print(json.dumps(dict(rss=sum(r['rss'] for r in rows),processes=rows)))
