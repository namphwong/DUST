"""Read-only inventory of Git-tracked DrawAI files, syntax, local doc links and duplicates."""
import ast
import collections
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote,urlsplit

ROOT=Path(__file__).resolve().parents[1]


def audit():
    names=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    names=sorted(name for name in names if name)
    tracked=set(names);groups=collections.defaultdict(list);errors=[];links=[];local_only=[]
    python_count=json_count=markdown_count=0
    snapshots=[]
    for name in names:
        path=ROOT/name
        if not path.is_file():errors.append(dict(path=name,error='tracked file missing'));continue
        payload=path.read_bytes();groups[hashlib.sha256(payload).hexdigest()].append(name)
        try:
            if path.suffix=='.py':ast.parse(payload.decode('utf-8-sig'),filename=name);python_count+=1
            elif path.suffix=='.json':json.loads(payload.decode('utf-8-sig'));json_count+=1
        except (ValueError,SyntaxError,UnicodeError) as error:errors.append(dict(path=name,error=str(error)))
        if path.suffix!='.md':continue
        markdown_count+=1
        text=payload.decode('utf-8-sig')
        for match in re.finditer(r'!?\[[^\]\n]*\]\(([^)\n]+)\)',text):
            href=match.group(1).strip()
            if href.startswith('<'):href=href[1:href.find('>')]
            else:href=href.split(' "',1)[0]
            url=urlsplit(href)
            if url.scheme or url.netloc or not url.path:continue
            target=(ROOT/unquote(url.path).lstrip('/')) if url.path.startswith('/') else path.parent/unquote(url.path)
            target=target.resolve()
            if not target.is_relative_to(ROOT):continue
            relative=target.relative_to(ROOT).as_posix()
            row=dict(source=name,line=text[:match.start()].count('\n')+1,href=href,target=relative)
            historical=name.startswith('artifacts/') and any(part in name.split('/') for part in ('pre-rerun','pre-p3','pre-p4'))
            if not target.exists():
                if historical:snapshots.append(row)
                else:links.append(row)
            elif target.is_file() and relative not in tracked:local_only.append(row)
    return dict(tracked_files=len(names),by_top_level=dict(sorted(collections.Counter(n.split('/')[0] for n in names).items())),
        python_syntax_checked=python_count,json_checked=json_count,markdown_checked=markdown_count,syntax_errors=errors,
        broken_local_links=links,historical_snapshot_relative_links=snapshots,local_only_link_targets=local_only,
        duplicate_groups=[dict(sha256=digest,paths=paths) for digest,paths in groups.items() if len(paths)>1],
        retired_project_code=[n for n in names if re.search(r'(motionsense|uci.har|csv.pipeline|video.fusion)',n,re.I)])


if __name__=='__main__':print(json.dumps(audit(),ensure_ascii=False,indent=2))
