"""Public procedural fixtures only: no Paint test/train images used in UI development."""
from pathlib import Path
import hashlib
import json
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.drawai.native_runtime import configure_windows_dlls
configure_windows_dlls()
from training.drawai.paint_canvas import pixels64

cases={
 'diagonal':[[{'x':80,'y':80},{'x':320,'y':320}]],
 'edge':[[{'x':0,'y':0},{'x':400,'y':0},{'x':400,'y':400}]],
 'two-strokes':[[{'x':51.25,'y':63.5},{'x':100.75,'y':275.125}],[{'x':300,'y':35},{'x':310,'y':340}]],
 'round-dot':[[{'x':200,'y':200}]],
 'duplicate':[[{'x':125.5,'y':78.125},{'x':125.5,'y':78.125},{'x':270.25,'y':301.75}]],
}
out=ROOT/'artifacts/drawai/phase7-image64';out.mkdir(parents=True,exist_ok=True)
fixtures=[]
for name,strokes in cases.items():
    fixtures.append(dict(name=name,drawing=dict(schema_version='drawai-strokes-v1',coordinate_system='canvas-y-down',
       canvas=dict(width=400,height=400),strokes=strokes,source=dict(kind='synthetic')),pixels_sha256=hashlib.sha256(pixels64(strokes).tobytes()).hexdigest()))
(out/'canvas-fixtures.json').write_text(json.dumps(dict(source='procedural; no human dataset samples',cases=fixtures),indent=2)+'\n',encoding='utf-8')
print(f'Wrote {len(fixtures)} reference fixtures')
