"""Write an explicitly scoped Swin report without recreating legacy CNN claims."""
import argparse
import json
from pathlib import Path
from .prepare_data import ROOT, write_json
from .verify_training import verify


def report(output, destination):
    output = Path(output).resolve()
    verify(output)
    selection = json.loads((output/'selection.json').read_text(encoding='utf-8'))
    evaluation = json.loads((output/'evaluation/evaluation.json').read_text(encoding='utf-8'))
    lines = ['# DrawAI — Swin baseline', '',
             'Current report contains Swin results only. Historical experiment metadata remains immutable.',
             'DINOv3 and private data results are not available in P1.', '',
             '| Run | Validation macro-F1 | Test accuracy | Test macro-F1 |',
             '|---|---:|---:|---:|']
    for result in evaluation['results']:
        if result['config']['model'] != 'swin':
            continue
        lines.append(f"| {result['run_id']} | {result['validation']['macro_f1']:.6f} | {result['test']['accuracy']:.6f} | {result['test']['macro_f1']:.6f} |")
    lines.extend(['', 'Deployment: `' + selection['deployment_run'] + '` (selected by validation).',
                  'Historical v1 verification is scoped to the selected Swin, not all archived runs.',
                  'PenDigits validation may share writers with train; these are not private-dataset metrics.'])
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    Path(destination).write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'artifacts/drawai/d3-local-v1')
    parser.add_argument('--destination',type=Path,default=ROOT/'artifacts/drawai/swin-report.md')
    args=parser.parse_args(); report(args.output_dir,args.destination)


if __name__ == '__main__': main()
