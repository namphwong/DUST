"""Audit Phase 6 evidence and rerun Python unit tests; never train/tune the model."""
import hashlib
import json
import unittest
from datetime import datetime, timezone
from prepare_uci_har import ROOT, write_json


def read(path):
    return json.loads((ROOT/path).read_text(encoding='utf-8'))


def main():
    base = 'artifacts/phase6/'
    golden = read(base+'python-reference.json')
    parity = read(base+'browser-parity.json')
    ui = read(base+'ui-verification.json')
    performance = read(base+'performance.json')
    browser = read(base+'playwright-results.json')
    release = read('web/public/models/har-swin-kaggle-v1/release.json')
    assert golden['version'] == release['version'] == 'har-swin-kaggle-v1'
    assert all(report['status'] == 'passed' for report in (golden,parity,ui,performance))
    assert [case['name'] for case in golden['cases']] == [case['name'] for case in parity['cases']]
    assert len(golden['cases']) == 17 and all(case['passed'] for case in parity['cases'])
    for case in golden['cases']:
        file = ROOT/(base+'fixtures/'+case['file'])
        assert hashlib.sha256(file.read_bytes()).hexdigest() == case['csv_sha256']
    assert browser['stats']['unexpected'] == 0 and browser['stats']['skipped'] == 0 and browser['stats']['flaky'] == 0
    assert browser['stats']['expected'] >= 29  # Original suite plus later UI regression tests.
    assert performance['sourceRows'] == 100000 and performance['windows'] == 1561
    assert performance['cancelMs'] <= 1000 and performance['maxHeartbeatGapMs'] < performance['heartbeatAcceptanceLimitMs']
    assert performance['staleResultsAbsent'] and performance['renderedRows'] == 20
    assert all(ui[key] for key in ('noNetworkWrites','confirmationRequired','unitChangeClearsResults','allWindowsExported','mobileNoHorizontalOverflow','desktopNoHorizontalOverflow'))
    for name,entry in release['files'].items():
        content = (ROOT/'web/public/models/har-swin-kaggle-v1'/name).read_bytes()
        assert hashlib.sha256(content).hexdigest() == entry['sha256'] and len(content) == entry['bytes']
    suite = unittest.defaultTestLoader.discover(str(ROOT/'training/tests'))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    assert result.wasSuccessful() and result.testsRun == 19
    windows = sum(len(case.get('windows',[])) for case in golden['cases'])
    report = dict(status='passed',phase=6,verifiedAt=datetime.now(timezone.utc).isoformat(),
                  version=release['version'],model_sha256=release['files']['model.onnx']['sha256'],
                  browser_tests=browser['stats']['expected'],python_tests=result.testsRun,
                  csv_fixtures=len(golden['cases']),golden_windows=windows,
                  normalization_max_error=max(case.get('inputMaxError',0) for case in parity['cases']),
                  logits_max_error=max(case.get('logitMaxError',0) for case in parity['cases']),
                  acceptance={f'WEB-{i:02}':'passed' for i in range(1,11)},
                  evidence=['playwright-results.json','python-reference.json','browser-parity.json','ui-verification.json','performance.json','desktop-proof.png','mobile-proof.png'],
                  scope='Local Chromium desktop + mobile viewport. Not a real mobile-device benchmark, accuracy improvement or public Vercel release.')
    write_json(ROOT/(base+'verification.json'),report)
    print(f'PASS: Phase 6, {report["browser_tests"]} browser tests, {result.testsRun} Python tests, {windows} golden windows. Model unchanged.')


if __name__ == '__main__':
    main()
