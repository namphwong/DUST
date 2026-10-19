"""Independent D6 local audit. Human collection and real touch remain NOT RUN."""
import json
import io
import unittest
import numpy as np

from .prepare_data import ROOT,digest,write_json
from .preprocessing import preprocess


def main():
    out=ROOT/'artifacts/drawai/verification';bench=ROOT/'artifacts/drawai/benchmark'
    stream=io.StringIO()
    suite=unittest.defaultTestLoader.loadTestsFromName('training.tests.test_drawai_canvas')
    unit=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    if not unit.wasSuccessful() or unit.testsRun!=4:raise ValueError(stream.getvalue())
    write_json(out/'canvas-gate-tests.json',dict(status='passed',tests=unit.testsRun,
               source_sha256=digest(ROOT/'training/tests/test_drawai_canvas.py'),output=stream.getvalue(),
               synthetic_temporary_fixtures=True,human_samples_created=0))
    tests=json.loads((out/'playwright-report.json').read_text())
    assert tests['stats']['expected']==34 and tests['stats']['unexpected']==tests['stats']['flaky']==tests['stats']['skipped']==0
    fixture_path=ROOT/'web/public/verification/drawai/d6-fixtures.json'
    fixtures=json.loads(fixture_path.read_text());provenance=json.loads((out/'fixture-provenance.json').read_text())
    assert digest(fixture_path)==provenance['fixture_sha256'] and provenance['human_data'] is False
    release=ROOT/'web/public/models/drawai-swin-v1/release.json'
    assert fixtures['release_sha256']==digest(release)
    summaries=[]
    for browser in ['chrome','edge']:
        evidence=json.loads((out/f'{browser}-parity.json').read_text());result=evidence['report']
        assert evidence['status']=='passed' and result['passed'] and evidence['releaseSha256']==digest(release)
        assert len(result['cases'])==17 and len(result['invalid'])==7 and all(c['passed'] for c in result['invalid'])
        assert result['modelSha256']==fixtures['model_sha256']
        tensor_error=0.;logit_error=0.
        for case,reference in zip(result['cases'],fixtures['cases']):
            assert case['name']==reference['name'] and case['passed']
            python=preprocess(reference['drawing']['strokes'],128).reshape(-1)
            np.testing.assert_array_equal(python,np.asarray(reference['input'],np.float32))
            np.testing.assert_allclose(case['input'],python,atol=2e-7,rtol=0)
            tensor_error=max(tensor_error,float(np.max(np.abs(np.asarray(case['input'])-python))))
            for logits in [reference['onnx_logits'],case['logits'],case['product']['logits']]:
                np.testing.assert_allclose(logits,reference['pytorch_logits'],atol=1e-5,rtol=1e-4)
                assert np.argmax(logits)==np.argmax(reference['pytorch_logits'])
                logit_error=max(logit_error,float(np.max(np.abs(np.asarray(logits)-reference['pytorch_logits']))))
            product=case['product'];assert product['drawing_revision']>=0 and product['model_version']=='drawai-swin-v1'
            expected=np.exp(np.asarray(product['logits'])-max(product['logits']));expected/=expected.sum()
            np.testing.assert_allclose(product['scores'],expected,atol=1e-12,rtol=1e-12)
        timings=result['timings'];assert len(timings)==100
        assert result['benchmark']==dict(warmup=5,iterations=100,input='train-digit-0',threads=1,percentile='nearest rank')
        assert all(np.isfinite(t[k]) and t[k]>=0 for t in timings for k in ['preprocessing','inference'])
        ordered=sorted(t['inference'] for t in timings);assert ordered[94]==result['p95Ms'] and (ordered[49]+ordered[50])/2==result['medianMs']
        assert result['p95Ms']<=100
        assert all(r['method']=='GET' and r['url'].startswith('http://127.0.0.1:4186/') for r in evidence['requests'])
        stages={s['stage']:s['at'] for s in result['stages']}
        assert 0<=stages['fetching']<=stages['verifying']<=stages['initializing']<=result['loadAndInitMs']
        summary=dict(browser=browser,version=evidence['browserVersion'],user_agent=evidence['userAgent'],
                     cpu=evidence['machine']['cpu'],tensor_max_error=tensor_error,logit_max_error=logit_error,
                     load_ms=stages['verifying']-stages['fetching'],verify_ms=stages['initializing']-stages['verifying'],
                     init_ms=result['loadAndInitMs']-stages['initializing'],preprocess_median_ms=float(np.median([t['preprocessing'] for t in timings])),
                     inference_median_ms=result['medianMs'],inference_p95_ms=result['p95Ms'])
        summaries.append(summary)
    model_bytes=(ROOT/'web/public/models/drawai-swin-v1/drawai_swin1d_pendigits_l128_fp32_v1.onnx').stat().st_size;assert model_bytes<=10*1024*1024
    tasks={'D6-01':'passed-local','D6-02':'passed-local','D6-03':'passed-local','D6-04':'passed-local',
           'D6-05':'desktop-passed; real-touch-not-run','D6-06':'not-run; no participants','D6-07':'not-run; no human data','D6-08':'local-report-complete; full-acceptance-pending'}
    review=dict(phase='D6',status='local-technical-passed; human-acceptance-pending',phase_complete=False,tasks=tasks,
                browser_tests_passed=34,python_canvas_gate_tests_passed=4,valid_json_fixtures=17,invalid_json_fixtures=7,
                model_bytes=model_bytes,benchmark=summaries,human_samples=0,real_touch_tested=False,
                release_sha256=digest(release),fixture_sha256=digest(fixture_path),
                evidence_files={str(p.relative_to(ROOT)):digest(p) for p in [out/'playwright-report.json',out/'chrome-parity.json',out/'edge-parity.json',bench/'chrome.json',bench/'edge.json',out/'canvas-gate-tests.json']},
                source_files={str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/'web/src/drawai').glob('*')) if p.is_file()})
    for p in [ROOT/'training/drawai/prepare_integration.py',ROOT/'training/drawai/verify_integration.py',
              ROOT/'training/drawai/evaluate_canvas.py',ROOT/'training/tests/test_drawai_canvas.py',
              ROOT/'web/tests/drawai-integration.spec.ts',ROOT/'web/playwright.drawai-integration.config.ts']:
        review['source_files'][str(p.relative_to(ROOT))]=digest(p)
    lines=['# DrawAI — Nghiệm thu D6','',
           'Ngày 11/10/2026. **Kiểm chứng kỹ thuật local đạt; D6 chưa hoàn thành nghiệm thu người thật.** Theo lựa chọn của người dùng, chưa có người/thiết bị để thu nét; không thay dữ liệu đó bằng fixture hoặc Pointer Events tự động.',
           '', '| Task | Trạng thái |','|---|---|']
    lines += [f'| {task} | {state} |' for task,state in tasks.items()]
    lines += ['', '## Kiểm chứng toàn tuyến và lỗi','',
              'Build TypeScript/Vite và 34 browser tests đạt trên Chrome/Edge cài trên máy (headless, profile mới). 17 fixture JSON gồm 7 hình học và 10 train PenDigits: một/nhiều nét, dọc/ngang, điểm trùng, singleton trong bản vẽ có chuyển động; thêm 7 JSON lỗi. Cùng JSON được xử lý bằng Python, đường preprocessing/inference dùng chung với sản phẩm và một InferenceClient/worker sản phẩm thật. Verifier độc lập so tensor, logits PyTorch/ORT/WASM, argmax và softmax.',
              'Race: nhận dạng rồi vẽ/xóa/undo/nạp mẫu trong lúc tải trì hoãn; response cũ bị bỏ, request tiếp theo hoạt động. Nhấn nhanh không chạy trùng, cancel/resize/DPR2/pointer cạnh tranh đạt. Worker crash được mô phỏng bằng ErrorEvent; timeout dùng đồng hồ page chạy nhanh qua 60 giây, sau reset/retry có inference thật. Đây là phép thử điều khiển lỗi, không phải đợi stall thực 60 giây hoặc chứng minh mọi tình huống browser crash.',
              'Tải model 404, hash sai, hợp đồng sai dù hash nhất quán, release cache cũ và WASM 404 đều bị từ chối/giữ nét; retry sau khi sửa nguồn hoạt động. Request inference/demo/technical chỉ GET static assets cùng origin, không có POST nét; worker giữ session cho request tiếp theo. Loader đã chuẩn hóa lỗi mạng thành FETCH, lỗi khởi tạo thành INIT; client kiểm tra thêm preprocessing version và timing hữu hạn; worker từ chối message null thay vì crash.',
              '', '## Benchmark laptop','',
              '| Browser / version | Load ms | Verify ms | Init ms | Preprocess median ms | Inference median ms | Inference p95 ms |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for row in summaries:lines.append(f"| {row['browser']} {row['version']} | {row['load_ms']:.2f} | {row['verify_ms']:.2f} | {row['init_ms']:.2f} | {row['preprocess_median_ms']:.2f} | {row['inference_median_ms']:.2f} | {row['inference_p95_ms']:.2f} |")
    lines += ['',f"ONNX {model_bytes:,} bytes ({model_bytes/1024**2:.2f} MiB), đạt ≤10 MiB; p95 đạt ≤100 ms ở cả hai browser. Máy {summaries[0]['cpu']}. WASM CPU 1 thread, cùng fixture train-digit-0, 5 warm-up + 100 inference; p95 nearest rank. Load/verify/init là một lần cold session trong profile mới, có thể chịu OS/network cache; không phải thống kê nhiều lần cold load. Benchmark dùng shared inference path trong worker, đo session.run riêng; request sản phẩm cùng đường này đã được đối chiếu logits. Không suy latency laptop thành latency điện thoại, không tính WASM runtime vào dung lượng ONNX.",
              '', '| Browser | Tensor max error | Logits max error (ORT/WASM vs PyTorch) |','|---|---:|---:|']
    for row in summaries:lines.append(f"| {row['browser']} | {row['tensor_max_error']:.3g} | {row['logit_max_error']:.3g} |")
    lines += ['', 'Tensor tolerance 2e-7; logits abs(error) ≤ 1e-5 + 1e-4×abs(PyTorch). Mọi argmax khớp. Chrome/Edge desktop, viewport 1366×768/390×844 và DPR1/2 đã kiểm tra; mobile viewport và synthetic pointer không phải điện thoại/cảm ứng/bút thật.',
              '', '## Nét người thật và điều kiện còn thiếu','',
              '**0 mẫu người tham gia; accuracy/macro-F1 canvas: NOT RUN.** [Protocol cố định](canvas-protocol.md), template manifest và CLI validate/evaluate đã chuẩn bị. Bốn tests dùng dữ liệu synthetic trong thư mục tạm kiểm tra đủ ma trận 250 mẫu, thiếu/duplicate, source/consent/attestation, timestamp và path traversal; chúng không tạo artifact người thật hoặc metric canvas.',
              'Cần ít nhất 5 người × 10 chữ số × 5 lần = 250 JSON với consent, ID ẩn danh và device metadata. Người cung cấp phải xác nhận nguồn người thật; công cụ không tự chứng minh tính xác thực. Toàn bộ tập giữ làm evaluation cho release đã khóa, không fine-tune hoặc chọn lại model bằng kết quả này. Cần thử thiết bị cảm ứng/bút thật riêng trước nghiệm thu D6 đầy đủ. D7 có thể chuẩn bị demo nghiên cứu nhưng không gọi D6 hoàn thành toàn bộ.',
              '', '## Tái lập','', '```powershell',
              "$env:DUST_DLL_DIRECTORY = 'C:/Users/admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin'",
              './.venv-phase1/Scripts/python.exe -m training.drawai.prepare_integration',
              './.venv-phase1/Scripts/python.exe -m unittest training.tests.test_drawai_canvas -v',
              'Set-Location web','npm run build','npx playwright test --config playwright.drawai-integration.config.ts','Set-Location ..',
              './.venv-phase1/Scripts/python.exe -m training.drawai.verify_integration','```',
              '', 'Cần Chrome/Edge cài local cho channels chrome/msedge; không tự đổi sang Chromium để gọi đó là Chrome/Edge. Python verification dùng requirements-verification.txt; Linux bỏ biến DLL. Tests port4186; preview4183 có `/drawai` và `/drawai/integration-check`. Evidence tại artifacts/drawai/verification/ và benchmark/; d6-review.json ghi hash, PASS/NOT RUN và phase_complete=false.']
    report=ROOT/'docs/drawai/acceptance-report.md';report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    review['report_sha256']=digest(report);write_json(out/'d6-review.json',review)
    print('PASS local D6: 34 browser tests, 17 fixtures/browser and benchmark audited; human acceptance remains NOT RUN.')


if __name__=='__main__':main()
