import unittest
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from csv_pipeline import parse_csv, prepare_csv, normalize
from verify_phase6_pipeline import fixture_specs, signal_csv


class CsvPipelineTests(unittest.TestCase):
    def test_fixture_contract(self):
        for name,times,unit,reorder,error in fixture_specs():
            with self.subTest(name=name):
                if error:
                    with self.assertRaisesRegex(ValueError,error):
                        prepare_csv(parse_csv(signal_csv(times,unit,reorder)),unit,name != 'incompatible-gravity')
                else:
                    summary, windows = prepare_csv(parse_csv(signal_csv(times,unit,reorder)),unit)
                    self.assertEqual(summary['windowCount'],len(windows))
                    for window in windows:
                        self.assertEqual(window['raw'].shape,(3,128))
                        self.assertEqual(window['endMs']-window['startMs'],2540)

    def test_boundaries_tail_and_identity(self):
        for n,expected,tail in [(128,1,0),(191,1,63),(192,2,0)]:
            values = parse_csv(signal_csv(np.arange(n)*20))
            report, windows = prepare_csv(values)
            self.assertEqual((report['windowCount'],report['tailPoints'],report['interpolatedPoints']),(expected,tail,0))
            np.testing.assert_array_equal(windows[0]['raw'],values[:128,1:].T)

    def test_units_and_layout(self):
        _,g = prepare_csv(parse_csv(signal_csv(np.arange(192)*20)))
        _,ms = prepare_csv(parse_csv(signal_csv(np.arange(192)*20,'ms2')), 'ms2')
        np.testing.assert_allclose(g[0]['raw'],ms[0]['raw'],atol=1e-15)
        self.assertNotEqual(g[0]['raw'][0,0],g[0]['raw'][1,0])

    def test_no_cross_gap(self):
        report, windows = prepare_csv(parse_csv(signal_csv(np.r_[np.arange(192)*20,5000+np.arange(128)*20])))
        self.assertEqual(report['windowCount'],3)
        self.assertEqual([(w['segmentId'],w['startMs'],w['endMs']) for w in windows],[(0,0,2540),(0,1280,3820),(1,5000,7540)])

    def test_normalization_epsilon(self):
        out = normalize(np.zeros((3,128)),{'mean':[0,0,0],'std':[0,1,1]})
        self.assertTrue(np.isfinite(out).all())


if __name__ == '__main__':
    unittest.main()
