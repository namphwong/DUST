"""Geometry invariants and strict PenDigits parsing (stdlib unittest)."""
import unittest
import copy

import numpy as np

from training.drawai.prepare_data import parse_unipen, split_development
from training.drawai.preprocessing import DrawingError, contract, preprocess


def points(*pairs):
    return [dict(x=x, y=y) for x, y in pairs]


class GeometryTests(unittest.TestCase):
    def test_horizontal_line_matches_hand_calculation(self):
        value = preprocess([points((0, 0), (10, 0))])
        np.testing.assert_allclose(value[0], np.linspace(-0.5, 0.5, 64), atol=3e-8)
        np.testing.assert_array_equal(value[1], np.zeros(64))
        np.testing.assert_array_equal(value[2], [1] + [0] * 63)

    def test_vertical_line_and_source_axis(self):
        down = preprocess([points((0, 10), (0, 0))], y_axis='up')
        np.testing.assert_array_equal(down, preprocess([points((0, -10), (0, 0))]))
        np.testing.assert_allclose(down[1], np.linspace(-0.5, 0.5, 64), atol=3e-8)

    def test_translation_and_scale_invariance(self):
        strokes = [points((0, 0), (1, 2), (3, 1)), points((4, 3), (5, 1))]
        shifted = [[dict(x=p['x']*7+100, y=p['y']*7-40) for p in s] for s in strokes]
        np.testing.assert_allclose(preprocess(strokes), preprocess(shifted), atol=3e-8)

    def test_two_strokes_are_not_connected(self):
        value = preprocess([points((0, 0), (1, 0)), points((9, 0), (10, 0))])
        self.assertEqual(np.flatnonzero(value[2]).tolist(), [0, 32])
        self.assertAlmostEqual(float(value[0,31]), -0.4, places=6)
        self.assertAlmostEqual(float(value[0,32]), 0.4, places=6)
        self.assertFalse(np.any((value[0] > -0.39) & (value[0] < 0.39)))

    def test_largest_remainder_and_stable_tie(self):
        strokes = [points((i*3, 0), (i*3+1, 0)) for i in range(3)]
        value = preprocess(strokes)
        self.assertEqual(np.flatnonzero(value[2]).tolist(), [0, 22, 43])

    def test_singleton_preserved_not_repeated(self):
        value = preprocess([points((0, 0)), points((1, 0), (2, 0))])
        self.assertEqual(np.flatnonzero(value[2]).tolist(), [0, 1])
        self.assertEqual(value.shape, (3,64))

    def test_duplicates_do_not_change_geometry(self):
        a = [points((0,0), (0,0), (1,2), (1,2), (2,0))]
        b = [points((0,0), (1,2), (2,0))]
        np.testing.assert_array_equal(preprocess(a), preprocess(b))

    def test_input_is_not_mutated(self):
        strokes = [points((0,0), (0,0), (2,1)), points((4,0))]
        before = copy.deepcopy(strokes)
        preprocess(strokes,128,'up')
        self.assertEqual(strokes,before)

    def test_aspect_ratio_and_endpoints_are_preserved(self):
        for length in (64,128):
            value = preprocess([points((0,0),(4,2))],length)
            np.testing.assert_array_equal(value[:2,0],[-0.5,-0.25])
            np.testing.assert_array_equal(value[:2,-1],[0.5,0.25])
            np.testing.assert_allclose(value[1],value[0]/2,atol=3e-8)

    def test_maximum_strokes_and_points_are_accepted(self):
        strokes = [points((i*2,0),(i*2+1,1)) for i in range(16)]
        for length in (64,128):
            value = preprocess(strokes,length)
            self.assertEqual(value.shape,(3,length))
            self.assertEqual(value[2].sum(),16)
        self.assertEqual(preprocess([points(*[(i,0) for i in range(8192)])]).shape,(3,64))

    def test_point_limit_is_total_across_strokes(self):
        with self.assertRaisesRegex(DrawingError,'^POINTS:'):
            preprocess([points(*[(i,0) for i in range(4097)]),points(*[(i,1) for i in range(4096)])])

    def test_invalid_scalar_types_and_config(self):
        for scalar in (True,np.bool_(False),'1',[1],None,float('inf'),-float('inf')):
            with self.subTest(scalar=str(scalar)), self.assertRaisesRegex(DrawingError,'^NUMBER:'):
                preprocess([[dict(x=0,y=0),dict(x=scalar,y=1)]])
        for point in ({'x':1},None,[],4):
            with self.subTest(point=point), self.assertRaisesRegex(DrawingError,'^NUMBER:'):
                preprocess([[dict(x=0,y=0),point]])
        for length in (0,63,65,129):
            with self.assertRaisesRegex(DrawingError,'^LENGTH:'):
                preprocess([points((0,0),(1,1))],length)
            with self.assertRaisesRegex(DrawingError,'^LENGTH:'):
                contract(length)
        with self.assertRaisesRegex(DrawingError,'^AXIS:'):
            preprocess([points((0,0),(1,1))],y_axis='unknown')

    def test_rejections(self):
        for strokes in [[], [[]], [points((0,0))], [points((0,0)), points((1,1))],
                        [points((0,0), (float('nan'),1))], [points((0,0), (1e7,1))],
                        [dict(x=1,y=2)], [[dict(x='1', y=0)]],
                        [points((0,0), (1,1))]*17, [points(*[(i,0) for i in range(8193)])]]:
            with self.subTest(strokes=str(strokes)[:60]), self.assertRaises(DrawingError):
                preprocess(strokes)


class ParserTests(unittest.TestCase):
    def text(self):
        return '.INCLUDE dene.doc\n.SEGMENT DIGIT 0-1 ? "4"\n.COMMENT 4 9 1\n.PEN_DOWN\n0 10\n10 0\n.PEN_UP\n.DT 100\n.PEN_DOWN\n20 20\n.PEN_UP\n.DT 100\n'

    def test_two_strokes_not_two_samples(self):
        samples = parse_unipen(self.text(), 'fixture')
        self.assertEqual(len(samples), 1)
        self.assertEqual(len(samples[0]['strokes']), 2)
        self.assertEqual(samples[0]['label'], 4)
        self.assertIsNone(samples[0]['writer_id'])
        self.assertEqual(samples[0]['comment'], [4,9,1])

    def test_malformed_data_is_not_silently_skipped(self):
        cases = [self.text().replace('0-1', '0'), self.text().replace('4 9 1', '3 9 1'),
                 self.text().replace('.PEN_UP', '', 1), self.text().replace('0 10', 'nan 10'),
                 self.text().replace('0 10', '501 10'), self.text()+'1 2\n',
                 self.text().replace('.PEN_DOWN', '.UNKNOWN', 1),
                 self.text().replace('.INCLUDE', '.INCLUDE_FAKE'),
                 self.text().replace('.DT 100','.DT 99'),
                 self.text().replace('0-1','1-2'),
                 self.text().replace('.COMMENT 4 9 1',''),
                 self.text().replace('0 10','-1 10')]
        for text in cases:
            with self.subTest(text=text[:60]), self.assertRaises(ValueError):
                parse_unipen(text, 'fixture')

    def test_split_has_no_lost_or_overlapping_sample(self):
        samples = [dict(label=label) for label in range(10) for _ in range(20)]
        train, val = split_development(samples)
        self.assertEqual((len(train),len(val)), (160,40))
        self.assertFalse(set(train)&set(val))
        self.assertEqual(set(train)|set(val), set(range(200)))
        self.assertEqual((train,val), split_development(samples))

    def test_invalid_split_configuration_is_rejected(self):
        samples = [dict(label=label) for label in range(10) for _ in range(2)]
        for fraction in (0,1,-0.2,1.2):
            with self.assertRaises(ValueError):
                split_development(samples,fraction=fraction)
        for missing in ([],samples[:-2],[dict(label=label) for label in range(10)]):
            with self.assertRaises(ValueError):
                split_development(missing)


if __name__ == '__main__':
    unittest.main()
