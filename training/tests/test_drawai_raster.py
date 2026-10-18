"""Analytic geometry and contract tests independent of ONNX/pretrained access."""
import copy
import unittest
import numpy as np
from training.drawai.raster import render_rgb,preprocess,normalized_strokes,normalize_rgb,MEAN,STD
from training.drawai.preprocessing import DrawingError


class RasterTests(unittest.TestCase):
    def test_horizontal_centerline_caps_and_padding(self):
        strokes=[[dict(x=0,y=0),dict(x=10,y=0)]]
        image=render_rgb(strokes)
        self.assertEqual(image.shape,(224,224,3));self.assertEqual(image.dtype,np.uint8)
        self.assertTrue((image[111,24:200]==0).all())
        self.assertTrue((image[0]==255).all());self.assertTrue((image[100]==255).all())
        self.assertTrue((image[111,15]==255).all())
        # Half-pixel center means equal coverage in adjacent rows.
        np.testing.assert_array_equal(image[111],image[112])

    def test_no_pen_up_connector_and_singleton_is_preserved(self):
        strokes=[[dict(x=0,y=0),dict(x=0,y=10)], [dict(x=10,y=0),dict(x=10,y=10)], [dict(x=5,y=0)]]
        image=render_rgb(strokes)
        self.assertEqual(int(image[112,112,0]),255)
        self.assertEqual(int(image[24,112,0]),0)

    def test_translation_scale_axis_and_input_immutability(self):
        strokes=[[dict(x=0,y=0),dict(x=10,y=10),dict(x=7,y=10)]];before=copy.deepcopy(strokes)
        transformed=[[dict(x=p['x']*2+30,y=p['y']*2-15) for p in s] for s in strokes]
        np.testing.assert_array_equal(render_rgb(strokes),render_rgb(transformed))
        flipped=[[dict(x=p['x'],y=-p['y']) for p in s] for s in strokes]
        np.testing.assert_array_equal(render_rgb(strokes),render_rgb(flipped,'up'))
        self.assertEqual(strokes,before)

    def test_keep_aspect_ratio_and_duplicate_points(self):
        strokes=[[dict(x=0,y=0),dict(x=20,y=10)]]
        points=normalized_strokes(strokes)[0]
        self.assertAlmostEqual((points[1][0]-points[0][0])/(points[1][1]-points[0][1]),2)
        np.testing.assert_array_equal(render_rgb(strokes),render_rgb([[strokes[0][0]]+strokes[0]]))

    def test_channel_order_and_known_normalization(self):
        rgb=np.zeros((224,224,3),np.uint8);rgb[:,:,0]=255
        tensor=normalize_rgb(rgb)
        self.assertEqual(tensor.shape,(3,224,224));self.assertEqual(tensor.dtype,np.float32)
        expected=[(1-MEAN[0])/STD[0],-MEAN[1]/STD[1],-MEAN[2]/STD[2]]
        np.testing.assert_allclose(tensor[:,0,0],expected,atol=1e-6)

    def test_invalid_inputs(self):
        cases=[[],[[]],[[dict(x=1,y=1)]],[[dict(x=0,y=0)], [dict(x=1,y=1)]],
               [[dict(x=True,y=0),dict(x=1,y=1)]],[[dict(x='0',y=0),dict(x=1,y=1)]],
               [[dict(x=float('nan'),y=0),dict(x=1,y=1)]],
               [[dict(x=0,y=0),dict(x=1,y=float('inf'))]],
               [[dict(x=0,y=0),dict(x=1000001,y=1)]]]
        for value in cases:
            with self.subTest(value=value),self.assertRaises(DrawingError):render_rgb(value)
        with self.assertRaises(DrawingError):render_rgb([[dict(x=0,y=0)] for _ in range(17)])
        with self.assertRaises(DrawingError):render_rgb([[dict(x=i,y=i) for i in range(8193)]])
        with self.assertRaises(DrawingError):preprocess([[dict(x=0,y=0),dict(x=1,y=1)]],'sideways')


if __name__=='__main__':unittest.main()
