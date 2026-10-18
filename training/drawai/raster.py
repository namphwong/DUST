"""drawai-raster-v1: deterministic RGB rasterization, shared with TypeScript.

Pixel (column,row) is sampled at its integer coordinate (pixel centers),
round segment caps, union of segments, one-pixel analytic edge coverage.
No canvas screenshots, font engine, Pillow drawing, resize or crop is used.
"""
from numbers import Real
import numpy as np
from .preprocessing import DrawingError, MAX_POINTS, MAX_STROKES

VERSION = 'drawai-raster-v1'
SIZE, PADDING, STROKE_WIDTH = 224, 24, 8
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def contract():
    return dict(version=VERSION, size=SIZE, layout='NCHW', dtype='float32',
                channels=['R','G','B'], y_axis='down', padding_pixels=PADDING,
                stroke_width_pixels=STROKE_WIDTH, background_rgb=[255,255,255],
                ink_rgb=[0,0,0], preserve_aspect_ratio=True, connect_strokes=False,
                pixel_centers='integer coordinates 0..223; center 111.5',
                renderer='union of round-capped segments; coverage=clip(4.5-distance,0,1)',
                byte_quantization='floor(255*(1-coverage)+0.5)',
                rescale_factor=1/255, image_mean=MEAN, image_std=STD,
                normalization_source='Verified pinned timm DINOv3 qkvb publisher config: ImageNet mean/std; 224px stroke-raster adaptation',
                extra_resize=False, extra_crop=False, timestamps_used=False,
                max_strokes=MAX_STROKES, max_points=MAX_POINTS)


def normalized_strokes(strokes, y_axis='down'):
    if y_axis not in ('up','down'):
        raise DrawingError('AXIS: expected up or down')
    if not isinstance(strokes,(list,tuple)) or not strokes:
        raise DrawingError('EMPTY: draw a digit first')
    if len(strokes)>MAX_STROKES:
        raise DrawingError('STROKES: maximum 16 strokes')
    clean=[]; count=0; movement=False
    for stroke in strokes:
        if not isinstance(stroke,(list,tuple)) or not stroke:
            raise DrawingError('EMPTY_STROKE: stroke cannot be empty')
        count+=len(stroke)
        if count>MAX_POINTS: raise DrawingError('POINTS: maximum 8192 points')
        points=[]
        for p in stroke:
            if not isinstance(p,dict) or any(not isinstance(p.get(k),Real) or isinstance(p.get(k),(bool,np.bool_)) for k in ('x','y')):
                raise DrawingError('NUMBER: finite numeric coordinates required')
            try: x,y=float(p['x']),float(p['y'])
            except (OverflowError,ValueError): raise DrawingError('NUMBER: finite numeric coordinates required') from None
            if not np.isfinite([x,y]).all() or max(abs(x),abs(y))>1e6:
                raise DrawingError('NUMBER: coordinates outside supported range')
            point=(x,-y if y_axis=='up' else y)
            if not points or point!=points[-1]: points.append(point)
        movement |= len(points)>1
        clean.append(points)
    if not movement: raise DrawingError('TOO_SHORT: need a moving stroke')
    all_points=[p for stroke in clean for p in stroke]
    minimum=[min(p[k] for p in all_points) for k in (0,1)]
    maximum=[max(p[k] for p in all_points) for k in (0,1)]
    center=[(minimum[k]+maximum[k])/2 for k in (0,1)]
    span=max(maximum[k]-minimum[k] for k in (0,1))
    # Operation order mirrors JS exactly; all geometry uses float64.
    extent=SIZE-1-2*PADDING
    return [[((x-center[0])/span*extent+(SIZE-1)/2,
              (y-center[1])/span*extent+(SIZE-1)/2) for x,y in stroke] for stroke in clean]


def render_rgb(strokes, y_axis='down'):
    clean=normalized_strokes(strokes,y_axis)
    distance=np.full((SIZE,SIZE),np.inf,dtype=np.float64)
    radius=STROKE_WIDTH/2+0.5
    for stroke in clean:
        segments=list(zip(stroke,stroke[1:])) if len(stroke)>1 else [(stroke[0],stroke[0])]
        for (ax,ay),(bx,by) in segments:
            left=max(0,int(np.floor(min(ax,bx)-radius))); right=min(SIZE-1,int(np.ceil(max(ax,bx)+radius)))
            top=max(0,int(np.floor(min(ay,by)-radius))); bottom=min(SIZE-1,int(np.ceil(max(ay,by)+radius)))
            xx,yy=np.meshgrid(np.arange(left,right+1,dtype=np.float64),np.arange(top,bottom+1,dtype=np.float64))
            dx,dy=bx-ax,by-ay; denominator=dx*dx+dy*dy
            t=np.clip(((xx-ax)*dx+(yy-ay)*dy)/denominator,0,1) if denominator else np.zeros_like(xx)
            d=np.sqrt((xx-(ax+t*dx))**2+(yy-(ay+t*dy))**2)
            view=distance[top:bottom+1,left:right+1]
            np.minimum(view,d,out=view)
    coverage=np.clip(radius-distance,0,1)
    gray=np.floor(255*(1-coverage)+0.5).astype(np.uint8)
    return np.repeat(gray[:,:,None],3,axis=2)


def normalize_rgb(rgb):
    if rgb.dtype!=np.uint8 or rgb.shape!=(SIZE,SIZE,3):
        raise DrawingError('IMAGE: expected uint8 RGB 224x224')
    channels=rgb.transpose(2,0,1).astype(np.float32)/np.float32(255)
    return ((channels-np.asarray(MEAN,dtype=np.float32)[:,None,None])/
            np.asarray(STD,dtype=np.float32)[:,None,None]).astype(np.float32)


def preprocess(strokes,y_axis='down'):
    return normalize_rgb(render_rgb(strokes,y_axis))
