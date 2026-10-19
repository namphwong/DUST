"""Independent Python reference for the P7 fixed-canvas grayscale coverage adapter."""
import math
import numpy as np


def pixels64(strokes):
    mask=np.zeros((256,256),dtype=np.uint8);scale=256/400;radius=3*scale
    for stroke in strokes:
        for i,q in enumerate(stroke):
            p=stroke[max(0,i-1)]
            ax,ay,bx,by=(p['x']*scale,p['y']*scale,q['x']*scale,q['y']*scale)
            dx=bx-ax;dy=by-ay;den=dx*dx+dy*dy
            for y in range(max(0,math.floor(min(ay,by)-radius)),min(255,math.ceil(max(ay,by)+radius))+1):
                for x in range(max(0,math.floor(min(ax,bx)-radius)),min(255,math.ceil(max(ax,bx)+radius))+1):
                    t=max(0,min(1,((x+.5-ax)*dx+(y+.5-ay)*dy)/den)) if den else 0
                    if (x+.5-ax-t*dx)**2+(y+.5-ay-t*dy)**2<=radius*radius:mask[y,x]=1
    ink=mask.reshape(64,4,64,4).sum((1,3))
    return np.floor(255*(16-ink)/16+.5).astype(np.uint8)
