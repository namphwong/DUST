/** drawai-raster-v1: deterministic pixel-center geometry, not canvas rendering. */
import {type Stroke} from './preprocessing';
export const RASTER_VERSION='drawai-raster-v1';
export const SIZE=224,PADDING=24,STROKE_WIDTH=8;
export const MEAN=[0.485,0.456,0.406],STD=[0.229,0.224,0.225];

export function normalizedStrokes(strokes:Stroke[],yAxis='down'):number[][][]{
  if(!['up','down'].includes(yAxis))throw new Error('AXIS: expected up or down');
  if(!Array.isArray(strokes)||!strokes.length)throw new Error('EMPTY: draw a digit first');
  if(strokes.length>16)throw new Error('STROKES: maximum 16 strokes');
  let count=0,movement=false;
  const clean=strokes.map(stroke=>{
    if(!Array.isArray(stroke)||!stroke.length)throw new Error('EMPTY_STROKE: stroke cannot be empty');
    if((count+=stroke.length)>8192)throw new Error('POINTS: maximum 8192 points');
    const points:number[][]=[];
    for(const p of stroke){
      if(!p||typeof p.x!=='number'||typeof p.y!=='number'||!Number.isFinite(p.x)||!Number.isFinite(p.y)||Math.max(Math.abs(p.x),Math.abs(p.y))>1e6)
        throw new Error('NUMBER: finite numeric coordinates required');
      const point=[p.x,yAxis==='up'?-p.y:p.y],last=points.at(-1);
      if(!last||last[0]!==point[0]||last[1]!==point[1])points.push(point);
    }
    movement ||= points.length>1;return points;
  });
  if(!movement)throw new Error('TOO_SHORT: need a moving stroke');
  const all=clean.flat(),minimum=[0,1].map(k=>Math.min(...all.map(p=>p[k]))),maximum=[0,1].map(k=>Math.max(...all.map(p=>p[k])));
  const center=[0,1].map(k=>(minimum[k]+maximum[k])/2),span=Math.max(maximum[0]-minimum[0],maximum[1]-minimum[1]);
  return clean.map(stroke=>stroke.map(([x,y])=>[(x-center[0])/span*(SIZE-1-2*PADDING)+(SIZE-1)/2,(y-center[1])/span*(SIZE-1-2*PADDING)+(SIZE-1)/2]));
}

export function renderRgb(strokes:Stroke[],yAxis='down'):Uint8Array{
  const clean=normalizedStrokes(strokes,yAxis),distance=new Float64Array(SIZE*SIZE).fill(Infinity),radius=STROKE_WIDTH/2+0.5;
  for(const stroke of clean){
    for(let index=0;index<Math.max(1,stroke.length-1);index++){
      const [ax,ay]=stroke[index],[bx,by]=stroke.length>1?stroke[index+1]:stroke[0],dx=bx-ax,dy=by-ay,denominator=dx*dx+dy*dy;
      const left=Math.max(0,Math.floor(Math.min(ax,bx)-radius)),right=Math.min(SIZE-1,Math.ceil(Math.max(ax,bx)+radius));
      const top=Math.max(0,Math.floor(Math.min(ay,by)-radius)),bottom=Math.min(SIZE-1,Math.ceil(Math.max(ay,by)+radius));
      for(let y=top;y<=bottom;y++)for(let x=left;x<=right;x++){
        const t=denominator?Math.min(1,Math.max(0,((x-ax)*dx+(y-ay)*dy)/denominator)):0;
        const ex=x-(ax+t*dx),ey=y-(ay+t*dy),d=Math.sqrt(ex*ex+ey*ey),i=y*SIZE+x;
        if(d<distance[i])distance[i]=d;
      }
    }
  }
  const rgb=new Uint8Array(SIZE*SIZE*3);
  for(let i=0;i<distance.length;i++){
    const gray=Math.floor(255*(1-Math.min(1,Math.max(0,radius-distance[i])))+0.5);
    rgb[3*i]=rgb[3*i+1]=rgb[3*i+2]=gray;
  }
  return rgb;
}

export function normalizeRgb(rgb:Uint8Array):Float32Array{
  if(!(rgb instanceof Uint8Array)||rgb.length!==SIZE*SIZE*3)throw new Error('IMAGE: expected uint8 RGB 224x224');
  const values=new Float32Array(SIZE*SIZE*3),n=SIZE*SIZE;
  for(let c=0;c<3;c++)for(let i=0;i<n;i++){
    const rescaled=Math.fround(rgb[i*3+c]/255),centered=Math.fround(rescaled-Math.fround(MEAN[c]));
    values[c*n+i]=Math.fround(centered/Math.fround(STD[c]));
  }
  return values;
}

export function rasterize(strokes:Stroke[],yAxis='down'):Float32Array{return normalizeRgb(renderRgb(strokes,yAxis));}
