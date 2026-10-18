/** drawai-geometry-v1: must match training/drawai/preprocessing.py. */
export type Point = { x: number; y: number; t?: number };
export type Stroke = Point[];
export const GEOMETRY_VERSION = 'drawai-geometry-v1';

export function preprocess(strokes: Stroke[], length = 64, yAxis = 'down'): Float32Array {
  if (![64,128].includes(length)) throw new Error('LENGTH: expected 64 or 128');
  if (!['up','down'].includes(yAxis)) throw new Error('AXIS: expected up or down');
  if (!Array.isArray(strokes) || !strokes.length) throw new Error('EMPTY: draw a digit first');
  if (strokes.length > 16) throw new Error('STROKES: maximum 16 strokes');
  let count = 0;
  const clean = strokes.map(stroke => {
    if (!Array.isArray(stroke) || !stroke.length) throw new Error('EMPTY_STROKE: stroke cannot be empty');
    count += stroke.length;
    if (count > 8192) throw new Error('POINTS: maximum 8192 points');
    const points: [number,number][] = [];
    for (const point of stroke) {
      if (!point || typeof point.x !== 'number' || typeof point.y !== 'number' ||
          !Number.isFinite(point.x) || !Number.isFinite(point.y) || Math.max(Math.abs(point.x),Math.abs(point.y)) > 1e6) {
        throw new Error('NUMBER: finite numeric x/y within +/-1e6 required');
      }
      const value: [number,number] = [point.x, yAxis === 'up' ? -point.y : point.y];
      const previous = points.at(-1);
      if (!previous || previous[0] !== value[0] || previous[1] !== value[1]) points.push(value);
    }
    return points;
  });
  const all = clean.flat();
  const minX = Math.min(...all.map(p=>p[0])), maxX = Math.max(...all.map(p=>p[0]));
  const minY = Math.min(...all.map(p=>p[1])), maxY = Math.max(...all.map(p=>p[1]));
  const scale = Math.max(maxX-minX,maxY-minY);
  if (scale <= 0) throw new Error('TOO_SHORT: need a stroke with movement');
  const normalized = clean.map(stroke=>stroke.map(([x,y])=>[(x-(minX+maxX)/2)/scale,(y-(minY+maxY)/2)/scale]));
  const distances = normalized.map(stroke=> {
    const cumulative = [0];
    for (let i=1;i<stroke.length;i++) cumulative.push(cumulative[i-1]+Math.hypot(stroke[i][0]-stroke[i-1][0],stroke[i][1]-stroke[i-1][1]));
    return cumulative;
  });
  const arcs = distances.map(d=>d.at(-1)!);
  const total = arcs.reduce((a,b)=>a+b,0);
  if (total <= 1e-12) throw new Error('TOO_SHORT: isolated points are not a moving stroke');
  const budgets = normalized.map(s=>s.length > 1 ? 2 : 1);
  const remaining = length-budgets.reduce((a,b)=>a+b,0);
  if (remaining < 0) throw new Error('BUDGET: not enough tokens to retain all strokes');
  const shares = arcs.map(arc=>remaining*arc/total);
  const extras = shares.map(Math.floor);
  const order = shares.map((share,i)=>({i,fraction:share-extras[i]})).sort((a,b)=>b.fraction-a.fraction || a.i-b.i);
  for (let i=0;i<remaining-extras.reduce((a,b)=>a+b,0);i++) budgets[order[i].i]++;
  extras.forEach((extra,i)=>budgets[i]+=extra);
  const result = new Float32Array(3*length);
  let offset = 0;
  normalized.forEach((stroke,index)=> {
    const budget = budgets[index], distance = distances[index];
    result[2*length+offset] = 1;
    let right = 1;
    for (let i=0;i<budget;i++) {
      let point = stroke[0];
      if (stroke.length > 1) {
        const target = i === budget-1 ? arcs[index] : i*arcs[index]/(budget-1);
        while (right < stroke.length-1 && distance[right] < target) right++;
        const factor = (target-distance[right-1])/(distance[right]-distance[right-1]);
        point = [0,1].map(c=>stroke[right-1][c]+factor*(stroke[right][c]-stroke[right-1][c]));
      }
      result[offset] = point[0];
      result[length+offset] = point[1];
      offset++;
    }
  });
  if (offset !== length || result.some(v=>!Number.isFinite(v))) throw new Error('INTERNAL: incorrect tensor');
  return result;
}
