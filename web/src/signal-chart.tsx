import type { SignalPoint } from './csv-preview';

export function SignalChart({ points, unit, sourceRows, gaps }: { points: SignalPoint[]; unit: 'g' | 'ms2'; sourceRows?: number; gaps?: [SignalPoint,SignalPoint][] }) {
  const scale = unit === 'ms2' ? 9.80665 : 1;
  const first = points[0].timestampMs;
  const duration = Math.max(1, points[points.length-1].timestampMs - first);
  let segment = 0;
  const selected = points.map((point, i) => {
    if (point.segmentId !== undefined) segment = point.segmentId;
    else if (i && point.timestampMs-points[i-1].timestampMs > 100) segment++;
    return { point, segment };
  }).filter((_, i) => i % Math.max(1, Math.ceil(points.length/600)) === 0 || i === points.length-1);
  let maxAbs = 1;
  for (const { point } of selected) for (const value of point.acc) maxAbs = Math.max(maxAbs, Math.abs(value/scale));
  const x = (point: SignalPoint) => 52 + (point.timestampMs-first)/duration*680;
  const y = (value: number) => 118 - (value/scale/maxAbs)*80;
  const colors = ['#0c8368', '#5684d1', '#b27648'];
  const gapBounds: [SignalPoint, SignalPoint][] = [];
  for (let i=1; i<points.length; i++) if ((points[i].segmentId !== undefined ? points[i].segmentId !== points[i-1].segmentId : points[i].timestampMs-points[i-1].timestampMs > 100) && gapBounds.length < 200) gapBounds.push([points[i-1], points[i]]);
  return <figure className="signal-figure">
    <div className="chart-heading"><div><h3>Gia tốc ba trục</h3><p>Đơn vị hiển thị g · thời gian tương đối</p></div><div className="legend">{['X', 'Y', 'Z'].map((label, i) => <span key={label}><i style={{ background: colors[i] }}/>{label}</span>)}</div></div>
    <svg viewBox="0 0 760 250" role="img" aria-label="Biểu đồ gia tốc X, Y, Z theo thời gian; vùng vàng biểu thị khoảng mất dữ liệu">
      {[38, 78, 118, 158, 198].map((height, i) => <g key={height}><line x1="52" y1={height} x2="732" y2={height} stroke="#e7ebe5"/><text x="42" y={height+4} textAnchor="end">{((2-i)*maxAbs/2).toPrecision(2)}</text></g>)}
      {(gaps ?? gapBounds).map(([a,b], i) => <rect key={i} x={x(a)} y="28" width={Math.max(1, x(b)-x(a))} height="180" fill="#f2d38f" opacity=".45"/>)}
      {[0,1,2].map(channel => {
        let path = '';
        for (let i=0; i<selected.length; i++) {
          const p = selected[i].point;
          // Never draw interpolation across a gap >100 ms in the original data.
          const discontinuity = !i || selected[i-1].segment !== selected[i].segment;
          path += `${discontinuity ? 'M' : 'L'}${x(p).toFixed(2)},${y(p.acc[channel]).toFixed(2)} `;
        }
        return <path key={channel} d={path} fill="none" stroke={colors[channel]} strokeWidth="1.7"/>;
      })}
      {[0, .25, .5, .75, 1].map(f => <text key={f} x={52+680*f} y="234" textAnchor="middle">{(duration*f/1000).toFixed(2)} s</text>)}
    </svg>
    <figcaption>Preview {(sourceRows ?? points.length).toLocaleString('vi-VN')} điểm nguồn; biểu đồ rút gọn tối đa ~600 điểm. Không nối giữa các đoạn; tô nền tối đa 200 khoảng mất mẫu từ timestamp gốc. Tín hiệu nguồn hiển thị trước resample.</figcaption>
  </figure>;
}
