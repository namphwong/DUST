import { CsvError, MAX_ROWS, type CsvPreview, type SignalPoint, channelMajor } from './csv-preview';
import type { WindowPrediction } from './inference-artifact';
export type Unit = 'g' | 'ms2';
export type Prepared = {
  segments: { id: number; points: SignalPoint[] }[];
  sourceRows: number; resampledRows: number; segmentCount: number;
  interpolatedPoints: number; droppedSegments: number; droppedPoints: number;
  tailPoints: number; windowCount: number; sourceUnit: Unit;
};
export type PreparationSummary = Omit<Prepared, 'segments'>;
export type CsvResult = WindowPrediction & { segmentId: number; windowIndex: number; startMs: number; endMs: number };
export type CompactPreview = CsvPreview & { sourceRows: number; firstRows: SignalPoint[]; gapBounds: [SignalPoint,SignalPoint][] };

export function compactPreview(preview: CsvPreview): CompactPreview {
  const stride = Math.max(1, Math.ceil(preview.points.length / 600));
  let segmentId = 0;
  const points: SignalPoint[] = [];
  const gapBounds: [SignalPoint,SignalPoint][] = [];
  preview.points.forEach((p, i) => {
    if (i && p.timestampMs - preview.points[i-1].timestampMs > 100) {
      segmentId++;
      if (gapBounds.length < 200) gapBounds.push([preview.points[i-1],p]);
    }
    if (i < 5 || i % stride === 0 || i === preview.points.length-1) points.push({ ...p, segmentId });
  });
  return { ...preview, points, gapBounds, sourceRows: preview.points.length, firstRows: preview.points.slice(0,5) };
}
export function prepareCsv(preview: CsvPreview, unit: Unit, gravity: boolean, demo = false): Prepared {
  if (!gravity) throw new CsvError('INCOMPATIBLE_SIGNAL', 'Cần gia tốc bao gồm trọng lực (total acceleration).');
  if (unit !== 'g' && unit !== 'ms2') throw new CsvError('INVALID_UNIT', 'Chọn g hoặc m/s².');
  const origin = preview.points[0].timestampMs;
  const groups: SignalPoint[][] = [];
  for (const p of preview.points) {
    const relative = { ...p, timestampMs: p.timestampMs-origin };
    if (!groups.length || relative.timestampMs - groups[groups.length-1].at(-1)!.timestampMs > 100) groups.push([]);
    groups[groups.length-1].push(relative);
  }
  if (demo && (unit !== 'g' || groups.length !== 1 || preview.points.length !== 128 || preview.points.some((p,i)=>p.timestampMs !== i*20))) throw new CsvError('INVALID_TIMESTAMP', 'Mẫu UCI phải giữ nguyên cửa sổ 128 điểm / 50 Hz.');
  // UCI is already pre-windowed. Never run its samples through interpolation.
  if (demo) return { segments: [{id:0,points:preview.points}], sourceRows:128,resampledRows:128,segmentCount:1,
    interpolatedPoints:0,droppedSegments:0,droppedPoints:0,tailPoints:0,windowCount:1,sourceUnit:'g' };
  const scale = unit === 'ms2' ? 9.80665 : 1;
  const prepared: Prepared = { segments: [], sourceRows: preview.points.length, resampledRows: 0, segmentCount: groups.length,
    interpolatedPoints: 0, droppedSegments: 0, droppedPoints: 0, tailPoints: 0, windowCount: 0, sourceUnit: unit };
  groups.forEach((group, id) => {
    const count = Math.floor((group.at(-1)!.timestampMs-group[0].timestampMs)/20)+1;
    if (!Number.isSafeInteger(count) || prepared.resampledRows+count > MAX_ROWS) throw new CsvError('TOO_MANY_ROWS', 'Sau resample vượt 100.000 điểm; hãy cắt phiên nhỏ hơn.');
    prepared.resampledRows += count;
    if (count < 128) { prepared.droppedSegments++; prepared.droppedPoints += count; return; }
    const intervals = group.slice(1).map((p,i)=>p.timestampMs-group[i].timestampMs).sort((a,b)=>a-b);
    const middle = Math.floor(intervals.length/2);
    const median = intervals.length%2 ? intervals[middle] : (intervals[middle-1]+intervals[middle])/2;
    if (median < 10 || median > 25) throw new CsvError('UNSUPPORTED_FREQUENCY', `Đoạn ${id}: median ${median.toFixed(3)} ms; cần nguồn 40–100 Hz (10–25 ms).`);
    const points: SignalPoint[] = [];
    let left = 0;
    for (let i=0; i<count; i++) {
      const absolute = group[0].timestampMs+i*20;
      while (left+1 < group.length && group[left+1].timestampMs <= absolute) left++;
      const a = group[left], b = group[Math.min(left+1,group.length-1)];
      const ratio = absolute === a.timestampMs ? 0 : (absolute-a.timestampMs)/(b.timestampMs-a.timestampMs);
      if (ratio !== 0) prepared.interpolatedPoints++;
      const acc = a.acc.map((v,c)=>(ratio === 0 ? v : v+(b.acc[c]-v)*ratio)/scale) as [number,number,number];
      if (acc.some(v=>!Number.isFinite(v))) throw new CsvError('INVALID_NUMBER','Giá trị không hữu hạn sau nội suy/chuyển đơn vị.');
      points.push({ timestampMs: absolute, acc, segmentId: id });
    }
    const countWindows = Math.floor((count-128)/64)+1;
    prepared.windowCount += countWindows;
    prepared.tailPoints += count-((countWindows-1)*64+128);
    prepared.segments.push({ id, points });
  });
  if (!prepared.windowCount) throw new CsvError('TOO_SHORT', `Không có đoạn đủ 128 điểm sau resample (${prepared.resampledRows} điểm, ${groups.length} đoạn).`);
  return prepared;
}
export function summary(prepared: Prepared): PreparationSummary {
  const { segments: _, ...rest } = prepared;
  return rest;
}
export function* windows(prepared: Prepared) {
  let windowIndex = 0;
  for (const segment of prepared.segments) for (let start=0; start+128<=segment.points.length; start+=64) {
    const points = segment.points.slice(start,start+128);
    yield { segmentId: segment.id, windowIndex: windowIndex++, startMs: points[0].timestampMs, endMs: points[127].timestampMs, raw: channelMajor(points) };
  }
}
export function resultsCsv(results: CsvResult[], unit: Unit): string {
  const header = ['segment_id','window_index','start_ms','end_ms','predicted_id','predicted_label',...Array.from({length:6},(_,i)=>`score_${i}`),'model_version','contract_version','source_unit'];
  return [header.join(','), ...results.map(r=>[r.segmentId,r.windowIndex,r.startMs,r.endMs,r.predictedId,r.predictedCode,...r.scores,r.version,'1',unit === 'ms2' ? 'm/s²' : 'g'].join(','))].join('\r\n')+'\r\n';
}
