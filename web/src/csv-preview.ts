export const MAX_BYTES = 10 * 1024 * 1024;
export const MAX_ROWS = 100_000;
export type SignalPoint = { timestampMs: number; acc: [number, number, number]; segmentId?: number };
export type CsvPreview = { points: SignalPoint[]; extraColumns: string[]; estimatedHz: number | null; gapCount: number; durationMs: number };
export class CsvError extends Error {
  constructor(public code: string, message: string, public line?: number, public column?: string) {
    super(`${code}: ${message}${line ? ` · dòng ${line}` : ''}${column ? ` · cột ${column}` : ''}`);
  }
}

// Strict CSV quoting, including escaped quotes and CRLF/newlines inside quoted fields.
function rowsFromCsv(text: string): { fields: string[]; line: number }[] {
  const rows: { fields: string[]; line: number }[] = [];
  let nonBlankRows = 0;
  let fields: string[] = [], field = '', quoted = false, closed = false, line = 1, rowLine = 1;
  const emit = () => {
    fields.push(field); rows.push({ fields, line: rowLine });
    if (fields.some(v => v.trim())) nonBlankRows++;
    if (nonBlankRows > MAX_ROWS + 1) throw new CsvError('TOO_MANY_ROWS', 'Tối đa 100.000 dòng dữ liệu.');
    fields = []; field = ''; closed = false;
  };
  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    if (quoted) {
      if (char === '"') {
        if (text[i+1] === '"') { field += '"'; i++; }
        else { quoted = false; closed = true; }
      } else { field += char; if (char === '\n') line++; }
    } else if (char === ',') { fields.push(field); field = ''; closed = false; }
    else if (char === '\n' || char === '\r') {
      emit(); if (char === '\r' && text[i+1] === '\n') i++;
      line++; rowLine = line;
    } else if (char === '"') {
      if (field.length || closed) throw new CsvError('INVALID_CSV', 'Dấu ngoặc kép không đúng vị trí.', line);
      quoted = true;
    } else {
      if (closed) throw new CsvError('INVALID_CSV', 'Ký tự sau ô có ngoặc kép không hợp lệ.', line);
      field += char;
    }
  }
  if (quoted) throw new CsvError('INVALID_CSV', 'Ô có ngoặc kép chưa đóng.', rowLine);
  if (field.length || fields.length || closed) emit();
  while (rows.length && rows[rows.length-1].fields.every(v => !v.trim())) rows.pop();
  return rows;
}

export function parseCsvPreview(text: string): CsvPreview {
  if (new TextEncoder().encode(text).byteLength > MAX_BYTES) throw new CsvError('FILE_TOO_LARGE', 'File phải không quá 10 MiB.');
  const rows = rowsFromCsv(text.replace(/^\uFEFF/, ''));
  if (rows.length < 2) throw new CsvError('EMPTY_FILE', 'Cần header và ít nhất một dòng dữ liệu.');
  const header = rows[0].fields.map(v => v.trim());
  if (new Set(header).size !== header.length) throw new CsvError('DUPLICATE_COLUMN', 'Tên cột bị trùng sau khi trim.', 1);
  const required = ['timestamp_ms', 'acc_x', 'acc_y', 'acc_z'];
  const missing = required.filter(name => !header.includes(name));
  if (missing.length) throw new CsvError('MISSING_COLUMN', `Thiếu ${missing.join(', ')}.`, 1);
  const offsets = required.map(name => header.indexOf(name));
  const points: SignalPoint[] = [];
  const numberPattern = /^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/;
  for (const row of rows.slice(1)) {
    if (row.fields.length !== header.length) throw new CsvError('INVALID_CSV', 'Số ô khác số cột header.', row.line);
    const values = offsets.map((offset, i) => {
      const cell = row.fields[offset].trim();
      const value = Number(cell);
      if (!numberPattern.test(cell) || !Number.isFinite(value)) throw new CsvError('INVALID_NUMBER', 'Ô phải là số hữu hạn, không được trống.', row.line, required[i]);
      return value;
    });
    const [timestampMs, x, y, z] = values;
    if (timestampMs < 0 || (points.length && timestampMs <= points[points.length-1].timestampMs)) throw new CsvError('INVALID_TIMESTAMP', 'Timestamp phải không âm và tăng nghiêm ngặt.', row.line, 'timestamp_ms');
    points.push({ timestampMs, acc: [x, y, z] });
  }
  const intervals = points.slice(1).map((p, i) => p.timestampMs - points[i].timestampMs);
  const sorted = intervals.filter(v => v <= 100).sort((a, b) => a-b);
  const middle = Math.floor(sorted.length / 2);
  const median = sorted.length ? (sorted.length % 2 ? sorted[middle] : (sorted[middle-1]+sorted[middle])/2) : null;
  return { points, extraColumns: header.filter(name => !required.includes(name)), estimatedHz: median ? 1000/median : null,
    gapCount: intervals.filter(v => v > 100).length, durationMs: points[points.length-1].timestampMs - points[0].timestampMs };
}

export function channelMajor(points: SignalPoint[]): number[] {
  if (points.length !== 128) throw new CsvError('TOO_SHORT', 'Mẫu UCI phải có đúng 128 điểm.');
  return [0, 1, 2].flatMap(channel => points.map(p => p.acc[channel]));
}
