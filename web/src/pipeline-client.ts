export class PipelineClient {
  private worker = new Worker(new URL('./pipeline-worker.ts', import.meta.url), { type: 'module' });
  private next = 0;
  private reject?: (reason: Error) => void;
  request<T>(action: string, payload: Record<string,unknown> = {}, progress?: (type: string, data: any) => void): Promise<T> {
    const id = ++this.next;
    return new Promise((resolve,reject)=>{
      this.reject = reject;
      this.worker.onerror = event=>{ this.reject = undefined; reject(new Error(event.message || 'Worker không khởi động được.')); };
      this.worker.onmessage = event=>{
        if (event.data.id !== id) return;
        const { type, data } = event.data;
        if (type === 'done' || type === 'error') {
          this.reject = undefined;
          if (type === 'error') reject(new Error(data)); else resolve(data as T);
        } else progress?.(type,data);
      };
      this.worker.postMessage({ id, action, ...payload });
    });
  }
  terminate() { this.worker.terminate(); this.reject?.(new Error('Đã hủy tác vụ.')); this.reject = undefined; }
}
