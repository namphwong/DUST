/** Local P7 stays opt-in independent of the public deployment profile. */
export const PUBLIC_PAINT=import.meta.env.VITE_DRAWAI_PUBLIC==='true';
export function publicApiBase(value:string|undefined=import.meta.env.VITE_API_BASE_URL):string {
  if(!value)throw new Error('CONFIG: Chưa cấu hình địa chỉ backend DrawAI.');
  const u=new URL(value);
  if(u.protocol!=='https:'||u.username||u.password||u.search||u.hash||u.pathname!=='/'||
     /^(localhost|127\.|0\.|\[::1\])/.test(u.hostname))throw new Error('CONFIG: Backend phải là origin HTTPS public.');
  return u.origin;
}
