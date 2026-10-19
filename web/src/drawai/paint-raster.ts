import {validateDrawing,type Drawing} from './types';
/** Fixed viewport, 6 logical px round black strokes, 4x coverage, no crop or centering.
 * Independent of DOM canvas size, CSS color, browser rasterizer, pointer type and DPR.
 * Paint training images have no trajectory; this is a new adapter, not recovered data.
 */
export function paintPixels(drawing:Drawing):Uint8Array {
  validateDrawing(drawing);
  if(drawing.canvas.width!==400||drawing.canvas.height!==400)throw new Error('CANVAS: Cần viewport logic 400×400.');
  const mask=new Uint8Array(256*256),pixels=new Uint8Array(4096);
  const scale=256/400,radius=3*scale;
  for(const stroke of drawing.strokes){
    for(let i=0;i<stroke.length;i++){
      const p=stroke[Math.max(0,i-1)],q=stroke[i];
      if([p.x,p.y,q.x,q.y].some(v=>v<0||v>400))throw new Error('CANVAS: Điểm ngoài vùng vẽ.');
      const ax=p.x*scale,ay=p.y*scale,bx=q.x*scale,by=q.y*scale,dx=bx-ax,dy=by-ay,den=dx*dx+dy*dy;
      for(let y=Math.max(0,Math.floor(Math.min(ay,by)-radius));y<=Math.min(255,Math.ceil(Math.max(ay,by)+radius));y++)
        for(let x=Math.max(0,Math.floor(Math.min(ax,bx)-radius));x<=Math.min(255,Math.ceil(Math.max(ax,bx)+radius));x++){
          const t=den?Math.max(0,Math.min(1,((x+.5-ax)*dx+(y+.5-ay)*dy)/den)):0;
          if((x+.5-ax-t*dx)**2+(y+.5-ay-t*dy)**2<=radius*radius)mask[y*256+x]=1;
        }
    }
  }
  for(let y=0;y<64;y++)for(let x=0;x<64;x++){
    let ink=0;for(let sy=0;sy<4;sy++)for(let sx=0;sx<4;sx++)ink+=mask[(y*4+sy)*256+x*4+sx];
    pixels[y*64+x]=Math.floor(255*(16-ink)/16+.5);
  }
  if(!pixels.some(v=>v<255))throw new Error('EMPTY: Nét quá nhỏ để tạo ảnh.');
  return pixels;
}
export function pixelsBase64(pixels:Uint8Array){return btoa(String.fromCharCode(...pixels));}
