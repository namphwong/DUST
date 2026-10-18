import { useEffect,useRef, type PointerEvent } from 'react';
import { type Stroke } from './preprocessing';
export function DrawingCanvas({strokes,onChange,onDrawing}:{strokes:Stroke[];onChange:(s:Stroke[])=>void;onDrawing:(v:boolean)=>void}) {
  const canvas=useRef<HTMLCanvasElement>(null),live=useRef(strokes),pointer=useRef<number>(),before=useRef<Stroke[]>([]);
  live.current=strokes;
  function paint(){
    const node=canvas.current;if(!node)return;const bounds=node.getBoundingClientRect(),dpr=window.devicePixelRatio||1;
    node.width=Math.round(bounds.width*dpr);node.height=Math.round(bounds.height*dpr);
    const ctx=node.getContext('2d')!;ctx.scale(node.width/400,node.height/400);ctx.clearRect(0,0,400,400);
    ctx.lineWidth=6;ctx.strokeStyle='#162e3b';ctx.lineCap='round';ctx.lineJoin='round';
    for(const stroke of live.current){ctx.beginPath();stroke.forEach((p,i)=>i?ctx.lineTo(p.x,p.y):ctx.moveTo(p.x,p.y));ctx.stroke();
      if(stroke.length===1){ctx.beginPath();ctx.arc(stroke[0].x,stroke[0].y,3,0,Math.PI*2);ctx.fillStyle='#162e3b';ctx.fill();}}
  }
  useEffect(()=>{paint();},[strokes]);
  useEffect(()=>{const observer=new ResizeObserver(paint);observer.observe(canvas.current!);window.addEventListener('resize',paint);return()=>{observer.disconnect();window.removeEventListener('resize',paint);};},[]);
  function position(e:PointerEvent<HTMLCanvasElement>){const b=e.currentTarget.getBoundingClientRect();return {x:Math.max(0,Math.min(400,(e.clientX-b.left)/b.width*400)),y:Math.max(0,Math.min(400,(e.clientY-b.top)/b.height*400))};}
  function update(s:Stroke[]){live.current=s;onChange(s);}
  function start(e:PointerEvent<HTMLCanvasElement>){
    if(pointer.current!==undefined||e.button!==0||!e.isPrimary||live.current.length>=16)return;
    e.preventDefault();pointer.current=e.pointerId;before.current=live.current;e.currentTarget.setPointerCapture(e.pointerId);onDrawing(true);update([...live.current,[position(e)]]);
  }
  function move(e:PointerEvent<HTMLCanvasElement>){
    if(pointer.current!==e.pointerId)return;e.preventDefault();
    if(live.current.reduce((n,s)=>n+s.length,0)>=8192)return;
    const p=position(e),last=live.current.at(-1)!.at(-1)!;if(Math.hypot(p.x-last.x,p.y-last.y)<.5)return;
    update([...live.current.slice(0,-1),[...live.current.at(-1)!,p]]);
  }
  function end(e:PointerEvent<HTMLCanvasElement>,cancel=false){
    if(pointer.current!==e.pointerId)return;
    if(cancel)update(before.current);else move(e);
    pointer.current=undefined;onDrawing(false);
    if(e.currentTarget.hasPointerCapture(e.pointerId))e.currentTarget.releasePointerCapture(e.pointerId);
  }
  return <canvas ref={canvas} aria-label="Vùng vẽ chữ số" role="img" onPointerDown={start} onPointerMove={move} onPointerUp={e=>end(e)} onPointerCancel={e=>end(e,true)} onLostPointerCapture={e=>end(e,true)} />;
}
