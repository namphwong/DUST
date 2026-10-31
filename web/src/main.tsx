import { createRoot } from 'react-dom/client';
import './drawai/base.css';
import { DrawAIApp } from './drawai/drawai-app';
import { DrawAIFeasibility } from './drawai/feasibility-app';
import { IntegrationCheck } from './drawai/integration-check';
import { DinoV3Check } from './drawai/dinov3-check';
import { CollectApp } from './drawai/collect-app';
import { PaintCompare } from './drawai/paint-compare';
import {PUBLIC_PAINT} from './drawai/paint-deployment';

const path = location.pathname.replace(/\/$/, '') || '/';
const publicAllowed=['/','/drawai','/drawai/compare'].includes(path);
const page = PUBLIC_PAINT&&!publicAllowed ? <main><h1>Không tìm thấy trang</h1><a href="/">Về DrawAI</a></main>
  : path === '/collect' ? <CollectApp/>
  : path === '/drawai/compare' ? <PaintCompare/>
  : path === '/drawai/legacy' ? <DrawAIApp legacy/>
  : path === '/drawai/dinov3-check' ? <DinoV3Check/>
  : path === '/drawai/integration-check' ? <IntegrationCheck/>
  : path === '/drawai/release-check' ? <DrawAIFeasibility trained/>
  : path === '/drawai/check' ? <DrawAIFeasibility/>
  : path === '/' || path === '/drawai' ? <DrawAIApp/>
  : <main><h1>Không tìm thấy trang</h1><a href="/">Về DrawAI</a></main>;

createRoot(document.getElementById('root')!).render(page);
