import { createRoot } from 'react-dom/client';
import './drawai/base.css';
import { DrawAIApp } from './drawai/drawai-app';
import { DrawAIFeasibility } from './drawai/feasibility-app';
import { IntegrationCheck } from './drawai/integration-check';
import { DinoV3Check } from './drawai/dinov3-check';

const path = location.pathname.replace(/\/$/, '') || '/';
const page = path === '/drawai/dinov3-check' ? <DinoV3Check/>
  : path === '/drawai/integration-check' ? <IntegrationCheck/>
  : path === '/drawai/release-check' ? <DrawAIFeasibility trained/>
  : path === '/drawai/check' ? <DrawAIFeasibility/>
  : path === '/' || path === '/drawai' ? <DrawAIApp/>
  : <main><h1>Không tìm thấy trang</h1><a href="/">Về DrawAI</a></main>;

createRoot(document.getElementById('root')!).render(page);
