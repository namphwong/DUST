/** P5 validation selection; anchored to the verified P6 bundle, never selected by test. */
export const PAINT_BUNDLE_SHA256='86121e4def30d53f52b4e21126efceca3fff66b4d24a591c915c312a7db5eae2';
export const CANVAS_ADAPTER='drawai-canvas400-gray64-v1';
export const PAINT_PREPROCESSING='drawai-paint-image64-v1';
export const DEFAULT_PAINT_MODEL='dinov3-vits16';
export const PAINT_MODELS={
  'swin1d-image':{name:'Swin Transformer 1D',format:'onnx',shape:[1,64,64]},
  'dinov3-vits16':{name:'DINOv3 ViT-S/16',format:'torchscript',shape:[1,3,224,224]},
} as const;
export type PaintModel=keyof typeof PAINT_MODELS;
