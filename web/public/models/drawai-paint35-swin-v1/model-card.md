# DrawAI swin1d-image - Paint35 FP32 onnx

Trained checkpoint seed43, epoch29, selected using validation before test.
Input row_pixels float32 [1, 64, 64]; output logits float32[1,10], classes0-9.
Preprocessing is external: see preprocessing.json. Source uint8 grayscale64 with white background/black ink.
Swin uses spatial rows, not writing trajectories. DINOv3 includes the full pretrained backbone and fitted head.
Test1000: accuracy 0.8440, macro-F1 0.842804; same argmax as P5.
37 near-groups across main splits remain. No leakage-free/generalization claim; owner-attested Paint origin/identities/permissions.
Snapshot split is retrospective; original source already public. No device metadata or pen trajectories.
FP32 artifact format: onnx; verified full CPU parity. ONNX diagnostic status is in report.json. No canvas integration in P6.
trained=true, artifact_verified=true, release_ready=false until P7 product/domain/runtime acceptance.
Do not use the P2 untrained-head artifact in place of this model. Do not tune from final test results.
Checkpoint SHA256: 3daa475f97e34c00d10380ea16157c9543de6495799ca3ee66cac642f2864619. Artifact SHA256: 9bb8bf5478cb2d8131936a79888505dcbc4f05e331027ca4e2117db7715df59d.
