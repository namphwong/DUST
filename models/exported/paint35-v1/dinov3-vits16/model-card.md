# DrawAI dinov3-vits16 - Paint35 FP32 torchscript

Trained checkpoint seed42, epoch98, selected using validation before test.
Input pixel_values float32 [1, 3, 224, 224]; output logits float32[1,10], classes0-9.
Preprocessing is external: see preprocessing.json. Source uint8 grayscale64 with white background/black ink.
Swin uses spatial rows, not writing trajectories. DINOv3 includes the full pretrained backbone and fitted head.
Test1000: accuracy 0.9880, macro-F1 0.988017; same argmax as P5.
37 near-groups across main splits remain. No leakage-free/generalization claim; owner-attested Paint origin/identities/permissions.
Snapshot split is retrospective; original source already public. No device metadata or pen trajectories.
FP32 artifact format: torchscript; verified full CPU parity. ONNX diagnostic status is in report.json. No canvas integration in P6.
trained=true, artifact_verified=true, release_ready=false until P7 product/domain/runtime acceptance.
Do not use the P2 untrained-head artifact in place of this model. Do not tune from final test results.
Checkpoint SHA256: 1b14dc81786010d10f4bb3c1d77c8ef3e1af44f52a2186881e616dd19906c4f3. Artifact SHA256: 5846bd2331f546d1a97e617a0733f7cafd72a953ee8988dd7538940ff2cc6baa.
