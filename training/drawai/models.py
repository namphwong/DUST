"""Two DrawAI families; DINOv3 dependencies are optional and lazily imported."""
from models.SwinTransformer_cls import SwinTransformer_1D


def swin_config(length=64):
    if length not in (64, 128):
        raise ValueError('Unsupported length')
    return dict(seq_len=length, in_chans=3, num_classes=10, patch_size=2,
                window_size=8, embed_dim=32, depths=[2,2,2], num_heads=[2,4,8],
                drop_rate=0.1, drop_path_rate=0.1)


def build_model(kind='swin', length=64, *, weights_dir=None, smoke_only=False):
    if kind == 'swin':
        return SwinTransformer_1D(**swin_config(length))
    if kind == 'dinov3_vits16':
        from .dinov3_model import DinoDigitClassifier, WeightsUnavailableError
        if smoke_only:
            if weights_dir is not None:
                raise ValueError('Do not combine architecture smoke and pretrained weights')
            return DinoDigitClassifier.architecture_smoke()
        if weights_dir is None:
            raise WeightsUnavailableError('Provide a verified local public timm or authorized Meta snapshot; random initialization requires explicit smoke_only=True')
        return DinoDigitClassifier.from_pretrained(weights_dir)
    raise ValueError('Unsupported model family; use swin or dinov3_vits16')
