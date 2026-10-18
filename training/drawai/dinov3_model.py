"""DINOv3 ViT-S/16 image backbone + digit classifier; no silent fallback."""
import hashlib
from pathlib import Path
import torch
from torch import nn

MODEL_ID = 'facebook/dinov3-vits16-pretrain-lvd1689m'
REVISION = '114c1379950215c8b35dfcd4e90a5c251dde0d32'
OFFICIAL_WEIGHTS_SHA256 = '4610ad75edef83e75afdebf162d148dc628045ea6cbb83d67d4708c709c4f91d'
OFFICIAL_CONFIG_BLOBS = {
    'config.json':'fdff48569812f3cbadf253762565c4ce9dacddac',
    'preprocessor_config.json':'0126173c1ab7b75bbd9102e37de8a56ffe0a012b',
}
TIMM_MODEL_ID = 'timm/vit_small_patch16_dinov3_qkvb.lvd1689m'
TIMM_REVISION = '2c7705788ac282557562465d6443606664a55f05'
TIMM_WEIGHTS_SHA256 = '71aedea5ea914657b8fd641beadbd97c012f87e3116d4d081eb143aa520ab53c'
TIMM_CONFIG_BLOB = '719caee8b50053d542aebd3bcb04d515ce89172e'


class TimmDinoBackbone(nn.Module):
    """Preserve native timm DINOv3 inference, expose the existing token API."""
    def __init__(self,model):
        super().__init__()
        from types import SimpleNamespace
        self.model=model
        self.config=SimpleNamespace(hidden_size=384,num_register_tokens=4)

    def forward(self,pixel_values):
        from types import SimpleNamespace
        return SimpleNamespace(last_hidden_state=self.model.forward_features(pixel_values))


class WeightsUnavailableError(RuntimeError):
    pass


def architecture_config():
    from transformers import DINOv3ViTConfig
    # Public Meta hub/backbones.py vits16 recipe. Not a downloaded HF config.
    config=DINOv3ViTConfig(image_size=224,patch_size=16,hidden_size=384,
        intermediate_size=1536,num_hidden_layers=12,num_attention_heads=6,
        num_register_tokens=4,layerscale_value=1e-5,pos_embed_rescale=2.0,
        layer_norm_eps=1e-5,rope_theta=100.0)
    config._attn_implementation='eager'
    return config


class DinoDigitClassifier(nn.Module):
    """CLS + mean of patch tokens, excluding all register tokens; 768 -> 10."""
    def __init__(self, backbone, provenance, freeze_backbone=True):
        super().__init__()
        self.backbone=backbone
        self.provenance=provenance
        self.freeze_backbone=freeze_backbone
        self.head=nn.Linear(2*backbone.config.hidden_size,10)
        if freeze_backbone: self.backbone.requires_grad_(False)
        self.train(True)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone: self.backbone.eval()
        return self

    def forward_features(self,pixel_values):
        if not torch.jit.is_tracing() and (pixel_values.ndim!=4 or tuple(pixel_values.shape[1:])!=(3,224,224)):
            raise ValueError('DINOv3 contract requires NCHW [B,3,224,224]')
        sequence=self.backbone(pixel_values=pixel_values).last_hidden_state
        patches=sequence[:,1+self.backbone.config.num_register_tokens:,:]
        return torch.cat((sequence[:,0,:],patches.mean(dim=1)),dim=1)

    def forward(self,pixel_values):
        return self.head(self.forward_features(pixel_values))

    @classmethod
    def architecture_smoke(cls,seed=42):
        from transformers import DINOv3ViTModel
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            backbone=DINOv3ViTModel(architecture_config())
            return cls(backbone,dict(model_id=MODEL_ID,revision=REVISION,
                pretrained_loaded=False,smoke_only=True,release_ready=False,
                initialization='random; public Meta architecture recipe',seed=seed))

    @classmethod
    def from_pretrained(cls,weights_dir):
        """Load only an authorized official HF-format local snapshot.

        Download is a separate explicit command. Reject wrong hash, missing keys
        and incompatible configuration instead of initializing missing weights.
        """
        from transformers import DINOv3ViTModel
        folder=Path(weights_dir).expanduser().resolve()
        import json
        if (folder/'config.json').is_file():
            configuration=json.loads((folder/'config.json').read_text(encoding='utf-8'))
            if configuration.get('architecture')=='vit_small_patch16_dinov3_qkvb':
                return cls.from_pretrained_timm(folder)
        files=['config.json','preprocessor_config.json','model.safetensors']
        if not all((folder/name).is_file() for name in files):
            raise WeightsUnavailableError('Official HF snapshot missing. Request access, authenticate locally, then run training.drawai.dinov3_weights. No random fallback.')
        h=hashlib.sha256()
        with (folder/'model.safetensors').open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
        if h.hexdigest()!=OFFICIAL_WEIGHTS_SHA256:
            raise ValueError('Weights hash differs from pinned official HF revision')
        for name,expected_blob in OFFICIAL_CONFIG_BLOBS.items():
            payload=(folder/name).read_bytes()
            git_blob=hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()
            if git_blob!=expected_blob:
                raise ValueError(f'Official config blob mismatch: {name}; use the complete pinned snapshot')
        backbone,info=DINOv3ViTModel.from_pretrained(str(folder),local_files_only=True,
            attn_implementation='eager',output_loading_info=True,torch_dtype=torch.float32)
        if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
            raise ValueError('Pretrained backbone did not load completely')
        c=backbone.config
        if (c.hidden_size,c.patch_size,c.num_hidden_layers,c.num_attention_heads,c.num_register_tokens)!=(384,16,12,6,4):
            raise ValueError('Not the pinned DINOv3 ViT-S/16 architecture')
        if c.rope_theta!=100 or c.pos_embed_rescale!=2.0 or c.intermediate_size!=1536:
            raise ValueError('Positional encoding/MLP configuration differs from the pinned recipe')
        import json
        processor=json.loads((folder/'preprocessor_config.json').read_text(encoding='utf-8'))
        from .raster import MEAN,STD
        if processor.get('image_mean')!=MEAN or processor.get('image_std')!=STD:
            raise ValueError('Official processor normalization differs; update raster contract before training')
        if processor.get('do_rescale') is not True or processor.get('do_normalize') is not True or abs(processor.get('rescale_factor',0)-1/255)>1e-12:
            raise ValueError('Official processor rescale/normalize contract differs')
        return cls(backbone,dict(model_id=MODEL_ID,revision=REVISION,
            weights_sha256=h.hexdigest(),pretrained_loaded=True,smoke_only=False,
            release_ready=False,head_initialization='random; digit head must be trained',
            source_directory=str(folder),official_config_verified=True,
            processor_sha256=hashlib.sha256((folder/'preprocessor_config.json').read_bytes()).hexdigest()))

    @classmethod
    def from_pretrained_timm(cls,folder):
        """Strict native loading from the public timm publisher, not an HF mirror.

        No conversion to Meta's gated artifact is claimed. timm's documented
        float32 RoPE implementation and QKV-bias variant remain intact.
        """
        import json
        import timm
        from timm.layers import set_fused_attn
        from safetensors.torch import load_file
        from .raster import MEAN,STD
        folder=Path(folder).resolve()
        source=folder/'model.safetensors'
        if not source.is_file():raise WeightsUnavailableError('Pinned timm pretrained weights are missing')
        sha=hashlib.sha256(source.read_bytes()).hexdigest()
        if sha!=TIMM_WEIGHTS_SHA256:raise ValueError('Weights hash differs from pinned timm publisher revision')
        payload=(folder/'config.json').read_bytes()
        blob=hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()
        if blob!=TIMM_CONFIG_BLOB:raise ValueError('Pinned timm config blob mismatch')
        config=json.loads(payload);data=config['pretrained_cfg']
        if data['mean']!=MEAN or data['std']!=STD or data['fixed_input_size'] or data['min_input_size']!=[3,128,128]:
            raise ValueError('timm preprocessing/dynamic-size contract mismatch')
        if timm.__version__!='1.0.22':raise ValueError('Use verified timm==1.0.22 for this pinned adapter')
        set_fused_attn(False)
        native=timm.create_model('vit_small_patch16_dinov3_qkvb',pretrained=False,num_classes=0,img_size=224)
        weights=load_file(str(source))
        native.load_state_dict(weights,strict=True)
        if not all(torch.equal(native.state_dict()[name],value) for name,value in weights.items()):
            raise ValueError('Loaded native tensors differ from downloaded checkpoint')
        if native.num_prefix_tokens!=5 or tuple(native.reg_token.shape)!=(1,4,384):
            raise ValueError('Not the expected CLS + 4 register DINOv3 S/16')
        return cls(TimmDinoBackbone(native),dict(model_id=TIMM_MODEL_ID,revision=TIMM_REVISION,
            weights_sha256=sha,pretrained_loaded=True,smoke_only=False,release_ready=False,
            source_directory=str(folder),publisher='timm / Hugging Face PyTorch Image Models',
            source_kind='public-maintainer-pretrained-release',library='timm',library_version=timm.__version__,
            official_config_verified=True,config_git_blob=blob,processor_sha256=hashlib.sha256(payload).hexdigest(),
            native_state_all_tensors_equal=True,loaded_tensor_count=len(weights),
            backbone_parameters=sum(p.numel() for p in native.parameters()),
            input_adaptation='224 RGB glyph; publisher supports variable size >=128; mean/std unchanged',
            publisher_default_input_size=data['input_size'],rope='native timm float32 periods; no claim of bit-identical Meta BF16 buffers',
            difference_from_hf='No unused masked-image token (384 parameters); native timm keys/rotary implementation',
            license='DINOv3 custom license; public downloadable weights are not an OSI license',
            head_initialization='random; digit head is not trained by loading a pretrained backbone'))
