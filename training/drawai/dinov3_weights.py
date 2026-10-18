"""Download a pinned public timm DINOv3 release or an authorized Meta snapshot."""
import argparse
import json
from pathlib import Path
from .dinov3_model import MODEL_ID, REVISION, TIMM_MODEL_ID, TIMM_REVISION, DinoDigitClassifier, WeightsUnavailableError


def download(destination,source='timm'):
    destination=Path(destination)
    # Existing verified local files make reruns idempotent, without network use.
    required=['config.json','model.safetensors','README.md','LICENSE.md']
    if source=='meta':required.append('preprocessor_config.json')
    if all((destination/name).is_file() for name in required):
        model=DinoDigitClassifier.from_pretrained(destination)
        if model.provenance['model_id']!=(TIMM_MODEL_ID if source=='timm' else MODEL_ID):
            raise WeightsUnavailableError('Existing snapshot does not match the requested publisher')
        (destination/'drawai-provenance.json').write_text(json.dumps(model.provenance,indent=2),encoding='utf-8')
        print('Verified existing publisher snapshot; no network download or random fallback.')
        return
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError
    try:
        snapshot_download(TIMM_MODEL_ID if source=='timm' else MODEL_ID,
            revision=TIMM_REVISION if source=='timm' else REVISION,local_dir=str(destination),
            token=False if source=='timm' else None,
            allow_patterns=['config.json','preprocessor_config.json','model.safetensors','LICENSE.md','README.md'])
    except GatedRepoError as error:
        raise WeightsUnavailableError('DINOv3 requires approved access. Request access on the official Hugging Face model page and authenticate with `hf auth login` locally. Do not put a token in source code or chat.') from error
    model=DinoDigitClassifier.from_pretrained(destination)
    (destination/'drawai-provenance.json').write_text(json.dumps(model.provenance,indent=2),encoding='utf-8')
    print('Verified pinned publisher pretrained backbone; digit head remains untrained.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',choices=['timm','meta'],default='timm')
    parser.add_argument('--destination',type=Path)
    args=parser.parse_args()
    destination=args.destination or Path('models/pretrained')/('dinov3-vits16-timm-qkvb' if args.source=='timm' else 'dinov3-vits16')
    download(destination,args.source)
