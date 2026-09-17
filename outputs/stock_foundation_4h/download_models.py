"""Download pinned public weights into ignored local model storage."""
from pathlib import Path
from huggingface_hub import snapshot_download
ROOT=Path(__file__).resolve().parents[2]
for name,revision,dest in [('clapAI/Fin-ModernBERT','31d3d96d5839a03dccd030bea40b77c2649a9a01','modern'),('amazon/chronos-2','29ec3766d36d6f73f0696f85560a422f50e8498c','chronos')]:
 snapshot_download(name,revision=revision,local_dir=ROOT/'work/stock-data/foundation_4h/models'/dest,allow_patterns=['*.json','*.safetensors','README.md'])
