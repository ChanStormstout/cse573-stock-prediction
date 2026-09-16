"""Download one pinned public model into the ignored local work directory."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'work/stock-data/model_compare'
os.environ['HF_HOME'] = str(BASE / 'hf-home')
os.environ['HF_HUB_DISABLE_XET'] = '1'
from huggingface_hub import snapshot_download

if __name__ == '__main__':
    print(snapshot_download(
        'mlx-community/Qwen3.5-9B-4bit',
        revision='8b2b98c00a6b4d291155e4890773ca8f769aee53',
        local_dir=BASE / 'model', max_workers=2), flush=True)
