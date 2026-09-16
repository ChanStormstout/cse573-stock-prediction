"""Run with the same interpreter as inference; does not initialize the GPU."""
import importlib.metadata as metadata
import json
import platform
import sys
from pathlib import Path

if __name__=='__main__':
    result={'python':sys.version,'platform':platform.platform(),'machine':platform.machine(),
            'packages':{name:metadata.version(name) for name in ['mlx','mlx-lm','transformers','tokenizers','numpy','huggingface-hub']},
            'execution':'Local MLX frozen inference; no paid API, no Sol, no weight training'}
    Path(__file__).with_name('ENVIRONMENT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
