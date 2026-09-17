"""Refresh Git review manifests without modifying experimental artifacts."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    paths = [x.strip() for x in (ROOT/'scripts/repository_results.txt').read_text().splitlines() if x.strip() and not x.startswith('#')]
    entries = []
    for name in paths:
        path = ROOT/name
        if not path.is_file() or not path.resolve().is_relative_to(ROOT):
            raise ValueError(f'Missing or invalid result path: {name}')
        entries.append({'path': name, 'bytes': path.stat().st_size, 'sha256': digest(path)})
    (ROOT/'docs/RESULT_FILES.json').write_text(json.dumps(entries, indent=2)+'\n')
    ignore = ROOT/'.gitignore'
    old = ignore.read_text().split('# BEGIN selected results')[0].rstrip()
    ignore.write_text(old+'\n\n# BEGIN selected results\n'+''.join('!/'+x+'\n' for x in paths))
    # Inventory experiment binaries only, never scan virtual environments or credentials.
    assets = []
    for path in sorted((ROOT/'outputs').rglob('*')):
        if path.is_file() and path.suffix in {'.pkl','.joblib','.pt','.npy','.npz'}:
            assets.append({'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size, 'sha256': digest(path), 'in_git': False})
    (ROOT/'docs/LOCAL_ARTIFACTS.json').write_text(json.dumps({'scope':'Experiment model and cache binaries under outputs; raw course data and environments excluded from this inventory.', 'files':assets}, indent=2)+'\n')
    dirs = sorted(x for x in (ROOT/'outputs').iterdir() if x.is_dir() and x.name.startswith('stock_'))
    lines = ['# 实验索引','', '当前入口是 **stock_finbert_event_adapter_4h/v1**；历史目录保留原名称，不能将一小时结果与四小时结果直接比较。','']
    for directory in dirs:
        docs = [x for x in directory.glob('*.md') if x.name in {'README.md','REPORT.md','PROTOCOL.md','FINAL_STATUS.md'}]
        if not docs:
            docs = list(directory.glob('*.md'))[:3]
        links = ' · '.join(f'[{x.name}](../{x.relative_to(ROOT).as_posix()})' for x in sorted(docs))
        lines.append(f'- **{directory.name}**: {links or "Python source archive"}')
    (ROOT/'docs/EXPERIMENT_INDEX.md').write_text('\n'.join(lines)+'\n')
    print(f'Refreshed {len(entries)} result files, {len(assets)} local artifact hashes, {len(dirs)} experiment directories.')

if __name__ == '__main__':
    main()
