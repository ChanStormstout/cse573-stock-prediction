"""Lightweight repository hygiene checks, not model validation."""
from pathlib import Path
import ast
import hashlib
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def main():
    names = subprocess.check_output(['git','ls-files','-z','--cached','--others','--exclude-standard'], cwd=ROOT).decode().split('\0')
    paths = sorted({ROOT/x for x in names if x and (ROOT/x).is_file()})
    problems = []
    count = 0
    secret_patterns = [rb'gh[pousr]_[A-Za-z0-9]{30,}', rb'github_pat_[A-Za-z0-9_]{40,}', rb'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----']
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        if relative.startswith(('work/','.playwright-mcp/')) or path.suffix in {'.zip','.pkl','.joblib','.pt','.npy','.npz','.pdf','.ppt','.pptx'}:
            problems.append(f'Forbidden artifact: {relative}')
        if path.stat().st_size > 10*1024*1024:
            problems.append(f'File over 10 MiB: {relative}')
        data = path.read_bytes()
        if any(re.search(pattern,data) for pattern in secret_patterns):
            problems.append(f'Possible credential: {relative}')
        if path.suffix == '.py':
            try:
                ast.parse(data, filename=relative)
                count += 1
            except SyntaxError as exc:
                problems.append(f'Invalid Python: {relative}: {exc.lineno}')
    for entry in json.loads((ROOT/'docs/RESULT_FILES.json').read_text()):
        path = ROOT/entry['path']
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            problems.append(f'Result manifest mismatch: {entry["path"]}')
    if problems:
        raise SystemExit('\n'.join(problems))
    print(f'PASS: {len(paths)} files; {count} Python sources parse; selected result hashes match; no prohibited artifacts or known token patterns.')

if __name__ == '__main__':
    main()
