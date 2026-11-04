#!/usr/bin/env python3
"""Check the release inventory and file hashes using Python's standard library."""
import hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def main():
    expected = {}
    for line in (ROOT / 'checksums.sha256').read_text().splitlines():
        digest, relative = line.split('  ', 1)
        path = (ROOT / relative).resolve()
        assert ROOT in path.parents, 'Path outside the release'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, 'Changed file: ' + relative
        expected[relative] = digest
    actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()
              and not any(x in {'outputs','__pycache__','.git'} for x in p.relative_to(ROOT).parts)
              and p.name not in {'checksums.sha256','.DS_Store'}}
    assert actual == set(expected), 'Release inventory differs from checksums'
    print(f'File inventory and {len(expected)} SHA-256 checks passed.')


if __name__ == '__main__':
    main()
