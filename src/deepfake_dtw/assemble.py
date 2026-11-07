"""Assemble ordered private checkpoint parts only after full integrity verification."""
import argparse
import hashlib
import os
from pathlib import Path
import tempfile


ARTIFACTS = {
    'whisper': ('tiny.en.pt', 75_571_315, 'd3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03'),
    # Supplied by the user; this is not an independently published upstream hash.
    'dinov2': ('dinov2_vits14_pretrain.pth', 88_283_115, 'b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9'),
}


def assemble(parts, destination, *, expected_size, expected_sha256):
    parts = [Path(p) for p in parts]
    destination = Path(destination)
    missing = [p.name for p in parts if not p.is_file()]
    if missing:
        raise ValueError('Missing checkpoint parts: ' + ', '.join(missing))
    if sum(p.stat().st_size for p in parts) != expected_size:
        raise ValueError('Checkpoint parts do not match the expected total size')
    if destination.exists():
        raise FileExistsError('Refusing to overwrite an existing checkpoint')
    digest = hashlib.sha256()
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', prefix='.checkpoint-', dir=destination.parent, delete=False) as stream:
            temp_path = Path(stream.name)
            for part in parts:
                with part.open('rb') as source:
                    for block in iter(lambda: source.read(1024 * 1024), b''):
                        stream.write(block)
                        digest.update(block)
        if digest.hexdigest() != expected_sha256:
            raise ValueError('Reassembled checkpoint SHA-256 mismatch')
        # Atomic create without replacing a file created by another process.
        os.link(temp_path, destination)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=ARTIFACTS)
    parser.add_argument('--parts-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    name, size, digest = ARTIFACTS[args.kind]
    root = Path(args.parts_dir)
    parts = [root / f'{name}.part.{i:03d}' for i in range(4)]
    try:
        actual = assemble(parts, Path(args.output_dir) / name, expected_size=size, expected_sha256=digest)
    except (ValueError, OSError) as exc:
        parser.exit(2, f'Checkpoint not assembled: {exc}\n')
    print(f'{name}: {size} bytes; SHA-256 {actual}')
    if args.kind == 'dinov2':
        print('DINOv2 expected hash was supplied by the user; it is not an upstream-published checksum.')


if __name__ == '__main__':
    main()
