"""Verify a private LAV-DF tar and extract only selected, regular MP4 members.

No tar extraction API is used: untrusted paths and links are never followed.
"""
import argparse
import hashlib
import json
import os
import tarfile
from pathlib import Path, PurePosixPath

from .lavdf import _safe_file


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def extract_selected(archive, records, media_root, expected_sha256):
    if len(expected_sha256) != 64 or any(c not in '0123456789abcdef' for c in expected_sha256.lower()):
        raise ValueError('Expected archive SHA-256 must be 64 hexadecimal characters')
    if sha256(archive) != expected_sha256.lower():
        raise ValueError('Archive SHA-256 mismatch; no members extracted')
    root = Path(media_root).resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    selected = {}
    for record in records:
        name = _safe_file(record['provider_file'])
        member = name if name.startswith('LAV-DF/') else 'LAV-DF/' + name
        destination = root / name
        if str(destination.resolve()) != str(Path(record['video_path']).resolve()):
            raise ValueError('Plan video path does not match the media root and provider file')
        if member in selected:
            raise ValueError('Duplicate selected archive member')
        selected[member] = destination
    if not selected:
        raise ValueError('Plan must select at least one video')
    with tarfile.open(archive, 'r:') as tar:
        found = {}
        for member in tar:
            if member.name not in selected:
                continue
            if member.name in found:
                raise ValueError('Archive contains duplicate selected member names')
            if not member.isfile() or member.size <= 0:
                raise ValueError('Selected member must be a nonempty regular file')
            found[member.name] = member
        missing = selected.keys() - found.keys()
        if missing:
            raise ValueError(f'{len(missing)} selected members are absent from archive')
        result = {'extracted': 0, 'reused_verified': 0}
        for name in sorted(selected):
            dest = selected[name]
            # Do not follow symlinked paths from an existing private directory.
            current = root
            for part in PurePosixPath(dest.relative_to(root).as_posix()).parts[:-1]:
                current = current / part
                if current.is_symlink():
                    raise ValueError('Symlink in destination path')
                current.mkdir(exist_ok=True, mode=0o700)
            if dest.is_symlink():
                raise ValueError('Symlink destination is not allowed')
            source = tar.extractfile(found[name])
            if source is None:
                raise ValueError('Cannot read selected member')
            with source:
                if dest.exists():
                    digest = hashlib.sha256()
                    for block in iter(lambda: source.read(1024 * 1024), b''):
                        digest.update(block)
                    if not dest.is_file() or sha256(dest) != digest.hexdigest():
                        raise ValueError(f'Existing member differs from archive: {name}; preserve and inspect it')
                    result['reused_verified'] += 1
                else:
                    fd = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                    try:
                        with os.fdopen(fd, 'wb') as output:
                            for block in iter(lambda: source.read(1024 * 1024), b''):
                                output.write(block)
                    except BaseException:
                        dest.unlink(missing_ok=True)
                        raise
                    result['extracted'] += 1
            print(f"Processed {result['extracted'] + result['reused_verified']}/{len(selected)} selected videos", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--media-root', required=True)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    with open(args.plan, encoding='utf-8') as stream:
        records = json.load(stream)
    print('Verifying full archive SHA-256; this can take several minutes.', flush=True)
    print(json.dumps(extract_selected(args.archive, records, args.media_root, args.sha256), indent=2))


if __name__ == '__main__':
    main()
