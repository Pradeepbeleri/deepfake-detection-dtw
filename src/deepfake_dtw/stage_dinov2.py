"""Stage a private DINOv2 checkout locally and verify it before use."""
import argparse
import os
import shutil
import tempfile
from pathlib import Path

from .encoders import DINOV2_COMMIT, verify_dinov2_source


def _verify_local_checkout(path):
    for root, dirs, files in os.walk(path, followlinks=False):
        if any((Path(root) / name).is_symlink() for name in dirs + files):
            raise ValueError('DINOv2 staged checkout contains a symlink; use a self-contained checkout')
    alternates = path / '.git' / 'objects' / 'info' / 'alternates'
    if alternates.exists():
        raise ValueError('DINOv2 staged checkout uses an external Git object store')
    verify_dinov2_source(str(path))


def stage_dinov2_source(source, destination):
    """Copy a full checkout, verify its local copy, and never overwrite a destination.

    A previously staged destination is reused only after fresh verification. The
    Drive checkout is never modified. Failed/incomplete copies are not published.
    """
    source = Path(source).resolve()
    destination = Path(destination).absolute()
    if not source.is_dir():
        raise ValueError('DINOv2 source checkout directory is missing')
    if not (source / '.git').is_dir() or (source / '.git').is_symlink():
        raise ValueError('DINOv2 source must contain its own .git directory, not a linked worktree')
    if source == destination.resolve() or destination.resolve().is_relative_to(source):
        raise ValueError('DINOv2 staging destination must be outside the source checkout')
    if destination.is_symlink():
        raise ValueError('DINOv2 staging destination cannot be a symlink')
    if destination.exists():
        _verify_local_checkout(destination)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix='.dinov2-stage-', dir=destination.parent) as directory:
            staged = Path(directory) / 'checkout'
            shutil.copytree(source, staged, symlinks=True)
            # Never rely on a Drive-backed symlink or external Git object store.
            _verify_local_checkout(staged)
            if destination.exists() or destination.is_symlink():
                raise FileExistsError('DINOv2 destination appeared while staging; no checkout was overwritten')
            staged.rename(destination)
            _verify_local_checkout(destination)
    except FileExistsError:
        raise
    except OSError as exc:
        raise ValueError(f'Could not stage the DINOv2 source locally: {exc.strerror or type(exc).__name__}') from exc
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, help='Trusted private source checkout')
    parser.add_argument('--destination', required=True, help='Local staging path outside the source')
    args = parser.parse_args()
    staged = stage_dinov2_source(args.source, args.destination)
    print(f'Local DINOv2 checkout verified at pinned commit {DINOV2_COMMIT}: {staged}', flush=True)


if __name__ == '__main__':
    main()
