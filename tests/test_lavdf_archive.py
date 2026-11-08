import hashlib
import io
import tarfile

import pytest

from deepfake_dtw.lavdf_archive import extract_selected


def archive(path, members):
    with tarfile.open(path, 'w') as tar:
        for name, data, kind in members:
            info = tarfile.TarInfo(name)
            info.type = kind
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data) if kind == tarfile.REGTYPE else None)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(root, name):
    return {'provider_file': name, 'video_path': str(root / name)}


def test_extract_and_verify_restart(tmp_path):
    tar = tmp_path/'fixture.tar'; root = tmp_path/'media'
    digest = archive(tar, [('LAV-DF/train/a.mp4', b'video', tarfile.REGTYPE), ('LAV-DF/train/b.mp4', b'other', tarfile.REGTYPE)])
    records = [record(root, 'train/a.mp4')]
    assert extract_selected(tar, records, root, digest) == {'extracted': 1, 'reused_verified': 0}
    assert (root/'train/a.mp4').read_bytes() == b'video'
    assert not (root/'train/b.mp4').exists()
    assert extract_selected(tar, records, root, digest) == {'extracted': 0, 'reused_verified': 1}
    (root/'train/a.mp4').write_bytes(b'changed')
    with pytest.raises(ValueError, match='differs'):
        extract_selected(tar, records, root, digest)


def test_reject_bad_hash_traversal_links_duplicates_and_missing(tmp_path):
    tar = tmp_path/'fixture.tar'; root = tmp_path/'media'
    digest = archive(tar, [('LAV-DF/train/a.mp4', b'video', tarfile.REGTYPE)])
    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        extract_selected(tar, [record(root, 'train/a.mp4')], root, '0'*64)
    with pytest.raises(ValueError, match='safe relative'):
        extract_selected(tar, [record(root, '../a.mp4')], root, digest)
    with pytest.raises(ValueError, match='absent'):
        extract_selected(tar, [record(root, 'train/missing.mp4')], root, digest)
    digest = archive(tar, [('LAV-DF/train/a.mp4', b'', tarfile.SYMTYPE)])
    with pytest.raises(ValueError, match='regular file'):
        extract_selected(tar, [record(root, 'train/a.mp4')], root, digest)
    digest = archive(tar, [('LAV-DF/train/a.mp4', b'1', tarfile.REGTYPE), ('LAV-DF/train/a.mp4', b'2', tarfile.REGTYPE)])
    with pytest.raises(ValueError, match='duplicate'):
        extract_selected(tar, [record(root, 'train/a.mp4')], root, digest)
