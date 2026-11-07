import hashlib

import pytest

from deepfake_dtw.assemble import assemble


def test_assemble_requires_complete_parts_and_verified_hash(tmp_path):
    parts = [tmp_path / f'part.{i}' for i in range(4)]
    for p, content in zip(parts[:3], (b'a', b'bc', b'd')):
        p.write_bytes(content)
    out = tmp_path / 'model.pt'
    digest = hashlib.sha256(b'abcde').hexdigest()
    with pytest.raises(ValueError, match='Missing checkpoint parts'):
        assemble(parts, out, expected_size=5, expected_sha256=digest)
    assert not out.exists()
    parts[3].write_bytes(b'e')
    with pytest.raises(ValueError, match='total size'):
        assemble(parts, out, expected_size=6, expected_sha256=digest)
    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        assemble(parts, out, expected_size=5, expected_sha256='0' * 64)
    assert not out.exists()
    assert not list(tmp_path.glob('.checkpoint-*'))
    assert assemble(parts, out, expected_size=5, expected_sha256=digest) == digest
    assert out.read_bytes() == b'abcde'
    assert out.stat().st_mode & 0o077 == 0
    with pytest.raises(FileExistsError):
        assemble(parts, out, expected_size=5, expected_sha256=digest)
