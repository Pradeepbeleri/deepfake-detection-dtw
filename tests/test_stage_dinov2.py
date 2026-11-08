"""Local Git fixtures only; no model weights or external services."""
import subprocess
from pathlib import Path

import pytest

from deepfake_dtw import encoders
from deepfake_dtw.stage_dinov2 import stage_dinov2_source


def checkout(tmp_path, monkeypatch):
    source = tmp_path/'drive'/'dinov2-src'
    source.mkdir(parents=True)
    subprocess.run(['git', 'init', '-q', str(source)], check=True)
    (source/'hubconf.py').write_text('fixture = True\n')
    subprocess.run(['git', '-C', str(source), 'add', 'hubconf.py'], check=True)
    subprocess.run(['git', '-C', str(source), '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.test', 'commit', '-qm', 'fixture'], check=True)
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    monkeypatch.setattr(encoders, 'DINOV2_COMMIT', commit)
    return source


def test_stage_clean_checkout_and_reuse_after_restart(tmp_path, monkeypatch):
    source = checkout(tmp_path, monkeypatch)
    dest = tmp_path/'local'/'dinov2-src'
    assert stage_dinov2_source(source, dest) == dest
    assert (dest/'hubconf.py').read_text() == 'fixture = True\n'
    assert stage_dinov2_source(source, dest) == dest
    (dest/'untracked.py').write_text('changed')
    with pytest.raises(ValueError, match='clean working tree'):
        stage_dinov2_source(source, dest)
    assert (dest/'untracked.py').exists()


def test_reject_dirty_and_wrong_commit_without_publishing(tmp_path, monkeypatch):
    source = checkout(tmp_path, monkeypatch); dest = tmp_path/'local'/'dinov2-src'
    (source/'hubconf.py').write_text('changed')
    with pytest.raises(ValueError, match='clean working tree'):
        stage_dinov2_source(source, dest)
    assert not dest.exists()
    subprocess.run(['git', '-C', str(source), 'checkout', '--', 'hubconf.py'], check=True)
    monkeypatch.setattr(encoders, 'DINOV2_COMMIT', '0'*40)
    with pytest.raises(ValueError, match='pinned commit'):
        stage_dinov2_source(source, dest)
    assert not dest.exists()


def test_reject_untracked_symlink_and_nested_destination(tmp_path, monkeypatch):
    source = checkout(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='outside the source'):
        stage_dinov2_source(source, source/'local')
    (source/'link').symlink_to('/tmp')
    with pytest.raises(ValueError, match='symlink'):
        stage_dinov2_source(source, tmp_path/'local'/'dest')


def test_git_timeout_is_actionable_not_invalid_checkout(tmp_path, monkeypatch):
    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs['timeout'])
    monkeypatch.setattr(subprocess, 'run', timeout)
    with pytest.raises(ValueError, match='timed out.*Stage.*local storage'):
        encoders.verify_dinov2_source(str(tmp_path))


def test_git_status_timeout_reports_operation(tmp_path, monkeypatch):
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            return subprocess.CompletedProcess(command, 0, stdout=str(tmp_path))
        if len(calls) == 2:
            return subprocess.CompletedProcess(command, 0, stdout=encoders.DINOV2_COMMIT)
        raise subprocess.TimeoutExpired(command, kwargs['timeout'])
    monkeypatch.setattr(subprocess, 'run', run)
    with pytest.raises(ValueError, match='working-tree status timed out.*Stage.*local storage'):
        encoders.verify_dinov2_source(str(tmp_path))
    assert calls[-1][-2:] == ['--porcelain', '--untracked-files=all']


def test_untracked_source_file_never_published(tmp_path, monkeypatch):
    source = checkout(tmp_path, monkeypatch)
    (source/'untracked.py').write_text('do not execute')
    dest = tmp_path/'local'/'checkout'
    with pytest.raises(ValueError, match='clean working tree'):
        stage_dinov2_source(source, dest)
    assert not dest.exists() and (source/'untracked.py').exists()
