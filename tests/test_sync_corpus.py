import pytest
from deepfake_dtw.sync_corpus import audit


def record(path, sample, part='train', group=None):
    return {'sample_id':sample,'video_path':str(path),'partition':part,'label':'real','consented':True,'source_group':group or sample}


def test_test_rows_and_source_leak_rejected_before_read(tmp_path):
    a=record(tmp_path/'missing','a')
    with pytest.raises(ValueError,match='test is excluded'):
        audit([a,record(tmp_path/'test','b','test')])
    with pytest.raises(ValueError,match='spans'):
        audit([a,record(tmp_path/'missing2','b','validation','a')])


def test_audit_retains_failures_and_rejects_duplicates(tmp_path):
    p=tmp_path/'a.mp4'; p.write_bytes(b'fixture')
    def runner(path): raise ValueError('fixture quality failure')
    result=audit([record(p,'a')],runner=runner)
    assert result['failure_rate']==1 and result['outcomes'][0]['status']=='failed'
    assert not result['identity_disjoint_verified']
    with pytest.raises(ValueError,match='duplicate media'):
        audit([record(p,'a'),record(p,'b')],runner=runner)


def test_valid_report_bound_to_hash(tmp_path):
    p=tmp_path/'a.mp4'; p.write_bytes(b'fixture')
    def runner(path): return {'video_sha256':'0'*64}
    with pytest.raises(ValueError,match='source hash'):
        audit([record(p,'a')],runner=runner)


def test_aggregates_per_recording_and_retains_quality_failures(tmp_path):
    import hashlib
    paths = [tmp_path / name for name in ('a.mp4', 'b.mp4', 'c.mp4')]
    for index, path in enumerate(paths): path.write_bytes(bytes([index]))
    def runner(path):
        if path.endswith('c.mp4'): raise ValueError('no single-face track')
        error = 10. if path.endswith('a.mp4') else 30.
        summary = {method: {'mean_absolute_response_error_ms': error} for method in ('fixed_zero', 'cross_correlation', 'dtw')}
        measurements = {name: {'summary': summary} for name in ('rms_aperture_v1', 'rms_aperture_velocity_v1')}
        return {'video_sha256': hashlib.sha256(open(path,'rb').read()).hexdigest(), 'measurements': measurements}
    result = audit([record(path, str(i)) for i, path in enumerate(paths)], runner=runner)
    assert result['successful'] == 2 and result['failed'] == 1
    assert result['failure_rate'] == pytest.approx(1/3)
    assert result['aggregate']['rms_aperture_v1']['dtw']['mean_of_recording_mae_ms'] == 20
    assert result['outcomes'][-1]['reason'] == 'no single-face track'
