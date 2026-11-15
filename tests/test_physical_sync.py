"""Synthetic media verifies physical shift plumbing, not real sync accuracy."""
import subprocess
import wave
import numpy as np
import pytest
from deepfake_dtw.media import read_mp4
from deepfake_dtw.physical_sync_experiment import make_shifted, verify_output_timeline, _metrics
from deepfake_dtw.local_drift_experiment import make_stretched


def test_physical_audio_delay_and_advance(tmp_path):
    wav=tmp_path/'audio.wav'; data=np.zeros(16000*4,dtype='<i2'); data[32000:32160]=20000
    with wave.open(str(wav),'wb') as out:
        out.setnchannels(1); out.setsampwidth(2); out.setframerate(16000); out.writeframes(data.tobytes())
    src=tmp_path/'source.mp4'
    subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','color=size=64x64:rate=25:duration=4',
                    '-i',str(wav),'-c:v','mpeg4','-c:a','aac',str(src)],check=True,capture_output=True)
    for shift,expected in [(1.,3.),(-1.,1.),(0.,2.)]:
        target=tmp_path/f'shift-{shift}.mp4'
        make_shifted(src,target,shift,4.)
        timeline = verify_output_timeline(target,4.)
        assert timeline['container_duration_seconds'] == pytest.approx(4., abs=.1)
        audio,video,meta=read_mp4(target)
        peak=float(audio.timestamps[0])+int(np.argmax(np.abs(audio.values.numpy())))/16000
        assert peak == pytest.approx(expected,abs=.08)
        assert meta['audio_minus_video_seconds'] == pytest.approx(0,abs=.05)
        assert len(video.values) == 100
        with pytest.raises(FileExistsError): make_shifted(src,target,shift,4.)
    for tempo in (.95, 1.05):
        target=tmp_path/f'tempo-{tempo}.mp4'
        make_stretched(src,target,tempo,4.)
        assert verify_output_timeline(target,4.)['container_duration_seconds'] == pytest.approx(4.,abs=.1)
        audio, _, _ = read_mp4(target)
        peak=float(audio.timestamps[0])+int(np.argmax(np.abs(audio.values.numpy())))/16000
        # FFmpeg's windowed atempo is approximate at an individual transient.
        assert peak == pytest.approx(2/tempo,abs=.2)


def test_rejection_counts_are_not_hidden():
    rows=[{'methods':{'m':{'absolute_relative_error_seconds':.04,'direction_correct':True}}},
          {'methods':{'m':{'status':'rejected','boundary_hit':True}}}]
    result=_metrics(rows,'m')
    assert result['attempted']==2 and result['measured']==1 and result['rejected_or_failed']==1
    assert result['within_ms_measured_only']['40']==1
    assert result['within_ms_all_attempted']['40']==.5
    assert result['boundary_hits']==1
