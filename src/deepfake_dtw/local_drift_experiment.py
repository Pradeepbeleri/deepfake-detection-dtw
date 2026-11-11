"""Private exploratory audio time-stretch experiment; not a classifier."""
import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
import numpy as np
from .dtw_validation import extract_signals
from .global_sync import analyze_signals
from .media import _run, read_mp4
from .physical_sync_experiment import make_shifted, verify_output_timeline
from .resources import peak_rss_kib_linux


def make_stretched(source, destination, factor, duration):
    """Stretch audio using FFmpeg atempo; trim/pad to original video duration."""
    factor=float(factor); duration=float(duration); destination=Path(destination)
    if not np.isfinite([factor,duration]).all() or not .9<=factor<=1.1 or not 0<duration<=30:
        raise ValueError('Invalid tempo or duration')
    if destination.exists(): raise FileExistsError('Refusing to overwrite media')
    _run(['ffmpeg','-v','error','-xerror','-protocol_whitelist','file,pipe','-i',str(source),
          '-map','0:v:0','-map','0:a:0','-c:v','copy','-af',f'asetpts=PTS-STARTPTS,atempo={factor:.6f},apad',
          '-c:a','aac','-ac','1','-ar','16000','-t',f'{duration:.6f}','-n',str(destination)])


def run(source):
    path=Path(source)
    with path.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
    start=time.perf_counter(); _,_,meta=read_mp4(path); duration=meta['container_duration_seconds']
    if abs(meta['audio_minus_video_seconds'])>.05 or duration<9.5:
        raise ValueError('Source timing or duration unsuitable for the comparison')
    rows=[]
    with tempfile.TemporaryDirectory(prefix='local-drift-') as work:
        for factor in (1., .95, 1.05):
            row={'tempo_factor':factor}; dest=Path(work)/f'{factor:.2f}.mp4'; ts=time.perf_counter()
            try:
                if factor==1.: make_shifted(path,dest,0.,duration)
                else: make_stretched(path,dest,factor,duration)
                row['playback_timeline']=verify_output_timeline(dest,duration)
                a,e,v,m=extract_signals(dest); result=analyze_signals(a,e,v,m)
                row['global']=result['global']; local=result['local_dtw']
                # Keep raw paths private and bounded; report summary only.
                row['local_dtw']={k:v for k,v in local.items() if k not in ('audio_times_seconds','original_video_times_seconds')}
            except (ValueError,RuntimeError,OSError,ImportError) as exc:
                row['failure']={'type':type(exc).__name__,'reason':str(exc)}
            row['runtime_seconds']=time.perf_counter()-ts; rows.append(row)
            print(f'Tempo {factor:.2f}: '+('failed' if 'failure' in row else row['global']['status']),flush=True)
    with path.open('rb') as f:
        if hashlib.file_digest(f,'sha256').hexdigest()!=digest: raise ValueError('Source changed during experiment')
    return {'version':'local-drift-v1','source_sha256':digest,'rows':rows,'elapsed_seconds':time.perf_counter()-start,
       'peak_process_rss_kib_linux':peak_rss_kib_linux(),
       'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('physical_sync_experiment.py'),Path(__file__).with_name('global_sync.py'),Path(__file__).with_name('dtw.py')]},
       'interpretation':'Pitch-preserving atempo 0.95 and 1.05 are physically generated audio edits, trimmed/padded to the original duration. Approximate drift is not verified at individual speech events. Endpoint-normalized costs have different denominators; lower DTW cost is not proof of ground-truth alignment or deepfake detection.'}


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('video');p.add_argument('output');p.add_argument('--consented-non-test',action='store_true',required=True);args=p.parse_args()
    fd=os.open(args.output,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:
            try: data=run(args.video); code=0
            except (ValueError,RuntimeError,OSError,ImportError) as exc: data={'status':'failed','reason':str(exc),'error_type':type(exc).__name__};code=2
            json.dump(data,f,indent=2,allow_nan=False)
    except BaseException:
        Path(args.output).unlink(missing_ok=True);raise
    print('Private drift report written; classifier and held-out test were not run.')
    return code


if __name__=='__main__':raise SystemExit(main())
