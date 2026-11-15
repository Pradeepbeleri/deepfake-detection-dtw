import { afterEach, describe, expect, it, vi } from 'vitest';
import { deleteJob, getHealth, getJob, upload, validateFile, validateVideoMetadata } from './api';

afterEach(() => vi.unstubAllGlobals());

describe('video input validation', () => {
  it('accepts only nonempty MP4 files within the service limit', () => {
    expect(validateFile({ name: 'talking.MP4', size: 100_000_000 })).toBeNull();
    expect(validateFile({ name: 'talking.mov', size: 10 })).toMatch(/MP4/);
    expect(validateFile({ name: 'talking.mp4', size: 0 })).toMatch(/empty/);
    expect(validateFile({ name: 'talking.mp4', size: 100_000_001 })).toMatch(/100 MB/);
  });
  it('checks browser metadata without treating it as a stream integrity check', () => {
    expect(validateVideoMetadata(30, 1920, 1080)).toBeNull();
    expect(validateVideoMetadata(30.01, 640, 480)).toMatch(/30-second/);
    expect(validateVideoMetadata(NaN, 640, 480)).toMatch(/duration/);
    expect(validateVideoMetadata(2, 3840, 2160)).toMatch(/dimensions/);
  });
});

describe('research API integration', () => {
  it('reads real health and authenticated job responses', async () => {
    const fetchMock = vi.fn().mockResolvedValueOnce(new Response(JSON.stringify({service:'available', model:'unavailable'})))
      .mockResolvedValueOnce(new Response(JSON.stringify({status:'processing'})));
    vi.stubGlobal('fetch', fetchMock);
    expect((await getHealth()).model).toBe('unavailable');
    expect((await getJob('a'.repeat(32), 'private-token')).status).toBe('processing');
    expect(fetchMock.mock.calls[1][1].headers.Authorization).toBe('Bearer private-token');
  });
  it('returns the server error without making up a result', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({detail:'Model unavailable or not authorized'}), {status:503})));
    await expect(getHealth()).rejects.toThrow('Model unavailable or not authorized');
  });
  it('sends raw MP4 and reports actual upload progress', async () => {
    let sent = false;
    class XHR {
      status = 202; responseText = JSON.stringify({job_id:'a'.repeat(32)}); upload = {onprogress: (_event: {lengthComputable:boolean;loaded:number;total:number}) => {}};
      onload = () => {}; onerror = () => {};
      open(method: string, url: string) { expect(method).toBe('POST'); expect(url).toBe('/api/jobs'); }
      setRequestHeader(_key:string, _value:string) {}
      send() { sent = true; this.upload.onprogress({lengthComputable:true,loaded:5,total:10}); this.onload(); }
    }
    vi.stubGlobal('XMLHttpRequest', XHR);
    const progress: (number|null)[] = [];
    expect(await upload({} as File, 'test', value => progress.push(value))).toBe('a'.repeat(32));
    expect(sent).toBe(true); expect(progress).toEqual([50]);
  });
});


describe('private result deletion', () => {
  it('sends an authenticated delete request', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(null, {status:204}));
    vi.stubGlobal('fetch', fetchMock);
    await deleteJob('a'.repeat(32), 'private-token');
    expect(fetchMock.mock.calls[0][1].method).toBe('DELETE');
    expect(fetchMock.mock.calls[0][1].headers.Authorization).toBe('Bearer private-token');
  });
  it('surfaces a running-job conflict without pretending it was deleted', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({detail:'Wait until finished'}), {status:409})));
    await expect(deleteJob('a'.repeat(32), 'private-token')).rejects.toThrow('Wait until finished');
  });
});

describe('result integrity', () => {
  it('rejects malformed scores instead of rendering a fabricated result', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({status:'completed_research_only',result:{classification:{classification:'real',score:null,threshold:.5,score_kind:'uncalibrated sigmoid',warning:'test'},timing:{status:'unavailable'}}}))));
    await expect(getJob('a'.repeat(32), 'test')).rejects.toThrow('incomplete or unsupported result');
  });
  it('rejects missing experimental timing paths', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({status:'completed_research_only',result:{classification:{classification:'real',score:.2,threshold:.5,score_kind:'uncalibrated sigmoid',warning:'test'},timing:{status:'experimental_unvalidated'}}}))));
    await expect(getJob('a'.repeat(32), 'test')).rejects.toThrow('invalid experimental timing path');
  });
  it('rejects malformed signal arrays', async () => {
    const result = {classification:{classification:'real',score:.2,threshold:.5,score_kind:'uncalibrated sigmoid',warning:'test'},timing:{status:'experimental_unvalidated',audio_times_seconds:[0,1],video_times_seconds:[0,1],signals:{audio_times_seconds:[0,1],audio_rms:[.1],video_times_seconds:[0,1],lip_aperture:[.2,.3]}}};
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({status:'completed_research_only',result}))));
    await expect(getJob('a'.repeat(32), 'test')).rejects.toThrow('invalid timing signals');
  });
});
