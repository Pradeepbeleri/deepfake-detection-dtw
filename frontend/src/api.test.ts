import { afterEach, describe, expect, it, vi } from 'vitest';
import { getHealth, getJob, upload, validateFile } from './api';

afterEach(() => vi.unstubAllGlobals());

describe('video input validation', () => {
  it('accepts only nonempty MP4 files within the service limit', () => {
    expect(validateFile({ name: 'talking.MP4', size: 100_000_000 })).toBeNull();
    expect(validateFile({ name: 'talking.mov', size: 10 })).toMatch(/MP4/);
    expect(validateFile({ name: 'talking.mp4', size: 0 })).toMatch(/empty/);
    expect(validateFile({ name: 'talking.mp4', size: 100_000_001 })).toMatch(/100 MB/);
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
