export type Health = { service: string; model: 'unavailable' | 'configured_unverified'; note: string };
export type Analysis = { classification: { classification: 'real' | 'fake'; score: number; threshold: number; score_kind: string; warning: string }; timing: { status: string; audio_times_seconds?: number[]; video_times_seconds?: number[]; median_video_minus_audio_seconds?: number; normalized_cost?: number; reason?: string; caveat?: string }; warning: string };
export type Job = { status: 'queued' | 'processing' | 'interrupted' | 'failed' | 'completed_research_only'; error?: string; result?: Analysis };

export function validateFile(file: Pick<File, 'name' | 'size'>): string | null {
  if (!file.name.toLowerCase().endsWith('.mp4')) return 'Choose an MP4 video to continue.';
  if (file.size === 0) return 'This file is empty. Choose another video.';
  if (file.size > 100_000_000) return 'This video exceeds the 100 MB limit.';
  return null;
}

async function checked(response: Response) {
  if (!response.ok) {
    let message = `Request failed (${response.status}).`;
    try { const data = await response.json(); if (typeof data.detail === 'string') message = data.detail; } catch { /* Leave generic message. */ }
    throw new Error(message);
  }
  return response.json();
}
export async function getHealth(signal?: AbortSignal): Promise<Health> { return checked(await fetch('/api/health', { signal })); }
export async function getJob(id: string, token: string, signal?: AbortSignal): Promise<Job> {
  const value = await checked(await fetch(`/api/jobs/${encodeURIComponent(id)}`, { headers: { Authorization: `Bearer ${token}` }, signal }));
  if (!value || !['queued', 'processing', 'interrupted', 'failed', 'completed_research_only'].includes(value.status)) throw new Error('The service returned an invalid job state.');
  if (value.status === 'completed_research_only') {
    const classification = value.result?.classification, timing = value.result?.timing;
    if (!classification || !['real', 'fake'].includes(classification.classification) ||
        !Number.isFinite(classification.score) || classification.score < 0 || classification.score > 1 ||
        !Number.isFinite(classification.threshold) || classification.score_kind !== 'uncalibrated sigmoid' ||
        typeof classification.warning !== 'string' || !timing || typeof timing.status !== 'string') throw new Error('The service returned an incomplete or unsupported result.');
    if (timing.status === 'experimental_unvalidated' && (!Array.isArray(timing.audio_times_seconds) || !Array.isArray(timing.video_times_seconds) || timing.audio_times_seconds.length === 0 || timing.audio_times_seconds.length !== timing.video_times_seconds.length || !timing.audio_times_seconds.every(Number.isFinite) || !timing.video_times_seconds.every(Number.isFinite))) throw new Error('The service returned an invalid experimental timing path.');
  }
  return value as Job;
}
export async function deleteJob(id: string, token: string): Promise<void> { const response = await fetch(`/api/jobs/${encodeURIComponent(id)}`, { method: 'DELETE', headers: { Authorization: `Bearer ${token}` } }); if (!response.ok) { let message = `Delete failed (${response.status}).`; try { const body = await response.json(); if (typeof body.detail === 'string') message = body.detail; } catch { /* Keep generic message. */ } throw new Error(message); } }
export function upload(file: File, token: string, onProgress: (value: number | null) => void): Promise<string> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest(); xhr.open('POST', '/api/jobs');
    xhr.setRequestHeader('Content-Type', 'video/mp4'); xhr.setRequestHeader('Authorization', `Bearer ${token}`);
    xhr.upload.onprogress = e => onProgress(e.lengthComputable ? Math.round(e.loaded / e.total * 100) : null);
    xhr.onerror = () => reject(new Error('Could not reach the local research service.'));
    xhr.onload = () => {
      try { const body = JSON.parse(xhr.responseText); if (xhr.status === 202 && typeof body.job_id === 'string') resolve(body.job_id); else reject(new Error(typeof body.detail === 'string' ? body.detail : `Upload failed (${xhr.status}).`)); }
      catch { reject(new Error(`Upload failed (${xhr.status}).`)); }
    };
    xhr.send(file);
  });
}
