import React from 'react';
import type { Analysis } from './api';

function OffsetChart({ audio, video }: { audio: number[]; video: number[] }) {
  if (!audio.length || audio.length !== video.length) return null;
  const start = audio[0], span = Math.max(audio[audio.length - 1] - start, .01);
  const points = audio.map((t, i) => `${((t-start)/span*100).toFixed(2)},${(50 - Math.max(-.5, Math.min(.5, video[i]-t))*80).toFixed(2)}`).join(' ');
  return <div className="chart-wrap"><svg className="chart" viewBox="0 0 100 100" preserveAspectRatio="none" role="img" aria-label="Experimental DTW path offset over audio time. The center line represents zero delay."><line x1="0" y1="50" x2="100" y2="50" className="chart-axis"/><polyline points={points} className="chart-line"/></svg><div className="chart-labels"><span>{start.toFixed(1)}s</span><span>Audio timeline</span><span>{audio[audio.length-1].toFixed(1)}s</span></div></div>;
}

export function ResultDetails({ result }: { result: Analysis }) {
  return <><div className="result-heading"><span className="result-symbol" aria-hidden="true">✓</span><div><span className="micro">CLASSIFIER ASSESSMENT</span><h4>{result.classification.classification === 'fake' ? 'Model label: fake' : 'Model label: real'}</h4></div></div><p className="result-disclaimer">{result.classification.warning}</p><div className="metrics"><div><span>FAKE-CLASS SCORE</span><strong>{result.classification.score.toFixed(3)}</strong><small>Uncalibrated · not a probability</small></div><div><span>DECISION THRESHOLD</span><strong>{result.classification.threshold.toFixed(3)}</strong><small>Validation-selected</small></div></div><div className="timing-title"><div><span className="micro">TIMING ANALYSIS</span><h4>Audio / lip alignment</h4></div><span className="outline-tag">EXPERIMENTAL</span></div>{result.timing.status === 'experimental_unvalidated' && result.timing.audio_times_seconds && result.timing.video_times_seconds ? <><OffsetChart audio={result.timing.audio_times_seconds} video={result.timing.video_times_seconds}/><div className="timing-meta"><span>Median video − audio <b>{result.timing.median_video_minus_audio_seconds?.toFixed(3)} s</b></span><span>Path cost <b>{result.timing.normalized_cost?.toFixed(3)}</b></span></div><p className="result-disclaimer">DTW is not used by the classifier. This timing signal is experimental and has not been validated as synchronization evidence. {result.timing.caveat}</p></> : <div className="notice">Timing analysis unavailable{result.timing.reason ? `: ${result.timing.reason}` : '.'}</div>}</>;
}
