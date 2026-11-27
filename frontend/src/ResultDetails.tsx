import React from 'react';
import type { Analysis } from './api';

function SignalTimelines({ signals }: { signals: NonNullable<Analysis['timing']['signals']> }) {
  const start = Math.min(signals.audio_times_seconds[0], signals.video_times_seconds[0]);
  const end = Math.max(signals.audio_times_seconds.at(-1)!, signals.video_times_seconds.at(-1)!);
  const points = (times: number[], values: number[], top: number) => {
    const lo = Math.min(...values), span = Math.max(Math.max(...values) - lo, 1e-12);
    return times.map((t, i) => `${((t-start)/Math.max(end-start, .01)*100).toFixed(2)},${(top+30-(values[i]-lo)/span*25).toFixed(2)}`).join(' ');
  };
  return <div className="signal-timelines"><span className="micro">EXTRACTED SIGNALS · SAME TIME AXIS</span><div className="signal-legend"><span>Audio RMS</span><span>Lip aperture</span></div><svg className="signal-chart" viewBox="0 0 100 80" preserveAspectRatio="none" role="img" aria-label="Extracted audio energy and lip aperture plotted against the same media timeline; each vertical scale is normalized separately."><line x1="0" y1="40" x2="100" y2="40" className="chart-axis"/><polyline points={points(signals.audio_times_seconds, signals.audio_rms, 0)} className="chart-line"/><polyline points={points(signals.video_times_seconds, signals.lip_aperture, 40)} className="chart-line signal-visual"/></svg><div className="chart-labels"><span>{start.toFixed(1)}s</span><span>Media time</span><span>{end.toFixed(1)}s</span></div><p className="chart-caption">Each vertical scale is normalized separately. These signals are not validated as equivalent, and their shape does not identify manipulation.</p></div>;
}


function OffsetChart({ audio, video }: { audio: number[]; video: number[] }) {
  if (!audio.length || audio.length !== video.length) return null;
  const start = audio[0], span = Math.max(audio[audio.length - 1] - start, .01);
  const points = audio.map((t, i) => `${((t-start)/span*100).toFixed(2)},${(50 - Math.max(-.5, Math.min(.5, t-video[i]))*80).toFixed(2)}`).join(' ');
  return <div className="chart-wrap"><svg className="chart" viewBox="0 0 100 100" preserveAspectRatio="none" role="img" aria-label="Experimental DTW path offset over audio time. The center line represents zero delay."><line x1="0" y1="50" x2="100" y2="50" className="chart-axis"/><polyline points={points} className="chart-line"/></svg><div className="chart-labels"><span>{start.toFixed(1)}s</span><span>Audio timeline</span><span>{audio[audio.length-1].toFixed(1)}s</span></div></div>;
}

function StreamStartOffset({ seconds }: { seconds: number }) {
  const milliseconds = Math.round(Math.abs(seconds) * 1000);
  const description = milliseconds === 0
    ? 'Audio and video start together (within 1 ms)'
    : `Audio starts ${milliseconds} ms ${seconds > 0 ? 'after' : 'before'} video`;
  return <div className="timing-meta">
    <span>Stream timestamps <b>{description}</b></span>
    <small>This measures when playback streams start. Audio content can still be delayed within the stream.</small>
  </div>;
}

export function ResultDetails({ result }: { result: Analysis }) {
  const src = result.classification.prediction_source;
  const dtwDrove = result.classification.dtw_used_for_prediction === true;
  const displayScore = (result.classification.score * 100).toFixed(1);
  let sourceLabel = `Temporal model (${result.classification.mode ?? 'unknown'})`;
  if (src === 'baseline_v1_embeddings') sourceLabel = 'Baseline V1 — Whisper + DINOv2 embeddings';
  if (src === 'drive_demo_adapted') sourceLabel = 'Drive demo model — trained on the demo clips';
  return <>{src && <div className={`prediction-source ${dtwDrove ? 'dtw' : ''}`}>
      <span className="micro">PREDICTION PRODUCED BY</span>
      <strong>{sourceLabel}</strong>
      {dtwDrove
        ? <small>DTW-derived timing features are inputs to this prediction.</small>
        : <small>DTW does <b>not</b> affect this prediction.</small>}
      {result.temporal_model?.status === 'unavailable' &&
        <small>The configured temporal model could not score this clip. The label uses the model shown above.</small>}
      {typeof result.classification.baseline_v1_score === 'number' &&
        <small>Baseline V1 score for comparison: {(result.classification.baseline_v1_score * 100).toFixed(1)} / 100</small>}
    </div>}<div className="result-heading"><span className="result-symbol" aria-hidden="true">✓</span><div><span className="micro">CLASSIFIER ASSESSMENT</span><h4>{result.classification.classification === 'fake' ? 'Model label: fake' : 'Model label: real'}</h4></div></div><p className="result-disclaimer">{result.classification.warning}</p><div className="metrics"><div><span>FAKE-CLASS SCORE</span><strong>{displayScore} / 100</strong><small>Uncalibrated · not a probability</small></div></div><div className="score-scale"><div className="scale-heading"><span>Fake-class score out of 100</span></div><div className="score-track" role="img" aria-label={`Uncalibrated fake-class score ${displayScore} out of 100`}><span className="score-fill" style={{width: `${result.classification.score * 100}%`}}/></div><div className="scale-heading"><span>0</span><span>100</span></div><p>Scores indicate model output, not the probability that a video is authentic or fake.</p></div><div className="timing-title"><div><span className="micro">TIMING ANALYSIS</span><h4>Audio / lip alignment</h4></div><span className="outline-tag">EXPERIMENTAL</span></div>
    <TimingSummary timing={result.timing} dtwDrove={dtwDrove}/>
  </>;
}

function TimingSummary({ timing, dtwDrove }: { timing: Analysis['timing']; dtwDrove: boolean }) {
  const audio = timing.audio_times_seconds;
  const video = timing.video_times_seconds;
  if (timing.status !== 'experimental_unvalidated' || !audio?.length || !video?.length) {
    return <div className="notice">Timing analysis unavailable{timing.reason ? `: ${timing.reason}` : '.'}</div>;
  }

  // The backend reports video minus audio. Reverse the sign for audio lag.
  const offset = timing.median_video_minus_audio_seconds;
  const lagMs = typeof offset === 'number' && Number.isFinite(offset) ? -offset * 1000 : null;
  const magnitude = lagMs === null ? null : Math.round(Math.abs(lagMs));
  let summary = 'Estimated lag unavailable';
  if (magnitude === 0) summary = '0 ms estimated lag';
  if (magnitude && lagMs !== null) summary = `Audio about ${magnitude} ms ${lagMs > 0 ? 'behind' : 'ahead of'} lip movement`;

  return <>
    <div className="notice">
      <span className="micro">ESTIMATED DTW LAG</span>
      <div><strong>{summary}</strong></div>
      <small>{magnitude === 0
        ? 'The matched timestamps have a median offset of zero. A delayed clip can also produce this estimate.'
        : 'Median of the matched audio and video timestamps in this upload. This experimental estimate can differ from the actual delay.'}</small>
    </div>
    <p className="result-disclaimer">{dtwDrove
      ? 'The temporal model uses timing features in its prediction.'
      : 'The classifier gives the Real/Fake label separately.'}</p>
    <details>
      <summary>Show timing chart</summary>
      <OffsetChart audio={audio} video={video}/>
      <p className="chart-caption">Positive offsets mean matched audio is later than matched video.
        {typeof timing.max_warp_seconds === 'number' && ` The path search was limited to ±${(timing.max_warp_seconds * 1000).toFixed(0)} ms.`}
        {!!timing.isolated_missing_frames && ` The timing analysis skipped ${timing.isolated_missing_frames} isolated frames with no detected face.`}
      </p>
      {timing.signals && <SignalTimelines signals={timing.signals}/>}
      {typeof timing.stream_start_audio_minus_video_seconds === 'number' &&
        <StreamStartOffset seconds={timing.stream_start_audio_minus_video_seconds}/>}
    </details>
  </>;
}
