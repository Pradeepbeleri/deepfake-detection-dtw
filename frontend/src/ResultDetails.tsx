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


const FEATURE_LABELS: Record<string, string> = {
  dtw_normalized_cost: 'DTW alignment cost',
  dtw_gain_over_direct: 'Gain over no-warp',
  direct_alignment_cost: 'Direct (no-warp) cost',
  dtw_median_offset_seconds: 'Median offset',
  dtw_mean_abs_offset_seconds: 'Mean |offset|',
  dtw_offset_iqr_seconds: 'Offset spread (IQR)',
  dtw_offset_drift_seconds: 'Offset drift',
  dtw_boundary_fraction: 'Warp-band contact',
  dtw_path_length_ratio: 'Path length ratio',
  xcorr_peak: 'Cross-correlation peak',
  xcorr_lag_seconds: 'Cross-correlation lag',
  xcorr_peak_margin: 'Peak margin',
  audio_energy_cv: 'Audio energy variation',
  aperture_std: 'Aperture spread',
  aperture_velocity_mean_abs: 'Mean |aperture velocity|',
  overlap_seconds: 'Usable overlap',
};
const SECONDS = /_seconds$/;

function DtwFeatureTable({ features }: { features: NonNullable<Analysis['timing']['dtw_features']> }) {
  const rows = Object.entries(features);
  if (!rows.length) return null;
  return <div className="dtw-features"><span className="micro">DTW FEATURE VECTOR · CLASSIFIER INPUT</span>
    <table className="feature-table"><tbody>
      {rows.map(([key, value]) => <tr key={key}>
        <th scope="row">{FEATURE_LABELS[key] ?? key}</th>
        <td>{value === null ? <span className="unavailable">not computed</span>
          : SECONDS.test(key) ? `${(value * 1000).toFixed(0)} ms` : value.toFixed(3)}</td>
      </tr>)}
    </tbody></table>
    <p className="chart-caption">These are the measured values the temporal model consumes. A high
      warp-band contact means the alignment was pinned to its limit, so the offset is a lower bound.</p>
  </div>;
}


function SyncPanel({ sync }: { sync: NonNullable<Analysis['timing']['global_sync']> }) {
  const off = sync.offset_seconds;
  const reportable = sync.confidence !== 'unreliable' && off !== null && off !== undefined;
  const ms = reportable ? Math.round(off! * 1000) : null;
  const inSync = reportable && Math.abs(ms!) <= 100;
  return <div className={`sync-panel ${reportable ? (inSync ? 'ok' : 'offset') : 'unknown'}`}>
    <span className="micro">AUDIO / VIDEO SYNCHRONIZATION · ±3s SEARCH</span>
    {reportable
      ? <><strong>{inSync ? 'No significant offset detected'
          : `Audio is ${ms! < 0 ? 'ahead of' : 'behind'} the picture by ${Math.abs(ms!)} ms`}</strong>
         <div className="sync-meta">
           <span>Offset <b>{(off! >= 0 ? '+' : '') + off!.toFixed(2)} s</b></span>
           <span>Correlation <b>{sync.correlation?.toFixed(3)}</b></span>
           <span>Confidence <b>{sync.confidence}</b></span>
         </div></>
      : <><strong>Offset could not be determined</strong>
         <small>Audio energy and lip motion did not correlate strongly enough on this clip
           {sync.correlation !== undefined ? ` (peak ${sync.correlation.toFixed(3)})` : ''}.
           No offset is reported rather than guessing one.</small></>}
    <small>{sync.note ?? 'A playback offset is a synchronization issue, not evidence of manipulation.'}</small>
  </div>;
}

function OffsetChart({ audio, video }: { audio: number[]; video: number[] }) {
  if (!audio.length || audio.length !== video.length) return null;
  const start = audio[0], span = Math.max(audio[audio.length - 1] - start, .01);
  const points = audio.map((t, i) => `${((t-start)/span*100).toFixed(2)},${(50 - Math.max(-.5, Math.min(.5, video[i]-t))*80).toFixed(2)}`).join(' ');
  return <div className="chart-wrap"><svg className="chart" viewBox="0 0 100 100" preserveAspectRatio="none" role="img" aria-label="Experimental DTW path offset over audio time. The center line represents zero delay."><line x1="0" y1="50" x2="100" y2="50" className="chart-axis"/><polyline points={points} className="chart-line"/></svg><div className="chart-labels"><span>{start.toFixed(1)}s</span><span>Audio timeline</span><span>{audio[audio.length-1].toFixed(1)}s</span></div></div>;
}

export function ResultDetails({ result }: { result: Analysis }) {
  const src = result.classification.prediction_source;
  const dtwDrove = result.classification.dtw_used_for_prediction === true;
  return <>{src && <div className={`prediction-source ${dtwDrove ? 'dtw' : ''}`}>
      <span className="micro">PREDICTION PRODUCED BY</span>
      <strong>{src === 'baseline_v1_embeddings' ? 'Baseline V1 — Whisper + DINOv2 embeddings'
        : `Temporal model (${result.classification.mode ?? 'dtw'}) — DTW features in the decision`}</strong>
      {dtwDrove
        ? <small>DTW-derived timing features are inputs to this prediction.</small>
        : <small>DTW is computed and shown below but does <b>not</b> affect this prediction.</small>}
      {typeof result.classification.baseline_v1_score === 'number' &&
        <small>Baseline V1 score for comparison: {result.classification.baseline_v1_score.toFixed(3)}</small>}
    </div>}<div className="result-heading"><span className="result-symbol" aria-hidden="true">✓</span><div><span className="micro">CLASSIFIER ASSESSMENT</span><h4>{result.classification.classification === 'fake' ? 'Model label: fake' : 'Model label: real'}</h4></div></div><p className="result-disclaimer">{result.classification.warning}</p><div className="metrics"><div><span>FAKE-CLASS SCORE</span><strong>{result.classification.score.toFixed(3)}</strong><small>Uncalibrated · not a probability</small></div><div><span>DECISION THRESHOLD</span><strong>{result.classification.threshold.toFixed(3)}</strong><small>Validation-selected</small></div></div><div className="score-scale"><div className="scale-heading"><span>Fake-class score on a 0–1 scale</span><span>Threshold {result.classification.threshold.toFixed(3)}</span></div><div className="score-track" role="img" aria-label={`Uncalibrated fake-class score ${result.classification.score.toFixed(3)} on a zero to one scale; decision threshold ${result.classification.threshold.toFixed(3)}`}><span className="score-fill" style={{width: `${result.classification.score * 100}%`}}/><span className="threshold-mark" style={{left: `${Math.max(0, Math.min(1, result.classification.threshold)) * 100}%`}}/></div><div className="scale-heading"><span>0</span><span>1</span></div><p>Scores indicate model output, not the probability that a video is authentic or fake.</p></div><div className="timing-title"><div><span className="micro">TIMING ANALYSIS</span><h4>Audio / lip alignment</h4></div><span className="outline-tag">EXPERIMENTAL</span></div>{result.timing.status === 'experimental_unvalidated' && result.timing.audio_times_seconds && result.timing.video_times_seconds ? <>{result.timing.global_sync && <SyncPanel sync={result.timing.global_sync}/>}{result.timing.signals && <SignalTimelines signals={result.timing.signals}/>}{result.timing.dtw_features && <DtwFeatureTable features={result.timing.dtw_features}/>}<p className="result-disclaimer">Each point compares a pair of actual DTW path timestamps. The line is not a manipulated-segment timeline.</p><OffsetChart audio={result.timing.audio_times_seconds} video={result.timing.video_times_seconds}/><p className="chart-caption">Vertical scale: −0.5 to +0.5 seconds; center is zero. Positive values mean the paired video time is later.</p><div className="timing-meta"><span>Median video − audio <b>{result.timing.median_video_minus_audio_seconds?.toFixed(3)} s</b></span><span>Path cost <b>{result.timing.normalized_cost?.toFixed(3)}</b></span></div>{result.timing.cross_correlation?.status === "experimental_unvalidated" && <div className="method-comparison"><span>Fixed-zero reference: 0.000 s</span><span>Cross-correlation lag: {result.timing.cross_correlation.offset_seconds?.toFixed(3)} s{result.timing.cross_correlation.search_boundary ? " · search boundary" : ""}</span><small>Both methods and DTW are exploratory; their outputs are not validated estimates of manipulation.</small></div>}<p className="result-disclaimer">{dtwDrove ? 'These timing features are inputs to the temporal model above. They are not a validated measure of audiovisual synchronization, and the measured gain from adding them to the embedding baseline was within run-to-run noise.' : 'DTW is not used by the classifier.'} This timing signal is experimental and has not been validated as synchronization evidence. {result.timing.caveat}</p></> : <div className="notice">Timing analysis unavailable{result.timing.reason ? `: ${result.timing.reason}` : '.'}</div>}</>;
}
