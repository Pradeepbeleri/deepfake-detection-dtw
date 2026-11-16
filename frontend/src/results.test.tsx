import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { ResultDetails } from './ResultDetails';
import type { Analysis } from './api';

describe('result rendering: synthetic fixtures only', () => {
  const fixture: Analysis = { classification: { classification:'real', score:.114, threshold:.495,
    score_kind:'uncalibrated sigmoid', warning:'Not identity verification' },
    timing: { status:'experimental_unvalidated', audio_times_seconds:[0,1], video_times_seconds:[0,.9],
      median_video_minus_audio_seconds:-.1, normalized_cost:.4, caveat:'Timing is not a prediction.' }, warning:'Research only' };
  it('labels the supplied fake-class score and keeps timing separate', () => {
    const html=renderToStaticMarkup(<ResultDetails result={fixture}/>);
    expect(html).toContain('Model label: real');
    expect(html).toContain('0.114');
    expect(html).toContain('Uncalibrated · not a probability');
    expect(html).toContain('decision threshold 0.495');
    expect(html).toContain('not a manipulated-segment timeline');
    expect(html).toContain('EXPERIMENTAL');
    expect(html).toContain('not been validated as synchronization evidence');
    expect(html).toContain('DTW is not used by the classifier.');
  });
  it('shows an unavailable timing state instead of a path', () => {
    const html=renderToStaticMarkup(<ResultDetails result={{...fixture,timing:{status:'unavailable',reason:'no reliable face'}}}/>);
    expect(html).toContain('Timing analysis unavailable');
    expect(html).not.toContain('<polyline');
  });
  it('plots only supplied signal samples and labels the separate scales', () => {
    const timing = {...fixture.timing, signals: {audio_times_seconds:[0,1], audio_rms:[.2,.4], video_times_seconds:[0,1], lip_aperture:[.1,.3]}, cross_correlation:{status:'experimental_unvalidated',offset_seconds:.2,correlation:.4}};
    const html=renderToStaticMarkup(<ResultDetails result={{...fixture,timing}}/>);
    expect(html).toContain('EXTRACTED SIGNALS');
    expect(html).toContain('Each vertical scale is normalized separately');
    expect(html).toContain('Cross-correlation lag: 0.200 s');
  });
});

describe('DTW feature display and prediction provenance', () => {
  const base: Analysis = { classification: { classification:'real', score:.114, threshold:.495,
    score_kind:'uncalibrated sigmoid', warning:'Not identity verification' },
    timing: { status:'experimental_unvalidated', audio_times_seconds:[0,1], video_times_seconds:[0,.9],
      median_video_minus_audio_seconds:-.1, normalized_cost:.4, caveat:'Timing is not a prediction.' },
    warning:'Research only' };

  it('says plainly when DTW did not affect the prediction', () => {
    const result = { ...base, classification: { ...base.classification,
      prediction_source:'baseline_v1_embeddings', dtw_used_for_prediction:false } };
    const html = renderToStaticMarkup(<ResultDetails result={result}/>);
    expect(html).toContain('Baseline V1');
    expect(html).toContain('does <b>not</b> affect this prediction');
  });

  it('says plainly when DTW did drive the prediction, and shows the baseline for comparison', () => {
    const result = { ...base, classification: { ...base.classification,
      classification:'fake' as const, score:.81, prediction_source:'temporal_fusion',
      mode:'fusion', dtw_used_for_prediction:true, baseline_v1_score:.23 } };
    const html = renderToStaticMarkup(<ResultDetails result={result}/>);
    expect(html).toContain('Temporal model (fusion)');
    expect(html).toContain('are inputs to this prediction');
    expect(html).toContain('0.230');
  });

  it('renders measured DTW features and marks uncomputed ones rather than printing zero', () => {
    const result = { ...base, timing: { ...base.timing, dtw_features: {
      dtw_normalized_cost: 0.4796, dtw_boundary_fraction: 0.043,
      dtw_median_offset_seconds: -0.12, xcorr_peak: null } } };
    const html = renderToStaticMarkup(<ResultDetails result={result}/>);
    expect(html).toContain('DTW FEATURE VECTOR');
    expect(html).toContain('0.480');            // alignment cost, 3dp
    expect(html).toContain('-120 ms');          // seconds rendered as ms
    expect(html).toContain('not computed');     // null is never shown as 0
    expect(html).not.toContain('>0.000<');
  });
});

describe('timing disclaimer matches whether DTW was actually used', () => {
  const base: Analysis = { classification: { classification:'real', score:.1, threshold:.5,
    score_kind:'uncalibrated sigmoid', warning:'w' },
    timing: { status:'experimental_unvalidated', audio_times_seconds:[0,1], video_times_seconds:[0,.9],
      median_video_minus_audio_seconds:-.1, normalized_cost:.4, caveat:'c' }, warning:'r' };

  it('says DTW is unused only when it is unused', () => {
    const html = renderToStaticMarkup(<ResultDetails result={{...base,
      classification:{...base.classification, prediction_source:'baseline_v1_embeddings',
        dtw_used_for_prediction:false}}}/>);
    expect(html).toContain('DTW is not used by the classifier');
  });

  it('never claims DTW is unused when it drove the prediction', () => {
    const html = renderToStaticMarkup(<ResultDetails result={{...base,
      classification:{...base.classification, prediction_source:'temporal_fusion', mode:'fusion',
        dtw_used_for_prediction:true}}}/>);
    expect(html).not.toContain('DTW is not used by the classifier');
    expect(html).toContain('inputs to the temporal model');
    expect(html).toContain('within run-to-run noise');
  });
});

describe('audio/video synchronization panel', () => {
  const base: Analysis = { classification: { classification:'real', score:.3, threshold:.45,
    score_kind:'uncalibrated sigmoid', warning:'w' },
    timing: { status:'experimental_unvalidated', audio_times_seconds:[0,1], video_times_seconds:[0,.9],
      median_video_minus_audio_seconds:-.04, normalized_cost:.4, caveat:'c' }, warning:'r' };
  const withSync = (gs: NonNullable<Analysis['timing']['global_sync']>) =>
    renderToStaticMarkup(<ResultDetails result={{...base, timing:{...base.timing, global_sync: gs}}}/>);

  it('reports a measured offset in milliseconds with the right direction', () => {
    const html = withSync({status:'experimental', confidence:'measured', offset_seconds:-1.64,
      correlation:0.455, peak_margin:0.09});
    expect(html).toContain('Audio is ahead of the picture by 1640 ms');
    expect(html).toContain('-1.64 s');
    expect(html).toContain('0.455');
  });

  it('says picture lags when the offset is positive', () => {
    const html = withSync({status:'experimental', confidence:'measured', offset_seconds:1.2,
      correlation:0.5, peak_margin:0.1});
    expect(html).toContain('behind the picture by 1200 ms');
  });

  it('calls a near-zero offset in sync rather than inventing a delay', () => {
    const html = withSync({status:'experimental', confidence:'measured', offset_seconds:-0.04,
      correlation:0.41, peak_margin:0.08});
    expect(html).toContain('No significant offset detected');
  });

  it('refuses to report an offset it cannot measure', () => {
    const html = withSync({status:'rejected', confidence:'unreliable', offset_seconds:null,
      correlation:0.22, peak_margin:0.01});
    expect(html).toContain('Offset could not be determined');
    expect(html).toContain('rather than guessing one');
    expect(html).not.toContain('No significant offset detected');
  });

  it('never presents a delay as evidence of manipulation', () => {
    const html = withSync({status:'experimental', confidence:'measured', offset_seconds:-1.64,
      correlation:0.455, peak_margin:0.09});
    expect(html).toContain('not evidence of manipulation');
  });
});
