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
    expect(html).toContain('11.4 / 100');
    expect(html).toContain('Uncalibrated · not a probability');
    expect(html).toContain('decision threshold 49.5 out of 100');
    expect(html).toContain('Audio about 100 ms behind lip movement');
    expect(html).toContain('EXPERIMENTAL');
    expect(html).toContain('This experimental estimate can differ from the actual delay.');
    expect(html).toContain('The classifier gives the Real/Fake label separately.');
  });
  it('shows an unavailable timing state instead of a path', () => {
    const html=renderToStaticMarkup(<ResultDetails result={{...fixture,timing:{status:'unavailable',reason:'no reliable face'}}}/>);
    expect(html).toContain('Timing analysis unavailable');
    expect(html).not.toContain('<polyline');
  });
  it('shows a delayed stream start alongside a real model label and DTW path', () => {
    const timing = { ...fixture.timing, stream_start_audio_minus_video_seconds: .2,
      max_warp_seconds: .4, isolated_missing_frames: 2 };
    const html = renderToStaticMarkup(<ResultDetails result={{ ...fixture, timing }}/>);
    expect(html).toContain('Model label: real');
    expect(html).toContain('Audio starts 200 ms after video');
    expect(html).toContain('Audio content can still be delayed within the stream');
    expect(html).toContain('limited to ±400 ms');
    expect(html).toContain('skipped 2 isolated frames with no detected face');
    expect(html).toContain('<polyline');
  });
  it('plots only supplied signal samples and labels the separate scales', () => {
    const timing = {...fixture.timing, signals: {audio_times_seconds:[0,1], audio_rms:[.2,.4], video_times_seconds:[0,1], lip_aperture:[.1,.3]}, cross_correlation:{status:'experimental_unvalidated',offset_seconds:.2,correlation:.4}};
    const html=renderToStaticMarkup(<ResultDetails result={{...fixture,timing}}/>);
    expect(html).toContain('EXTRACTED SIGNALS');
    expect(html).toContain('Each vertical scale is normalized separately');
    expect(html).toContain('Show timing chart');
    expect(html).not.toContain('Cross-correlation lag');
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

  it('identifies the model adapted using the Drive demo clips', () => {
    const result = { ...base, classification: { ...base.classification,
      prediction_source: 'drive_demo_adapted', dtw_used_for_prediction: false } };
    const html = renderToStaticMarkup(<ResultDetails result={result}/>);
    expect(html).toContain('Drive demo model — trained on the demo clips');
    expect(html).toContain('does <b>not</b> affect this prediction');
  });

  it('says plainly when DTW did drive the prediction, and shows the baseline for comparison', () => {
    const result = { ...base, classification: { ...base.classification,
      classification:'fake' as const, score:.81, prediction_source:'temporal_fusion',
      mode:'fusion', dtw_used_for_prediction:true, baseline_v1_score:.23 } };
    const html = renderToStaticMarkup(<ResultDetails result={result}/>);
    expect(html).toContain('Temporal model (fusion)');
    expect(html).toContain('are inputs to this prediction');
    expect(html).toContain('23.0 / 100');
  });

  it('shows the measured lag direction and hides the feature vector', () => {
    const result = { ...base, timing: { ...base.timing, video_times_seconds: [.12,1.12],
      median_video_minus_audio_seconds: .12, dtw_features: { dtw_normalized_cost: .4796 } } };
    const html = renderToStaticMarkup(<ResultDetails result={result}/>);
    expect(html).toContain('Audio about 120 ms ahead of lip movement');
    expect(html).not.toContain('DTW FEATURE VECTOR');
  });

  it('distinguishes measured zero from a missing lag', () => {
    const zero = { ...base, timing: { ...base.timing, video_times_seconds: [0,1], median_video_minus_audio_seconds: 0 } };
    expect(renderToStaticMarkup(<ResultDetails result={zero}/>)).toContain('Estimated lag: 0 ms');
    const missing = { ...base, timing: { ...base.timing, median_video_minus_audio_seconds: undefined } };
    expect(renderToStaticMarkup(<ResultDetails result={missing}/>)).toContain('Estimated lag unavailable');
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
    expect(html).toContain('The classifier gives the Real/Fake label separately');
  });

  it('never claims DTW is unused when it drove the prediction', () => {
    const html = renderToStaticMarkup(<ResultDetails result={{...base,
      classification:{...base.classification, prediction_source:'temporal_fusion', mode:'fusion',
        dtw_used_for_prediction:true}}}/>);
    expect(html).not.toContain('The classifier gives the Real/Fake label separately');
    expect(html).toContain('The temporal model uses timing features in its prediction.');
  });
});
