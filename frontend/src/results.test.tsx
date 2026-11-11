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
});
