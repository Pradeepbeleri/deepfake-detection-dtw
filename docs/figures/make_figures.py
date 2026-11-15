"""Render report figures from the recorded experiment numbers only.

Every value below is transcribed from docs/dtw-research-validation.md and the
recorded held-out evaluation. Nothing here is simulated, fitted or interpolated.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from pathlib import Path

OUT = Path('/home/user/deepfake-detection-dtw/docs/figures')
OUT.mkdir(parents=True, exist_ok=True)

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK2 = '#52514e'
MUTED = '#8a8880'
GRID = '#e4e3df'
S1, S2, S3 = '#2a78d6', '#eb6834', '#1baf7a'   # validated slots 1-3
SEQ = ['#cde2fb', '#9ec5f4', '#5598e7', '#2a78d6', '#1c5cab', '#0d366b']

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE,
    'axes.edgecolor': GRID, 'axes.labelcolor': INK2,
    'xtick.color': INK2, 'ytick.color': INK2,
    'axes.linewidth': 0.8, 'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'svg.fonttype': 'none',
})


def recessive(ax, grid_axis='y'):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- figure 1
# Recorded held-out confusion matrix: rows actual real/fake, cols predicted.
cm = np.array([[20, 10], [4, 26]])
fig, (ax, axt) = plt.subplots(1, 2, figsize=(7.6, 3.1),
                              gridspec_kw={'width_ratios': [1, 1.15]})
vmax = cm.max()
for i in range(2):
    for j in range(2):
        frac = cm[i, j] / vmax
        colour = SEQ[min(int(frac * (len(SEQ) - 1) + 0.5), len(SEQ) - 1)]
        # 2px surface gap between adjacent cells
        ax.add_patch(plt.Rectangle((j + 0.015, -i - 0.985), 0.97, 0.97,
                                   facecolor=colour, edgecolor=SURFACE, linewidth=2))
        ax.text(j + 0.5, -i - 0.5, str(cm[i, j]), ha='center', va='center',
                fontsize=19, fontweight='bold',
                color='#ffffff' if frac > 0.55 else INK)
ax.set_xlim(0, 2); ax.set_ylim(-2, 0)
ax.set_xticks([0.5, 1.5]); ax.set_xticklabels(['predicted\nreal', 'predicted\nfake'])
ax.set_yticks([-0.5, -1.5]); ax.set_yticklabels(['actual\nreal', 'actual\nfake'])
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title('Held-out confusion matrix (60 videos)', color=INK,
             fontsize=10, fontweight='bold', pad=10, loc='left')

axt.axis('off')
rows = [('Accuracy', '76.67 %'), ('Precision', '72.22 %'), ('Recall', '86.67 %'),
        ('F1', '78.79 %'), ('ROC-AUC', '0.8256'), ('False-positive rate', '33.33 %')]
axt.text(0, 1.0, 'Recorded held-out metrics', color=INK, fontsize=10,
         fontweight='bold', va='top')
for k, (name, val) in enumerate(rows):
    y = 0.84 - k * 0.133
    axt.text(0.0, y, name, color=INK2, fontsize=9, va='center')
    axt.text(0.62, y, val, color=INK, fontsize=9.5, va='center', fontweight='bold')
axt.text(0, -0.1, 'One recorded run · 30 real / 30 fake · uncalibrated scores\n'
                  'threshold 0.4947 selected on validation · identity-disjointness not verified',
         color=MUTED, fontsize=7.6, va='top')
fig.tight_layout()
fig.savefig(OUT / 'fig1-heldout-confusion.svg', bbox_inches='tight')
plt.close(fig)


# ---------------------------------------------------------------- figure 2
# Experiment 1: controlled audio-timestamp shifts, RMS / aperture representation.
shifts = np.array([-400, -200, -100, 0, 100, 200, 400])
ideal = -shifts
fixed = np.zeros_like(shifts)
xcorr = np.array([400, 200, 120, 0, -80, -200, -400])
dtw = np.array([-40, 120, 80, 0, -40, -80, 80])

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.6),
                              gridspec_kw={'width_ratios': [1.25, 1]})
recessive(ax)
ax.axhline(0, color=GRID, linewidth=0.9)
ax.plot(shifts, ideal, linestyle=(0, (5, 3)), color=MUTED, linewidth=2,
        zorder=2, label='Ideal response (= −shift)')
for y, c, lbl, mk in ((fixed, S3, 'Fixed zero', 's'),
                      (xcorr, S1, 'Cross-correlation', 'o'),
                      (dtw, S2, 'DTW median path offset', '^')):
    ax.plot(shifts, y, color=c, linewidth=2, marker=mk, markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=3, label=lbl)
ax.set_xlabel('Imposed audio-timestamp shift (ms)')
ax.set_ylabel('Measured relative response (ms)')
ax.set_xticks([-400, -200, 0, 200, 400])
ax.set_xlim(-480, 780)
ax.set_title('Experiment 1 — controlled timestamp shifts',
             color=INK, fontsize=10, fontweight='bold', loc='left', pad=10)
# direct labels at the right-hand end of each series, clear of the marks
for yval, c, txt in ((-400, S1, 'Cross-\ncorrelation'), (80, S2, 'DTW'), (0, S3, 'Fixed zero')):
    ax.text(425, yval, txt, color=c, fontsize=8, fontweight='bold', va='center')
ax.text(425, 400, 'Ideal', color=MUTED, fontsize=8, fontweight='bold', va='center')
ax.legend(frameon=False, fontsize=7.5, loc='lower left', labelcolor=INK2)

recessive(ax2)
methods = ['Fixed zero', 'Cross-\ncorrelation', 'DTW']
mean_err = [200.00, 5.71, 171.43]
cols = [S3, S1, S2]
bars = ax2.bar(methods, mean_err, color=cols, width=0.56, zorder=3,
               edgecolor=SURFACE, linewidth=2)
for b, v in zip(bars, mean_err):
    ax2.text(b.get_x() + b.get_width() / 2, v + 6, f'{v:.2f} ms',
             ha='center', color=INK, fontsize=9, fontweight='bold')
ax2.set_ylabel('Mean absolute response error (ms)')
ax2.set_ylim(0, 245)
ax2.set_title('Lower is better', color=INK, fontsize=10, fontweight='bold',
              loc='left', pad=10)
ax2.text(0, -0.30, 'DTW path touched its ±400 ms warping boundary on all six nonzero shifts.',
         transform=ax2.transAxes, color=MUTED, fontsize=7.6)
fig.tight_layout()
fig.savefig(OUT / 'fig2-controlled-shift-response.svg', bbox_inches='tight')
plt.close(fig)


# ---------------------------------------------------------------- figure 3
# Experiment 2: physical playback shifts. None = rejected / unavailable.
imposed = [-3000, -2000, -1000, -500, -200, 0, 200, 500, 1000, 2000, 3000]
corr_err = [640, 640, None, 2100, 0, 0, 1680, None, 4920, 4920, None]
dtw_err = [560, 600, None, 2100, 40, 0, 1720, None, 4800, 4800, None]

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.8),
                              gridspec_kw={'width_ratios': [1.6, 1]})
recessive(ax)
x = np.arange(len(imposed))
w = 0.38
for off, data, c, lbl in ((-w / 2 - 0.012, corr_err, S1, 'Global correlation'),
                          (w / 2 + 0.012, dtw_err, S2, 'Global correlation + local DTW')):
    vals = [0 if v is None else v for v in data]
    ax.bar(x + off, vals, width=w, color=c, zorder=3,
           edgecolor=SURFACE, linewidth=2, label=lbl)
for k, v in enumerate(corr_err):
    if v is None:
        ax.text(k, 170, 'rejected', ha='center', color=MUTED, fontsize=7.4,
                rotation=90, va='bottom', fontweight='bold')
    elif v == 0 and dtw_err[k] in (0, 40):
        # An exact-zero error is a measurement, not a gap: say so.
        ax.text(k, 170, 'measured ≈ 0', ha='center', color=INK2, fontsize=7.4,
                rotation=90, va='bottom', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'{v/1000:+.1f}'.replace('+0.0', '0') for v in imposed])
ax.set_xlabel('Imposed physical audio delay (s)')
ax.set_ylabel('Absolute error vs known shift (ms)')
ax.set_title('Experiment 2 — physical ±3 s playback shifts, one clip',
             color=INK, fontsize=10, fontweight='bold', loc='left', pad=10)
ax.legend(frameon=False, fontsize=8, loc='upper left', labelcolor=INK2)
ax.set_ylim(0, 5600)

recessive(ax2)
m2 = ['Fixed\nzero', 'Global\ncorrelation', 'Correlation\n+ DTW']
mean2 = [1218.18, 1862.50, 1827.50]
cols2 = [S3, S1, S2]
bars = ax2.bar(m2, mean2, color=cols2, width=0.56, zorder=3,
               edgecolor=SURFACE, linewidth=2)
for b, v, n in zip(bars, mean2, ['11/11', '8/11', '8/11']):
    ax2.text(b.get_x() + b.get_width() / 2, v + 55, f'{v:.0f} ms',
             ha='center', color=INK, fontsize=9, fontweight='bold')
    ax2.text(b.get_x() + b.get_width() / 2, 70, n, ha='center',
             color='#ffffff', fontsize=8, fontweight='bold')
ax2.set_ylabel('Mean absolute error, measured cases (ms)')
ax2.set_ylim(0, 2300)
ax2.set_title('Estimating nothing beats estimating badly',
              color=INK, fontsize=10, fontweight='bold', loc='left', pad=10)
ax2.text(0, -0.32, 'Bar labels show measured / attempted. Fixed zero never fails, so it is\n'
                   'averaged over all 11; the other two are averaged over the 8 they accepted.',
         transform=ax2.transAxes, color=MUTED, fontsize=7.6)
fig.tight_layout()
fig.savefig(OUT / 'fig3-physical-shift-error.svg', bbox_inches='tight')
plt.close(fig)

print('wrote:')
for p in sorted(OUT.glob('*.svg')):
    print(' ', p, p.stat().st_size, 'bytes')
