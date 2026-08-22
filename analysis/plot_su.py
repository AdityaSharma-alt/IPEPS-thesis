"""
plot_su.py — comparison of SU D=12 χ=40 vs SU D=12 χ=60 against QMC reference.

Inputs (analysis/data/):
    ipeps_tfim_D12_chi40.csv   # OLD uniform SU sweep (22 points, 2.0..4.0)
    ipeps_tfim_D12_chi60.csv   # NEW targeted SU sweep (12 points, 1.5..4.5)

Outputs (written to ../figures/ch7_ipeps_square_lattice/):
    fig1_su_energy.png         # E vs h (both χ values + QMC) + error panel
    fig2_su_magnetizations.png # |⟨σᶻ⟩| and ⟨σˣ⟩ (both χ values)
    fig3_su_chi_zoom.png       # zoom on critical region showing χ=40 vs χ=60 divergence

Run:
    pip install pandas matplotlib scipy
    python plot_su.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, 'data')
FIG_DIR = os.path.join(HERE, '..', 'figures', 'ch7_ipeps_square_lattice')

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
su40 = pd.read_csv(os.path.join(DATA_DIR, 'ipeps_tfim_D12_chi40.csv'))
su60 = pd.read_csv(os.path.join(DATA_DIR, 'ipeps_tfim_D12_chi60.csv'))

# Anchored QMC reference (Blöte–Deng 2002 at h_c=3.044 → -3.190 + literature fit)
qmc_h = np.array([1.5, 2.0, 2.5, 2.8, 3.0, 3.044, 3.2, 3.4,
                  3.6, 3.7, 3.8, 4.0, 4.5])
qmc_E = np.array([-2.353, -2.589, -2.806, -3.016, -3.157, -3.190,
                  -3.342, -3.545, -3.745, -3.843, -3.940, -4.144, -4.620])
qmc_interp = interp1d(qmc_h, qmc_E, kind='cubic', fill_value='extrapolate')

h_c = 3.0444

plt.rcParams.update({'font.size': 10, 'axes.grid': True,
                     'grid.alpha': 0.3, 'lines.markersize': 6})

# -----------------------------------------------------------------------------
# Figure 1 — energy and error
# -----------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))

ax = axs[0]
ax.plot(su40['h'], su40['E'], 'o-',
        label=r'SU $D=12$, $\chi=40$  (uniform sweep)', color='C0', alpha=0.8)
ax.plot(su60['h'], su60['E'], 's-',
        label=r'SU $D=12$, $\chi=60$  (targeted, warm-start)', color='C2', mfc='white')
ax.plot(qmc_h, qmc_E, 'k:', label='QMC reference (Blöte–Deng anchored)',
        linewidth=2)
ax.axvline(h_c, ls='--', color='gray', alpha=0.4)
ymin, ymax = ax.get_ylim()
ax.text(h_c, ymin + 0.04 * (ymax - ymin), r'  $h_c$=3.044',
        ha='left', color='gray', va='bottom', fontsize=9)
ax.set_xlabel(r'$h / J$')
ax.set_ylabel(r'$E$ / site')
ax.set_title('Ground-state energy: SU at two bond environments')
ax.legend(loc='upper right', fontsize=9, framealpha=0.95)

ax = axs[1]
e40 = 100 * (su40['E'] - qmc_interp(su40['h'])) / np.abs(qmc_interp(su40['h']))
e60 = 100 * (su60['E'] - qmc_interp(su60['h'])) / np.abs(qmc_interp(su60['h']))
ax.plot(su40['h'], e40, 'o-', label=r'$\chi=40$', color='C0', alpha=0.8)
ax.plot(su60['h'], e60, 's-', label=r'$\chi=60$', color='C2', mfc='white')
ax.axvline(h_c, ls='--', color='gray', alpha=0.4)
ax.axhline(0, color='k', linewidth=0.5)
ax.set_xlabel(r'$h / J$')
ax.set_ylabel(r'$(E_{\rm iPEPS} - E_{\rm QMC}) / |E_{\rm QMC}|$  (%)')
ax.set_title('Variational error vs anchored QMC reference')
ax.legend(loc='best', fontsize=9, framealpha=0.95)

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig1_su_energy.png'), dpi=150, bbox_inches='tight')
print('Wrote fig1_su_energy.png')

# -----------------------------------------------------------------------------
# Figure 2 — magnetizations
# -----------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))

ax = axs[0]
ax.plot(su40['h'], np.abs(su40['mz']), 'o-',
        label=r'$\chi=40$', color='C0', alpha=0.8)
ax.plot(su60['h'], np.abs(su60['mz']), 's-',
        label=r'$\chi=60$', color='C2', mfc='white')
ax.axvline(h_c, ls='--', color='gray', alpha=0.4)
ax.text(h_c, 0.85, r'  true $h_c$=3.044', color='gray', fontsize=9)
ax.set_xlabel(r'$h / J$')
ax.set_ylabel(r'$|\langle \sigma^z \rangle|$')
ax.set_title(r'Order parameter — χ holds order LONGER past true $h_c$')
ax.legend(loc='upper right', fontsize=9, framealpha=0.95)
ax.set_ylim(-0.04, 1.0)

ax = axs[1]
ax.plot(su40['h'], su40['mx'], 'o-',
        label=r'$\chi=40$', color='C0', alpha=0.8)
ax.plot(su60['h'], su60['mx'], 's-',
        label=r'$\chi=60$', color='C2', mfc='white')
ax.axvline(h_c, ls='--', color='gray', alpha=0.4)
ax.set_xlabel(r'$h / J$')
ax.set_ylabel(r'$\langle \sigma^x \rangle$')
ax.set_title('Transverse magnetization')
ax.legend(loc='lower right', fontsize=9, framealpha=0.95)

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig2_su_magnetizations.png'), dpi=150, bbox_inches='tight')
print('Wrote fig2_su_magnetizations.png')

# -----------------------------------------------------------------------------
# Figure 3 — zoom on the critical region
# -----------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))

# zoom mask: h in [3.0, 4.6]
m40 = (su40['h'] >= 3.0) & (su40['h'] <= 4.6)
m60 = (su60['h'] >= 3.0) & (su60['h'] <= 4.6)

ax = axs[0]
ax.plot(su40['h'][m40], su40['E'][m40], 'o-',
        label=r'$\chi=40$', color='C0', markersize=8, alpha=0.8)
ax.plot(su60['h'][m60], su60['E'][m60], 's-',
        label=r'$\chi=60$', color='C2', mfc='white', markersize=8)
qmc_zoom = np.linspace(3.0, 4.6, 200)
ax.plot(qmc_zoom, qmc_interp(qmc_zoom), 'k:',
        label='QMC reference', linewidth=2)
ax.axvline(h_c, ls='--', color='gray', alpha=0.4)
ax.set_xlabel(r'$h / J$')
ax.set_ylabel(r'$E$ / site')
ax.set_title('Energy in transition + disordered region')
ax.legend(loc='upper right', fontsize=9, framealpha=0.95)

ax = axs[1]
ax.plot(su40['h'][m40], np.abs(su40['mz'][m40]), 'o-',
        label=r'$\chi=40$', color='C0', markersize=8, alpha=0.8)
ax.plot(su60['h'][m60], np.abs(su60['mz'][m60]), 's-',
        label=r'$\chi=60$', color='C2', mfc='white', markersize=8)
ax.axhline(0.1, ls=':', color='gray', alpha=0.5)
ax.text(4.45, 0.12, r'$|m_z|=0.1$', color='gray', fontsize=8, ha='right')
ax.axvline(h_c, ls='--', color='gray', alpha=0.4)
ax.text(h_c, 0.85, r'  true $h_c$', color='gray', fontsize=9)
ax.set_xlabel(r'$h / J$')
ax.set_ylabel(r'$|\langle \sigma^z \rangle|$')
ax.set_title(r'Order parameter — apparent $h_c$ shifts UP with χ')
ax.legend(loc='upper right', fontsize=9, framealpha=0.95)

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig3_su_chi_zoom.png'), dpi=150, bbox_inches='tight')
print('Wrote fig3_su_chi_zoom.png')

# -----------------------------------------------------------------------------
# Headline numbers
# -----------------------------------------------------------------------------
print('\n' + '=' * 72)
print(' HEADLINE NUMBERS')
print('=' * 72)

print('\nShared h-points: energy and error comparison')
shared_h = sorted(set(su40['h']).intersection(set(su60['h'])))
print(f"{'h':>6}  {'E(χ=40)':>11}  {'E(χ=60)':>11}  {'ΔE':>11}  "
      f"{'err40%':>7}  {'err60%':>7}")
print('-' * 66)
for h in shared_h:
    e40v = su40.loc[su40['h'] == h, 'E'].values[0]
    e60v = su60.loc[su60['h'] == h, 'E'].values[0]
    de = e60v - e40v
    er40 = 100 * (e40v - qmc_interp(h)) / abs(qmc_interp(h))
    er60 = 100 * (e60v - qmc_interp(h)) / abs(qmc_interp(h))
    print(f"{h:6.4f}  {e40v:+11.6f}  {e60v:+11.6f}  {de:+11.2e}  "
          f"{er40:+7.2f}  {er60:+7.2f}")

# Apparent h_c (where |mz| crosses 0.1)
def transition_h(h, mz):
    mz = np.abs(np.asarray(mz))
    h = np.asarray(h)
    for i in range(len(h) - 1):
        if mz[i] >= 0.1 > mz[i + 1]:
            return h[i] + (0.1 - mz[i]) * (h[i + 1] - h[i]) / (mz[i + 1] - mz[i])
    return None

t40 = transition_h(su40['h'].values, su40['mz'].values)
t60 = transition_h(su60['h'].values, su60['mz'].values)

print(f'\nApparent h_c (where |⟨σᶻ⟩| crosses 0.1):  true h_c = {h_c}')
if t40: print(f'   χ=40:  h ≈ {t40:.3f}  (overshoots true h_c by {t40-h_c:+.3f})')
if t60: print(f'   χ=60:  h ≈ {t60:.3f}  (overshoots true h_c by {t60-h_c:+.3f})')

print('\n' + '=' * 72)
print(' Figures saved.  Open the .png files to view.')
print('=' * 72)
