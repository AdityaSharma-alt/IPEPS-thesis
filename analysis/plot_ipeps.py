"""
plot_ipeps.py — comparison figures for the iPEPS TFIM benchmark.

Inputs (analysis/data/):
    ipeps_tfim_D12_chi40.csv   # SU sweep at D=12, χ=40 (22 h-points)
    fu_sweep_D4_chi20.csv      # FU sweep at D=4, χ=20 (22 h-points, with paired SU baseline)

Outputs (written to ../figures/ch7_ipeps_square_lattice/):
    fig1_energy.png            # energy vs h, with QMC reference + error panel
    fig2_magnetizations.png    # |⟨σᶻ⟩| and ⟨σˣ⟩ vs h
    fig3_fu_benefit.png        # ΔE = E_SU − E_FU at fixed D=4, and S_b comparison

Run:
    pip install pandas matplotlib scipy   # if needed
    python plot_ipeps.py
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
# Load data
# -----------------------------------------------------------------------------
su = pd.read_csv(os.path.join(DATA_DIR, 'ipeps_tfim_D12_chi40.csv'))
fu = pd.read_csv(os.path.join(DATA_DIR, 'fu_sweep_D4_chi20.csv'))

# QMC reference from the SU job's stdout (printed as part of benchmark summary)
qmc_h = np.array([2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9,
                  3.0, 3.0444, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8,
                  3.9, 4.0])
qmc_E = np.array([-2.685, -2.730, -2.775, -2.820, -2.865, -2.910,
                  -2.983, -3.056, -3.129, -3.202, -3.275, -3.310396,
                  -3.368333, -3.4725, -3.576667, -3.680833, -3.785,
                  -3.888, -3.991, -4.094, -4.197, -4.300])
qmc_interp = interp1d(qmc_h, qmc_E, kind='cubic', fill_value='extrapolate')

h_c = 3.0444  # exact critical point

# Common styling
plt.rcParams.update({'font.size': 10, 'axes.grid': True,
                     'grid.alpha': 0.3, 'lines.markersize': 5})

# -----------------------------------------------------------------------------
# Figure 1 — energy
# -----------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))

ax = axs[0]
ax.plot(su['h'], su['E'], 'o-', label=r'SU $D=12$, $\chi=40$', color='C0')
ax.plot(fu['h'], fu['E_SU'], 's--', label=r'SU $D=4$, $\chi=20$', color='C3', alpha=0.55)
ax.plot(fu['h'], fu['E_FU'], '^-', label=r'FU $D=4$, $\chi=20$', color='C2')
ax.plot(qmc_h, qmc_E, 'k:', label='QMC reference', linewidth=2)
ax.axvline(h_c, ls='--', color='gray', alpha=0.4)
ax.text(h_c, ax.get_ylim()[1] * 0.04 + ax.get_ylim()[0] * 0.96, r' $h_c$',
        ha='left', color='gray', va='bottom')
ax.set_xlabel(r'$h / J$')
ax.set_ylabel(r'$E$ / site')
ax.set_title('Ground-state energy')
ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

ax = axs[1]
su_err = 100 * (su['E'] - qmc_interp(su['h'])) / np.abs(qmc_interp(su['h']))
fu_err = 100 * (fu['E_FU'] - qmc_interp(fu['h'])) / np.abs(qmc_interp(fu['h']))
su4_err = 100 * (fu['E_SU'] - qmc_interp(fu['h'])) / np.abs(qmc_interp(fu['h']))
ax.plot(su['h'], su_err, 'o-', label=r'SU $D=12$', color='C0')
ax.plot(fu['h'], su4_err, 's--', label=r'SU $D=4$', color='C3', alpha=0.55)
ax.plot(fu['h'], fu_err, '^-', label=r'FU $D=4$', color='C2')
ax.axhline(0, color='k', linewidth=0.5)
ax.axvline(h_c, ls='--', color='gray', alpha=0.4)
ax.set_xlabel(r'$h / J$')
ax.set_ylabel(r'$(E - E_{\rm QMC})/|E_{\rm QMC}|$  (%)')
ax.set_title('Variational error vs QMC')
ax.legend(loc='best', fontsize=9, framealpha=0.9)

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig1_energy.png'), dpi=150, bbox_inches='tight')
print('Wrote fig1_energy.png')

# -----------------------------------------------------------------------------
# Figure 2 — magnetizations
# -----------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))

ax = axs[0]
ax.plot(su['h'], np.abs(su['mz']), 'o-', label=r'SU $D=12$, $\chi=40$', color='C0')
ax.plot(fu['h'], np.abs(fu['mz_SU']), 's--', label=r'SU $D=4$, $\chi=20$', color='C3', alpha=0.55)
ax.plot(fu['h'], np.abs(fu['mz_FU']), '^-', label=r'FU $D=4$, $\chi=20$', color='C2')
ax.axvline(h_c, ls='--', color='gray', alpha=0.4)
ax.text(h_c, 0.92, r' $h_c$', ha='left', color='gray')
ax.set_xlabel(r'$h / J$')
ax.set_ylabel(r'$|\langle \sigma^z \rangle|$')
ax.set_title('Order parameter (Z$_2$ symmetry-broken)')
ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

ax = axs[1]
ax.plot(su['h'], su['mx'], 'o-', label=r'SU $D=12$, $\chi=40$', color='C0')
ax.plot(fu['h'], fu['mx_SU'], 's--', label=r'SU $D=4$, $\chi=20$', color='C3', alpha=0.55)
ax.plot(fu['h'], fu['mx_FU'], '^-', label=r'FU $D=4$, $\chi=20$', color='C2')
ax.axvline(h_c, ls='--', color='gray', alpha=0.4)
ax.set_xlabel(r'$h / J$')
ax.set_ylabel(r'$\langle \sigma^x \rangle$')
ax.set_title('Transverse magnetization')
ax.legend(loc='lower right', fontsize=9, framealpha=0.9)

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig2_magnetizations.png'), dpi=150, bbox_inches='tight')
print('Wrote fig2_magnetizations.png')

# -----------------------------------------------------------------------------
# Figure 3 — FU's variational benefit and entanglement
# -----------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))

ax = axs[0]
# -dE so that "FU is better" appears as positive
ax.plot(fu['h'], -fu['dE'], 'o-', color='C2')
ax.axvline(h_c, ls='--', color='gray', alpha=0.4)
ax.axhline(0, color='k', linewidth=0.5)
ax.set_xlabel(r'$h / J$')
ax.set_ylabel(r'$E_{\rm SU} - E_{\rm FU}$ per site')
ax.set_title(r'FU variational improvement at fixed $D=4$')

ax = axs[1]
ax.plot(fu['h'], fu['Sb_SU'], 's--', label=r'SU $D=4$', color='C3', alpha=0.7)
ax.plot(fu['h'], fu['Sb_FU'], '^-', label=r'FU $D=4$', color='C2')
ax.axvline(h_c, ls='--', color='gray', alpha=0.4)
ax.set_xlabel(r'$h / J$')
ax.set_ylabel(r'$S_b$  (boundary entropy)')
ax.set_title(r'Entanglement preservation: SU vs FU at $D=4$')
ax.legend(loc='best', fontsize=9, framealpha=0.9)

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig3_fu_benefit.png'), dpi=150, bbox_inches='tight')
print('Wrote fig3_fu_benefit.png')

# -----------------------------------------------------------------------------
# Headline numbers (printed to terminal)
# -----------------------------------------------------------------------------
print('\n' + '=' * 72)
print(' HEADLINE NUMBERS')
print('=' * 72)

# 1) where FU improves most over SU at D=4
i_max = fu['dE'].abs().idxmax()
print(f'\nLargest |ΔE_FU − ΔE_SU| at D=4:')
print(f'   h = {fu["h"][i_max]:.4f}   ΔE = {fu["dE"][i_max]:+.4e}')

# 2) FU at D=4 vs SU at D=12 at h_c
hc_idx_su = (su['h'] - h_c).abs().idxmin()
hc_idx_fu = (fu['h'] - h_c).abs().idxmin()
print(f'\nAt h_c ≈ {h_c}:')
print(f'   SU D=12, χ=40:  E = {su["E"][hc_idx_su]:+.6f}')
print(f'   FU D= 4, χ=20:  E = {fu["E_FU"][hc_idx_fu]:+.6f}')
print(f'   QMC reference:  E = {qmc_interp(h_c):+.6f}')
print(f'   D=4 FU vs D=12 SU: {fu["E_FU"][hc_idx_fu] - su["E"][hc_idx_su]:+.6f}  '
      f'(negative = FU(D=4) BEATS SU(D=12))')

# 3) Where the transitions happen (where |mz| first drops below 0.1)
def transition_h(h, mz):
    mz = np.abs(mz)
    for i in range(len(h) - 1):
        if mz[i] >= 0.1 > mz[i + 1]:
            return h[i] + (0.1 - mz[i]) * (h[i + 1] - h[i]) / (mz[i + 1] - mz[i])
    return None

t_su12 = transition_h(su['h'].values, su['mz'].values)
t_su4  = transition_h(fu['h'].values, fu['mz_SU'].values)
t_fu4  = transition_h(fu['h'].values, fu['mz_FU'].values)
print(f'\nEffective h_c (where |⟨σᶻ⟩| crosses 0.1):  true h_c = {h_c}')
print(f'   SU D=12, χ=40: h ≈ {t_su12:.3f}  (overshoots by {t_su12 - h_c:+.3f})')
print(f'   SU D= 4, χ=20: h ≈ {t_su4:.3f}  (overshoots by {t_su4 - h_c:+.3f})')
print(f'   FU D= 4, χ=20: h ≈ {t_fu4:.3f}  (overshoots by {t_fu4 - h_c:+.3f})')

print('\n' + '=' * 72)
print(' All figures saved.  Open the .png files to view.')
print('=' * 72)
