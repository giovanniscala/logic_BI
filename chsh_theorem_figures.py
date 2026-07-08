import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

#outdir = Path('/mnt/data/chsh_imperfect_measurements')
#outdir.mkdir(parents=True, exist_ok=True)
# Directory where THIS script is located
outdir = Path(__file__).parent

outdir.mkdir(parents=True, exist_ok=True)
# ============================================================
# Theorem 2: optimized CHSH with imperfect transverse visibility
# ============================================================

def smax(v, p):
    """Optimized asymptotic CHSH value for Theorem 2.

    v: transverse measurement visibility
    p: deletion/loss probability, eta = 1-p
    """
    eta = 1.0 - p
    transverse_sq = (v**4) * (eta**2)  # (v^2 eta)^2
    longitudinal_sq = (1.0 - 2.0 * eta * (1.0 - eta))**2
    return 2.0 * np.sqrt(transverse_sq + np.maximum(transverse_sq, longitudinal_sq))

# Grid
v_grid = np.linspace(0.6, 1.0, 401)
p_grid = np.linspace(0.0, 0.3, 401)
P, V = np.meshgrid(p_grid, v_grid)
S = smax(V, P)

# Threshold p_c(v) by bisection over p in [0, 0.3].
def threshold_pc(v, pmax=0.3):
    f0 = smax(v, 0.0) - 2.0
    f1 = smax(v, pmax) - 2.0
    if f0 <= 0:
        return 0.0
    if f1 > 0:
        return np.nan  # above the plotting domain
    lo, hi = 0.0, pmax
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if smax(v, mid) > 2.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

pc_vals = np.array([threshold_pc(v) for v in v_grid])
ideal_pc = 1.0 - 1.0 / np.sqrt(2.0)

def pc_roots_poly(v):
    """Polynomial check for the mixed-branch threshold.

    In p variables, the boundary solves
        (1 - 2p + 2p^2)^2 + v^4(1-p)^2 = 1.
    """
    coeff = [4.0, -8.0, 8.0 + v**4, -4.0 - 2.0 * v**4, v**4]
    roots = np.roots(coeff)
    real = np.real(roots[np.isclose(np.imag(roots), 0.0, atol=1e-9)])
    real = real[(real >= -1e-12) & (real <= 0.3000000001)]
    if len(real) == 0:
        return np.nan
    return float(np.min(real[real >= -1e-12]))

pc_poly = np.array([pc_roots_poly(v) for v in v_grid])

# Determine specific values
v_samples = np.array([0.7, 0.8, 0.9, 1.0, np.sqrt(2 / np.pi)])
pc_samples = {float(v): threshold_pc(float(v)) for v in v_samples}
S_at_p0 = {float(v): smax(float(v), 0.0) for v in v_samples}

# Figure (a): Phase diagram
plt.figure(figsize=(6.4, 4.8))
im = plt.pcolormesh(P, V, S, shading='auto')
cb = plt.colorbar(im)
cb.set_label(r'$S_{\max}(v,p)$')
cs = plt.contour(P, V, S, levels=[2.0], linewidths=1.8)
plt.clabel(cs, fmt={2.0: r'$S=2$'}, inline=True)
plt.xlabel(r'loss probability $p$')
plt.ylabel(r'measurement visibility $v$')
plt.title(r'Optimized CHSH value under imperfect transverse measurements')
plt.tight_layout()
phase_png = outdir / 'phase_diagram_Smax.png'
phase_pdf = outdir / 'phase_diagram_Smax.pdf'
plt.savefig(phase_png, dpi=300, bbox_inches='tight')
plt.savefig(phase_pdf, bbox_inches='tight')
plt.close()

# Figure (b): Threshold curve
plt.figure(figsize=(6.4, 4.8))
plt.plot(v_grid, pc_vals, label=r'optimized threshold $p_c(v)$')
plt.axhline(ideal_pc, linestyle='--', label=r'ideal threshold $1-1/\sqrt{2}$')
plt.axvline(np.sqrt(2 / np.pi), linestyle=':', label=r'$v=\sqrt{2/\pi}$')
plt.xlabel(r'measurement visibility $v$')
plt.ylabel(r'loss threshold $p_c$')
plt.title(r'Loss tolerance versus measurement visibility')
plt.ylim(0.0, 0.31)
plt.xlim(0.6, 1.0)
plt.legend()
plt.tight_layout()
thr_png = outdir / 'threshold_curve_pc_v.png'
thr_pdf = outdir / 'threshold_curve_pc_v.pdf'
plt.savefig(thr_png, dpi=300, bbox_inches='tight')
plt.savefig(thr_pdf, bbox_inches='tight')
plt.close()

# Figure (c): Cross sections
plt.figure(figsize=(6.4, 4.8))
for v in [0.7, 0.8, 0.9, 1.0]:
    plt.plot(p_grid, smax(v, p_grid), label=rf'$v={v:.1f}$')
plt.axhline(2.0, linestyle='--', label=r'local bound $S=2$')
plt.xlabel(r'loss probability $p$')
plt.ylabel(r'$S_{\max}(v,p)$')
plt.title(r'Cross-sections of optimized CHSH value')
plt.xlim(0.0, 0.3)
plt.legend()
plt.tight_layout()
cross_png = outdir / 'cross_sections_Smax.png'
cross_pdf = outdir / 'cross_sections_Smax.pdf'
plt.savefig(cross_png, dpi=300, bbox_inches='tight')
plt.savefig(cross_pdf, bbox_inches='tight')
plt.close()

# Save numerical threshold data
csv_path = outdir / 'threshold_curve_data.csv'
with csv_path.open('w') as f:
    f.write('v,p_c\n')
    for v, pc in zip(v_grid, pc_vals):
        f.write(f'{v:.8f},{pc:.10f}\n')

# Create a small summary text file
summary_path = outdir / 'summary_values.txt'
with summary_path.open('w') as f:
    f.write(f'Ideal threshold p_ideal = {ideal_pc:.10f}\n')
    for v, pc in pc_samples.items():
        f.write(f'v = {v:.10f}, p_c = {pc:.10f}, S(p=0) = {S_at_p0[v]:.10f}\n')

# ============================================================
# Theorem 1: sparse-deletion lower bound for square binomial PI family
# ============================================================
# Square family:
#     n = M^2, g = M, t = M-1, p = c/M.
# Main bound:
#     S_lb = sqrt(2) (1 - eps_q)^2 (1 + mu_*^2).
# For the theorem-certified vertical line, c = 1/4, i.e. p = 1/(4M).


def log_binom(n, k):
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def central_binom_ratio(M):
    """binom(M, floor(M/2)) / 2^M, computed stably."""
    k = M // 2
    return math.exp(log_binom(M, k) - M * math.log(2.0))


def mu_star_square(M):
    """mu_* for the square family t=M-1.

    mu_* = exp[-2t/(M-1)^2] [1 - 2 binom(M,floor(M/2))/2^M - 4 exp(-M/32)]_+
    with t = M-1.
    """
    M = int(M)
    t = M - 1
    prefactor = math.exp(-2.0 * t / ((M - 1) ** 2))
    bracket = 1.0 - 2.0 * central_binom_ratio(M) - 4.0 * math.exp(-M / 32.0)
    return prefactor * max(0.0, bracket)


def eps_q_square(M, c):
    """Bernstein-type tail term eps_q for square family at p=c/M."""
    M = float(M)
    c = float(c)
    n = M**2
    t = M - 1.0
    p = c / M
    np_mean = n * p  # = c M
    gap = t - np_mean
    if gap <= 0.0:
        return 1.0
    denom = 2.0 * np_mean * (1.0 - p) + (2.0 / 3.0) * gap
    return math.exp(-(gap**2) / denom)


def theorem1_s_lower_bound(M, c=0.25):
    mu = mu_star_square(int(round(M)))
    eps = eps_q_square(M, c)
    return math.sqrt(2.0) * (1.0 - eps) ** 2 * (1.0 + mu**2)


def theorem1_row(M, c=0.25):
    M = int(M)
    n = M * M
    g = M
    t = M - 1
    p = c / M
    return {
        'M': M,
        'n': n,
        'g': g,
        't': t,
        'p_1_over_4M': 1.0 / (4.0 * M),
        'np_at_threshold': n * (1.0 / (4.0 * M)),
        'mu_star': mu_star_square(M),
        'eps_q': eps_q_square(M, 0.25),
        'S_lower_bound': theorem1_s_lower_bound(M, 0.25),
    }

# Representative values used in the internal report/PRL supplement.
theorem1_M_samples = [51, 75, 101, 151, 201, 301, 501]
theorem1_csv = outdir / 'theorem1_square_family_values.csv'
with theorem1_csv.open('w') as f:
    f.write('M,n,g,t,p_1_over_4M,np_at_threshold,mu_star,eps_q,S_lower_bound\n')
    for M in theorem1_M_samples:
        row = theorem1_row(M)
        f.write(
            f"{row['M']},{row['n']},{row['g']},{row['t']},"
            f"{row['p_1_over_4M']},{row['np_at_threshold']},"
            f"{row['mu_star']},{row['eps_q']},{row['S_lower_bound']}\n"
        )

# Smooth odd-M curve for square-family plots.
M_grid = np.arange(51, 502, 2)
mu_grid = np.array([mu_star_square(int(M)) for M in M_grid])
eps_grid = np.array([eps_q_square(float(M), 0.25) for M in M_grid])
S_lb_grid = np.array([theorem1_s_lower_bound(float(M), 0.25) for M in M_grid])
neg_log_eps = -np.log10(eps_grid)
M_cert = 101

# Figure (d): Theorem 1 square-family lower bound
plt.figure(figsize=(6.4, 4.8))
plt.plot(M_grid, S_lb_grid, label=r'$S_{\rm lb}(M,p=1/(4M))$')
plt.axhline(2.0, linestyle='--', label=r'local bound $S=2$')
plt.axvline(M_cert, linestyle=':', label=rf'$M={M_cert}$')
plt.xlabel(r'$M$ in square family $(n=M^2,\; g=M)$')
plt.ylabel(r'certified lower bound')
plt.title(r'Theorem 1 square-family lower bound')
plt.legend()
plt.tight_layout()
t1_bound_png = outdir / 'theorem1_square_bound.png'
t1_bound_pdf = outdir / 'theorem1_square_bound.pdf'
plt.savefig(t1_bound_png, dpi=300, bbox_inches='tight')
plt.savefig(t1_bound_pdf, bbox_inches='tight')
plt.close()

# Figure (e): Components entering Theorem 1 lower bound
fig, ax1 = plt.subplots(figsize=(6.4, 4.8))
line_mu, = ax1.plot(M_grid, mu_grid, label=r'$\mu_*$')
line_cert = ax1.axvline(M_cert, linestyle=':', label=rf'$M={M_cert}$')
ax1.set_xlabel(r'$M$')
ax1.set_ylabel(r'$\mu_*$')
ax1.set_ylim(-0.02, 1.02)

ax2 = ax1.twinx()
line_eps, = ax2.plot(M_grid, neg_log_eps, linestyle='--', label=r'$-\log_{10}\varepsilon_q$')
ax2.set_ylabel(r'$-\log_{10}\varepsilon_q$')

lines = [line_mu, line_cert, line_eps]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='lower right')
plt.title(r'Components entering Theorem 1 lower bound')
fig.tight_layout()
t1_components_png = outdir / 'theorem1_components.png'
t1_components_pdf = outdir / 'theorem1_components.pdf'
plt.savefig(t1_components_png, dpi=300, bbox_inches='tight')
plt.savefig(t1_components_pdf, bbox_inches='tight')
plt.close(fig)

# Figure (f): Sparse-deletion certification region for square family
# Axes: c = p M, M = sqrt(n). The theorem corollary uses c = 1/4.
M_phase = np.arange(50, 301, 1)
c_grid = np.linspace(0.05, 0.95, 451)
C, MM = np.meshgrid(c_grid, M_phase)

# Vectorize the scalar functions for the grid. This keeps the code dependency-free.
mu_vec = np.vectorize(lambda x: mu_star_square(int(round(x))))
eps_vec = np.vectorize(lambda x, y: eps_q_square(float(x), float(y)))
Mu = mu_vec(MM)
Eps = eps_vec(MM, C)
S_phase = np.sqrt(2.0) * (1.0 - Eps) ** 2 * (1.0 + Mu**2)

plt.figure(figsize=(6.4, 4.8))
im = plt.pcolormesh(C, MM, S_phase, shading='auto')
cb = plt.colorbar(im)
cb.set_label(r'$S_{\rm lb}(M,c),\; p=c/M$')
cs = plt.contour(C, MM, S_phase, levels=[2.0], colors='black', linewidths=1.8)
plt.clabel(cs, fmt={2.0: r'$S=2$'}, inline=True)
plt.axvline(0.25, linestyle='--', linewidth=1.8, color='white', label=r'$c=1/4$')
plt.xlabel(r'scaled deletion probability $c=pM$')
plt.ylabel(r'$M$ $(n=M^2)$')
plt.title(r'Square-family sparse-deletion certification region')
plt.legend(loc='lower right')
plt.tight_layout()
t1_phase_png = outdir / 'theorem1_sparse_phase.png'
t1_phase_pdf = outdir / 'theorem1_sparse_phase.pdf'
plt.savefig(t1_phase_png, dpi=300, bbox_inches='tight')
plt.savefig(t1_phase_pdf, bbox_inches='tight')
plt.close()

# Print all generated files.
generated_paths = [
    phase_png, phase_pdf,
    thr_png, thr_pdf,
    cross_png, cross_pdf,
    csv_path, summary_path,
    t1_bound_png, t1_bound_pdf,
    t1_components_png, t1_components_pdf,
    t1_phase_png, t1_phase_pdf,
    theorem1_csv,
]

print('Generated files:')
for path in generated_paths:
    print(path)

print('\nTheorem 2 sample thresholds:')
for v in [0.7, 0.8, np.sqrt(2 / np.pi), 0.9, 1.0]:
    print(f'v={v:.6f}, p_c={threshold_pc(v):.6f}, S(v,0)={smax(v, 0):.6f}')

print('\nTheorem 1 square-family samples:')
for M in theorem1_M_samples:
    row = theorem1_row(M)
    print(
        f"M={M:3d}, n={row['n']:6d}, p=1/(4M)={row['p_1_over_4M']:.8f}, "
        f"mu_*={row['mu_star']:.6f}, eps_q={row['eps_q']:.3e}, "
        f"S_lb={row['S_lower_bound']:.6f}"
    )
