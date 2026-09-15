#!/usr/bin/env python3
"""Reproducible numerical material for *Recovery-Free CHSH Nonlocality with
Particle Loss*.

The script follows the notation of the current manuscript.  It evaluates the
visibility-dependent one-excitation protocol, the vacuum-anchored
``|0>``--``|N>`` pure-loss family, the exact golden-ratio balance, and the
certified lower bound for the square sparse-binomial PI family.  It also
contains the finite-Fock search used as a numerical benchmark: anchored,
single-coherence, band-limited, and unrestricted-observable strategies.

Run ``python logic_chsh_computations.py --help`` for options.  ``figures``
(the default) is quick and creates all analytical figures and data.  The
finite-Fock search is deliberately opt-in because the manuscript's ``serious``
configuration is computationally demanding:

    python logic_chsh_computations.py --mode search --profile serious
    python logic_chsh_computations.py --mode all --profile serious

All labels use Matplotlib mathtext with Computer-Modern-like fonts; no local
LaTeX installation is required.

MORE:
For running the finite-Fock search, the output directory must exist and be writable.  
cd /workspace/scratch/46843dfc7af6
python3 logic_chsh_computations.py --mode search --profile serious --output-dir logic_chsh_outputs

# Run everything: figures plus serious search
python3 logic_chsh_computations.py --mode all --profile serious --output-dir logic_chsh_outputs

# Quick sanity search, much faster
python3 logic_chsh_computations.py --mode search --profile quick --output-dir logic_chsh_outputs

# Verify formulas only
python3 logic_chsh_computations.py --mode verify
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ETA_G = (math.sqrt(5.0) - 1.0) / 2.0
ETA_ONE_EXCITATION = 1.0 / math.sqrt(2.0)
ETA_HALF = 0.5
RNG_SEED = 20_260_716
VIOLATION_TOLERANCE = 1.0e-7
CONVERGENCE_TOLERANCE = 1.0e-12

# Note: ETA_G = 1/GOLDEN_RATIO, but it is kept as an independent literal above
# since it plays a physical role (the golden-ratio survival threshold), while
# GOLDEN_RATIO below is purely a plotting convention.
GOLDEN_RATIO = (1.0 + math.sqrt(5.0)) / 2.0


def _golden_figsize(width: float) -> tuple[float, float]:
    """Return a Matplotlib ``figsize`` with a golden-ratio aspect (width:height = phi:1).

    Every figure in this module is created by passing its intended width (in
    inches, chosen to fit a single/double manuscript column) through this
    helper instead of hand-picking a height, so all figures share one
    consistent, deliberate aspect ratio rather than ad hoc values.
    """
    return (width, width / GOLDEN_RATIO)

SURVIVAL_GRID = (
    ETA_ONE_EXCITATION,
    ETA_G,
    0.60,
    0.58,
    0.56,
    0.54,
    0.52,
    0.505,
)
SINGLE_COHERENCE_CUTOFFS = (2, 3, 4, 5, 6)
BAND_LIMITED_CUTOFFS = (3, 4, 5, 6, 8)
BANDWIDTHS = (1, 2, 3, 4)
UNRESTRICTED_CUTOFFS = (3, 4, 5, 6, 8)

PUBLICATION_RC: dict[str, Any] = {
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 9.0,
    "axes.labelsize": 10.0,
    "axes.titlesize": 10.0,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.0,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.25,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


@dataclass(frozen=True)
class SearchProfile:
    """Fixed finite-Fock search settings documented in the manuscript."""

    random_single_coherence_codes: int
    random_band_limited_codes: int
    random_unrestricted_codes: int
    band_limited_starts: int
    band_limited_iterations: int
    unrestricted_starts: int
    unrestricted_iterations: int


SEARCH_PROFILES: dict[str, SearchProfile] = {
    "quick": SearchProfile(32, 8, 16, 4, 50, 8, 80),
    # This is the configuration in Table S of the current manuscript.
    "serious": SearchProfile(256, 64, 128, 12, 150, 24, 250),
    # Retained from the latest search implementation for longer investigations.
    "stress": SearchProfile(2048, 512, 1024, 32, 400, 64, 600),
}


# ---------------------------------------------------------------------------
# Shared validation and numerical utilities
# ---------------------------------------------------------------------------

ScalarOrArray = float | np.ndarray


def _require_unit_interval(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1], got {value!r}.")


def _require_positive_integer(value: int, name: str, minimum: int = 1) -> None:
    if not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}, got {value!r}.")


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str, *, dpi: int = 300) -> list[Path]:
    """Save one figure in PNG and vector-PDF form and close it."""
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    fig.savefig(paths[0], dpi=dpi, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(paths[1], bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    return paths


def _write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)
    return path


# ---------------------------------------------------------------------------
# One-excitation protocol with transverse visibility
# ---------------------------------------------------------------------------

def one_excitation_visible_chsh(visibility: ScalarOrArray, eta: ScalarOrArray) -> ScalarOrArray:
    """Optimized CHSH value from Eq. (Smax-vp) of the Supplement.

    The sharp longitudinal correlation is ``c=1-2 eta(1-eta)``.  Each local
    transverse measurement has visibility ``v``, so the transverse singular
    values have magnitude ``v**2 eta``.
    """
    v = np.asarray(visibility, dtype=float)
    survival = np.asarray(eta, dtype=float)
    transverse_squared = v**4 * survival**2
    longitudinal_squared = (1.0 - 2.0 * survival * (1.0 - survival)) ** 2
    return 2.0 * np.sqrt(transverse_squared + np.maximum(transverse_squared, longitudinal_squared))


def critical_loss_one_excitation(visibility: float, maximum_loss: float = 0.3) -> float:
    """Return ``p_c`` where the visible one-excitation value reaches two."""
    _require_unit_interval(visibility, "visibility")
    _require_unit_interval(maximum_loss, "maximum_loss")
    if one_excitation_visible_chsh(visibility, 1.0) <= 2.0:
        return 0.0
    if one_excitation_visible_chsh(visibility, 1.0 - maximum_loss) > 2.0:
        return math.nan
    low, high = 0.0, maximum_loss
    for _ in range(80):
        midpoint = 0.5 * (low + high)
        if one_excitation_visible_chsh(visibility, 1.0 - midpoint) > 2.0:
            low = midpoint
        else:
            high = midpoint
    return 0.5 * (low + high)


def mixed_branch_loss_root(visibility: float) -> float:
    """Polynomial cross-check for the mixed-branch one-excitation threshold."""
    _require_unit_interval(visibility, "visibility")
    v4 = visibility**4
    coefficients = (4.0, -8.0, 8.0 + v4, -4.0 - 2.0 * v4, v4)
    roots = np.roots(coefficients)
    real_roots = np.real(roots[np.isclose(np.imag(roots), 0.0, atol=1.0e-9)])
    admissible = real_roots[(real_roots >= -1.0e-12) & (real_roots <= 0.3000000001)]
    return float(np.min(admissible)) if admissible.size else math.nan


# ---------------------------------------------------------------------------
# Vacuum-anchored finite-Fock protocol
# ---------------------------------------------------------------------------

def anchored_complete_erasure_probability(eta: ScalarOrArray, excitation_number: int) -> ScalarOrArray:
    """Return ``q_N=(1-eta)^N`` for the vacuum-anchored family."""
    _require_positive_integer(excitation_number, "excitation_number")
    return (1.0 - np.asarray(eta, dtype=float)) ** excitation_number


def anchored_longitudinal_correlation(eta: ScalarOrArray, excitation_number: int) -> ScalarOrArray:
    """Return ``c_N=1-2q_N(1-q_N)``."""
    q_n = anchored_complete_erasure_probability(eta, excitation_number)
    return 1.0 - 2.0 * q_n * (1.0 - q_n)


def anchored_chsh(eta: ScalarOrArray, excitation_number: int, visibility: float = 1.0) -> ScalarOrArray:
    """Return ``S_{N,v}=2 sqrt(c_N^2+v^4 eta^(2N))``."""
    _require_positive_integer(excitation_number, "excitation_number")
    _require_unit_interval(visibility, "visibility")
    survival = np.asarray(eta, dtype=float)
    c_n = anchored_longitudinal_correlation(survival, excitation_number)
    return 2.0 * np.sqrt(c_n**2 + visibility**4 * survival ** (2 * excitation_number))


def ideal_one_excitation_chsh(eta: ScalarOrArray) -> ScalarOrArray:
    """Asymptotic direct one-excitation value ``S=2 sqrt(2) eta``."""
    return 2.0 * math.sqrt(2.0) * np.asarray(eta, dtype=float)


def critical_survival_anchored(
    excitation_number: int,
    visibility: float,
    eta_minimum: float = 0.5,
    eta_maximum: float = 1.0,
    grid_size: int = 4001,
) -> float:
    """First survival value above which the selected anchored curve violates."""
    _require_positive_integer(excitation_number, "excitation_number")
    _require_unit_interval(visibility, "visibility")
    _require_unit_interval(eta_minimum, "eta_minimum")
    _require_unit_interval(eta_maximum, "eta_maximum")
    if eta_minimum >= eta_maximum or grid_size < 2:
        raise ValueError("Require eta_minimum < eta_maximum and grid_size >= 2.")
    eta_grid = np.linspace(eta_minimum, eta_maximum, grid_size)
    excess = anchored_chsh(eta_grid, excitation_number, visibility) - 2.0
    indices = np.flatnonzero(excess > 1.0e-13)
    if not indices.size:
        return math.nan
    first = int(indices[0])
    if first == 0:
        return eta_minimum
    low, high = float(eta_grid[first - 1]), float(eta_grid[first])
    for _ in range(90):
        midpoint = 0.5 * (low + high)
        if anchored_chsh(midpoint, excitation_number, visibility) > 2.0:
            high = midpoint
        else:
            low = midpoint
    return 0.5 * (low + high)


# ---------------------------------------------------------------------------
# Exact golden-ratio balance
# ---------------------------------------------------------------------------

def anchored_log_components(eta: ScalarOrArray, excitation_number: ScalarOrArray) -> tuple[np.ndarray, np.ndarray]:
    """Return stable natural logarithms of ``G_N`` and ``D_N``.

    ``G_N=eta^(2N)`` is the squared coherent contribution and
    ``D_N=1-c_N^2`` is the longitudinal deficit.  The decomposition avoids
    cancellation when both quantities are exponentially small.
    """
    survival = np.asarray(eta, dtype=float)
    n_value = np.asarray(excitation_number, dtype=float)
    if np.any((survival <= 0.0) | (survival >= 1.0)):
        raise ValueError("The logarithmic balance requires 0 < eta < 1.")
    if np.any(n_value < 1.0):
        raise ValueError("excitation_number must be positive.")
    log_q = n_value * np.log1p(-survival)
    q_n = np.exp(log_q)
    log_gain = 2.0 * n_value * np.log(survival)
    log_deficit = math.log(4.0) + log_q + np.log1p(-q_n) + np.log1p(-q_n + q_n**2)
    return log_gain, log_deficit


def anchored_log10_ratio(eta: ScalarOrArray, excitation_number: ScalarOrArray) -> np.ndarray:
    """Return ``Lambda_N=log10(G_N/D_N)`` exactly within floating precision."""
    log_gain, log_deficit = anchored_log_components(eta, excitation_number)
    return (log_gain - log_deficit) / math.log(10.0)


def anchored_log10_ratio_asymptotic(eta: float, excitation_numbers: np.ndarray) -> np.ndarray:
    """Return the large-``N`` form of ``Lambda_N``."""
    _require_unit_interval(eta, "eta")
    if eta in (0.0, 1.0):
        raise ValueError("The asymptotic logarithm requires 0 < eta < 1.")
    return excitation_numbers * math.log10(eta**2 / (1.0 - eta)) - math.log10(4.0)


def first_violating_excitation_number(eta: float, maximum_n: int = 100_000) -> Optional[int]:
    """Return the first finite ``N`` with ``S_N>2``, or ``None`` if absent."""
    _require_unit_interval(eta, "eta")
    _require_positive_integer(maximum_n, "maximum_n")
    if eta <= ETA_G:
        return None
    excitation_numbers = np.arange(1, maximum_n + 1)
    violating = np.flatnonzero(anchored_log10_ratio(eta, excitation_numbers) > 0.0)
    return int(violating[0] + 1) if violating.size else None


def optimum_anchored_margin(eta: float, maximum_n: int = 100_000) -> tuple[Optional[int], Optional[float], Optional[float]]:
    """Return the best finite ``N``, ``S_N-2``, and its base-10 logarithm."""
    _require_unit_interval(eta, "eta")
    _require_positive_integer(maximum_n, "maximum_n")
    if eta <= ETA_G:
        return None, None, None
    ratio_base = eta**2 / (1.0 - eta)
    continuous_guess = math.log(4.0 * math.log(1.0 - eta) / math.log(eta**2)) / math.log(ratio_base)
    upper = min(maximum_n, max(50, int(math.ceil(continuous_guess + 25.0))))
    excitation_numbers = np.arange(1, upper + 1)
    log_gain, log_deficit = anchored_log_components(eta, excitation_numbers)
    valid = log_gain > log_deficit
    if not np.any(valid):
        return None, None, None
    log_delta = np.full(excitation_numbers.shape, -np.inf)
    gap = log_deficit[valid] - log_gain[valid]
    log_delta[valid] = log_gain[valid] + np.log1p(-np.exp(gap))
    index = int(np.argmax(log_delta))
    delta = math.exp(float(log_delta[index])) if log_delta[index] > -745.0 else 0.0
    log_margin = float(log_delta[index] + math.log(2.0 / (math.sqrt(1.0 + delta) + 1.0)))
    margin = math.exp(log_margin) if log_margin > -745.0 else 0.0
    return int(excitation_numbers[index]), margin, log_margin / math.log(10.0)


# ---------------------------------------------------------------------------
# Certified square sparse-binomial PI bound
# ---------------------------------------------------------------------------

def _log_binomial(n: int, k: int) -> float:
    if not 0 <= k <= n:
        return -math.inf
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def central_binomial_ratio(m_order: int) -> float:
    """Return ``binom(M,floor(M/2))/2^M`` using log-gamma arithmetic."""
    _require_positive_integer(m_order, "m_order", minimum=1)
    return math.exp(_log_binomial(m_order, m_order // 2) - m_order * math.log(2.0))


def square_sparse_mu_star(m_order: int) -> float:
    """Uniform coherence bound ``mu_*`` for ``g=M``, ``t=M-1``, ``n=M^2``."""
    _require_positive_integer(m_order, "m_order", minimum=2)
    t_value = m_order - 1
    prefactor = math.exp(-2.0 * t_value / (m_order - 1) ** 2)
    bracket = 1.0 - 2.0 * central_binomial_ratio(m_order) - 4.0 * math.exp(-m_order / 32.0)
    return prefactor * max(0.0, bracket)


def square_sparse_tail_epsilon(m_order: float, scaled_deletion: float) -> float:
    """Bernstein upper-tail term for ``p=c/M`` in the square family."""
    if m_order <= 1.0:
        raise ValueError("m_order must exceed 1.")
    if scaled_deletion < 0.0:
        raise ValueError("scaled_deletion c must be nonnegative.")
    particle_number = m_order**2
    t_value = m_order - 1.0
    deletion_probability = scaled_deletion / m_order
    if deletion_probability > 1.0:
        raise ValueError("p=c/M must not exceed one.")
    mean = particle_number * deletion_probability
    gap = t_value - mean
    if gap <= 0.0:
        return 1.0
    denominator = 2.0 * mean * (1.0 - deletion_probability) + (2.0 / 3.0) * gap
    return math.exp(-(gap**2) / denominator)


def square_sparse_chsh_lower_bound(m_order: float, scaled_deletion: float = 0.25) -> float:
    """Certified lower bound ``sqrt(2)(1-epsilon_q)^2(1+mu_*^2)``."""
    m_integer = int(round(m_order))
    mu_star = square_sparse_mu_star(m_integer)
    epsilon_q = square_sparse_tail_epsilon(float(m_order), scaled_deletion)
    return math.sqrt(2.0) * (1.0 - epsilon_q) ** 2 * (1.0 + mu_star**2)


def square_sparse_row(m_order: int, scaled_deletion: float = 0.25) -> dict[str, float | int]:
    """Return a self-describing row for the square sparse-binomial family."""
    _require_positive_integer(m_order, "m_order", minimum=2)
    return {
        "M": m_order,
        "n": m_order**2,
        "g": m_order,
        "t": m_order - 1,
        "c": scaled_deletion,
        "p": scaled_deletion / m_order,
        "mu_star": square_sparse_mu_star(m_order),
        "epsilon_q": square_sparse_tail_epsilon(m_order, scaled_deletion),
        "S_lower_bound": square_sparse_chsh_lower_bound(m_order, scaled_deletion),
    }


# ---------------------------------------------------------------------------
# Analytical figures and data
# ---------------------------------------------------------------------------

def make_visibility_figures(output_dir: Path) -> list[Path]:
    """Create the one-excitation visibility phase map, curve, and sections."""
    paths: list[Path] = []
    visibility_grid = np.linspace(0.6, 1.0, 401)
    loss_grid = np.linspace(0.0, 0.3, 401)
    loss_mesh, visibility_mesh = np.meshgrid(loss_grid, visibility_grid)
    values = one_excitation_visible_chsh(visibility_mesh, 1.0 - loss_mesh)
    critical_loss = np.array([critical_loss_one_excitation(float(v)) for v in visibility_grid])
    polynomial_loss = np.array([mixed_branch_loss_root(float(v)) for v in visibility_grid])

    with plt.rc_context(PUBLICATION_RC):
        fig, axis = plt.subplots(figsize=_golden_figsize(5.0))
        image = axis.pcolormesh(loss_mesh, visibility_mesh, values, shading="auto")
        colorbar = fig.colorbar(image, ax=axis, pad=0.02)
        colorbar.set_label(r"$S_{\max}(v,\eta)$", labelpad=2)
        contour = axis.contour(loss_mesh, visibility_mesh, values, levels=[2.0], colors="black", linewidths=1.1)
        axis.clabel(contour, fmt={2.0: r"$S=2$"}, inline=True, fontsize=8)
        axis.set(xlabel=r"deletion probability $p=1-\eta$", ylabel=r"local transverse visibility $v$")
        paths += _save_figure(fig, output_dir, "phase_diagram_Smax")

        fig, axis = plt.subplots(figsize=_golden_figsize(4.85))
        axis.plot(visibility_grid, critical_loss, label=r"numerical $p_c(v)$")
        axis.plot(visibility_grid, polynomial_loss, "--", label="mixed-branch polynomial check")
        axis.axhline(1.0 - ETA_ONE_EXCITATION, color="0.2", linestyle=":", label=r"$1-1/\sqrt{2}$")
        axis.axvline(math.sqrt(2.0 / math.pi), color="0.35", linestyle="-.", label=r"$\sqrt{2/\pi}$")
        axis.set(xlim=(0.6, 1.0), ylim=(0.0, 0.31), xlabel=r"local transverse visibility $v$", ylabel=r"critical deletion probability $p_c$")
        axis.legend(frameon=False, loc="upper left")
        paths += _save_figure(fig, output_dir, "threshold_curve_pc_v")

        fig, axis = plt.subplots(figsize=_golden_figsize(4.85))
        for visibility in (0.7, 0.8, 0.9, 1.0):
            axis.plot(loss_grid, one_excitation_visible_chsh(visibility, 1.0 - loss_grid), label=rf"$v={visibility:.1f}$")
        axis.axhline(2.0, color="0.2", linestyle="--", label=r"local bound $S=2$")
        axis.set(xlim=(0.0, 0.3), xlabel=r"deletion probability $p=1-\eta$", ylabel=r"$S_{\max}$")
        axis.legend(frameon=False, ncol=2)
        paths += _save_figure(fig, output_dir, "cross_sections_Smax")

    paths.append(_write_csv(
        output_dir / "threshold_curve_data.csv",
        ("v", "p_c", "p_c_mixed_polynomial"),
        ((f"{v:.8f}", f"{p:.12f}", f"{q:.12f}") for v, p, q in zip(visibility_grid, critical_loss, polynomial_loss)),
    ))
    return paths


def make_hierarchy_and_robustness_figures(output_dir: Path) -> list[Path]:
    """Create the manuscript hierarchy and visibility-robustness figures."""
    paths: list[Path] = []
    eta_grid = np.linspace(0.45, 1.0, 1201)
    representative_n = (3, 4, 6, 8)
    anchored_curves = {n_value: anchored_chsh(eta_grid, n_value) for n_value in representative_n}

    with plt.rc_context({**PUBLICATION_RC, "legend.fontsize": 8.5}):
        fig, axis = plt.subplots(figsize=_golden_figsize(3.4))
        axis.plot(eta_grid, ideal_one_excitation_chsh(eta_grid), label=r"$N=1$")
        for n_value in representative_n:
            axis.plot(eta_grid, anchored_curves[n_value], label=rf"$N={n_value}$")
        axis.axhline(2.0, color="0.2", linestyle="--", linewidth=1.0)
        for eta_value, label in ((ETA_HALF, r"$1/2$"), (ETA_G, r"$\eta_G$"), (ETA_ONE_EXCITATION, r"$1/\sqrt{2}$")):
            axis.axvline(eta_value, color="0.35", linestyle=":", linewidth=0.8)
            axis.text(eta_value, 1.52, label, ha="center", va="bottom", fontsize=8.5)
        axis.set(xlim=(0.45, 1.0), ylim=(1.45, 2.88), xlabel=r"$\eta$", ylabel=r"$S$")
        axis.legend(ncol=2, frameon=False, loc="lower right", handlelength=1.6, columnspacing=0.9, handletextpad=0.45)
        axis.tick_params(direction="out", length=3.0, width=0.8, pad=2.0)
        fig.tight_layout(pad=0.35)
        paths += _save_figure(fig, output_dir, "main_threshold_hierarchy_2", dpi=600)

    paths.append(_write_csv(
        output_dir / "main_threshold_hierarchy_data.csv",
        ("eta", "S_one_excitation", *(f"S_anchored_N{n_value}" for n_value in representative_n)),
        ((f"{eta:.10f}", f"{ideal_one_excitation_chsh(eta):.12f}", *(f"{anchored_chsh(eta, n_value):.12f}" for n_value in representative_n)) for eta in eta_grid),
    ))

    visibility_grid = np.linspace(0.6, 1.0, 401)
    eta_critical_one = 1.0 - np.array([critical_loss_one_excitation(float(v)) for v in visibility_grid])
    eta_approximation = 1.0 - visibility_grid**4 / 4.0
    selected_n = (3, 4, 6, 8, 12)
    eta_critical_by_n = {
        n_value: np.array([critical_survival_anchored(n_value, float(v)) for v in visibility_grid])
        for n_value in selected_n
    }
    envelope = np.array([
        np.nanmin([critical_survival_anchored(n_value, float(v)) for n_value in range(1, 31)])
        for v in visibility_grid
    ])

    with plt.rc_context(PUBLICATION_RC):
        fig, axis = plt.subplots(figsize=_golden_figsize(6.4))
        axis.plot(visibility_grid, eta_critical_one, label=r"one-excitation $\eta_c^{\rm 1ex}(v)$")
        axis.plot(visibility_grid, eta_approximation, "--", label=r"$1-v^4/4$")
        for n_value in selected_n:
            axis.plot(visibility_grid, eta_critical_by_n[n_value], label=rf"$N={n_value}$")
        axis.plot(visibility_grid, envelope, linewidth=2.2, color="0.30", label=r"best $1\leq N\leq30$")
        axis.axhline(ETA_ONE_EXCITATION, color="0.3", linestyle=":", label=r"$1/\sqrt{2}$")
        axis.axhline(ETA_G, color="0.3", linestyle="-.", label=r"$\eta_G$")
        axis.set(xlim=(0.60, 1.00), ylim=(0.60, 1.00), xlabel=r"local transverse visibility $v$", ylabel=r"critical survival probability $\eta_c$")
        axis.set_xticks(np.arange(0.60, 1.001, 0.05))
        axis.set_yticks(np.arange(0.60, 1.001, 0.05))
        axis.tick_params(direction="out", length=3.0, width=0.8, pad=2.0)
        fig.subplots_adjust(left=0.12, right=0.68, bottom=0.15, top=0.97)
        legend = axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, handlelength=2.0, labelspacing=0.55)
        paths += _save_figure(fig, output_dir, "supplement_combined_robustness", dpi=600)
        del legend  # Keep the deliberate external-legend layout explicit.

    paths.append(_write_csv(
        output_dir / "supplement_combined_robustness_data.csv",
        ("v", "eta_c_one_excitation", "eta_c_approximation", *(f"eta_c_anchored_N{n_value}" for n_value in selected_n), "eta_c_best_N_le_30"),
        ((f"{v:.10f}", f"{eta_critical_one[i]:.12f}", f"{eta_approximation[i]:.12f}", *(f"{eta_critical_by_n[n_value][i]:.12f}" for n_value in selected_n), f"{envelope[i]:.12f}") for i, v in enumerate(visibility_grid)),
    ))
    return paths


def make_golden_ratio_figure(output_dir: Path) -> list[Path]:
    """Create the exact gain-deficit crossover figure used in the Supplement."""
    paths: list[Path] = []
    golden_rc = {**PUBLICATION_RC, "font.size": 8.2, "axes.labelsize": 9.2, "legend.fontsize": 7.2, "xtick.labelsize": 8.0, "ytick.labelsize": 8.0}
    with plt.rc_context(golden_rc):
        figure = plt.figure(figsize=_golden_figsize(7.15), constrained_layout=True)
        grid = figure.add_gridspec(1, 2, width_ratios=(1.28, 1.0))
        axis = figure.add_subplot(grid[0, 0])
        eta_grid = np.linspace(0.55, 0.72, 681)
        n_grid = np.arange(1, 1001)
        eta_mesh, n_mesh = np.meshgrid(eta_grid, n_grid)
        log_ratio = anchored_log10_ratio(eta_mesh, n_mesh)
        image = axis.pcolormesh(eta_grid, n_grid, np.clip(log_ratio, -4.0, 4.0), shading="auto", cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-4.0, vcenter=0.0, vmax=4.0), rasterized=True)
        axis.contour(eta_grid, n_grid, log_ratio, levels=[0.0], colors="black", linewidths=1.15)
        axis.text(0.700, 3.0, r"$\mathbf{S}_N=\mathbf{2}$", fontsize=9.8, fontweight="bold", rotation=-20, ha="center", va="bottom")
        axis.axvline(ETA_G, color="#D69E00", linestyle="--", linewidth=1.25)
        axis.text(ETA_G + 0.002, 7.0, r"$\eta_G=1/\varphi$", color="#7A5900", rotation=90, va="top", ha="left", fontsize=9.4)
        for eta_value, offset in ((0.620, -13), (0.640, 2)):
            n_min = first_violating_excitation_number(eta_value)
            assert n_min is not None
            axis.plot(eta_value, n_min, marker="o", markersize=3.6, color="black", zorder=5)
            axis.annotate(rf"$\mathbf{{N}}_{{\min}}=\mathbf{{{n_min}}}$", xy=(eta_value, n_min), xytext=(7, offset), textcoords="offset points", fontsize=9.4, fontweight="bold", arrowprops={"arrowstyle": "-", "lw": 0.55, "color": "black"})
        axis.text(0.580, 20.0, "erasure deficit\ndominates", color="white", ha="center", va="center", fontsize=9.2, fontweight="bold")
        axis.text(0.680, 50.0, "coherence term\ndominates", color="white", ha="center", va="center", fontsize=9.2, fontweight="bold")
        axis.set_yscale("log")
        axis.set(xlim=(eta_grid[0], eta_grid[-1]), ylim=(1, 1000), xlabel=r"survival probability $\eta$", ylabel=r"excitation number $N$")
        axis.set_yticks((1, 3, 10, 30, 100, 300, 1000))
        axis.set_yticklabels(("1", "3", "10", "30", "100", "300", "1000"))
        axis.text(-0.13, 1.055, "(a)", transform=axis.transAxes, fontweight="bold", fontsize=9.5)
        colorbar = figure.colorbar(image, ax=axis, fraction=0.055, pad=0.02)
        colorbar.set_ticks((-4, -2, 0, 2, 4))
        colorbar.set_ticklabels((r"$\leq-4$", "-2", "0", "2", r"$\geq4$"))
        colorbar.set_label(r"$\Lambda_N$", labelpad=-2)

        axis = figure.add_subplot(grid[0, 1])
        n_cross = np.arange(1, 121)
        for eta_value, color, label in ((0.600, "#3569A8", r"$\eta=0.600<\eta_G$"), (ETA_G, "#C78D00", r"$\eta=\eta_G$"), (0.640, "#B23A3A", r"$\eta=0.640>\eta_G$")):
            axis.plot(n_cross, anchored_log10_ratio(eta_value, n_cross), color=color, linewidth=1.8, label=label)
            axis.plot(n_cross, anchored_log10_ratio_asymptotic(eta_value, n_cross), color=color, linewidth=0.9, linestyle="--", alpha=0.85)
        axis.axhline(0.0, color="black", linewidth=0.9)
        axis.text(116, 0.2, r"$S_N=2$", ha="right", va="bottom", fontsize=9.2)
        axis.annotate(r"$\Lambda_N\to-\log_{10}4$", xy=(96, -math.log10(4.0)), xytext=(41, -2.1), arrowprops={"arrowstyle": "->", "lw": 0.65}, fontsize=9.0)
        axis.legend(loc="upper left", frameon=False, handlelength=2.4)
        axis.set(xlim=(1, 120), ylim=(-6.6, 6.6), xlabel=r"excitation number $N$", ylabel=r"$\Lambda_N=\log_{10}(G_N/D_N)$")
        axis.grid(color="0.88", linewidth=0.45)
        axis.text(-0.15, 1.055, "(b)", transform=axis.transAxes, fontweight="bold", fontsize=9.5)
        paths += _save_figure(figure, output_dir, "golden_ratio_chsh_balance", dpi=420)

    records: list[tuple[str, ...]] = []
    for eta_value in (0.600, ETA_G, 0.619, 0.620, 0.625, 0.630, 0.640, 0.650, 0.670, 0.700):
        n_min = first_violating_excitation_number(eta_value)
        n_opt, margin, log_margin = optimum_anchored_margin(eta_value)
        regime = "below" if eta_value < ETA_G else "golden" if math.isclose(eta_value, ETA_G) else "above"
        records.append((f"{eta_value:.12f}", f"{1.0 - eta_value:.12f}", regime, "" if n_min is None else str(n_min), "" if n_opt is None else str(n_opt), "" if margin is None else f"{margin:.12e}", "" if log_margin is None else f"{log_margin:.8f}"))
    paths.append(_write_csv(output_dir / "golden_ratio_chsh_numerics.csv", ("eta", "p", "regime", "N_min_violation", "N_opt", "max_S_minus_2", "log10_max_S_minus_2"), records))
    return paths


def make_sparse_binomial_figures(output_dir: Path) -> list[Path]:
    """Create the current Supplement's sparse-binomial bound panels."""
    paths: list[Path] = []
    m_values = np.arange(51, 502, 2)
    c_reference = 0.25
    lower_bound = np.array([square_sparse_chsh_lower_bound(float(m_value), c_reference) for m_value in m_values])
    mu_values = np.array([square_sparse_mu_star(int(m_value)) for m_value in m_values])
    epsilon_values = np.array([square_sparse_tail_epsilon(float(m_value), c_reference) for m_value in m_values])

    with plt.rc_context(PUBLICATION_RC):
        m_phase = np.arange(50, 301)
        c_grid = np.linspace(0.05, 0.95, 451)
        c_mesh, m_mesh = np.meshgrid(c_grid, m_phase)
        mu_mesh = np.array([square_sparse_mu_star(int(m_value)) for m_value in m_phase])[:, None]
        epsilon_mesh = np.array([[square_sparse_tail_epsilon(float(m_value), float(c_value)) for c_value in c_grid] for m_value in m_phase])
        phase_bound = math.sqrt(2.0) * (1.0 - epsilon_mesh) ** 2 * (1.0 + mu_mesh**2)
        fig, axis = plt.subplots(figsize=_golden_figsize(4.4))
        image = axis.pcolormesh(c_mesh, m_mesh, phase_bound, shading="auto")
        # The black contour is the certified boundary S_lb=2.  Keeping it
        # unlabeled avoids a clipped annotation at the left plot boundary.
        axis.contour(c_mesh, m_mesh, phase_bound, levels=[2.0], colors="black", linewidths=1.1)
        axis.axvline(c_reference, color="white", linestyle="--", linewidth=1.2)
        axis.set(xlabel=r"scaled deletion parameter $c=pM$", ylabel=r"$M$ $(n=M^2)$")
        colorbar = fig.colorbar(image, ax=axis, pad=0.02)
        colorbar.set_label(r"$S_{\rm lb}(M,c)$", labelpad=2)
        paths += _save_figure(fig, output_dir, "SPARSE_PHASE")

        fig, axis = plt.subplots(figsize=_golden_figsize(4.4))
        axis.plot(m_values, lower_bound, label=r"$S_{\rm lb}(M,c=1/4)$")
        axis.axhline(2.0, color="0.2", linestyle="--", label=r"local bound $S=2$")
        axis.axvline(101, color="0.35", linestyle=":", label=r"$M=101$")
        axis.set(xlabel=r"square-family parameter $M$", ylabel=r"certified CHSH lower bound")
        axis.legend(frameon=False, loc="lower right")
        paths += _save_figure(fig, output_dir, "SQUARE_BOUND")

        fig, axis_mu = plt.subplots(figsize=_golden_figsize(4.4))
        line_mu = axis_mu.plot(m_values, mu_values, label=r"$\mu_*$")[0]
        line_cert = axis_mu.axvline(101, color="0.35", linestyle=":", label=r"$M=101$")
        axis_mu.set(xlabel=r"$M$", ylabel=r"$\mu_*$", ylim=(-0.02, 1.02))
        axis_epsilon = axis_mu.twinx()
        line_eps = axis_epsilon.plot(m_values, -np.log10(epsilon_values), "--", label=r"$-\log_{10}\varepsilon_q$")[0]
        axis_epsilon.set_ylabel(r"$-\log_{10}\varepsilon_q$")
        axis_mu.legend((line_mu, line_cert, line_eps), (line_mu.get_label(), line_cert.get_label(), line_eps.get_label()), frameon=False, loc="lower right")
        paths += _save_figure(fig, output_dir, "sparse_bound_components")

    paths.append(_write_csv(
        output_dir / "sparse_square_family_values.csv",
        ("M", "n", "g", "t", "c", "p", "mu_star", "epsilon_q", "S_lower_bound"),
        ((row["M"], row["n"], row["g"], row["t"], f"{row['c']:.12g}", f"{row['p']:.12g}", f"{row['mu_star']:.12g}", f"{row['epsilon_q']:.12g}", f"{row['S_lower_bound']:.12g}") for row in (square_sparse_row(m_value, c_reference) for m_value in (51, 75, 101, 151, 201, 301, 501))),
    ))
    return paths


# ---------------------------------------------------------------------------
# Finite-Fock pure-loss search
# ---------------------------------------------------------------------------

def hermitize(matrix: np.ndarray) -> np.ndarray:
    return (matrix + matrix.conj().T) / 2.0


def operator_norm(matrix: np.ndarray) -> float:
    return float(np.linalg.norm(matrix, 2))


def hermitian_sign(matrix: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(hermitize(matrix))
    signs = np.where(eigenvalues >= 0.0, 1.0, -1.0)
    return (eigenvectors * signs) @ eigenvectors.conj().T


def project_band(matrix: np.ndarray, bandwidth: int) -> np.ndarray:
    _require_positive_integer(bandwidth, "bandwidth", minimum=0)
    dimension = matrix.shape[0]
    indices = np.arange(dimension)
    return hermitize(np.where(np.abs(indices[:, None] - indices[None, :]) <= bandwidth, matrix, 0.0))


def random_hermitian_contraction(rng: np.random.Generator, dimension: int, bandwidth: Optional[int] = None) -> np.ndarray:
    matrix = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(size=(dimension, dimension))
    matrix = hermitize(matrix)
    if bandwidth is not None:
        matrix = project_band(matrix, bandwidth)
    norm = operator_norm(matrix)
    return hermitize(matrix / max(1.0, norm)) if norm > 1.0e-14 else matrix


def projected_band_update(weight: np.ndarray, bandwidth: int) -> np.ndarray:
    """Heuristic one-observable update used for the band-limited benchmark."""
    candidates: list[np.ndarray] = []
    for raw in (hermitian_sign(weight), weight):
        candidate = project_band(raw, bandwidth)
        norm = operator_norm(candidate)
        if norm > 1.0e-14:
            candidates.append(hermitize(candidate / max(1.0, norm)))
    if not candidates:
        return np.zeros_like(weight)
    scores = [float(np.real(np.trace(candidate @ weight))) for candidate in candidates]
    return candidates[int(np.argmax(scores))]


def random_code_isometry(rng: np.random.Generator, dimension: int) -> np.ndarray:
    matrix = rng.normal(size=(dimension, 2)) + 1j * rng.normal(size=(dimension, 2))
    isometry, _ = np.linalg.qr(matrix)
    for column in range(2):
        reference_index = int(np.argmax(np.abs(isometry[:, column])))
        isometry[:, column] *= np.exp(-1j * np.angle(isometry[reference_index, column]))
    return isometry[:, :2]


def anchored_code(cutoff: int, excitation_number: int) -> np.ndarray:
    _require_positive_integer(cutoff, "cutoff", minimum=1)
    if not 1 <= excitation_number <= cutoff:
        raise ValueError("excitation_number must lie between 1 and cutoff.")
    isometry = np.zeros((cutoff + 1, 2), dtype=complex)
    isometry[0, 0] = 1.0
    isometry[excitation_number, 1] = 1.0
    return isometry


def binomial_024_code(cutoff: int) -> np.ndarray:
    if cutoff < 4:
        raise ValueError("The binomial_024 candidate requires cutoff >= 4.")
    isometry = np.zeros((cutoff + 1, 2), dtype=complex)
    isometry[0, 0] = isometry[4, 0] = 1.0 / math.sqrt(2.0)
    isometry[2, 1] = 1.0
    return isometry


def deterministic_codes(cutoff: int) -> list[tuple[str, np.ndarray]]:
    codes = [(f"anchored_0_{n_value}", anchored_code(cutoff, n_value)) for n_value in range(1, cutoff + 1)]
    if cutoff >= 4:
        codes.append(("binomial_024", binomial_024_code(cutoff)))
    return codes


def pure_loss_kraus(cutoff: int, eta: float) -> list[np.ndarray]:
    """Kraus operators ``L_a|k>`` from the manuscript's pure-loss channel."""
    _require_positive_integer(cutoff, "cutoff", minimum=0)
    _require_unit_interval(eta, "eta")
    dimension = cutoff + 1
    operators: list[np.ndarray] = []
    for losses in range(cutoff + 1):
        operator = np.zeros((dimension, dimension), dtype=complex)
        for excitation in range(losses, cutoff + 1):
            coefficient = math.comb(excitation, losses) * (1.0 - eta) ** losses * eta ** (excitation - losses)
            operator[excitation - losses, excitation] = math.sqrt(coefficient)
        operators.append(operator)
    return operators


def pure_loss_output(code: np.ndarray, eta: float) -> np.ndarray:
    """Apply independent pure loss to both halves of the encoded Bell state."""
    if code.ndim != 2 or code.shape[1] != 2 or code.shape[0] < 2:
        raise ValueError("code must be a (K+1) by 2 isometry.")
    if not np.allclose(code.conj().T @ code, np.eye(2), atol=1.0e-10):
        raise ValueError("The code columns must be orthonormal.")
    cutoff = code.shape[0] - 1
    dimension = cutoff + 1
    amplitudes = (np.outer(code[:, 0], code[:, 0]) + np.outer(code[:, 1], code[:, 1])) / math.sqrt(2.0)
    state = np.zeros((dimension * dimension, dimension * dimension), dtype=complex)
    for left_operator in pure_loss_kraus(cutoff, eta):
        partial = left_operator @ amplitudes
        for right_operator in pure_loss_kraus(cutoff, eta):
            output_amplitudes = partial @ right_operator.T
            vector = output_amplitudes.reshape(dimension * dimension)
            state += np.outer(vector, vector.conj())
    state = hermitize(state)
    return state / float(np.real(np.trace(state)))


def partial_trace_b_weight(state: np.ndarray, observable: np.ndarray, dimension: int) -> np.ndarray:
    tensor = state.reshape(dimension, dimension, dimension, dimension)
    return hermitize(np.einsum("lj,ijkl->ik", observable, tensor, optimize=True))


def partial_trace_a_weight(state: np.ndarray, observable: np.ndarray, dimension: int) -> np.ndarray:
    tensor = state.reshape(dimension, dimension, dimension, dimension)
    return hermitize(np.einsum("ki,ijkl->jl", observable, tensor, optimize=True))


def chsh_value(state: np.ndarray, a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray) -> float:
    bell_operator = np.kron(a0, b0 + b1) + np.kron(a1, b0 - b1)
    return float(np.real(np.trace(state @ bell_operator)))


def observable_checks(observables: Sequence[np.ndarray], bandwidth: Optional[int] = None) -> dict[str, float]:
    antihermiticity = max(float(np.max(np.abs(observable - observable.conj().T))) for observable in observables)
    maximum_norm = max(operator_norm(observable) for observable in observables)
    band_violation = 0.0
    if bandwidth is not None:
        for observable in observables:
            dimension = observable.shape[0]
            for row in range(dimension):
                for column in range(dimension):
                    if abs(row - column) > bandwidth:
                        band_violation = max(band_violation, float(abs(observable[row, column])))
    return {"max_antihermiticity": antihermiticity, "max_operator_norm": maximum_norm, "max_band_violation": band_violation}


def seesaw_chsh(
    state: np.ndarray,
    dimension: int,
    rng: np.random.Generator,
    starts: int,
    iterations: int,
    bandwidth: Optional[int] = None,
) -> tuple[float, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Alternating CHSH optimization over four Hermitian contractions."""
    best_value = -math.inf
    best_observables: Optional[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None
    for _ in range(starts):
        a0, a1, b0, b1 = (random_hermitian_contraction(rng, dimension, bandwidth) for _ in range(4))
        previous = -math.inf
        for _ in range(iterations):
            update = hermitian_sign if bandwidth is None else lambda matrix: projected_band_update(matrix, bandwidth)
            a0 = update(partial_trace_b_weight(state, b0 + b1, dimension))
            a1 = update(partial_trace_b_weight(state, b0 - b1, dimension))
            b0 = update(partial_trace_a_weight(state, a0 + a1, dimension))
            b1 = update(partial_trace_a_weight(state, a0 - a1, dimension))
            value = abs(chsh_value(state, a0, a1, b0, b1))
            if abs(value - previous) < CONVERGENCE_TOLERANCE:
                break
            previous = value
        value = abs(chsh_value(state, a0, a1, b0, b1))
        if value > best_value:
            best_value = value
            best_observables = (a0, a1, b0, b1)
    assert best_observables is not None
    return best_value, best_observables


def anchored_observables(cutoff: int, excitation_number: int) -> tuple[np.ndarray, np.ndarray]:
    dimension = cutoff + 1
    z_n = -np.eye(dimension, dtype=complex)
    z_n[0, 0] = 1.0
    x_n = np.zeros((dimension, dimension), dtype=complex)
    x_n[0, excitation_number] = x_n[excitation_number, 0] = 1.0
    return z_n, x_n


def two_observable_chsh(state: np.ndarray, z_observable: np.ndarray, x_observable: np.ndarray) -> tuple[float, np.ndarray]:
    correlator = np.empty((2, 2), dtype=float)
    for row, left in enumerate((z_observable, x_observable)):
        for column, right in enumerate((z_observable, x_observable)):
            correlator[row, column] = float(np.real(np.trace(state @ np.kron(left, right))))
    singular_values = np.linalg.svd(correlator, compute_uv=False)
    return 2.0 * math.sqrt(float(np.sum(singular_values[:2] ** 2))), correlator


def _single_coherence_patterns(dimension: int, lower: int, upper: int) -> np.ndarray:
    other_indices = [index for index in range(dimension) if index not in (lower, upper)]
    patterns = np.empty((1 << len(other_indices), dimension), dtype=float)
    for mask in range(patterns.shape[0]):
        pattern = np.ones(dimension)
        pattern[lower], pattern[upper] = 1.0, -1.0
        for position, index in enumerate(other_indices):
            pattern[index] = 1.0 if (mask >> position) & 1 else -1.0
        patterns[mask] = pattern
    return patterns


def best_single_coherence(state: np.ndarray, dimension: int) -> tuple[float, dict[str, Any]]:
    """Exact finite enumeration over diagonal signs and one Fock coherence."""
    tensor = state.reshape(dimension, dimension, dimension, dimension)
    diagonal_probability = np.real(np.einsum("ijij->ij", tensor))
    best_value, best_data = -math.inf, {}
    for lower in range(dimension):
        for upper in range(lower + 1, dimension):
            patterns = _single_coherence_patterns(dimension, lower, upper)
            z_x = np.real(tensor[:, lower, :, upper].diagonal() + tensor[:, upper, :, lower].diagonal())
            x_z = np.real(tensor[lower, :, upper, :].diagonal() + tensor[upper, :, lower, :].diagonal())
            x_x = float(np.real(tensor[lower, lower, upper, upper] + tensor[lower, upper, upper, lower] + tensor[upper, lower, lower, upper] + tensor[upper, upper, lower, lower]))
            z_z = np.einsum("pi,ij,pj->p", patterns, diagonal_probability, patterns)
            values = 2.0 * np.sqrt(z_z**2 + (patterns @ z_x) ** 2 + (patterns @ x_z) ** 2 + x_x**2)
            index = int(np.argmax(values))
            if values[index] > best_value:
                best_value = float(values[index])
                best_data = {"pair": [lower, upper], "z_diagonal": patterns[index].tolist(), "T": [[float(z_z[index]), float((patterns @ z_x)[index])], [float((patterns @ x_z)[index]), x_x]]}
    return best_value, best_data


def _jsonable_complex_matrix(matrix: np.ndarray, digits: int = 8) -> list[list[list[float]]]:
    return [[[round(float(value.real), digits), round(float(value.imag), digits)] for value in row] for row in matrix]


def _jsonable_code(code: np.ndarray, digits: int = 10) -> dict[str, Any]:
    return {"dim": int(code.shape[0]), "columns": [[[round(float(value.real), digits), round(float(value.imag), digits)] for value in code[:, column]] for column in range(2)]}


def validate_anchored_implementation() -> list[dict[str, Any]]:
    """Compare direct Kraus propagation against the analytical anchored formula."""
    rows: list[dict[str, Any]] = []
    for eta_value in (ETA_ONE_EXCITATION, ETA_G, 0.60, 0.58):
        for n_value in (1, 2, 3, 4):
            state = pure_loss_output(anchored_code(n_value, n_value), eta_value)
            z_n, x_n = anchored_observables(n_value, n_value)
            direct_value, correlator = two_observable_chsh(state, z_n, x_n)
            formula_value = float(anchored_chsh(eta_value, n_value))
            rows.append({"eta": eta_value, "N": n_value, "S_formula": formula_value, "S_direct": direct_value, "abs_error": abs(formula_value - direct_value), "T": correlator.tolist()})
    return rows


def run_finite_fock_search(profile_name: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Execute the manuscript's reproducible finite-Fock numerical search."""
    if profile_name not in SEARCH_PROFILES:
        raise ValueError(f"Unknown profile {profile_name!r}; choose from {sorted(SEARCH_PROFILES)}.")
    profile = SEARCH_PROFILES[profile_name]
    rng = np.random.default_rng(RNG_SEED)
    validation = validate_anchored_implementation()
    rows: list[dict[str, Any]] = []
    detailed: dict[str, Any] = {"metadata": {"rng_seed": RNG_SEED, "profile": profile_name, "eta_G": ETA_G, "eta_one_excitation": ETA_ONE_EXCITATION, "violation_tolerance": VIOLATION_TOLERANCE}, "validation": validation, "best_candidates": {}, "violations_at_or_below_eta_G": []}
    start = time.time()

    for eta_value in SURVIVAL_GRID:
        for cutoff in sorted(set(SINGLE_COHERENCE_CUTOFFS + BAND_LIMITED_CUTOFFS + UNRESTRICTED_CUTOFFS)):
            n_value = max(range(1, cutoff + 1), key=lambda n: anchored_chsh(eta_value, n))
            rows.append({"eta": eta_value, "K": cutoff, "regime": "anchored", "B": "", "S": float(anchored_chsh(eta_value, n_value)), "code_name": f"anchored_0_{n_value}", "heuristic": False, "notes": "closed-form vacuum-anchored benchmark"})

    def random_code_list(cutoff: int, count: int) -> list[tuple[str, np.ndarray]]:
        codes = deterministic_codes(cutoff)
        codes += [(f"random_{index}", random_code_isometry(rng, cutoff + 1)) for index in range(count)]
        return codes

    def record_detail(key: str, value: float, data: dict[str, Any], eta_value: float) -> None:
        detailed["best_candidates"][key] = {**data, "S": value}
        if eta_value <= ETA_G + 1.0e-12 and value > 2.0 + VIOLATION_TOLERANCE:
            detailed["violations_at_or_below_eta_G"].append({**data, "S": value})

    for eta_value in SURVIVAL_GRID:
        for cutoff in SINGLE_COHERENCE_CUTOFFS:
            best_value, best_data = -math.inf, {}
            for name, code in random_code_list(cutoff, profile.random_single_coherence_codes):
                value, information = best_single_coherence(pure_loss_output(code, eta_value), cutoff + 1)
                if value > best_value:
                    best_value = value
                    best_data = {"eta": eta_value, "K": cutoff, "regime": "single_coherence", "code_name": name, "code": _jsonable_code(code), "single_coherence": information, "heuristic": False}
            rows.append({"eta": eta_value, "K": cutoff, "regime": "single_coherence", "B": "", "S": best_value, "code_name": best_data["code_name"], "heuristic": False, "notes": f"all diagonal signs; deterministic plus {profile.random_single_coherence_codes} random codes"})
            record_detail(f"single_eta={eta_value:.12f}_K={cutoff}", best_value, best_data, eta_value)

    for eta_value in SURVIVAL_GRID:
        for cutoff in BAND_LIMITED_CUTOFFS:
            for bandwidth in BANDWIDTHS:
                if bandwidth > cutoff:
                    continue
                best_value, best_data = -math.inf, {}
                for name, code in random_code_list(cutoff, profile.random_band_limited_codes):
                    value, observables = seesaw_chsh(pure_loss_output(code, eta_value), cutoff + 1, rng, profile.band_limited_starts, profile.band_limited_iterations, bandwidth)
                    if value > best_value:
                        best_value = value
                        best_data = {"eta": eta_value, "K": cutoff, "B": bandwidth, "regime": "band_limited", "code_name": name, "code": _jsonable_code(code), "checks": observable_checks(observables, bandwidth), "observables": {label: _jsonable_complex_matrix(observable) for label, observable in zip(("A0", "A1", "B0", "B1"), observables)}, "heuristic": True}
                rows.append({"eta": eta_value, "K": cutoff, "regime": "band_limited", "B": bandwidth, "S": best_value, "code_name": best_data["code_name"], "heuristic": True, "notes": f"projected see-saw; {profile.band_limited_starts} starts/code, deterministic plus {profile.random_band_limited_codes} random codes"})
                record_detail(f"band_eta={eta_value:.12f}_K={cutoff}_B={bandwidth}", best_value, best_data, eta_value)

    for eta_value in SURVIVAL_GRID:
        for cutoff in UNRESTRICTED_CUTOFFS:
            best_value, best_data = -math.inf, {}
            for name, code in random_code_list(cutoff, profile.random_unrestricted_codes):
                value, observables = seesaw_chsh(pure_loss_output(code, eta_value), cutoff + 1, rng, profile.unrestricted_starts, profile.unrestricted_iterations)
                if value > best_value:
                    best_value = value
                    best_data = {"eta": eta_value, "K": cutoff, "regime": "unrestricted", "code_name": name, "code": _jsonable_code(code), "checks": observable_checks(observables), "observables": {label: _jsonable_complex_matrix(observable) for label, observable in zip(("A0", "A1", "B0", "B1"), observables)}, "heuristic": False}
            rows.append({"eta": eta_value, "K": cutoff, "regime": "unrestricted", "B": "", "S": best_value, "code_name": best_data["code_name"], "heuristic": False, "notes": f"unrestricted see-saw; {profile.unrestricted_starts} starts/code, deterministic plus {profile.random_unrestricted_codes} random codes"})
            record_detail(f"unrestricted_eta={eta_value:.12f}_K={cutoff}", best_value, best_data, eta_value)

    detailed["metadata"]["runtime_seconds"] = time.time() - start
    return rows, detailed


def _typed_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    typed: list[dict[str, Any]] = []
    for row in rows:
        converted = dict(row)
        converted.update(eta=float(row["eta"]), K=int(row["K"]), S=float(row["S"]), B=None if row["B"] == "" else int(row["B"]))
        typed.append(converted)
    return typed


def save_search_results(output_dir: Path, rows: Sequence[dict[str, Any]], detailed: dict[str, Any]) -> list[Path]:
    """Write search summaries, candidates, and the three manuscript plots."""
    paths: list[Path] = []
    csv_path = output_dir / "finite_fock_results.csv"
    fields = ("eta", "K", "regime", "B", "S", "code_name", "heuristic", "notes")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
    paths.append(csv_path)
    json_path = output_dir / "finite_fock_best_candidates.json"
    json_path.write_text(json.dumps(detailed, indent=2), encoding="utf-8")
    paths.append(json_path)

    typed = _typed_rows(rows)
    regimes = ("anchored", "single_coherence", "band_limited", "unrestricted")
    labels = {"anchored": "anchored", "single_coherence": "single coherence", "band_limited": "best band-limited", "unrestricted": "unrestricted benchmark"}
    with plt.rc_context(PUBLICATION_RC):
        figure, axis = plt.subplots(figsize=_golden_figsize(5.1))
        for regime in regimes:
            x_values, y_values = [], []
            for eta_value in SURVIVAL_GRID:
                values = [row["S"] for row in typed if row["regime"] == regime and abs(row["eta"] - eta_value) < 1.0e-12]
                if values:
                    x_values.append(eta_value)
                    y_values.append(max(values))
            ordering = np.argsort(x_values)
            axis.plot(np.asarray(x_values)[ordering], np.asarray(y_values)[ordering], marker="o", label=labels[regime])
        axis.axhline(2.0, color="0.2", linestyle="--", label="local bound")
        axis.axvline(ETA_G, color="0.35", linestyle=":", label=r"$\eta_G$")
        axis.axvline(ETA_ONE_EXCITATION, color="0.35", linestyle="-.", label=r"$1/\sqrt{2}$")
        axis.set(xlabel=r"survival probability $\eta$", ylabel=r"largest found CHSH value $S$")
        axis.legend(frameon=False, fontsize=7.4)
        paths += _save_figure(figure, output_dir, "best_S_vs_eta")

        figure, axis = plt.subplots(figsize=_golden_figsize(5.1))
        for eta_value in SURVIVAL_GRID:
            x_values, y_values = [], []
            for bandwidth in BANDWIDTHS:
                values = [row["S"] for row in typed if row["regime"] == "band_limited" and row["B"] == bandwidth and abs(row["eta"] - eta_value) < 1.0e-12]
                if values:
                    x_values.append(bandwidth)
                    y_values.append(max(values))
            if x_values:
                axis.plot(x_values, y_values, marker="o", label=rf"$\eta={eta_value:.3f}$")
        axis.axhline(2.0, color="0.2", linestyle="--", label="local bound")
        axis.set(xlabel=r"bandwidth $B$", ylabel=r"largest found band-limited $S$")
        axis.legend(frameon=False, fontsize=7.0, ncol=2)
        paths += _save_figure(figure, output_dir, "performance_vs_bandwidth")

        figure, axis = plt.subplots(figsize=_golden_figsize(5.1))
        for regime in regimes:
            cutoffs = sorted({row["K"] for row in typed if row["regime"] == regime})
            thresholds = []
            for cutoff in cutoffs:
                violations = [row["eta"] for row in typed if row["regime"] == regime and row["K"] == cutoff and row["S"] > 2.0 + VIOLATION_TOLERANCE]
                thresholds.append(min(violations) if violations else math.nan)
            axis.plot(cutoffs, thresholds, marker="o", label=labels[regime])
        axis.axhline(ETA_G, color="0.35", linestyle=":", label=r"$\eta_G$")
        axis.axhline(ETA_ONE_EXCITATION, color="0.35", linestyle="-.", label=r"$1/\sqrt{2}$")
        axis.set(xlabel=r"Fock cutoff $K$", ylabel=r"lowest tested $\eta$ with $S>2+10^{-7}$")
        axis.legend(frameon=False, fontsize=7.4)
        paths += _save_figure(figure, output_dir, "threshold_estimate_vs_K")
    return paths


# ---------------------------------------------------------------------------
# Verification and command-line entry point
# ---------------------------------------------------------------------------

def run_verification() -> dict[str, Any]:
    """Perform deterministic numerical checks without running the expensive search."""
    validation = validate_anchored_implementation()
    maximum_error = max(float(row["abs_error"]) for row in validation)
    checks = {
        "max_anchored_formula_error": maximum_error,
        "visibility_v1_loss_threshold": critical_loss_one_excitation(1.0),
        "visibility_v1_threshold_error": abs(critical_loss_one_excitation(1.0) - (1.0 - ETA_ONE_EXCITATION)),
        "anchored_etaG_values": [float(anchored_chsh(ETA_G, n_value)) for n_value in (3, 4, 6, 8, 12, 20, 30)],
        "sparse_M101_lower_bound": square_sparse_chsh_lower_bound(101, 0.25),
        "N_min_eta_0_620": first_violating_excitation_number(0.620),
        "N_min_eta_0_640": first_violating_excitation_number(0.640),
    }
    checks["anchored_etaG_is_nonviolating"] = all(value <= 2.0 + 5.0e-12 for value in checks["anchored_etaG_values"])
    checks["sparse_M101_is_certified"] = checks["sparse_M101_lower_bound"] > 2.0
    checks["passed"] = maximum_error < 1.0e-12 and checks["visibility_v1_threshold_error"] < 1.0e-10 and checks["anchored_etaG_is_nonviolating"] and checks["sparse_M101_is_certified"]
    if not checks["passed"]:
        raise RuntimeError(f"Analytical verification failed: {checks}")
    return checks


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("figures", "search", "all", "verify"), default="figures", help="figures is quick; search uses the finite-Fock benchmark; all runs both.")
    parser.add_argument("--profile", choices=tuple(SEARCH_PROFILES), default="serious", help="finite-Fock search profile (default: serious, as documented in the manuscript).")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "logic_chsh_outputs", help="directory for all generated figures and data.")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    verification = run_verification()
    generated: list[Path] = []
    if arguments.mode in ("figures", "all"):
        generated += make_visibility_figures(output_dir)
        generated += make_hierarchy_and_robustness_figures(output_dir)
        generated += make_golden_ratio_figure(output_dir)
        generated += make_sparse_binomial_figures(output_dir)
    if arguments.mode in ("search", "all"):
        rows, detailed = run_finite_fock_search(arguments.profile)
        generated += save_search_results(output_dir, rows, detailed)
    verification_path = output_dir / "verification.json"
    verification_path.write_text(json.dumps(verification, indent=2), encoding="utf-8")
    generated.append(verification_path)
    print(f"Verification passed. Output directory: {output_dir}")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()