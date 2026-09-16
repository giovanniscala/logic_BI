# Recovery-Free CHSH Nonlocality with Particle Loss — Numerical Companion

[![arXiv](https://img.shields.io/badge/arXiv-2608.26407-b31b1b.svg)](https://arxiv.org/abs/2608.26407)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

This repository contains the complete, self-contained numerical code that generates
**every figure, table entry, and analytic sanity check** reported in the paper:

> G. Scala and C. Lupo, *"Recovery-Free CHSH Nonlocality with Particle Loss,"*
> [arXiv:2608.26407](https://arxiv.org/abs/2608.26407).

The paper asks whether CHSH nonlocality survives particle loss *without* an explicit
recovery step, and answers it by (i) proving a universal no-go for survival probability
`η ≤ 1/2`, and (ii) constructing three families of explicit, recovery-free measurement
protocols — built from permutation-invariant Dicke-state encodings — that violate CHSH
above that bound. This code implements the closed-form expressions behind those
protocols, reproduces every quantitative claim and plot in the main text and
Supplemental Material, and runs the finite-Fock numerical search used to stress-test
the main analytic result.

**One number to check the code against the paper directly:** running the script's
analytic self-check reproduces the exact float quoted in the Supplemental Material
(Sec. "Extended numerical search") for the agreement between the closed-form anchored
formula and the direct Kraus-channel simulation:

```
max_anchored_formula_error = 6.661338147750939e-16
```

This is machine-precision agreement (a handful of ULPs), and it is the same digit
string printed in the manuscript — the code in this repository *is* the code that
produced the paper's numbers.

---

## Table of contents

- [Repository contents](#repository-contents)
- [Requirements & installation](#requirements--installation)
- [Quick start](#quick-start)
- [How the code maps onto the paper](#how-the-code-maps-onto-the-paper)
  - [1. One-excitation protocol and its visibility robustness](#1-one-excitation-protocol-and-its-visibility-robustness)
  - [2. Anchored N-excitation protocol and the golden-ratio threshold](#2-anchored-n-excitation-protocol-and-the-golden-ratio-threshold)
  - [3. Sparse binomial PI construction (square family)](#3-sparse-binomial-pi-construction-square-family)
  - [4. Extended finite-Fock numerical search](#4-extended-finite-fock-numerical-search)
  - [5. Analytic results with no numerical component](#5-analytic-results-with-no-numerical-component)
- [Generated output files](#generated-output-files)
- [Reproducibility: seeds, tolerances, search profiles](#reproducibility-seeds-tolerances-search-profiles)
- [Command-line reference](#command-line-reference)
- [Citing this work](#citing-this-work)
- [License](#license)

---

## Repository contents

```
.
├── logic_chsh_computations2.py   # the entire numerical companion (single file, no local deps)
└── README.md                     # this file
```

The whole codebase lives in one dependency-light Python module. It is organized into
four kinds of content, in this order in the file:

| Block | Purpose |
|---|---|
| Configuration | physical constants (`ETA_G`, `ETA_ONE_EXCITATION`, …), plotting style, RNG seed, search profiles |
| Closed-form physics | the analytic CHSH expressions for each protocol (one-excitation, anchored `N`-excitation, sparse square family) |
| Figure/data generators | one function per manuscript figure or figure group, each producing `.png` + `.pdf` + a companion `.csv` |
| Finite-Fock numerical search | a from-scratch quantum simulation (Kraus channels, alternating/"see-saw" CHSH optimization) used as an independent numerical stress test of the main analytic theorem |

No third-party quantum-information library is used: qubit/Fock-space states,
partial-loss Kraus channels, partial traces, and the CHSH see-saw optimization are all
implemented directly with NumPy. The only dependencies are NumPy and Matplotlib.

## Requirements & installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy matplotlib
```

Tested with Python ≥ 3.10 (the script uses `from __future__ import annotations` and
PEP 604 `X | Y` type unions). Matplotlib is used with the non-interactive `Agg`
backend and Matplotlib's built-in "Computer Modern"-like mathtext, so **no local LaTeX
installation is required** to reproduce the manuscript-quality figures.

## Quick start

```bash
# All analytic figures, CSVs, and the module's built-in verification — a few seconds.
python3 logic_chsh_computations2.py --mode figures --output-dir outputs

# Only the fast analytic self-checks (no figures, no search).
python3 logic_chsh_computations2.py --mode verify

# The finite-Fock numerical search at the exact configuration reported in the
# manuscript's Table S ("serious" profile). This is the computationally demanding
# part — expect it to take from several minutes to a few hours depending on hardware.
python3 logic_chsh_computations2.py --mode search --profile serious --output-dir outputs

# A much smaller/faster search, useful for smoke-testing the pipeline.
python3 logic_chsh_computations2.py --mode search --profile quick --output-dir outputs

# Everything: all figures plus the serious search, in one run.
python3 logic_chsh_computations2.py --mode all --profile serious --output-dir outputs
```

Every run first executes `run_verification()`, which re-derives a handful of exact
analytic identities (see [below](#reproducibility-seeds-tolerances-search-profiles))
and **raises an error immediately if anything is inconsistent**, before any figure or
search routine runs. A successful run always ends by printing the full list of
generated files and writes the same checks to `verification.json`.

## How the code maps onto the paper

The paper develops three complementary recovery-free protocols plus one no-go bound.
The table below is the top-level map; each subsection after it gives the full detail
(closed-form expressions, which script functions implement them, and which manuscript
figure/table/equation they reproduce).

| Paper result | Survival threshold | Script coverage |
|---|---|---|
| Universal no-go ("Universal CHSH no-go for `η≤1/2`", label `thm:supp-half-loss-no-go`) | violation impossible for `η ≤ 1/2` | analytic only — no code needed (two-extendibility argument) |
| One-excitation PI protocol ("One-excitation CHSH value" / "…optimized visibility value", labels `thm:supp-one-ex-chsh` / `thm:supp-one-ex-visibility`) | `η > 1/√2 ≈ 0.7071` | §1 below |
| Anchored `N`-excitation PI family ("Anchored pure-loss golden-ratio threshold", label `thm:supp-anchored-golden`) | `η > η_G = (√5−1)/2 ≈ 0.6180` (golden ratio) as `N → ∞` | §2 below |
| Sparse binomial PI construction ("Sparse PI theorem, structural form" / "Stochastic sparse-deletion lower bound", labels `thm:supp-sparse-pi-structural` / `thm:supp-sparse-stochastic`, plus Corollary "Square-family consequence", label `cor:supp-square-family`) | tolerates `O(√n)` deletions | §3 below |
| Extended finite-Fock search (Supplemental Material, label `supp:numerics`) | numerical stress test of whether `η_G` can be beaten | §4 below |

### 1. One-excitation protocol and its visibility robustness

**Paper location:** main text §"One-excitation PI protocol" (`S_n(η) = 2√2·η`,
threshold `η > 1/√2`); Supplemental Material, theorems "One-excitation CHSH value"
(label `thm:supp-one-ex-chsh`) and "One-excitation optimized visibility value"
(label `thm:supp-one-ex-visibility`, the measurement-visibility-robust version).

**Physics.** Two parties share Dicke states `|D₀ⁿ⟩`, `|D₁ⁿ⟩`. After each particle
survives independently with probability `η`, the surviving block gives an ideal CHSH
value `2√2·η`. With an imperfect (visibility-`v`) transverse measurement, the optimized
value becomes

```
S_max(v, η) = 2·√[ (v⁴η²) + max(v⁴η², (1 − 2η(1−η))²) ]
```

whose `η → 1` limit at `v = 1` recovers `2√2·η`.

**Functions:**
- `one_excitation_visible_chsh(v, η)` — the `S_max(v, η)` expression above.
- `critical_loss_one_excitation(v)` — bisects for the critical loss `p_c(v)` at which `S_max = 2`.
- `mixed_branch_loss_root(v)` — an independent quartic-polynomial cross-check of the same threshold.
- `ideal_one_excitation_chsh(η) = 2√2·η` — the `v = 1` ideal curve.

**Figure generator:** `make_visibility_figures()` → reproduces the visibility phase
diagram, the `p_c(v)` threshold curve (with the independent polynomial cross-check
overlaid), and the CHSH cross-sections at fixed `v`.

### 2. Anchored N-excitation protocol and the golden-ratio threshold

**Paper location:** main text §"`N`-excitation PI protocols"; Supplemental Material,
theorems "Fixed-excitation PI deletion converges to pure loss" (label
`thm:supp-pi-to-pure-loss`) and "Anchored pure-loss golden-ratio threshold" (label
`thm:supp-anchored-golden`, the golden-ratio threshold itself), and
§"Numerical view of the golden-ratio crossover."

**Physics.** The vacuum-anchored encoding `|0_L⟩ = |0⟩`, `|1_L⟩ = |N⟩` reduces, in the
`n → ∞` limit, to a pure bosonic-loss channel. Writing the complete-erasure probability
`q_N = (1−η)^N`, the surviving longitudinal correlation `c_N = 1 − 2q_N(1−q_N)`, the
CHSH value is exactly

```
S_N(η) = 2·√( c_N(η)² + η^{2N} )
```

and `S_N(η) → 2` as `N → ∞` precisely at the golden-ratio point `η_G = (√5−1)/2`. The
paper's key diagnostic decomposes the squared excess `S_N²/4 − 1` into a *coherent
gain* term `G_N = η^{2N}` and a *complete-erasure deficit* `D_N = 1 − c_N²`, and tracks
their log-ratio `Λ_N(η) = log₁₀(G_N/D_N)`, whose sign flips exactly at `η_G`.

**Functions:**
- `anchored_complete_erasure_probability`, `anchored_longitudinal_correlation`, `anchored_chsh` — the `q_N`, `c_N`, `S_N(η, v)` chain (with an optional visibility parameter `v` for the robustness study).
- `critical_survival_anchored(N, v)` — the finite-`N` critical survival probability.
- `anchored_log_components`, `anchored_log10_ratio`, `anchored_log10_ratio_asymptotic` — the numerically stable `log₁₀(G_N/D_N)` computation (computed in log-space throughout to avoid cancellation at large `N`, where `G_N` and `D_N` both become exponentially small).
- `first_violating_excitation_number(η)` — smallest `N` with `S_N(η) > 2` (reproduces the `N_min = 121` / `N_min = 11` annotations in the manuscript figure).
- `optimum_anchored_margin(η)` — the best achievable `S_N − 2` over all `N`, and the `N` that attains it.

**Figure generators:**
- `make_hierarchy_and_robustness_figures()` → the single-panel `main_threshold_hierarchy_2` figure used in the **main text** (threshold hierarchy at `1/2`, `η_G`, `1/√2`), plus the visibility-robustness comparison figure `supplement_combined_robustness` combining the one-excitation and several anchored `N` curves.
- `make_golden_ratio_figure()` → the two-panel `golden_ratio_chsh_balance` Supplemental Material figure: panel (a) is the exact `Λ_N(η)` heat map with the `S_N = 2` contour and the `N_min` annotations; panel (b) is the large-`N` cross-section showing the slope of `Λ_N` changing sign at `η_G`.
- `make_threshold_hierarchy_schematic_figure()` → the earlier **two-panel** version of the threshold-hierarchy figure (qualitative bar schematic + CHSH curves). This figure is *not* referenced in the current manuscript (which uses only the compact `main_threshold_hierarchy_2` panel), and is kept here purely for continuity with earlier drafts and the reproducibility archive.

### 3. Sparse binomial PI construction (square family)

**Paper location:** main text §"Sparse binomial PI redundancy protocol"; Supplemental
Material §"Sparse binomial PI construction," theorems "Sparse PI theorem, structural
form" (label `thm:supp-sparse-pi-structural`) and "Stochastic sparse-deletion lower
bound" (label `thm:supp-sparse-stochastic`), and Corollary "Square-family consequence"
(label `cor:supp-square-family`, the explicit `M ≥ 101` guarantee).

**Physics.** Instead of increasing the excitation number, this protocol adds
*particle-number redundancy*: a GNU-type permutation-invariant binomial code with
spacing `g = t+1` and order `M`, on `n = gM` particles, tolerates up to `t` deletions.
Specializing to the "square family" `g = M`, `t = M−1`, `n = M²`, and scaled deletion
parameter `c = pM`, the certified CHSH lower bound is

```
S_lb(M, c) = √2 · (1 − ε_q(M,c))² · (1 + μ*(M)²)
```

where `μ*(M)` is a uniform coherence bound and `ε_q(M,c)` is a Bernstein-type tail
bound on the probability of exceeding `t` deletions. At `c = 1/4` (i.e.
`p = 1/(4M)`), every odd `M ≥ 101` certifies `S_lb > 2` — the explicit statement of
the Corollary, and the `M = 101` marker drawn on the figures below.

**Functions:**
- `central_binomial_ratio(M)` — `binom(M, ⌊M/2⌋)/2^M`, via `_log_binomial` (log-gamma arithmetic, stable for `M` in the hundreds).
- `square_sparse_mu_star(M)` — the uniform coherence bound `μ*`.
- `square_sparse_tail_epsilon(M, c)` — the Bernstein tail term `ε_q`.
- `square_sparse_chsh_lower_bound(M, c)` — assembles `S_lb(M, c)` above.
- `square_sparse_row(M, c)` — a self-describing summary row (`M, n, g, t, c, p, μ*, ε_q, S_lb`) used for the CSV export.

**Figure generator:** `make_sparse_binomial_figures()` → the two-panel Supplemental
Material figure (`SPARSE_PHASE`: the `(c, M)` phase diagram with the `S_lb = 2`
contour and the `c = 1/4` reference line; `SQUARE_BOUND`: the cross-section along
`c = 1/4` with the `M = 101` marker) plus a diagnostic figure (`sparse_bound_components`,
showing `μ*(M)` and `−log₁₀ε_q(M)` on twin axes — not itself reproduced in the paper,
but the two ingredients that combine into `S_lb`).

The built-in verification (`run_verification()`) explicitly checks
`square_sparse_chsh_lower_bound(101, 0.25) > 2`, i.e. that the `M = 101` case
certified in the Corollary is numerically confirmed on every run.

### 4. Extended finite-Fock numerical search

**Paper location:** Supplemental Material §"Extended numerical search"
(`supp:numerics`), Table "Configuration of the extended numerical search," Table
"Best value by survival probability and regime," Table "Best entries at or below
`η_G` by regime," and Figures `best_S_vs_eta` / `performance_vs_bandwidth`.

**Purpose.** The golden-ratio threshold is exact for the specific vacuum-anchored family, but does not
by itself prove optimality over *all* finite-Fock codes and measurements. This part of
the code is an independent numerical stress test: it truncates the local Hilbert space
at Fock cutoff `K`, implements the exact pure-loss Kraus channel, and searches — with
four progressively more permissive measurement classes — for any finite-Fock strategy
that beats `η_G`.

**The four search classes**, from most to least constrained, matching the paper's
paragraph structure exactly:

1. **Anchored Fock-pair benchmark** — the closed-form `S_N(η)` of §2 above, evaluated over all `1 ≤ N ≤ K` (no numerical optimization; this is the analytic family reproduced numerically as a baseline).
2. **Single-coherence search** — arbitrary orthonormal logical states in `H_K`, but the transverse observable is restricted to one Fock-level coherence `X_{rs} = |r⟩⟨s| + |s⟩⟨r|` at a time; `best_single_coherence()` enumerates all pairs `(r,s)` and diagonal sign patterns.
3. **Band-limited search** — each of the four CHSH observables may contain several Fock-level coherences, but only between levels closer than a bandwidth `B` (`project_band`, `projected_band_update`); optimized by an alternating ("see-saw") procedure with *projected* sign/linear updates (a heuristic, not an SDP-certified optimum — the paper is explicit about this distinction, and so is this code).
4. **General-observable (unrestricted) benchmark** — arbitrary Hermitian contractions on `H_K`, optimized by the same see-saw procedure with no bandwidth restriction.

**Core simulation primitives:**
- `pure_loss_kraus(K, η)`, `pure_loss_output(code, η)` — the truncated pure-loss channel `L_a|k⟩ = √[C(k,a)(1−η)^a η^{k−a}]|k−a⟩` and its action on an encoded Bell state.
- `random_code_isometry`, `anchored_code`, `binomial_024_code`, `deterministic_codes`, `random_code_list` — the deterministic and randomly sampled logical encodings searched over (including the `binomial_024` candidate `(|0⟩+|4⟩)/√2, |2⟩` mentioned explicitly in the manuscript).
- `chsh_value`, `partial_trace_a_weight`, `partial_trace_b_weight` — CHSH value and the reduced effective operators used by the see-saw update.
- `seesaw_chsh(...)` — the alternating-optimization loop; implements exactly the closed-form sign update `A₀ ← sign(Tr_B[ρ(I⊗(B₀+B₁))])` (and the analogous updates for `A₁, B₀, B₁`) derived in the manuscript, plus the "projected sign" / "projected linear" heuristics for the bandwidth-restricted case.
- `observable_checks`, `hermitize`, `operator_norm`, `hermitian_sign` — Hermiticity/operator-norm validity checks on every candidate observable.
- `run_finite_fock_search(profile_name)` — orchestrates the full grid (survival probabilities × cutoffs × bandwidths × codes × random restarts) exactly as configured in the manuscript's search table.
- `save_search_results(...)` — writes the results table/JSON and the two search figures.

**Reported outcome (matches the manuscript):** within every tested cutoff, code, and
measurement class, **no strategy was found with `S > 2 + 10⁻⁷` for `η ≤ η_G`**, while
broader measurement classes *do* find larger violations above `η_G` (e.g. `S ≈ 2.1724`
at `η = 1/√2`, versus the anchored family's `S ≈ 2.0332`) — evidence that the failure
to beat `η_G` is not simply an artifact of an overly narrow search.

### 5. Analytic results with no numerical component

The theorem "Universal CHSH no-go for `η≤1/2`" (label `thm:supp-half-loss-no-go`,
via two-extendibility and CHSH monogamy) and "Fixed-excitation PI deletion converges
to pure loss" (label `thm:supp-pi-to-pure-loss`) are pure operator-algebra /
channel-convergence arguments. They are not computed by this
script — the only place they touch the code is that the latter is *why* the pure-loss
Kraus channel `pure_loss_kraus`/`pure_loss_output` is the correct object to use
throughout §2 and §4 above, rather than the full finite-`n` PI-deletion channel.

## Generated output files

Running `--mode figures` (or `--mode all`) produces the following files. The "Paper
figure/table" column gives the exact manuscript label where applicable.

| File | Produced by | Paper figure/table |
|---|---|---|
| `phase_diagram_Smax.{png,pdf}` | `make_visibility_figures` | supporting visibility phase diagram (§S4.2 discussion) |
| `threshold_curve_pc_v.{png,pdf}` | `make_visibility_figures` | supporting `p_c(v)` curve |
| `cross_sections_Smax.{png,pdf}` | `make_visibility_figures` | supporting cross-sections |
| `threshold_curve_data.csv`, `summary_values.txt` | `make_visibility_figures` | numerical data behind the above |
| `main_threshold_hierarchy_2.{png,pdf}` | `make_hierarchy_and_robustness_figures` | main text, `fig:threshold-hierarchy` |
| `supplement_combined_robustness.{png,pdf}` + `.csv` | `make_hierarchy_and_robustness_figures` | `fig:supp-combined-robustness` |
| `main_threshold_hierarchy.{png,pdf}` | `make_threshold_hierarchy_schematic_figure` | earlier two-panel draft figure (not in current manuscript) |
| `figure_captions.txt` | `make_threshold_hierarchy_schematic_figure` | plain-text captions for continuity/archival use |
| `golden_ratio_chsh_balance.{png,pdf}` + `golden_ratio_chsh_numerics.csv` | `make_golden_ratio_figure` | `fig:supp-golden-ratio-balance` |
| `SPARSE_PHASE.{png,pdf}`, `SQUARE_BOUND.{png,pdf}` | `make_sparse_binomial_figures` | `fig:supp-sparse-numerical-results` (a),(b) |
| `sparse_bound_components.{png,pdf}` | `make_sparse_binomial_figures` | diagnostic (μ*, ε_q vs. M), not in paper |
| `sparse_square_family_values.csv` | `make_sparse_binomial_figures` | numerical data behind `S_lb(M, c)` |
| `verification.json` | `run_verification` | analytic self-checks (every run) |

Running `--mode search` (or `--mode all`) additionally produces:

| File | Produced by | Paper figure/table |
|---|---|---|
| `finite_fock_results.csv` | `save_search_results` | underlies Tables "Best value by survival probability…" and "Best entries at or below `η_G`…" |
| `finite_fock_best_candidates.json` | `save_search_results` | full detail (codes, observables, checks) for every best-found candidate |
| `best_S_vs_eta.{png,pdf}` | `save_search_results` | `fig:supp-best-S-vs-eta` |
| `performance_vs_bandwidth.{png,pdf}` | `save_search_results` | `fig:supp-performance-bandwidth` |
| `threshold_estimate_vs_K.{png,pdf}` | `save_search_results` | diagnostic (not in paper) |

## Reproducibility: seeds, tolerances, search profiles

All randomness in the finite-Fock search is controlled by a single fixed seed,
`RNG_SEED = 20_260_716`, so every run of a given `--profile` is bit-for-bit
reproducible.

| Constant | Value | Meaning |
|---|---|---|
| `ETA_G` | `(√5 − 1)/2 ≈ 0.618034` | the golden-ratio threshold |
| `ETA_ONE_EXCITATION` | `1/√2 ≈ 0.707107` | the one-excitation threshold |
| `VIOLATION_TOLERANCE` | `1e-7` | a search result counts as a genuine violation only if `S > 2 + 10⁻⁷`; smaller excesses are numerical roundoff, exactly as stated in the manuscript |
| `CONVERGENCE_TOLERANCE` | `1e-12` | termination tolerance for the see-saw alternating optimization |
| `SURVIVAL_GRID` | `{1/√2, η_G, 0.60, 0.58, 0.56, 0.54, 0.52, 0.505}` | the survival-probability grid used throughout the search, identical to the manuscript's Table S |

Search profiles (`--profile {quick, serious, stress}`), each a
`(random_single_coherence_codes, random_band_limited_codes, random_unrestricted_codes,
band_limited_starts, band_limited_iterations, unrestricted_starts,
unrestricted_iterations)` tuple:

| Profile | Config | Notes |
|---|---|---|
| `quick` | `(32, 8, 16, 4, 50, 8, 80)` | fast smoke test, not the manuscript configuration |
| `serious` | `(256, 64, 128, 12, 150, 24, 250)` | **exactly the configuration in the manuscript's search table** |
| `stress` | `(2048, 512, 1024, 32, 400, 64, 600)` | larger, for further exploration beyond the manuscript |

`run_verification()` (executed automatically at the start of every run, and available
standalone via `--mode verify`) checks, among other things:

- the closed-form anchored formula agrees with the direct Kraus-channel evaluation to floating-point precision (`max_anchored_formula_error`, matching the manuscript's quoted `6.661338147750939e-16`);
- `critical_loss_one_excitation(1.0)` matches the exact ideal threshold `1 − 1/√2`;
- `S_N(η_G, N) ≤ 2` (no anchored violation at or below the golden ratio) for `N ∈ {3,4,6,8,12,20,30}`;
- the `N_min` values at `η = 0.620` and `η = 0.640` match the manuscript's annotated `N_min = 121` and `N_min = 11`;
- `square_sparse_chsh_lower_bound(101, 0.25) > 2`, confirming the Corollary's `M = 101` guarantee.

If any check fails, `run_verification()` raises a `RuntimeError` before any figure is
produced, so a broken numerical claim can never silently propagate into a plot.

## Command-line reference

```
usage: logic_chsh_computations2.py [-h] [--mode {figures,search,all,verify}]
                                    [--profile {quick,serious,stress}]
                                    [--output-dir OUTPUT_DIR]

--mode figures   (default) all analytic figures and CSVs — fast (seconds).
--mode verify    only the analytic self-checks; no figures, no search.
--mode search    the finite-Fock numerical search; slow, opt-in.
--mode all       figures + search in one run.
--profile        quick | serious | stress (default: serious, the manuscript config).
--output-dir     directory for all generated files (created if missing).
```

## Citing this work

If you use this code, please cite the paper:

```bibtex
@article{ScalaLupo2026RecoveryFree,
  title   = {Recovery-Free CHSH Nonlocality with Particle Loss},
  author  = {Scala, Giovanni and Lupo, Cosmo},
  journal = {arXiv preprint arXiv:2608.26407},
  year    = {2026},
  eprint  = {2608.26407},
  archivePrefix = {arXiv},
  url     = {https://arxiv.org/abs/2608.26407}
}
```

## License

Add your preferred license here (e.g. MIT) before publishing the repository.

---

*Questions about the correspondence between a specific manuscript equation/figure and
a specific function should be resolved by first-hand cross-reference with the
Supplemental Material, whose section labels (`supp:...`, `thm:supp-...`) are quoted
throughout this README exactly as they appear in the paper source.*
