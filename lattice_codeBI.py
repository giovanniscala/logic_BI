import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb

# -------------------------------------------------------------------------
# 1. Define Code Parameters
# To satisfy the finite-difference identity constraint (r <= t < M),
# we choose: M = 3 (code redundancy), g = 3 (spacing), r = 2 (deletions).
# -------------------------------------------------------------------------
M = 3  
g = 3  
r = 2  
n = g * M  # Total initial particles (9)

# -------------------------------------------------------------------------
# 2. Mathematical Functions
# -------------------------------------------------------------------------
def Q_hypergeometric(l, a, r, g, M):
    """Computes the transition probability Q_{l,a}^{(r)}"""
    n_total = g * M
    k_excitations = g * l
    
    numerator = comb(k_excitations, a) * comb(n_total - k_excitations, r - a)
    denominator = comb(n_total, r)
    
    # Handle physical impossibilities safely
    if denominator == 0:
        return 0.0
    return numerator / denominator

def modular_L(q, g):
    """Reconstructs the lattice sector index l from surviving weight q"""
    a_recovered = (-q) % g
    return (q + a_recovered) // g

# -------------------------------------------------------------------------
# 3. Compute Probabilities for Each Noise Syndrome Branch 'a'
# -------------------------------------------------------------------------
syndromes = np.arange(0, r + 1)
c_0 = np.zeros(len(syndromes))  # Probabilities for |0_L> (Even l)
c_1 = np.zeros(len(syndromes))  # Probabilities for |1_L> (Odd l)

# Norm factor 1 / 2^(M-1)
norm_factor = 1.0 / (2**(M - 1))

for idx, a in enumerate(syndromes):
    # Sum over Even sectors for |0_L>
    sum_even = 0.0
    for l in range(0, M + 1, 2):
        sum_even += comb(M, l) * Q_hypergeometric(l, a, r, g, M)
    c_0[idx] = norm_factor * sum_even
    
    # Sum over Odd sectors for |1_L>
    sum_odd = 0.0
    for l in range(1, M + 1, 2):
        sum_odd += comb(M, l) * Q_hypergeometric(l, a, r, g, M)
    c_1[idx] = norm_factor * sum_odd

# -------------------------------------------------------------------------
# 4. Plotting
# -------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- LEFT PANEL: Syndrome Indistinguishability ---
x = np.arange(len(syndromes))
width = 0.3

# Overlaying bars with a slight offset to show they are identical
ax1.bar(x - width/2, c_0, width, label=r'From $|0_L\rangle$ (Even $l$)', color='#1f77b4', alpha=0.85)
ax1.bar(x + width/2, c_1, width, label=r'From $|1_L\rangle$ (Odd $l$)', color='#ff7f0e', alpha=0.6, hatch='//')

ax1.set_xlabel('Syndrome $a$ (Number of Deleted Excitations)', fontsize=12)
ax1.set_ylabel('Branch Probability $c_{i,a}^{(r)}$', fontsize=12)
ax1.set_title('Environment Learning Syndrome but NOT Parity\n'
             r'($c_{0,a}^{(r)} = c_{1,a}^{(r)}$ via Finite Differences)', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels([f'a = {a}' for a in syndromes])
ax1.set_ylim(0, max(max(c_0), max(c_1)) * 1.3)
ax1.grid(axis='y', linestyle='--', alpha=0.5)
ax1.legend(fontsize=11, loc='upper right')

# Text annotation explaining the left panel
ax1.text(0.5, 0.1, "Perfect Overlap!\nInformation Leakage = 0", 
         transform=ax1.transAxes, ha='center', va='center', color='green',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#e2f0d9', edgecolor='green', alpha=0.8))


# --- RIGHT PANEL: Modular Lattice Remapping ---
# Show how states shift under a specific syndrome example (say, a = 1)
example_a = 1
ax2.axhline(0, color='gray', linestyle='-', alpha=0.3)

# Original Lattice points k = g*l
l_vals = np.arange(0, M + 1)
k_vals = g * l_vals

for l in l_vals:
    parity = 'Even' if l % 2 == 0 else 'Odd'
    color = '#1f77b4' if parity == 'Even' else '#ff7f0e'
    marker = 'o' if parity == 'Even' else 's'
    
    # 1. Plot original state weight
    ax2.scatter(l, k_vals[l], color=color, s=120, marker=marker, label=f'{parity} Sector' if l in [0, 1] else "")
    ax2.text(l, k_vals[l] + 0.3, f'$|D_{{{k_vals[l]}}}^{{{n}}}\\rangle$', ha='center', va='bottom', fontsize=10)
    
    # 2. Plot surviving state weight after losing 'example_a' excitations
    q_surviving = k_vals[l] - example_a
    if q_surviving >= 0:
        ax2.scatter(l, q_surviving, color=color, s=120, marker=marker, facecolors='none', linewidths=2)
        ax2.text(l, q_surviving - 0.5, f'$|D_{{{q_surviving}}}^{{{n-r}}}\\rangle$', ha='center', va='top', fontsize=10, style='italic')
        
        # Draw shift arrow
        ax2.annotate('', xy=(l, q_surviving), xytext=(l, k_vals[l]),
                     arrowprops=dict(arrowstyle="->", color='purple', lw=1.5, ls=':'))

# Visual adjustments for Right Panel
ax2.set_xlabel('Lattice Sector Index $l$', fontsize=12)
ax2.set_ylabel('Dicke State Excitation Weight ($k$ or $q$)', fontsize=12)
ax2.set_title(f'Tracking Shifts on the Weight Lattice (Example for $a={example_a}$)\n'
             r'Modular Mapping: $L(q) = \frac{q + (-q \text{mod} g)}{g} \equiv l$', fontsize=13, fontweight='bold')
ax2.set_xticks(l_vals)
ax2.set_xlim(-0.5, M + 0.5)
ax2.set_ylim(-1, n + 1)
ax2.grid(True, linestyle=':', alpha=0.5)

# Custom legend to distinguish original vs shifted
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='none', label='Even Sectors ($|0_L\\rangle$)', markerfacecolor='#1f77b4', markersize=10),
    Line2D([0], [0], marker='s', color='none', label='Odd Sectors ($|1_L\\rangle$)', markerfacecolor='#ff7f0e', markersize=10),
    Line2D([0], [0], marker='o', color='purple', label=f'Deletion Shift ($-a$)', linestyle=':')
]
ax2.legend(handles=legend_elements, loc='upper left')

plt.tight_layout()
plt.show()