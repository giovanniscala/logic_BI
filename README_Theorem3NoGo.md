# Lean formalization of the universal no-go theorem

This repository contains a minimal Lean 4 formalization of the universal no-go theorem for recovery-free permutationally invariant CHSH protocols under constant deletion/loss.

The Lean development is intentionally shallow: it does not formalize Hilbert spaces, density matrices, Dicke states, completely positive maps, or the full deletion channel. Instead, it isolates the proof of Theorem 3 into a small logical core. The physics input is represented by three explicit assumptions, and Lean checks that the stated no-go conclusion follows from them.

## Main file

- `Theorem3NoGo.lean`: Lean 4 source file containing the formal statement and proof skeleton for Theorem 3.

The two main theorem names are:

- `LogicalBell.theorem3_universal_no_go`
- `LogicalBell.theorem3_universal_no_go_eta`

## Mathematical content

The formalization captures the following proof structure:

1. A deletion/loss channel in the relevant constant-loss regime is two-extendible.
2. A recovery-free post-deletion state produced by a permutationally invariant CHSH protocol is therefore two-extendible.
3. Every two-extendible bipartite state satisfies the CHSH monogamy bound `S ≤ 2`.
4. Hence no such recovery-free protocol can produce a CHSH violation.

In schematic form:

```text
deletion two-extendibility
+ post-deletion two-extendibility
+ CHSH bound for two-extendible states
------------------------------------------------
S_CHSH ≤ 2
```

## Explicit assumptions

The file leaves the following three physics ingredients as explicit Lean assumptions:

- `deletion_twoExtendible`
- `postDeletion_twoExtendible`
- `twoExtendible_chsh_bound`

These correspond respectively to the two-extendibility of the deletion/loss channel, the preservation/transfer of two-extendibility to the post-deletion state in the recovery-free protocol, and the CHSH monogamy bound for two-extendible states.

All remaining reasoning in the final theorem is checked by Lean.

## Requirements

This project uses Lean 4 with Mathlib.

The project files are:

- `lean-toolchain`
- `lakefile.toml`
- `lake-manifest.json`
- `Theorem3NoGo.lean`

The current setup uses Mathlib `v4.30.0`.

## How to check the proof

From the repository root, run:

```bash
lake env lean Theorem3NoGo.lean
```

A successful check produces no error output.

Alternatively, build the Lake target:

```bash
lake build
```

## How to open in VS Code

1. Install the Lean 4 extension for VS Code.
2. Open the repository folder, not only the individual `.lean` file.
3. Open `Theorem3NoGo.lean`.
4. The Lean infoview should load the file using the toolchain specified in `lean-toolchain`.

## Scope and limitations

This is not a full formalization of the quantum information theory behind the theorem. It is a Lean-checked formalization of the final logical implication once the three named physics lemmas are accepted.

The purpose is to make the dependency structure of the universal no-go theorem explicit and mechanically check the final proof step.
