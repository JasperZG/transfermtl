# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository status

Planning-stage research repo. There is **no source code yet** — only two design documents:

- `plan.txt` — the research plan for the paper *Transfer Sign Heterogeneity in
  Multi-Task Learning* (molecular MTL, NeurIPS target).
- `codebase_plan.md` — the implementation architecture that will turn `plan.txt`
  into code. This is the source of truth for module layout, data contracts,
  and milestone gating.

Don't fabricate commands (build, lint, test) until the corresponding tooling is
actually added. When implementation starts, the layout, dependencies, and
configuration system are specified in `codebase_plan.md` §1, §4, §8.

## Reading `plan.txt`

The file is 5375 lines but heavily duplicated. Unique content is concentrated in
three regions:

- **Lines 1-754**: canonical Parts 1-14. Read this first.
- **Lines 2800-3168**: full §2.2-§2.14 methodology (preprocessing, encoder,
  training protocols, gradient extraction, hierarchical bootstrap, validity,
  null distribution, predictor evaluation). The first occurrence of Part 2
  (lines 70-84) is a stub; this is the real version.
- **Lines 4522-4700**: §2.15-§2.21 (local-affinity sharing architecture details,
  baseline implementations, utility metrics, compute estimates,
  reproducibility, decomposition, **pre-committed hyperparameter table**).

Everything else is verbatim repetition. Treat the three blocks above as
authoritative; cross-reference by section number rather than line number.

## Project shape (one-paragraph summary)

The paper claims that task affinity in multi-task learning is region-conditional
and sign-changing: the same task pair can show positive transfer in one chemical
region and negative transfer in another. The contribution stack is
phenomenon → failure mode (global affinity cancellation) → diagnostic
(region-conditional gradient cosine `G_ij(r)`) → utility (sharing decisions).
The pilot (Phase 0, plan §5) is the gating decision; both green-light criteria
in §5.7 must pass before scaling up. The primary scientific risk is that scaffold
similarity explains everything, decided by the **incremental R² test** in
Phase 2 (plan §2.14, §7); kill criterion is incremental R² < 0.05.

## Hard constraints to respect

These are pre-committed by the research plan and changing them post-hoc is
p-hacking, not engineering preference:

- **Hyperparameters in plan §2.21** are locked: encoder (3-layer GCN, hidden
  256), AdamW lr=1e-3 wd=1e-2, batch 32, 70/15/15 scaffold-stratified split with
  splitting seed 42, M=5 default regions, n_min=50, ε=1.5 AUC / 0.10 RMSE,
  bootstrap B=1000, random partitions B'=200, pilot 3 seeds / final 5 seeds.
- **Validity criteria in §2.11 and §2.12** lock the prevalence denominator. Do
  not relax them after seeing data.
- **Hierarchical bootstrap** is required for every reported scalar. Plain iid
  bootstrap is wrong here because compounds within scaffolds are not
  independent.
- **Random-partition null** must accompany every claim about scaffold-based
  prevalence (plan §2.13).
- **Decision tree in §13**: respect the kill criteria. Pilot fails →
  workshop or drop; incremental R² fails → reframe; do not silently continue.

## Where to look first when implementing

When starting work on a module, the relevant section of `codebase_plan.md`
maps to the matching section of `plan.txt`:

| Module | codebase_plan.md | plan.txt |
|---|---|---|
| Data pipeline | §2.1 | §2.2 (line ~2803) |
| Encoders | §2.3 | §2.3 (line ~2828), §9.3 |
| Training | §2.4-2.5 | §2.4-2.5 (line ~2847) |
| Region partitioning | §2.6 | §2.6 (line ~2894) |
| Gradient affinity | §2.7 | §2.7 (line ~2950) |
| Bootstrap | §2.10 | §2.10 (line ~3030) |
| Predictor (incremental R²) | §2.14 | §2.14 (line ~3099) |
| Local-affinity sharing | §2.15 | §2.15 (line ~3143, continued at ~4522) |
| Baselines | §2.16 | §2.16 (line ~4558) |
| Pre-committed hparams | §4 | §2.21 (line ~4675) |

## Working with `plan.txt`

If you edit `plan.txt`, dedupe it first — the repeated blocks make diffs
unreadable and waste context. Confirm with the user before doing so; the file
may be intentionally redundant from a copy/paste workflow they want to keep.

When quoting plan.txt to the user, cite by **section number** (e.g.
"plan §2.14"), not line number, so the reference survives a future dedupe.
