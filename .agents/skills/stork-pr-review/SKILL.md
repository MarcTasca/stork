---
name: stork-pr-review
description: Review Stork pull requests, commits, or branch diffs for consequential defects in scientific and PyTorch code. Use when asked to review, audit, preflight, or assess changes before merge, including numerical correctness, neuron or state dynamics, autograd, tensor shape/device/dtype behavior, serialization, reproducibility, tests, documentation, notebooks, and research claims. Perform a read-only review and report prioritized actionable findings.
---

# Stork PR Review

Review the change as both software and executable scientific reasoning. Remain
read-only unless the user separately asks for fixes.

## Pin the review target

1. Use the base revision supplied by the user. Otherwise prefer `origin/main`
   when it resolves, then `main`.
2. Verify the revision and inspect `git diff <base>...HEAD`, the changed-file
   list, and `git log <base>..HEAD --oneline`.
3. Include staged, unstaged, or untracked work only when the user requests a
   working-tree review.
4. Read every applicable `AGENTS.md` as review guidance, including an untracked
   one. Do not treat it as part of the reviewed diff unless working-tree changes
   are in scope.
5. When read-only GitHub access is available, look for a pull request for the
   current branch and read its title, body, and linked issue. Otherwise use
   commit messages and relevant repository documentation, and state that
   external intent was unavailable.

Review defects introduced by the target change. Read surrounding and dependent
code as needed to establish impact, but do not report unrelated legacy issues.
For large diffs, triage changed library code and tests first, then inspect
documentation claims and integration surfaces. Review notebook source cells
and material outputs rather than walking notebook JSON or generated assets
line-by-line.

## Review along five axes

### Specification and mathematics

- Compare implementation behavior with public APIs, equations, units,
  assumptions, and claims in the README or other research documentation.
- Re-derive simple expected results independently; do not accept matching code,
  tests, and documentation as proof when all may share the same mistake.
- Check limiting cases, dimensional consistency, rounding rules, and whether
  empirical claims are presented with appropriate scope.

### Tensor, temporal, and state semantics

- Check shapes, broadcasting, batch dimensions, dtype/device propagation, and
  scalar/vector cases.
- Trace update order across time: inputs, spikes, resets, outputs, counters,
  and the exact boundary at which state changes.
- Check stateful versus stateless resets, derived buffers, save/load behavior,
  and reconfiguration after a time-step or batch-size change.
- Check autograd paths, in-place operations, detach behavior, and gradient
  behavior for both differentiable and non-differentiable reset modes.

### Reproducibility and evidence

- Check seed handling, RNG-state leakage, dataset splits, hidden environmental
  assumptions, and CPU/accelerator divergence.
- Distinguish deterministic verification from stochastic experiments. Require
  enough provenance for reported notebook results and generated figures to be
  reproducible without treating those outputs as CI tests.
- Check that tolerances, sample sizes, and benchmark comparisons support the
  strength of the stated conclusion.

### Tests

- Require tests for changed public behavior, boundary and error paths, and
  invariants with high regression risk.
- Check that expected values come from an independent oracle rather than a
  transcription of the implementation.
- Check determinism, isolation, execution time, and whether serialization,
  gradient, dtype/device, and batch behavior are covered when relevant.
- Treat notebook execution or one successful training run as supplementary
  evidence, not a replacement for focused automated tests.

### Integration and compatibility

- Check exports, constructor compatibility, defaults, packaging, supported
  Python versions, and interaction with existing Stork groups, connections,
  models, and monitors.
- Look for silent behavior changes, invalid states that are accepted, valid
  states that now fail, and changes whose cost scales poorly with batch, time,
  or neuron count.

## Verify without modifying

Run relevant read-only checks when the environment supports them. Prefer the
targeted suite, then:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Also inspect `git diff --check <base>...HEAD`, plus the working-tree equivalent
when that scope is requested. Report unavailable or failing verification
accurately; do not claim a check passed without fresh output.
If Matplotlib cannot write its default cache, point `MPLCONFIGDIR` at a
task-specific directory under `/tmp`; do not add cache files to the repo.

## Report findings

List findings before any summary. For each finding:

- Assign `P0` only for catastrophic merge blockers, `P1` for serious defects
  such as wrong scientific results, corrupted state, broken public behavior, or
  common valid-input failures, and `P2` for localized correctness or
  robustness defects worth fixing.
- Give a concise title, the narrowest changed file and line range, the failing
  scenario, and the concrete consequence.
- Explain why the issue is caused by the change and what direction would make
  it safe. Avoid full patch design unless needed to establish the defect.

Do not report formatting, subjective style, praise, speculative concerns
without a concrete trigger, or low-value nits. If there are no actionable
findings, say so and list the verification performed and any material coverage
gaps.
