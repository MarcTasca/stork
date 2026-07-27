---
name: stork-pr-tests
description: Add or improve tests for Stork pull requests and branch diffs, especially scientific PyTorch changes involving neuron dynamics, periodic resets, initializers, EFLOP metrics, readouts, notebooks, or numerical claims. Use when asked to write tests, add coverage, create regression tests, verify a PR, or test changes against a base branch. Preserve the repository's existing unittest framework unless migration is explicitly requested.
---

# Stork PR Tests

Add high-signal tests for completed changes without teaching the tests to merely
repeat the implementation.

## Establish the target

1. Use the base revision supplied by the user. Otherwise prefer `origin/main`
   when it resolves, then `main`.
2. Verify the revision and inspect `git diff <base>...HEAD`, the changed-file
   list, and `git log <base>..HEAD --oneline`.
3. Read every applicable `AGENTS.md` before planning or editing tests.
4. Read the changed implementation, its public call sites, existing tests, and
   the relevant equations or behavioral claims in repository documentation.
5. Treat notebooks and generated figures as evidence to inspect, not as unit
   tests or unquestioned specifications.

Keep the work scoped to the target diff. Do not modify production code,
dependencies, CI, or the testing framework unless the user explicitly expands
the task.

## Build the test plan

Map each changed behavior to an observable public outcome and an independent
oracle. Prioritize:

1. Previously broken behavior and regression cases.
2. Boundary transitions and off-by-one cases.
3. Invalid inputs and documented failure behavior.
4. Mathematical invariants over additional examples.
5. Integration with existing Stork abstractions and serialization.

Reject tests whose expected result is calculated by copying the production
algorithm. Prefer hand-worked examples, simple reference implementations,
closed-form results, conservation or monotonicity properties, and comparisons
between equivalent public execution paths.

## Cover scientific and PyTorch risks

Select checks that are relevant to the diff:

- For neuron and readout dynamics, exercise exact time traces, reset/spike
  precedence, the first and repeated reset boundaries, batch behavior, and
  stateful continuation across `reset_state`.
- For tensor code, exercise shape validation, broadcasting, dtype and device
  propagation, scalar versus vector parameters, empty dimensions where valid,
  and CPU execution. Add conditional accelerator coverage only when the
  behavior is device-specific.
- For differentiable behavior, verify forward values and the intended gradient
  path, including detach semantics such as `diff_reset`. Check gradients for
  shape, finiteness, and expected zero/nonzero behavior.
- For buffers and state, verify `state_dict` contents, save/load round trips,
  non-persistent derived state, and schedule reconstruction after loading.
- For stochastic initialization, isolate random state, use fixed seeds, test
  distribution-independent invariants first, and justify tolerances for any
  sampled statistic. Prefer `torch.random.fork_rng()` over leaking a seed into
  later tests.
- For numerical comparisons, use exact assertions only for discrete results.
  Use `torch.testing.assert_close` or an equivalent assertion with explicit,
  justified tolerances for floating-point results.
- For operation counters, compare small cases with a transparent reference
  enumerator and test non-negativity, additivity, threshold behavior, and
  broadcasting.
- For documented empirical results, separate deterministic smoke tests from
  expensive training or dataset reproduction. Never turn a single seeded
  accuracy result into a brittle unit-test threshold.

## Implement in the existing suite

Preserve `unittest.TestCase`, `subTest`, test discovery, and local naming
conventions unless the repository has deliberately migrated. Do not add
`pytest`, Hypothesis, coverage tooling, or other dependencies merely to write
these tests.

Keep each test focused on one behavior. Use small tensors and short traces.
Avoid network access, large datasets, notebooks, and training loops in the
default unit suite.

If a new test exposes a production defect:

- Do not weaken the assertion or mirror the defective behavior.
- Reduce it to the smallest convincing reproducer.
- Do not fix production code unless the user asks.
- Leave the failing regression test only when it clearly captures the intended
  contract, and report the failure prominently.

## Verify

1. Run the narrowest new or changed test first.
2. Run the full suite with:

   ```bash
   .venv/bin/python -m unittest discover -s tests -v
   ```

   If that interpreter is unavailable, inspect the project configuration and
   use its established equivalent rather than installing a new runner.
   If Matplotlib cannot write its default cache, point `MPLCONFIGDIR` at a
   task-specific directory under `/tmp`; do not add cache files to the repo.
3. Confirm repeated execution is deterministic.
4. Review the test diff for tautological assertions, shared-state leakage,
   excessive runtime, and accidental production changes.

Report the behaviors covered, commands and fresh results, any uncovered risks,
and any production defect revealed by the tests.
