# Stork Repository Guidance

## Environment and verification

- Treat Stork as a scientific PyTorch library supporting Python 3.10 through
  3.13.
- Preserve the existing standard-library `unittest` suite unless a test-runner
  migration is explicitly requested.
- Use the locked local environment when present. Run the default suite with
  `.venv/bin/python -m unittest discover -s tests -v`.
- If Matplotlib cannot write its default cache, use a task-specific
  `MPLCONFIGDIR` under `/tmp` instead of writing cache files into the repo.

## Scientific engineering

- Treat public APIs, documented neuron dynamics, equations, units, and stated
  assumptions as contracts, but derive test expectations independently of the
  implementation.
- Make tensor shape, batch, dtype, device, state, serialization, randomness,
  and autograd behavior explicit whenever a change can affect them.
- Keep deterministic unit verification separate from expensive notebook,
  dataset, training, and empirical-result reproduction.

## Code Review Rules

### Scientific correctness

- Flag changes that contradict documented equations, units, update ordering,
  limiting cases, or the stated scope of empirical claims. State the concrete
  input or experiment that demonstrates the discrepancy.

### Tensor and state semantics

- Flag consequential errors in broadcasting, batch handling, dtype/device
  propagation, reset boundaries, stateful continuation, buffer persistence,
  save/load reconstruction, or gradient flow. Distinguish intended detach
  behavior from accidental loss of gradients.

### Evidence and tests

- Require high-risk changed behavior to have deterministic tests with
  independent expected values. Notebooks, generated figures, and a single
  seeded training run are supplementary evidence, not substitutes for tests.
  Leave formatting and other deterministic mechanical checks to tooling.
