# PIF: periodic reset-and-fire neurons

This package contains four independent pieces:

1. non-leaky integrate-and-fire groups with periodic hard resets;
2. fluctuation-driven initialization for their finite integration windows;
3. a non-leaky accumulator readout for periodic-reset networks;
4. tensor-based effective operation counting for spikes and neuron updates.

## Discrete dynamics

Let \(u_{n,j}\) be the membrane of neuron \(j\) immediately before simulation
step \(n\), \(I_{n,j}\) its accumulated input in that step, and \(\vartheta_j\)
its threshold. The forward dynamics implemented by `PIFGroup` and
`HeterogeneousPIFGroup` are

$$
s_{n,j}=H(u_{n,j}-\vartheta_j),
\qquad
u_{n+1,j}=(u_{n,j}+I_{n,j})(1-\bar{s}_{n,j})(1-r_{n,j}).
$$

For the default `SuperSpike` activation, \(H(x)=\mathbf 1\{x>0\}\). In the
forward pass \(\bar{s}_{n,j}=s_{n,j}\); when `diff_reset=False`, only the
backward path through this reset factor is detached. Thus a spike or a
scheduled reset sets the next membrane to zero. Input arriving on a scheduled
reset step is discarded, while the spike returned on that step is determined
from the pre-update membrane \(u_{n,j}\).

For simulation step \(\Delta t\), continuous period \(\tau_j>0\), and sampled
phase \(\phi_j\sim\mathrm{Uniform}[0,1)\), the implementation uses

$$
p_j=\max\!\left(1,\mathrm{round}(\tau_j/\Delta t)\right),
\qquad
o_j=\lfloor \phi_jp_j\rfloor,
$$

and the exact integer-step reset schedule

$$
r_{n,j}=\mathbf 1\{n\ge o_j\}\,
\mathbf 1\{(n-o_j)\bmod p_j=0\}.
$$

Consequently, the realized reset period is \(p_j\Delta t\), not necessarily
exactly \(\tau_j\). `PIFGroup` sets every \(\tau_j=\tau\).
`HeterogeneousPIFGroup` samples

$$
\tau_j\sim\mathrm{Gamma}\!\left(k,\text{rate}=k/\tau\right),
\qquad
\mathbb E[\tau_j]=\tau,
\qquad
\mathrm{Var}(\tau_j)=\tau^2/k,
$$

where `concentration` is \(k\). The subsequent rounding changes the mean of
the realized integer periods slightly.

## Initialization: assumptions and derivation

The initializer is a finite-window extension of fluctuation-driven moment
matching. Its continuous-time derivation assumes:

- \(N\) independent presynaptic Poisson processes, each with rate \(\nu\);
- a fixed dense weight row \(w_1,\ldots,w_N\);
- no output spike during the window; and
- an observation age \(A\sim\mathrm{Uniform}(0,\tau)\), independent of
  the input.

Conditional on \(A=a\), let \(K_i(a)\sim\mathrm{Poisson}(\nu a)\) and
\(U(a)=\sum_i w_iK_i(a)\). With
\(S_1=\sum_iw_i\) and \(S_2=\sum_iw_i^2\), independence gives

$$
\mathbb E[U\mid A=a,w]=\nu aS_1,
\qquad
\mathrm{Var}(U\mid A=a,w)=\nu aS_2.
$$

Averaging these two conditional moments over reset phase yields

$$
\mathbb E_A\!\left[\mathbb E[U\mid A,w]\right]
=\frac{\nu\tau}{2}S_1,
\qquad
\mathbb E_A\!\left[\mathrm{Var}(U\mid A,w)\right]
=\frac{\nu\tau}{2}S_2.
$$

If weights are sampled independently with mean \(\mu_w\) and variance
\(\sigma_w^2\), then
\(\mathbb E[S_1]=N\mu_w\) and
\(\mathbb E[S_2]=N(\sigma_w^2+\mu_w^2)\). Matching target mean \(\mu_u\)
and target phase-averaged conditional variance \(\sigma_u^2\) gives exactly
the parameters used by `PeriodicResetFluctuationDrivenInitializer`:

$$
\boxed{\mu_w=\frac{2\mu_u}{N\nu\tau}},
\qquad
\boxed{\sigma_w^2=
\frac{2\sigma_u^2}{N\nu\tau}-\mu_w^2}.
$$

The requested parameters are feasible only when the expression for
\(\sigma_w^2\) is strictly positive; otherwise initialization raises
`ValueError`.

### What variance is being matched?

The formula above matches the reset-phase average of the variance conditional
on age. The full variance over both Poisson input and uniformly sampled age is,
for a fixed realized weight row,

$$
\mathrm{Var}_{A,K}(U\mid w)
=\frac{\nu\tau}{2}S_2
+\frac{\nu^2\tau^2}{12}S_1^2.
$$

The second term is the variance of the conditional mean across reset phase.
It vanishes when the row sum is zero. With `center_weights=True`, the sampler
enforces \(S_1=N\mu_w\) in every row, so the common choice \(\mu_u=\mu_w=0\)
removes this term exactly. Row centering correlates the sampled weights and,
without rescaling, changes the expected centered sum of squares from
\(N\sigma_w^2\) to \((N-1)\sigma_w^2\); this is a relative \(1/N\) finite-fan-in
correction when \(\mu_w=0\).

For a heterogeneous group, the initializer receives the scalar population
mean \(\tau\). It therefore matches population-average continuous moments
before discretization; it does not calibrate every sampled \(\tau_j\)
individually. The example notebook reports exact discrete predictions for the
realized \(p_j\) and weight rows.

```python
from stork.periodic_reset import (
    NonLeakyReadoutGroup,
    PeriodicResetFluctuationDrivenInitializer,
    PIFGroup,
)

group = PIFGroup(shape=128, tau=40e-3)
readout = NonLeakyReadoutGroup(shape=20, tau_mem=0.7)
initializer = PeriodicResetFluctuationDrivenInitializer(
    nu=15.8,
    tau=40e-3,
    mu_u=0.0,
    sigma_u=1.0,
    center_weights=True,
)
```

## Exact discrete moments used in the notebook

The initialization experiment uses one Bernoulli event per input bin, with
\(q=\nu\Delta t\), rather than an unbounded Poisson count. For an integer
period \(p_j\), the recorded integration age \(K_j\) is uniform on
\(\{0,\ldots,p_j-1\}\), so

$$
\mathbb E[K_j]=\frac{p_j-1}{2},
\qquad
\mathrm{Var}(K_j)=\frac{p_j^2-1}{12}.
$$

For a realized weight row with sums \(S_{1,j}\) and \(S_{2,j}\), the exact
cycle moments are

$$
\mathbb E[U_j]=q\,\frac{p_j-1}{2}S_{1,j},
$$

$$
\mathrm{Var}(U_j)
=q(1-q)\frac{p_j-1}{2}S_{2,j}
+q^2\frac{p_j^2-1}{12}S_{1,j}^2.
$$

The notebook evaluates these expressions for every neuron and averages the
per-neuron means and standard deviations, exactly matching its empirical
summary statistic.

## Effective-operation definitions

The EFLOP name and zero-skipping principle follow
[Narduzzi, Zenke, Liu, and Dunbar (2025)](https://doi.org/10.1088/2634-4386/addee8).
Their framework counts arithmetic and activation operations involving nonzero
operands, accounts for both weight and activation sparsity, and excludes memory
access. `EffectiveFlopsCounter` specializes that framework to explicit Stork
traces and extends it with periodic-reset PIF and non-leaky-readout rules. The
equations below define the exact convention implemented here; they should not
be read as equations quoted verbatim from the paper.

The counter consumes explicit tensors in `(batch, time, ...)` layout. Define

$$
A_{btj}=\mathbf 1\{X_{btj}\ne0\},\qquad
Q_{btj}=\mathbf 1\{\text{neuron }j\text{ receives at least one active synapse}\},
$$

where \(X\) is the recorded membrane or readout state. Let
\(P_{btj}=A_{b,t-1,j}\), with \(P_{b0j}=0\), and
\(N_{btj}=A_{btj}(1-P_{btj})\). For a PIF reset mask \(R\), the implemented
neuron conventions are

$$
C_{\mathrm{PIF}}
=2\sum A+\sum Q+\sum(PR)-2\sum N,
$$

$$
C_{\mathrm{LIF}}
=5\sum A+\sum Q-3\sum\bigl(Q(1-A)\bigr),
$$

$$
C_{\mathrm{readout}}
=2\sum A+\sum Q-2\sum N.
$$

The PIF reset term uses the pre-reset activity \(P\), because the state stored
on a reset step is already zero. For a dense weight matrix
\(W\in\mathbb R^{d_{\mathrm{out}}\times d_{\mathrm{in}}}\), define the active
fan-out of input \(i\) as

$$
d_i=\sum_j\mathbf 1\{|W_{ji}|>\varepsilon\}.
$$

For presynaptic trace \(S\), connection operations are

$$
C_{\mathrm{conn}}=\sum_{b,t,i}\mathbf 1\{S_{bti}\ne0\}d_i,
$$

and the total returned count is
\(C_{\mathrm{conn}}+C_{\mathrm{PIF}}+C_{\mathrm{LIF}}+C_{\mathrm{readout}}\)
over the supplied collections. These equations define an analytical,
event-driven convention. They are neither the number of dense PyTorch
instructions executed nor a measurement of runtime, energy, or hardware cost.

## Example

[`examples/06_PeriodicReset_SHD.ipynb`](../../examples/06_PeriodicReset_SHD.ipynb)
compares dense LIF, homogeneous PIF, and heterogeneous PIF networks on SHD.
It checks the initialization statistics, reports test accuracy, and counts
analytical effective operations per test sample.

The broader fluctuation-driven initialization framework is described by
[Rossbroich, Gygax, and Zenke (2022)](https://arxiv.org/abs/2206.10226).
The effective-operation methodology is based on
[Narduzzi, S., Zenke, F., Liu, S.-C., and Dunbar, L. A. (2025), *EFLOP: a
sparsity-aware metric for evaluating computational cost in spiking and
non-spiking neural networks*, Neuromorphic Computing and Engineering 5(3),
034011](https://doi.org/10.1088/2634-4386/addee8).
