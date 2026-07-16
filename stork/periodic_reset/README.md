# PIF: periodic reset-and-fire neurons

PIF is a lightweight alternative to a leaky integrate-and-fire (LIF) neuron.
A LIF neuron keeps leaky membrane and synaptic states. A PIF neuron keeps one
non-leaky membrane and clears it at fixed times. This removes the decay
operations while still preventing the membrane from growing forever.

This package provides:

- homogeneous and heterogeneous PIF groups;
- an initializer made for the PIF integration window;
- a non-leaky readout;
- EFLOP counting for connections, neurons, resets, and readouts.

## PIF dynamics

Let $u_{n,j}$ be the membrane of neuron $j$ before step $n$, $I_{n,j}$ its
input, and $r_{n,j}$ the scheduled-reset flag. The default forward update is

$$
s_{n,j}=\mathbf{1}[u_{n,j}>\theta_j],
\qquad
u_{n+1,j}=(u_{n,j}+I_{n,j})(1-s_{n,j})(1-r_{n,j}).
$$

The neuron therefore resets after a spike or at its next reset time. Input on
a reset step is discarded. `diff_reset` only changes the backward path; it
does not change this forward update.

The requested period $\tau_j$ is converted to an integer number of steps:

$$
p_j=\max(1,\mathrm{round}(\tau_j/\Delta t)).
$$

Each neuron receives a random phase in $0,\ldots,p_j-1$, which spreads reset
events across time. `PIFGroup` uses one shared value of $\tau$.
`HeterogeneousPIFGroup` draws one $\tau_j$ per neuron from a Gamma distribution
with mean $\tau$ and concentration $k$.

## Why PIF needs a different initializer

Stork's LIF initializer is based on an exponentially decaying response. PIF
does not have that response. It sums events from its last periodic reset, so
the relevant integration window has length $\tau$.

Assume $N$ independent input neurons, each firing as a Poisson process with
rate $\nu$. At age $a$ after a reset, the membrane is

$$
U(a)=\sum_{i=1}^{N}w_iK_i(a),
\qquad K_i(a)\sim\mathrm{Poisson}(\nu a).
$$

For fixed weights, this is a compound-Poisson sum. Its conditional moments are

$$
\mathbb{E}[U\mid a]=\nu a\sum_i w_i,
\qquad
\mathrm{Var}(U\mid a)=\nu a\sum_i w_i^2.
$$

If the reset phase is uniform, the mean age is $\tau/2$. Matching a target
membrane mean $\mu_u$ and standard deviation $\sigma_u$ gives the initializer
used here:

$$
\mu_w=\frac{2\mu_u}{N\nu\tau},
\qquad
\sigma_w^2=\frac{2\sigma_u^2}{N\nu\tau}-\mu_w^2.
$$

The requested variance must be positive. For a heterogeneous PIF layer, the
initializer uses the population mean $\tau$ rather than a separate value for
every neuron.

### Recentring the weights

With `center_weights=True`, every sampled row is shifted back to the requested
mean:

$$
w_i \leftarrow w_i-\frac{1}{N}\sum_k w_k+\mu_w.
$$

For the usual choice $\mu_u=0$, this makes $\sum_i w_i=0$ for each neuron.
The compound-Poisson sum then has no phase-dependent drift: a neuron does not
start each reset window with a systematic push up or down. Periodic resets
limit how long fluctuations can accumulate.

This combination is meant to keep activity in a useful range instead of
producing silent neurons or neurons that spike at every step. It is not a
guarantee for every dataset or trained model, so the notebook measures it. It
plots the test-set firing rate of every hidden neuron and reports the exact
number of zero-rate and always-spiking neurons.

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

## EFLOP counting

The counter follows the zero-skipping idea from Narduzzi et al. (2025). An
operation is counted when its operands are active and nonzero. Binary spikes
select weight additions, so an active nonzero connection costs one operation.
The counter also includes neuron-state updates and readout updates.

The PIF rules add the cost of a scheduled reset when the membrane was active.
The non-leaky readout rules are another small extension. These two extensions
are specific to this package; they are not equations copied from the paper.

The returned total is

$$
C_{\mathrm{total}}=C_{\mathrm{connections}}+C_{\mathrm{neurons}}+C_{\mathrm{readout}}.
$$

This is a hardware-independent operation count. It is not measured runtime,
memory traffic, or energy. See `metrics.py` and `tests/test_periodic_reset.py`
for the exact rules used by the implementation.

## Notebook

[`examples/06_PeriodicReset_SHD.ipynb`](../../examples/06_PeriodicReset_SHD.ipynb)
compares dense LIF, PIF, and heterogeneous PIF models on SHD. It checks the
initializer, trains or loads each model, plots hidden-neuron firing rates, and
reports test accuracy and EFLOPs per sample.

## References

- Rossbroich, J., Gygax, J., & Zenke, F. (2022). Fluctuation-driven
  initialization for spiking neural network training. *Neuromorphic Computing
  and Engineering, 2*, 044016.
  [https://doi.org/10.1088/2634-4386/ac97bb](https://doi.org/10.1088/2634-4386/ac97bb)
- Narduzzi, S., Zenke, F., Liu, S.-C., & Dunbar, L. A. (2025). EFLOP: a
  sparsity-aware metric for evaluating computational cost in spiking and
  non-spiking neural networks. *Neuromorphic Computing and Engineering, 5*,
  034011.
  [https://doi.org/10.1088/2634-4386/addee8](https://doi.org/10.1088/2634-4386/addee8)
