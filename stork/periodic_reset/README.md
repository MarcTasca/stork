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
does not have that response. It sums events since its last reset, so a small
imbalance in its inputs can build up instead of leaking away.

Assume $N$ independent input neurons, each firing as a Poisson process with
rate $\nu$. At time $t$ after a reset, and before the PIF neuron fires, its
membrane is

$$
U(t)=\sum_{i=1}^{N}w_iK_i(t),
\qquad K_i(t)\sim\mathrm{Poisson}(\nu t).
$$

For a fixed weight row,

$$
\mathbb{E}[U(t)\mid w]=\nu t\sum_i w_i,
\qquad
\mathrm{Var}(U(t)\mid w)=\nu t\sum_i w_i^2.
$$

This is a compound-Poisson accumulation. A spike or periodic reset truncates
it and starts a new window.

### Why zero mean is not enough

Suppose the weights are sampled independently with mean $\mu_w$ and variance
$\sigma_w^2$. Across sampled weight rows,

$$
\mathrm{Var}(U(t))
=N\nu t(\mu_w^2+\sigma_w^2)
+N\sigma_w^2(\nu t)^2.
$$

Even when $\mu_w=0$, a finite sampled row does not usually sum to exactly
zero. The non-leaky membrane accumulates this small error. Rows with a positive
sum drift up, while rows with a negative sum drift down. The last term above
is the variance caused by these different row sums, and it grows as $t^2$.

### Recentring the weights

With `center_weights=True`, every sampled row is shifted back to the requested
mean:

$$
w_i \leftarrow w_i-\frac{1}{N}\sum_k w_k+\mu_w.
$$

For the usual choice $\mu_u=0$, this makes $\sum_i w_i=0$ exactly for every
neuron. It removes the drift and the $t^2$ term while keeping the fluctuations
from the Poisson input. If the original sampled weights have variance
$\sigma_w^2$, the exact centered result is

$$
\mathrm{Var}(U_{\mathrm{centered}}(t))
=\nu t(N-1)\sigma_w^2.
$$

The centered variance is linear in time, so its standard-deviation envelope
grows as $\sqrt{t}$ instead of roughly linearly.

| independently sampled weights | row-recentered weights |
|:--:|:--:|
| ![Membrane trajectories with uncentered weights](assets/uncentered_membrane_dynamics.png) | ![Membrane trajectories with recentered weights](assets/centered_membrane_dynamics.png) |

The thin lines are membrane trajectories and the shaded area is $\pm1$ standard
deviation. The dashed line marks the firing threshold.

### Choosing the weight scale

The initializer also needs a finite integration time. With reset offsets spread
across the layer, the average time since reset is $\tau/2$. Matching a target
membrane mean $\mu_u$ and standard deviation $\sigma_u$ gives the parameters
used by the implementation:

$$
\mu_w=\frac{2\mu_u}{N\nu\tau},
\qquad
\sigma_w^2=\frac{2\sigma_u^2}{N\nu\tau}-\mu_w^2.
$$

The reset offsets are used to desynchronize neurons and the mean age $\tau/2$
sets the fluctuation scale. They are not the reason for recentering. The formula
uses $N$, while exact recentering gives the $N-1$ factor above. This small
finite-fan-in correction is checked in the notebook using the realized weight
rows.

The requested variance must be positive. For a heterogeneous PIF layer, the
initializer uses the population mean $\tau$ rather than a separate value for
every neuron.

This combination is meant to keep activity in a useful range instead of
producing silent neurons or neurons that spike at every step. It is not a
guarantee for every dataset or trained model, so the notebook measures it. It
plots the test-set firing rate of every hidden neuron and reports the exact
number of zero-rate and always-spiking neurons.

The following snapshots are qualitative examples from the dense, unpruned
500-epoch SHD experiments. Each column is one test sample and each row is one
hidden layer. The notebook provides the quantitative firing-rate comparison.

| LIF | PIF |
|:--:|:--:|
| ![LIF hidden-layer activity on five SHD samples](assets/lif_activity_snapshot.png) | ![PIF hidden-layer activity on five SHD samples](assets/pif_activity_snapshot.png) |

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
