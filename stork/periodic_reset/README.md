# PIF: periodic reset-and-fire neurons

This package contains four independent pieces:

1. non-leaky integrate-and-fire groups with periodic hard resets;
2. fluctuation-driven initialization for their finite integration windows;
3. a non-leaky accumulator readout for periodic-reset networks;
4. tensor-based effective operation counting for spikes and neuron updates.

## Dynamics

The discrete dynamics are

```text
s[t] = H(u[t] - threshold)
u[t + 1] = (u[t] + I[t]) * (1 - s[t]) * (1 - reset[t])
```

There is no membrane or synaptic leak. At a reset deadline, or after a spike,
the membrane is set to zero. ``tau`` is expressed in seconds and converted to
at least one simulation step when the group is configured. Random phase
offsets desynchronize reset events.

`PIFGroup` uses one shared period. `HeterogeneousPIFGroup`
samples a period per neuron from a Gamma distribution with mean ``tau``.

## Initialization

For fan-in `N`, presynaptic rate `nu`, reset period `tau`, target membrane
mean `mu_u`, and target standard deviation `sigma_u`, the initializer uses

```text
mu_w = 2 * mu_u / (N * nu * tau)
sigma_w^2 = 2 * sigma_u^2 / (N * nu * tau) - mu_w^2
```

The initializer receives one scalar `tau`. For a heterogeneous group, this is
the population mean reset period rather than each neuron's sampled period.

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

## Effective FLOPs

`EffectiveFlopsCounter` operates on explicit tensors instead of assuming a
model architecture or monitor order. It counts a nonzero outgoing weight for
each presynaptic spike and also exposes the observed spike-event count.

`count_pif_neurons` combines active membrane updates, incoming updates, active
periodic resets, and direct-assignment savings. `count_lif_neurons` counts the
leaky membrane and synaptic-current update convention.
`count_non_leaky_readout` preserves the accumulator convention used by the
original dense PIF experiments. The combined `count` method accepts separate
`pif_layers`, `lif_layers`, and `non_leaky_readouts` collections.

The result is an analytical operation count. It is not a wall-clock or
hardware-energy measurement.

## Example

[`examples/06_PeriodicReset_SHD.ipynb`](../../examples/06_PeriodicReset_SHD.ipynb)
compares dense LIF, homogeneous PIF, and heterogeneous PIF networks on SHD.
It checks the initialization statistics, reports test accuracy, and counts
analytical effective operations per test sample.
