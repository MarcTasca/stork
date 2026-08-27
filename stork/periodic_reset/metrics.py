from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EffectiveFlops:
    """Effective operation counts for a spiking-network execution."""

    connection_operations: int = 0
    neuron_operations: int = 0

    @property
    def total_operations(self):
        return self.connection_operations + self.neuron_operations


class EffectiveFlopsCounter:
    """Count operations that are active for a particular spike trace.

    The zero-skipping convention follows Narduzzi et al., "EFLOP: a
    sparsity-aware metric for evaluating computational cost in spiking and
    non-spiking neural networks" (2025), doi:10.1088/2634-4386/addee8. The
    periodic-reset and non-leaky-readout rules are extensions for this package.

    The counter consumes explicit tensors and has no dependency on monitor
    ordering, model architecture, datasets, or a particular accelerator.
    """

    def __init__(self, weight_epsilon=0.0):
        if weight_epsilon < 0:
            raise ValueError("weight_epsilon must be non-negative")
        self.weight_epsilon = weight_epsilon

    @staticmethod
    def count_spikes(spikes):
        """Count non-zero spike events in an observed activity tensor."""

        return int(torch.as_tensor(spikes).ne(0).sum().item())

    def count_connection(self, spikes, weights):
        """Count active synapses for one dense connection.

        ``spikes`` may have any leading dimensions, but its last dimension
        must match the input dimension of ``weights``. A non-zero spike
        activates each non-zero outgoing weight once.
        """

        if spikes.ndim < 1:
            raise ValueError("spikes must have at least one dimension")
        if weights.ndim != 2:
            raise ValueError("weights must have shape (outputs, inputs)")
        if spikes.shape[-1] != weights.shape[1]:
            raise ValueError("spikes and weights have incompatible input dimensions")

        active_weights = weights.abs() > self.weight_epsilon
        fan_out = active_weights.sum(dim=0).to(device=spikes.device)
        operations = (spikes.ne(0).to(torch.int64) * fan_out).sum()
        return int(operations.item())

    def count_connections(self, connections):
        """Count a sequence of ``(presynaptic_spikes, weights)`` pairs."""

        return sum(
            self.count_connection(spikes, weights) for spikes, weights in connections
        )

    @staticmethod
    def periodic_reset_mask(period_steps, offsets, num_steps, batch_size=1):
        """Build the reset schedule used by periodic-reset neuron groups."""

        if num_steps < 0:
            raise ValueError("num_steps must be non-negative")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        periods = torch.as_tensor(period_steps)
        phases = torch.as_tensor(offsets, device=periods.device)
        periods, phases = torch.broadcast_tensors(periods, phases)

        if torch.any(periods <= 0):
            raise ValueError("period_steps must contain only positive values")
        if torch.any(periods != periods.round()):
            raise ValueError("period_steps must contain integer step counts")
        if torch.any(phases != phases.round()):
            raise ValueError("offsets must contain integer step counts")
        if torch.any(phases < 0) or torch.any(phases >= periods):
            raise ValueError("offsets must satisfy 0 <= offset < period_steps")

        periods = periods.to(torch.int64)
        phases = phases.to(torch.int64)
        step_shape = (num_steps,) + (1,) * periods.ndim
        steps = torch.arange(num_steps, device=periods.device).reshape(step_shape)
        first_reset_steps = phases + periods
        elapsed = steps - first_reset_steps.unsqueeze(0)
        schedule = (elapsed >= 0) & (
            torch.remainder(elapsed, periods.unsqueeze(0)) == 0
        )
        return schedule.unsqueeze(0).expand((batch_size,) + schedule.shape)

    @staticmethod
    def count_pif_neurons(membranes, updating, reset_mask):
        """Count PIF membrane, update, and periodic-reset operations.

        Tensors use ``(batch, time, ...)`` layout. ``updating`` marks neurons
        receiving at least one active synapse, while ``reset_mask`` may omit
        the batch dimension and is broadcast across samples.
        """

        if membranes.ndim < 2:
            raise ValueError("membranes must have shape (batch, time, ...)")
        if updating.shape != membranes.shape:
            raise ValueError("updating must have the same shape as membranes")

        active = membranes.ne(0)
        updating = updating.to(device=membranes.device, dtype=torch.bool)
        reset_mask = torch.as_tensor(
            reset_mask, device=membranes.device, dtype=torch.bool
        )
        try:
            reset_mask = torch.broadcast_to(reset_mask, active.shape)
        except RuntimeError as error:
            raise ValueError("reset_mask is not broadcastable to membranes") from error

        previously_active = torch.zeros_like(active)
        previously_active[:, 1:] = active[:, :-1]
        newly_active = active & ~previously_active

        membrane_operations = 2 * active.sum()
        update_operations = updating.sum()
        reset_operations = (previously_active & reset_mask).sum()
        direct_assignment_savings = 2 * newly_active.sum()
        operations = (
            membrane_operations
            + update_operations
            + reset_operations
            - direct_assignment_savings
        )
        return int(operations.item())

    @staticmethod
    def count_lif_neurons(membranes, updating):
        """Count leaky integrate-and-fire neuron operations.

        Tensors use ``(batch, time, ...)`` layout. Synaptic-current activity is
        not tracked separately: membrane activity is the proxy for joint
        membrane and synaptic dynamics, and the five-operation active-state
        factor includes both. An incoming update adds one operation. When an
        inactive membrane receives an update, direct assignment saves three
        operations.
        """

        if membranes.ndim < 2:
            raise ValueError("membranes must have shape (batch, time, ...)")
        if updating.shape != membranes.shape:
            raise ValueError("updating must have the same shape as membranes")

        updating = updating.to(device=membranes.device, dtype=torch.bool)
        membrane_active = membranes.ne(0)
        active = membrane_active | updating
        inactive_but_updating = updating & ~membrane_active
        operations = (
            5 * active.sum()
            + updating.sum()
            - 3 * inactive_but_updating.sum()
        )
        return int(operations.item())

    @staticmethod
    def count_non_leaky_readout(outputs, updating):
        """Count a non-leaky accumulator readout.

        This preserves the effective-operation convention used by the dense
        PIF experiments: two operations for an active accumulator, one for an
        incoming update, and two saved operations when an inactive
        accumulator can receive its first value by direct assignment.

        Tensors use ``(batch, time, ...)`` layout.
        """

        if outputs.ndim < 2:
            raise ValueError("outputs must have shape (batch, time, ...)")
        if updating.shape != outputs.shape:
            raise ValueError("updating must have the same shape as outputs")

        active = outputs.ne(0)
        updating = updating.to(device=outputs.device, dtype=torch.bool)
        previously_active = torch.zeros_like(active)
        previously_active[:, 1:] = active[:, :-1]
        newly_active = active & ~previously_active

        operations = 2 * active.sum() + updating.sum() - 2 * newly_active.sum()
        return int(operations.item())

    def count(
        self,
        connections=(),
        pif_layers=(),
        lif_layers=(),
        non_leaky_readouts=(),
    ):
        """Return combined connection, neuron, and readout operation counts."""

        connection_operations = self.count_connections(connections)
        pif_operations = sum(
            self.count_pif_neurons(membranes, updating, reset_mask)
            for membranes, updating, reset_mask in pif_layers
        )
        lif_operations = sum(
            self.count_lif_neurons(membranes, updating)
            for membranes, updating in lif_layers
        )
        readout_operations = sum(
            self.count_non_leaky_readout(outputs, updating)
            for outputs, updating in non_leaky_readouts
        )
        neuron_operations = pif_operations + lif_operations + readout_operations
        return EffectiveFlops(connection_operations, neuron_operations)
