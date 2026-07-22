import torch

from stork import activations
from stork.nodes.base import CellGroup


class HeterogeneousPIFGroup(CellGroup):
    """Heterogeneous periodic reset-and-fire (PIF) neurons.

    Reset periods are sampled per neuron from a Gamma distribution. ``tau``
    is the mean reset period in seconds and is converted to simulation steps
    when the group is configured.
    """

    def __init__(
        self,
        shape,
        tau,
        concentration=2.0,
        threshold=1.0,
        diff_reset=False,
        activation=activations.SuperSpike,
        dropout_p=0.0,
        stateful=False,
        name="HeterogeneousPIFGroup",
        regularizers=None,
        **kwargs,
    ):
        if tau <= 0:
            raise ValueError("tau must be positive")
        if concentration <= 0:
            raise ValueError("concentration must be positive")
        if threshold <= 0:
            raise ValueError("threshold must be positive")

        super().__init__(
            shape,
            dropout_p=dropout_p,
            stateful=stateful,
            name=name,
            regularizers=regularizers,
            **kwargs,
        )
        self.spk_nl = activation.apply
        self.diff_reset = diff_reset
        self.mem = None

        distribution = torch.distributions.Gamma(
            concentration=torch.tensor(float(concentration)),
            rate=torch.tensor(float(concentration / tau)),
        )
        self.register_buffer("tau", distribution.sample(self.shape))
        self.register_buffer("threshold", torch.full(self.shape, float(threshold)))
        self.register_buffer("phase", torch.rand(self.shape))
        self.register_buffer(
            "period_steps", torch.empty(0, dtype=torch.int64), persistent=False
        )
        self.register_buffer(
            "offset", torch.zeros(self.shape, dtype=torch.int64), persistent=False
        )

    def _periods_in_steps(self, time_step, device):
        return torch.clamp(torch.round(self.tau.to(device) / time_step), min=1).to(
            torch.int64
        )

    def configure(self, batch_size, nb_steps, time_step, device, dtype):
        if time_step <= 0:
            raise ValueError("time_step must be positive")
        super().configure(batch_size, nb_steps, time_step, device, dtype)

    def get_spike_and_reset(self, membrane_minus_threshold):
        out = self.spk_nl(membrane_minus_threshold)
        reset = out if self.diff_reset else out.detach()
        return out, reset

    def forward(self):
        periodic_reset = self.current_step >= self.next_reset_step
        new_out, spike_reset = self.get_spike_and_reset(self.mem - self.threshold)
        new_mem = (
            (self.mem + self.input)
            * (1.0 - spike_reset)
            * (~periodic_reset).to(self.mem)
        )

        self.mem = self.states["mem"] = new_mem
        self.out = self.states["out"] = new_out
        self.next_reset_step += self.period_steps * periodic_reset
        self.current_step += 1

    def reset_state(self, batch_size=None):
        self.period_steps = self._periods_in_steps(self.time_step, self.device)
        self.offset = torch.floor(
            self.phase.to(self.device) * self.period_steps
        ).to(torch.int64)
        super().reset_state(batch_size)
        self.mem = self.get_state_tensor("mem", state=self.mem)
        self.out = self.states["out"] = torch.zeros(
            self.int_shape, device=self.device, dtype=self.dtype
        )
        continue_schedule = (
            self.stateful
            and hasattr(self, "next_reset_step")
            and hasattr(self, "current_step")
            and self.next_reset_step.shape == self.int_shape
        )
        if not continue_schedule:
            self.next_reset_step = (self.offset + self.period_steps).expand(
                self.int_shape
            ).clone()
            self.current_step = 0


class PIFGroup(HeterogeneousPIFGroup):
    """Periodic reset-and-fire neurons with a shared period and random phases."""

    def __init__(self, shape, tau, *args, **kwargs):
        kwargs.setdefault("name", "PIFGroup")
        super().__init__(shape, tau, *args, **kwargs)
        self.tau = torch.full(self.shape, float(tau))
