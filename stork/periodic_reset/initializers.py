from numbers import Real

import torch

from stork.initializers import Initializer


class PeriodicResetFluctuationDrivenInitializer(Initializer):
    """Fluctuation-driven initialization for periodic-reset IF groups.

    Under independent Poisson input and a uniformly observed integration age,
    the parameters match the phase-averaged conditional membrane mean and
    variance. They do not include the variance of the conditional mean across
    reset phase; see ``stork/periodic_reset/README.md`` for the derivation.

    ``tau`` is the scalar mean reset period in seconds. The same population
    mean is used for homogeneous and heterogeneous destination groups.
    """

    def __init__(
        self,
        nu,
        tau,
        mu_u=0.0,
        sigma_u=1.0,
        center_weights=False,
        **kwargs,
    ):
        super().__init__(scaling=None, **kwargs)
        if nu <= 0:
            raise ValueError("nu must be positive")
        if not isinstance(tau, Real) or tau <= 0:
            raise ValueError("tau must be a positive scalar")
        if sigma_u <= 0:
            raise ValueError("sigma_u must be positive")

        self.mu_u = mu_u
        self.sigma_u = sigma_u
        self.nu = nu
        self.tau = tau
        self.center_weights = center_weights

    def _get_weight_parameters_con(self, connection):
        weights = connection.op.weight
        if weights.ndim != 2:
            raise ValueError(
                "periodic-reset initialization requires a dense connection"
            )

        _, fan_in = weights.shape
        mu_weight = (2 * self.mu_u) / (fan_in * self.nu * self.tau)
        variance = (
            (2 * self.sigma_u**2) / (fan_in * self.nu * self.tau)
            - mu_weight**2
        )
        if variance <= 0:
            raise ValueError(
                "requested membrane statistics produce non-positive variance"
            )
        return mu_weight, variance**0.5

    def _get_weights(self, connection, mu_weight, sigma_weight):
        shape = connection.op.weight.shape
        weights = (
            torch.randn(
                shape,
                device=connection.op.weight.device,
                dtype=connection.op.weight.dtype,
            )
            * sigma_weight
            + mu_weight
        )
        if self.center_weights:
            weights = weights - weights.mean(dim=1, keepdim=True) + mu_weight
        return weights

    def initialize_connection(self, connection):
        parameters = self._get_weight_parameters_con(connection)
        self._set_weights_and_bias(
            connection, self._get_weights(connection, *parameters)
        )

    def initialize_layer(self, *args, **kwargs):
        raise NotImplementedError(
            "PeriodicResetFluctuationDrivenInitializer does not initialize layers"
        )
