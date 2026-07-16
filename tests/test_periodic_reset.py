import math
import unittest

import torch

from stork.connections import Connection
from stork.nodes import InputGroup
from stork.periodic_reset import (
    EffectiveFlops,
    EffectiveFlopsCounter,
    HeterogeneousPIFGroup,
    NonLeakyReadoutGroup,
    PIFGroup,
    PeriodicResetFluctuationDrivenInitializer,
)


class PeriodicResetTests(unittest.TestCase):
    def test_heterogeneous_periods_are_positive_and_configurable(self):
        torch.manual_seed(5)
        group = HeterogeneousPIFGroup(
            shape=32, tau=40e-3, concentration=2.0
        )

        self.assertEqual(group.tau.shape, (32,))
        self.assertTrue(torch.all(group.tau > 0))

        group.configure(1, 20, 2e-3, torch.device("cpu"), torch.float32)
        self.assertTrue(torch.all(group.period_steps >= 1))
        self.assertTrue(torch.all(group.offset >= 0))
        self.assertTrue(torch.all(group.offset < group.period_steps))

    def test_non_leaky_membrane_resets_on_schedule(self):
        group = PIFGroup(tau=6e-3, shape=1)
        group.configure(
            batch_size=1,
            nb_steps=8,
            time_step=2e-3,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        group.offset.fill_(2)
        group.reset_state()

        trace = []
        for _ in range(8):
            group.input.fill_(0.1)
            group.forward()
            trace.append(round(float(group.mem.item()), 4))

        self.assertEqual(int(group.period_steps.item()), 3)
        self.assertEqual(trace, [0.1, 0.2, 0.0, 0.1, 0.2, 0.0, 0.1, 0.2])
        self.assertFalse(hasattr(group, "crit_accum"))
        self.assertFalse(hasattr(group, "grad_accum"))

    def test_phase_is_preserved_when_time_step_changes(self):
        group = PIFGroup(tau=6e-3, shape=1)
        group.phase.fill_(0.5)

        group.configure(1, 8, 2e-3, torch.device("cpu"), torch.float32)
        self.assertEqual(int(group.period_steps.item()), 3)
        self.assertEqual(int(group.offset.item()), 1)

        group.configure(1, 16, 1e-3, torch.device("cpu"), torch.float32)
        self.assertEqual(int(group.period_steps.item()), 6)
        self.assertEqual(int(group.offset.item()), 3)
        self.assertEqual(set(group.state_dict()), {"tau", "threshold", "phase"})

    def test_initializer_uses_scalar_mean_period(self):
        torch.manual_seed(7)
        source = InputGroup(700)
        destination = PIFGroup(tau=40e-3, shape=128)
        destination.tau[:64] = 20e-3
        destination.tau[64:] = 80e-3
        connection = Connection(source, destination)
        initializer = PeriodicResetFluctuationDrivenInitializer(
            nu=15.8,
            tau=40e-3,
            mu_u=0.0,
            sigma_u=1.0,
            center_weights=True,
        )

        mean, standard_deviation = initializer._get_weight_parameters_con(connection)
        expected_standard_deviation = math.sqrt(2 / (700 * 15.8 * 40e-3))
        self.assertEqual(mean, 0.0)
        self.assertAlmostEqual(
            standard_deviation, expected_standard_deviation, places=8
        )

        initializer.initialize(connection)
        row_means = connection.op.weight.detach().mean(dim=1)
        self.assertLess(float(row_means.abs().max()), 1e-7)

    def test_eflops_counter_counts_spikes_and_periodic_updates(self):
        counter = EffectiveFlopsCounter()
        spikes = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]
        )
        weights = torch.tensor([[1.0, 1.0], [2.0, 0.0]])

        membranes = torch.tensor(
            [[[0.0, 0.2], [0.1, 0.3], [0.2, 0.0], [0.0, 0.4]]]
        )
        updating = torch.tensor(
            [[[True, False], [True, True], [False, False], [False, True]]]
        )
        reset_mask = counter.periodic_reset_mask(
            period_steps=torch.tensor([2, 3]),
            offsets=torch.tensor([0, 1]),
            num_steps=4,
            batch_size=1,
        )

        result = counter.count(
            connections=[(spikes, weights)],
            pif_layers=[(membranes, updating, reset_mask)],
        )

        expected_mask = torch.tensor(
            [[[True, False], [False, True], [True, False], [False, False]]]
        )
        self.assertTrue(torch.equal(reset_mask, expected_mask))
        self.assertEqual(counter.count_spikes(spikes), 4)
        self.assertEqual(
            result,
            EffectiveFlops(connection_operations=6, neuron_operations=10),
        )
        self.assertEqual(result.total_operations, 16)

    def test_eflops_counter_counts_lif_updates(self):
        membranes = torch.tensor(
            [[[0.0, 0.2], [0.1, 0.3], [0.2, 0.0]]]
        )
        updating = torch.tensor(
            [[[True, False], [True, False], [True, False]]]
        )

        counter = EffectiveFlopsCounter()
        operations = counter.count_lif_neurons(membranes, updating)
        result = counter.count(lif_layers=[(membranes, updating)])

        # Four active neurons and three updates, including one direct update
        # of an inactive neuron: 5 * 4 + 3 - 3 * 1.
        self.assertEqual(operations, 20)
        self.assertEqual(result, EffectiveFlops(neuron_operations=20))

    def test_counter_counts_reset_from_pre_reset_state(self):
        membranes = torch.tensor([[[0.2], [0.0]]])
        updating = torch.zeros_like(membranes, dtype=torch.bool)
        reset_mask = torch.tensor([[[False], [True]]])

        operations = EffectiveFlopsCounter.count_pif_neurons(
            membranes, updating, reset_mask
        )

        # The membrane stored at the reset step is already zero. The one
        # reset operation applies to the active state from the previous step.
        self.assertEqual(operations, 1)

    def test_non_leaky_readout_accumulates_without_decay(self):
        group = NonLeakyReadoutGroup(shape=2, gamma=0.5, initial_state=-1e-3)
        group.configure(
            batch_size=1,
            nb_steps=2,
            time_step=2e-3,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        group.input = group.states["input"] = torch.tensor([[2.0, -4.0]])
        group.forward()
        self.assertTrue(torch.allclose(group.out, torch.tensor([[0.999, -2.001]])))

        group.input = group.states["input"] = torch.tensor([[2.0, 2.0]])
        group.forward()
        self.assertTrue(torch.allclose(group.out, torch.tensor([[1.999, -1.001]])))

    def test_counter_counts_non_leaky_readout_and_combines_it(self):
        counter = EffectiveFlopsCounter()
        outputs = torch.tensor([[[0.0], [2.0], [2.0]]])
        updating = torch.tensor([[[False], [True], [False]]])

        self.assertEqual(counter.count_non_leaky_readout(outputs, updating), 3)

        result = counter.count(non_leaky_readouts=[(outputs, updating)])
        self.assertEqual(result, EffectiveFlops(neuron_operations=3))

    def test_counter_rejects_mismatched_non_leaky_readout_tensors(self):
        with self.assertRaisesRegex(ValueError, "same shape"):
            EffectiveFlopsCounter.count_non_leaky_readout(
                torch.zeros(1, 2, 3), torch.zeros(1, 2, 2)
            )


if __name__ == "__main__":
    unittest.main()
