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
        group.phase.fill_(0.75)
        group.configure(
            batch_size=1,
            nb_steps=11,
            time_step=2e-3,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        self.assertEqual(int(group.next_reset_step.item()), 5)
        trace = []
        for _ in range(11):
            group.input.fill_(0.1)
            group.forward()
            trace.append(round(float(group.mem.item()), 4))

        self.assertEqual(int(group.period_steps.item()), 3)
        self.assertEqual(
            trace,
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.0, 0.1, 0.2, 0.0, 0.1, 0.2],
        )
        self.assertFalse(hasattr(group, "crit_accum"))
        self.assertFalse(hasattr(group, "grad_accum"))

    def test_spike_reset_clears_membrane_and_discards_input(self):
        for diff_reset in (False, True):
            with self.subTest(diff_reset=diff_reset):
                group = PIFGroup(tau=20e-3, shape=1, diff_reset=diff_reset)
                group.phase.fill_(0.5)
                group.configure(
                    batch_size=1,
                    nb_steps=2,
                    time_step=2e-3,
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                )
                group.mem.fill_(1.1)

                group.input.fill_(0.4)
                group.forward()
                self.assertEqual(float(group.out.item()), 1.0)
                self.assertEqual(float(group.mem.item()), 0.0)

                group.input.fill_(0.2)
                group.forward()
                self.assertEqual(float(group.out.item()), 0.0)
                self.assertAlmostEqual(float(group.mem.item()), 0.2, places=7)

    def test_diff_reset_controls_gradient_through_spike_reset(self):
        def gradient_from_initial_membrane(diff_reset):
            group = PIFGroup(tau=20e-3, shape=1, diff_reset=diff_reset)
            group.configure(
                batch_size=1,
                nb_steps=1,
                time_step=2e-3,
                device=torch.device("cpu"),
                dtype=torch.float64,
            )
            initial_membrane = torch.tensor(
                [[1.1]], dtype=torch.float64, requires_grad=True
            )
            group.mem = group.states["mem"] = initial_membrane
            group.input.fill_(0.2)

            group.forward()
            group.mem.sum().backward()
            return float(initial_membrane.grad.item())

        detached_reset_gradient = gradient_from_initial_membrane(diff_reset=False)
        differentiable_reset_gradient = gradient_from_initial_membrane(
            diff_reset=True
        )

        # The neuron spikes because its initial membrane is above threshold.
        # Detaching the reset blocks that path; a differentiable reset keeps
        # the surrogate spike gradient connected to the initial membrane.
        self.assertEqual(detached_reset_gradient, 0.0)
        self.assertNotEqual(differentiable_reset_gradient, 0.0)

    def test_forward_keeps_states_in_the_configured_low_precision_dtype(self):
        for requested_dtype in (torch.float16, torch.bfloat16):
            with self.subTest(requested_dtype=requested_dtype):
                group = PIFGroup(tau=20e-3, shape=1)
                group.configure(
                    batch_size=1,
                    nb_steps=1,
                    time_step=2e-3,
                    device=torch.device("cpu"),
                    dtype=requested_dtype,
                )
                group.input.fill_(0.1)

                group.forward()

                self.assertEqual(group.threshold.dtype, requested_dtype)
                self.assertEqual(group.mem.dtype, requested_dtype)
                self.assertEqual(group.out.dtype, requested_dtype)

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

    def test_stateful_schedule_continues_across_state_resets(self):
        def make_group():
            group = PIFGroup(tau=6e-3, shape=1, stateful=True)
            group.phase.fill_(0.75)
            group.configure(
                batch_size=1,
                nb_steps=11,
                time_step=2e-3,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            return group

        def run_steps(group, num_steps):
            trace = []
            for _ in range(num_steps):
                group.input.fill_(0.1)
                group.forward()
                trace.append(round(float(group.mem.item()), 4))
            return trace

        continuous = make_group()
        continuous_trace = run_steps(continuous, 11)

        chunked = make_group()
        chunked_trace = run_steps(chunked, 4)
        self.assertEqual(chunked.current_step, 4)
        self.assertEqual(int(chunked.next_reset_step.item()), 5)

        chunked.reset_state()
        self.assertEqual(chunked.current_step, 4)
        self.assertEqual(int(chunked.next_reset_step.item()), 5)
        chunked_trace.extend(run_steps(chunked, 7))

        self.assertEqual(chunked_trace, continuous_trace)
        self.assertEqual(chunked.current_step, continuous.current_step)
        self.assertTrue(
            torch.equal(chunked.next_reset_step, continuous.next_reset_step)
        )

    def test_reset_state_refreshes_loaded_schedule(self):
        for group_class in (PIFGroup, HeterogeneousPIFGroup):
            with self.subTest(group_class=group_class.__name__):
                saved = group_class(shape=2, tau=40e-3)
                saved.tau.fill_(40e-3)
                saved.phase.copy_(torch.tensor([0.25, 0.75]))

                loaded = group_class(shape=2, tau=40e-3)
                loaded.phase.zero_()
                loaded.configure(
                    batch_size=1,
                    nb_steps=40,
                    time_step=2e-3,
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                )
                loaded.load_state_dict(saved.state_dict())
                loaded.reset_state()

                expected_periods = torch.tensor([20, 20])
                expected_offsets = torch.tensor([5, 15])
                self.assertTrue(torch.equal(loaded.period_steps, expected_periods))
                self.assertTrue(torch.equal(loaded.offset, expected_offsets))
                self.assertTrue(
                    torch.equal(
                        loaded.next_reset_step[0],
                        expected_offsets + expected_periods,
                    )
                )

    def test_stateful_group_rebuilds_schedule_from_loaded_phase(self):
        checkpoint_phase = torch.tensor([0.25, 0.75])
        expected_period_steps = torch.tensor([20, 20])
        expected_offsets = torch.tensor([5, 15])
        expected_next_reset_steps = torch.tensor([25, 35])

        for group_class in (PIFGroup, HeterogeneousPIFGroup):
            with self.subTest(group_class=group_class.__name__):
                checkpoint_group = group_class(
                    shape=2, tau=40e-3, stateful=True
                )
                checkpoint_group.tau.fill_(40e-3)
                checkpoint_group.phase.copy_(checkpoint_phase)

                restored_group = group_class(
                    shape=2, tau=40e-3, stateful=True
                )
                restored_group.tau.fill_(40e-3)
                restored_group.phase.zero_()
                restored_group.configure(
                    batch_size=1,
                    nb_steps=40,
                    time_step=2e-3,
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                )
                for _ in range(3):
                    restored_group.forward()

                self.assertEqual(restored_group.current_step, 3)
                torch.testing.assert_close(
                    restored_group.next_reset_step[0],
                    torch.tensor([20, 20]),
                    rtol=0,
                    atol=0,
                )

                restored_group.load_state_dict(checkpoint_group.state_dict())
                restored_group.reset_state()

                # Loading changes the phase-derived offsets. The old stateful
                # timeline must restart rather than continue with [20, 20].
                self.assertEqual(restored_group.current_step, 0)
                torch.testing.assert_close(
                    restored_group.period_steps,
                    expected_period_steps,
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    restored_group.offset,
                    expected_offsets,
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    restored_group.next_reset_step[0],
                    expected_next_reset_steps,
                    rtol=0,
                    atol=0,
                )

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

    def test_initializer_rejects_non_positive_weight_variance(self):
        connection = Connection(InputGroup(2), PIFGroup(tau=1.0, shape=1))
        initializer = PeriodicResetFluctuationDrivenInitializer(
            nu=1.0,
            tau=1.0,
            mu_u=10.0,
            sigma_u=1.0,
        )

        with self.assertRaisesRegex(ValueError, "non-positive variance"):
            initializer.initialize(connection)

    def test_initializer_rejects_centering_sparse_weights(self):
        with self.assertRaisesRegex(
            ValueError, "incompatible with sparse initialization"
        ):
            PeriodicResetFluctuationDrivenInitializer(
                nu=15.8,
                tau=40e-3,
                center_weights=True,
                sparseness=0.5,
            )

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
            [[[False, False], [False, False], [True, False], [False, False]]]
        )
        self.assertTrue(torch.equal(reset_mask, expected_mask))
        self.assertEqual(counter.count_spikes(spikes), 4)
        self.assertEqual(
            result,
            EffectiveFlops(connection_operations=6, neuron_operations=9),
        )
        self.assertEqual(result.total_operations, 15)

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

        # Four active membranes, plus one inactive membrane receiving an
        # update: 5 * 5 + 3 - 3 * 1.
        self.assertEqual(operations, 25)
        self.assertEqual(result, EffectiveFlops(neuron_operations=25))

    def test_eflops_counter_counts_inactive_lif_update(self):
        membranes = torch.zeros(1, 1, 1)
        updating = torch.ones_like(membranes, dtype=torch.bool)

        operations = EffectiveFlopsCounter.count_lif_neurons(
            membranes, updating
        )

        # The update activates the five-operation path, while direct
        # assignment saves three operations: 5 + 1 - 3.
        self.assertEqual(operations, 3)

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

    def test_periodic_reset_mask_is_identical_for_every_batch_item(self):
        mask = EffectiveFlopsCounter.periodic_reset_mask(
            period_steps=torch.tensor([2, 3]),
            offsets=torch.tensor([0, 1]),
            num_steps=7,
            batch_size=3,
        )

        # Neuron 0 first resets at step 2 and repeats every 2 steps. Neuron 1
        # has offset 1, so it first resets at 1 + 3 = step 4.
        expected_single_item = torch.tensor(
            [
                [False, False],
                [False, False],
                [True, False],
                [False, False],
                [True, True],
                [False, False],
                [True, False],
            ]
        )
        expected_batch = expected_single_item.unsqueeze(0).expand(3, -1, -1)

        torch.testing.assert_close(
            mask,
            expected_batch,
            rtol=0,
            atol=0,
        )

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
