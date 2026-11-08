from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from liquid_revision.models import TopologySpec, build_model
from liquid_revision.models.topology import recurrent_mask


def _config(topology: str = "dense"):
    return {
        "input_dim": 19,
        "state_dim": 4,
        "hidden_dim": 8,
        "topology": {"kind": topology, "sparsity": 0.5, "seed": 9},
        "ode": {"max_step": 0.02},
        "ctrnn": {"initial_tau": 0.2, "min_tau": 0.001},
    }


@pytest.mark.parametrize("name", ["mlp", "gru", "lstm", "ctrnn", "odernn"])
def test_native_policy_contract(name: str) -> None:
    torch.manual_seed(2)
    model = build_model(name, _config())
    x = torch.randn(3, 6, 19)
    dt = torch.full((3, 6), 0.02)
    action, state = model.forward_sequence(x, dt)
    assert action.shape == (3, 6, 4)
    assert torch.isfinite(action).all()
    assert state is not None or name == "mlp"


@pytest.mark.parametrize("name", ["ctrnn", "odernn"])
def test_time_aware_native_policy_uses_dt(name: str) -> None:
    torch.manual_seed(3)
    model = build_model(name, _config())
    x = torch.randn(2, 8, 19)
    action_a, _ = model.forward_sequence(x, torch.full((2, 8), 0.01))
    action_b, _ = model.forward_sequence(x, torch.full((2, 8), 0.04))
    assert float((action_a - action_b).abs().max().detach()) > 1e-8


def test_fixed_tau_control_is_not_trainable() -> None:
    model = build_model("ctrnn", _config())
    assert "fixed_tau" in dict(model.named_buffers())
    assert all("tau" not in name for name, _ in model.named_parameters())
    torch.testing.assert_close(model.tau, torch.full((8,), 0.2))


def test_random_sparse_mask_is_deterministic_and_connected() -> None:
    spec = TopologySpec(kind="random_sparse", sparsity=0.8, seed=4)
    first = recurrent_mask(12, spec)
    second = recurrent_mask(12, spec)
    assert torch.equal(first, second)
    assert torch.all(first.sum(dim=0) > 0)
    assert torch.all(first.sum(dim=1) > 0)


@pytest.mark.parametrize("name", ["ltc", "cfc"])
def test_ncps_policy_contract_and_dt_path(name: str) -> None:
    pytest.importorskip("ncps")
    torch.manual_seed(5)
    model = build_model(name, _config())
    x = torch.randn(2, 5, 19)
    action_a, _ = model.forward_sequence(x, torch.full((2, 5), 0.01))
    action_b, _ = model.forward_sequence(x, torch.full((2, 5), 0.04))
    assert action_a.shape == (2, 5, 4)
    assert torch.isfinite(action_a).all()
    assert float((action_a - action_b).abs().max().detach()) > 1e-8


@pytest.mark.parametrize("name", ["ltc", "cfc"])
def test_ncps_batch_safe_adapter_matches_batch_one_reference_and_gradients(name: str) -> None:
    pytest.importorskip("ncps")
    torch.manual_seed(17)
    adapted = build_model(name, _config())
    reference = build_model(name, _config())
    reference.load_state_dict(adapted.state_dict())
    x_adapted = torch.randn(1, 4, 19, requires_grad=True)
    x_reference = x_adapted.detach().clone().requires_grad_(True)
    dt = torch.tensor([[0.01, 0.02, 0.03, 0.04]])
    observed, _ = adapted.forward_sequence(x_adapted, dt)
    upstream, _ = reference.core(x_reference, timespans=dt)
    expected = reference.head(upstream)
    torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-6)
    observed.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(x_adapted.grad, x_reference.grad, rtol=2e-5, atol=2e-6)
    adapted_gradients = dict(adapted.named_parameters())
    reference_gradients = dict(reference.named_parameters())
    for parameter_name in adapted_gradients:
        left = adapted_gradients[parameter_name].grad
        right = reference_gradients[parameter_name].grad
        assert (left is None) == (right is None)
        if left is not None:
            torch.testing.assert_close(left, right, rtol=2e-5, atol=2e-6)
    assert adapted._timespan_mode == "batch_safe_cell_loop"


@pytest.mark.parametrize("name", ["ltc", "cfc"])
def test_ncps_each_batch_member_uses_its_own_dt(name: str) -> None:
    pytest.importorskip("ncps")
    torch.manual_seed(23)
    model = build_model(name, _config())
    shared = torch.randn(1, 5, 19).repeat(2, 1, 1)
    dt = torch.stack([torch.full((5,), 0.01), torch.full((5,), 0.04)])
    output, _ = model.forward_sequence(shared, dt)
    assert float((output[0] - output[1]).abs().max().detach()) > 1e-8

