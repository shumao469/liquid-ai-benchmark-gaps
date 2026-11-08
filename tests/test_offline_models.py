from __future__ import annotations

import pytest
import torch

from liquid_revision.offline.models import build_offline_model, resolve_hidden_dim


@pytest.mark.parametrize("name", ["lstm", "ode-rnn", "ltc", "cfc"])
def test_mandatory_offline_models_share_classifier_interface(name: str) -> None:
    torch.manual_seed(7)
    model = build_offline_model(
        name, input_dim=5, hidden_dim=6, output_dim=3, ode_max_step=0.5
    )
    values = torch.randn(2, 4, 5)
    dt = torch.full((2, 4), 0.25)
    logits = model(values, dt)
    assert logits.shape == (2, 3)
    assert torch.isfinite(logits).all()
    assert model.trainable_parameters > 0


@pytest.mark.parametrize("name", ["ode-rnn", "ltc", "cfc"])
def test_elapsed_time_changes_time_aware_model_output(name: str) -> None:
    torch.manual_seed(11)
    model = build_offline_model(
        name, input_dim=4, hidden_dim=5, output_dim=2, ode_max_step=0.5
    ).eval()
    values = torch.randn(1, 5, 4)
    short = torch.full((1, 5), 0.10)
    long = torch.full((1, 5), 0.85)
    with torch.inference_mode():
        short_output = model(values, short)
        long_output = model(values, long)
    assert model.uses_elapsed_time
    assert not torch.allclose(short_output, long_output, atol=1e-7, rtol=1e-6)


def test_lstm_is_explicitly_elapsed_time_agnostic() -> None:
    torch.manual_seed(13)
    model = build_offline_model("lstm", input_dim=4, hidden_dim=5, output_dim=2).eval()
    values = torch.randn(1, 5, 4)
    with torch.inference_mode():
        first = model(values, torch.full((1, 5), 0.1))
        second = model(values, torch.full((1, 5), 0.9))
    assert not model.uses_elapsed_time
    assert torch.equal(first, second)


def test_lengths_select_last_unpadded_state() -> None:
    torch.manual_seed(17)
    model = build_offline_model("lstm", input_dim=3, hidden_dim=4, output_dim=2).eval()
    values = torch.randn(2, 6, 3)
    dt = torch.ones(2, 6)
    with torch.inference_mode():
        features = model.forward_features(values, dt)
        logits = model(values, dt, lengths=torch.tensor([2, 5]))
        expected = model.head(torch.stack([features[0, 1], features[1, 4]]))
    assert torch.allclose(logits, expected)


def test_nonpositive_elapsed_time_is_rejected() -> None:
    model = build_offline_model("ode-rnn", input_dim=3, hidden_dim=4, output_dim=2)
    dt = torch.ones(1, 3)
    dt[0, 1] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        model(torch.randn(1, 3, 3), dt)


@pytest.mark.parametrize("name", ["ltc", "cfc"])
def test_vectorized_liquid_matches_ncps_batch1_output_and_gradients(name: str) -> None:
    torch.manual_seed(23)
    model = build_offline_model(name, input_dim=4, hidden_dim=5, output_dim=2)
    values = torch.randn(1, 6, 4)
    dt = torch.rand(1, 6) + 0.1

    vectorized = model.encoder(values, dt)
    vectorized.square().sum().backward()
    vectorized_gradients = {
        key: parameter.grad.detach().clone()
        for key, parameter in model.encoder.named_parameters()
        if parameter.grad is not None
    }
    model.encoder.zero_grad(set_to_none=True)

    reference, _ = model.encoder.core(values, timespans=dt)
    reference.square().sum().backward()
    reference_gradients = {
        key: parameter.grad.detach().clone()
        for key, parameter in model.encoder.named_parameters()
        if parameter.grad is not None
    }
    assert torch.allclose(vectorized, reference, atol=1e-6, rtol=1e-5)
    assert vectorized_gradients.keys() == reference_gradients.keys()
    for key in vectorized_gradients:
        assert torch.allclose(
            vectorized_gradients[key], reference_gradients[key], atol=1e-6, rtol=1e-5
        ), key


@pytest.mark.parametrize("name", ["ltc", "cfc"])
def test_vectorized_liquid_handles_batchwise_elapsed_times(name: str) -> None:
    torch.manual_seed(29)
    model = build_offline_model(name, input_dim=3, hidden_dim=5, output_dim=2).eval()
    shared = torch.randn(1, 5, 3).repeat(2, 1, 1)
    dt = torch.stack([torch.full((5,), 0.1), torch.full((5,), 0.9)])
    with torch.inference_mode():
        output = model(shared, dt)
    assert output.shape == (2, 2)
    assert not torch.allclose(output[0], output[1], atol=1e-7, rtol=1e-6)


def test_task_specific_hidden_dimension_resolution() -> None:
    config = {
        "hidden_dim": 8,
        "hidden_dims": {
            "har": {"lstm": 64, "cfc": 32},
            "physionet2012": {"lstm": 64, "cfc": 51},
        },
    }
    assert resolve_hidden_dim(config, "har", "cfc") == 32
    assert resolve_hidden_dim(config, "physionet2012", "cfc") == 51
    assert resolve_hidden_dim(config, "har", "ode-rnn") == 8

