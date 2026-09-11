import importlib.util
import sys

import numpy as np
import pytest

from goav.gradient import influence_from_token_scores, score_geometry, score_geometry_microbatched
from goav.loss import causal_lm_token_mean_loss, response_token_mask, token_mean_loss
from goav.models import DeterministicPolicyBackend, load_registered_model, load_transformers_backend
from goav.trainer import attach_lora


def test_influence_uses_one_global_trainer_token_denominator():
    token_gradients = np.array([
        [[1.0, 2.0], [3.0, 5.0], [100.0, 100.0]],
        [[7.0, 11.0], [100.0, 100.0], [100.0, 100.0]],
    ])
    mask = np.array([[1, 1, 0], [1, 0, 0]])
    influence = influence_from_token_scores(token_gradients, mask)
    # Negative score contributions divided by exactly three active trainer tokens.
    np.testing.assert_allclose(influence, [[-4 / 3, -7 / 3], [-7 / 3, -11 / 3]])
    rewards = np.array([0.25, 0.75])
    direct = -np.array([1 * 0.25 + 3 * 0.25 + 7 * 0.75, 2 * 0.25 + 5 * 0.25 + 11 * 0.75]) / 3
    np.testing.assert_allclose(influence @ rewards, direct)


def test_response_mask_uses_shifted_causal_positions_and_attention():
    attention = np.array([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])
    mask = response_token_mask(attention, np.array([2, 4]))
    np.testing.assert_array_equal(mask, [[0, 1, 1, 0], [0, 0, 0, 1]])
    left_padded = np.array([[0, 0, 1, 1, 1, 1]])
    np.testing.assert_array_equal(response_token_mask(left_padded, np.array([2])), [[0, 0, 0, 1, 1]])


def test_optional_model_stack_is_lazy_and_missing_dependency_is_actionable():
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules
    if importlib.util.find_spec("transformers") is None:
        with pytest.raises(RuntimeError, match="transformers.*model"):
            load_transformers_backend("Qwen/Qwen2.5-Coder-7B-Instruct", "abc1234")
        with pytest.raises(RuntimeError, match="transformers.*model"):
            load_registered_model("primary", "abc1234", quantization="nf4")
    if importlib.util.find_spec("peft") is None:
        with pytest.raises(RuntimeError, match="PEFT.*QLoRA"):
            attach_lora(object())


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="optional torch test; CPU CI downloads nothing")
def test_tiny_torch_l_times_y_equals_direct_clean_token_mean_gradient():
    import torch
    from types import SimpleNamespace

    torch.manual_seed(0)
    class TinyCausalLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(7, 3)
            self.head = torch.nn.Linear(3, 7, bias=False)

        def forward(self, input_ids):
            return SimpleNamespace(logits=self.head(self.embedding(input_ids)))

    model = TinyCausalLM()
    inputs = torch.tensor([[1, 2, 3, 4, 0], [1, 5, 6, 2, 3]])
    attention = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])
    mask = response_token_mask(attention, torch.tensor([2, 4]))
    rewards = torch.tensor([0.2, 0.9])
    geometry = score_geometry(model, inputs, mask)
    direct_loss = causal_lm_token_mean_loss(model(inputs), inputs, rewards, mask)
    direct = torch.autograd.grad(direct_loss, tuple(model.parameters()))
    flat_direct = torch.cat([item.reshape(-1) for item in direct]).detach().numpy()
    np.testing.assert_allclose(geometry @ rewards.numpy(), flat_direct, atol=1e-6)
    streamed = score_geometry_microbatched(model, inputs, mask, microbatch_size=1)
    np.testing.assert_allclose(streamed, geometry, atol=1e-6)
    from goav.trainer import OnlineTrainingBatch, causal_qlora_one_update
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    before = torch.cat([parameter.detach().reshape(-1) for parameter in model.parameters()]).clone()
    evidence = causal_qlora_one_update(model, optimizer, OnlineTrainingBatch(inputs, rewards, mask))
    after = torch.cat([parameter.detach().reshape(-1) for parameter in model.parameters()])
    assert evidence.parameter_changed and evidence.active_tokens == 3
    assert not torch.equal(before, after)


def test_deterministic_backend_returns_real_shapes_without_optional_imports():
    backend = DeterministicPolicyBackend(feature_dim=3)
    result = backend.analyze_candidate_ids(("a", "b", "c", "d"))
    assert result.imputed.shape == (4,)
    assert result.covariance.shape == (4, 4)
    assert result.influence.shape == (3, 4)
    assert result.candidate_token_ids.shape == result.response_mask.shape
    assert result.loss_denominator == int(result.response_mask.sum())
