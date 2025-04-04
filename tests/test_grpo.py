import copy

import torch

from grpo_math.grpo import Rollout, group_advantages, grpo_loss, sample_groups
from grpo_math.policy import TinyPolicy, Tokenizer


def test_group_relative_normalization_and_zero_variance():
    rewards = torch.tensor([1.0, 3.0, 5.0, 7.0])
    advantages = group_advantages(rewards)
    assert torch.allclose(advantages, (rewards - 4) / (rewards.std(unbiased=False) + 1e-6))
    assert advantages.mean().abs() < 1e-6
    assert torch.equal(group_advantages(torch.ones(4)), torch.zeros(4))


def test_rollout_log_probs_match_re_evaluation_and_include_eos():
    torch.manual_seed(5)
    policy = TinyPolicy(hidden_size=16)
    tokenizer = Tokenizer()
    ids, old_logp = policy.rollout("2+2?", tokenizer, 5, torch.Generator().manual_seed(3))
    new_logp, entropy, all_logp = policy.evaluate("2+2?", ids, tokenizer)
    assert len(old_logp) == len(ids) + 1
    assert torch.allclose(old_logp, new_logp)
    assert entropy.shape == old_logp.shape
    assert all_logp.shape == (len(old_logp), len(tokenizer.tokens))


def test_clipped_grpo_loss_has_real_advantage_gradient_and_metrics():
    torch.manual_seed(8)
    policy = TinyPolicy(hidden_size=16)
    reference = copy.deepcopy(policy)
    tokenizer = Tokenizer()
    rollouts = []
    for text, advantage in [("12", 1.0), ("9", -1.0)]:
        ids = tokenizer.encode(text)
        old_logp, _, _ = policy.evaluate("question", ids, tokenizer)
        rollouts.append(Rollout("question", "12", ids, old_logp.detach(), 1.0, advantage))
    loss, metrics = grpo_loss(policy, reference, rollouts, beta=0.1)
    loss.backward()
    assert loss.isfinite()
    assert metrics["kl"] == 0.0
    assert metrics["entropy"] > 0
    assert metrics["length"] == 2.5
    assert metrics["advantage_sum"] == 0.0
    assert policy.head.weight.grad is not None
    assert policy.head.weight.grad.abs().sum() > 0


def test_clipping_caps_large_positive_ratio():
    torch.manual_seed(9)
    policy = TinyPolicy(hidden_size=16)
    reference = copy.deepcopy(policy)
    ids = Tokenizer().encode("12")
    new_logp, _, _ = policy.evaluate("question", ids, Tokenizer())
    old_logp = new_logp.detach() - torch.log(torch.tensor(2.0))
    rollout = Rollout("question", "12", ids, old_logp, 1.0, 1.0)
    _, metrics = grpo_loss(policy, reference, [rollout], beta=0.0)
    assert abs(metrics["objective"] + 1.2) < 1e-6


def test_sampled_groups_use_verifier_and_center_each_group():
    policy = TinyPolicy(hidden_size=16)
    examples = [{"question": "six plus six", "answer": "12"}]
    rollouts = sample_groups(policy, examples, group_size=4, max_length=4, seed=1)
    assert len(rollouts) == 4
    assert abs(sum(item.advantage for item in rollouts)) < 1e-5
    assert all(item.reward in (0.0, 1.0, 1.2) for item in rollouts)
