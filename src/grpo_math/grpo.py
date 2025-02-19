from dataclasses import dataclass

import torch

from .policy import TinyPolicy, Tokenizer
from .rewards import arithmetic_reward


@dataclass
class Rollout:
    prompt: str
    expected: str
    ids: list[int]
    old_log_probs: torch.Tensor
    reward: float
    advantage: float = 0.0


def group_advantages(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if rewards.ndim != 1 or len(rewards) < 2:
        raise ValueError("GRPO needs a one-dimensional group of at least two rewards")
    centered = rewards - rewards.mean()
    scale = rewards.std(unbiased=False)
    if scale < eps:
        return torch.zeros_like(rewards)
    return centered / (scale + eps)


def sample_groups(
    policy: TinyPolicy,
    examples: list[dict[str, str]],
    group_size: int,
    max_length: int,
    seed: int,
) -> list[Rollout]:
    tokenizer = Tokenizer()
    generator = torch.Generator().manual_seed(seed)
    rollouts = []
    with torch.no_grad():
        for example in examples:
            group = []
            for _ in range(group_size):
                ids, log_probs = policy.rollout(
                    example["question"], tokenizer, max_length, generator
                )
                completion = tokenizer.decode(ids)
                group.append(
                    Rollout(
                        example["question"],
                        example["answer"],
                        ids,
                        log_probs,
                        arithmetic_reward(completion, example["answer"]),
                    )
                )
            advantages = group_advantages(torch.tensor([item.reward for item in group]))
            for item, advantage in zip(group, advantages, strict=True):
                item.advantage = float(advantage)
            rollouts.extend(group)
    return rollouts


def grpo_loss(
    policy: TinyPolicy,
    reference: TinyPolicy,
    rollouts: list[Rollout],
    clip: float = 0.2,
    beta: float = 0.02,
) -> tuple[torch.Tensor, dict[str, float]]:
    tokenizer = Tokenizer()
    objectives, kls, entropies, lengths = [], [], [], []
    with torch.no_grad():
        reference.eval()
    for rollout in rollouts:
        new_logp, entropy, all_logp = policy.evaluate(rollout.prompt, rollout.ids, tokenizer)
        with torch.no_grad():
            ref_logp, _, ref_all_logp = reference.evaluate(rollout.prompt, rollout.ids, tokenizer)
        old_logp = rollout.old_log_probs
        if len(old_logp) != len(new_logp):
            raise ValueError("behavior and training token lengths differ")
        ratio = (new_logp - old_logp).exp()
        advantage = torch.tensor(rollout.advantage)
        unclipped = ratio * advantage
        clipped = ratio.clamp(1 - clip, 1 + clip) * advantage
        objectives.append(-torch.minimum(unclipped, clipped))
        delta = ref_logp - new_logp
        kls.append(delta.exp() - delta - 1)
        entropies.append(entropy)
        lengths.append(len(new_logp))
    tokens = sum(lengths)
    objective = torch.cat(objectives).sum() / tokens
    sampled_kl = torch.cat(kls).sum() / tokens
    loss = objective + beta * sampled_kl
    with torch.no_grad():
        exact_kl_numerator = torch.tensor(0.0)
        for rollout in rollouts:
            _, _, policy_logp = policy.evaluate(rollout.prompt, rollout.ids, tokenizer)
            _, _, ref_logp = reference.evaluate(rollout.prompt, rollout.ids, tokenizer)
            exact_kl_numerator += (policy_logp.exp() * (policy_logp - ref_logp)).sum()
    metrics = {
        "reward": sum(item.reward for item in rollouts) / len(rollouts),
        "accuracy": sum(item.reward > 0 for item in rollouts) / len(rollouts),
        "kl": float(exact_kl_numerator / tokens),
        "entropy": float(torch.cat(entropies).mean()),
        "length": sum(lengths) / len(lengths),
        "advantage_sum": sum(item.advantage for item in rollouts),
        "objective": float(objective.detach()),
    }
    return loss, metrics
