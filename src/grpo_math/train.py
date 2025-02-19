import copy

import torch

from .grpo import grpo_loss, sample_groups
from .policy import TinyPolicy


def train_grpo(
    examples: list[dict[str, str]],
    steps: int = 5,
    group_size: int = 4,
    max_length: int = 12,
    lr: float = 1e-3,
    beta: float = 0.02,
    seed: int = 7,
) -> tuple[TinyPolicy, list[dict[str, float]]]:
    torch.manual_seed(seed)
    policy = TinyPolicy()
    reference = copy.deepcopy(policy).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    history = []
    for step in range(steps):
        rollouts = sample_groups(policy, examples, group_size, max_length, seed + step)
        loss, metrics = grpo_loss(policy, reference, rollouts, beta=beta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        metrics["step"] = step
        metrics["loss"] = float(loss.detach())
        history.append(metrics)
    return policy, history
