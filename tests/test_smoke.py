import torch


def test_smoke():
    policy = torch.nn.Linear(4, 3)
    assert policy(torch.ones(2, 4)).shape == (2, 3)
