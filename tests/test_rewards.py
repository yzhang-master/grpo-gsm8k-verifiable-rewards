from fractions import Fraction

import pytest

from grpo_math.rewards import arithmetic_reward, numeric_value


@pytest.mark.parametrize(
    ("completion", "expected", "reward"),
    [
        ("The answer is 12", "12", 1.0),
        ("Work here #### 12", "12", 1.2),
        ("12,000 pennies", "12000", 1.0),
        ("Result -2.5", "-2.5", 1.0),
        ("The answer is 13", "12", 0.0),
        ("no number", "12", 0.0),
        ("#### 12 then 13", "12", 0.0),
    ],
)
def test_exact_verifiable_reward(completion, expected, reward):
    assert arithmetic_reward(completion, expected) == reward


def test_numeric_parser_is_exact():
    assert numeric_value("1,200.50") == Fraction(2401, 2)
    assert numeric_value("nothing") is None
