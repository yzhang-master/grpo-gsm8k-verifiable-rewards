import re
from fractions import Fraction

NUMBER = re.compile(r"-?(?:\d+(?:\.\d*)?|\.\d+)")


def numeric_value(text: str) -> Fraction | None:
    match = NUMBER.search(text.replace(",", "").strip())
    if match is None:
        return None
    try:
        return Fraction(match.group())
    except (ValueError, ZeroDivisionError):
        return None


def arithmetic_reward(completion: str, expected: str) -> float:
    """Exact numeric match, using the explicitly marked answer when present."""
    marked = completion.rsplit("####", 1)
    candidate = marked[1] if len(marked) == 2 else completion
    matches = NUMBER.findall(candidate.replace(",", ""))
    if not matches:
        return 0.0
    try:
        predicted = Fraction(matches[-1])
        target = numeric_value(expected)
    except (ValueError, ZeroDivisionError):
        return 0.0
    if target is None or predicted != target:
        return 0.0
    return 1.2 if len(marked) == 2 else 1.0
