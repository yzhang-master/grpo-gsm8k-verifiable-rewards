import argparse
import json
from pathlib import Path

from grpo_math.policy import Tokenizer
from grpo_math.train import train_grpo


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny CPU GRPO training with arithmetic rewards")
    parser.add_argument("--data", type=Path, default=Path("data/examples.json"))
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    examples = json.loads(args.data.read_text())
    policy, history = train_grpo(
        examples,
        steps=args.steps,
        group_size=args.group_size,
        max_length=args.max_length,
        lr=args.lr,
        beta=args.beta,
        seed=args.seed,
    )
    tokenizer = Tokenizer()
    samples = []
    for index, example in enumerate(examples):
        ids, _ = policy.rollout(
            example["question"],
            tokenizer,
            args.max_length,
            __import__("torch").Generator().manual_seed(args.seed + 1000 + index),
        )
        samples.append({"question": example["question"], "completion": tokenizer.decode(ids)})
    result = {
        "scope": "tiny CPU policy on local GSM8K-style examples; not GSM8K benchmark accuracy",
        "examples": len(examples),
        "history": history,
        "samples": samples,
    }
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
