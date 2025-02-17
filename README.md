# grpo-gsm8k-verifiable-rewards

> **Era:** 2025+; GRPO reimplemented in 2026.

A tiny CPU policy for grouped relative policy optimization with verifiable arithmetic rewards. Full GSM8K/MATH language-model results are pending.

## Reproduce

```bash
pip install -e ".[dev]"
ruff check . && pytest -q
```

## Limitations

Local examples are GSM8K-style, not the GSM8K benchmark. No LLM benchmark has run.

## References

- DeepSeekMath / GRPO: https://arxiv.org/abs/2402.03300
- GSM8K: https://arxiv.org/abs/2110.14168

## License

MIT
