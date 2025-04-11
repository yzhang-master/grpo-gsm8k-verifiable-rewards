# grpo-gsm8k-verifiable-rewards

> **Era:** 2025+ · Staff AI Engineer post-training. Technique from 2024, reimplemented in 2026 on PyTorch 2.x.

## Problem

Outcome-only math fine-tuning needs rewards that cannot be fooled by fluent but incorrect prose. Grouped Relative Policy Optimization (GRPO) removes a learned value critic by comparing several sampled answers to the same prompt. This repository makes that loop executable on CPU while keeping benchmark claims separate from the toy setting.

## Method

For each question, sample `G` token completions from a recurrent tiny policy. The verifier extracts the text after the final `####` when present (otherwise the final number), removes thousands separators and compares exact rational values. Correct answers receive 1.0, or 1.2 with explicit answer formatting; incorrect/unparseable answers receive 0.

For reward group `r_i`, the per-completion advantage is

```text
A_i = (r_i - mean(r)) / (population_std(r) + 1e-6)
```

A zero-variance group receives zero advantages rather than artificial learning signal. At every generated token, including end-of-sequence, the implementation computes the behavior-policy ratio and clipped objective:

```text
ratio = exp(new_logp - old_logp)
L_pg = -min(ratio*A, clamp(ratio, 0.8, 1.2)*A)
L = mean_tokens(L_pg) + beta * mean_tokens(k3_KL)
k3_KL = exp(ref_logp-new_logp) - (ref_logp-new_logp) - 1
```

The frozen reference is the initial policy. Metrics include exact categorical policy/reference KL, token entropy, completion length, reward, accuracy and objective. Mixed-reward unit tests verify nonzero gradients; identical-reward groups correctly produce no policy-gradient update.

## Results

Local verification: **15 tests passed**, including exact reward parsing, group normalization, EOS log-probability consistency, trust-region clipping, clipped GRPO gradients and real two-step training. A two-step, two-question smoke run completed with finite KL/entropy/length metrics; its random rollout rewards were zero, so no learning improvement is claimed.

| Evaluation | Model | Accuracy/reward | Status |
|---|---|---|---|
| Local 8 GSM8K-style arithmetic questions | Tiny recurrent CPU policy | emitted by training JSON | smoke only |
| GSM8K test | LLM | **pending** | no LLM run |
| MATH | LLM | **pending** | no LLM run |

Run the included proxy benchmark and retain its complete reward/KL/entropy/length curves:

```bash
python scripts/train.py --steps 20 --group-size 8 --max-length 16 \
  --output results/tiny-grpo.json
```

The same CLI accepts another JSON file through `--data`, but it remains the tiny policy. Full LLM/GSM8K/MATH integration and metrics are pending; the table does not relabel this proxy as an LLM evaluation.

## Reproduce

```bash
python -m pip install -e ".[dev]"
ruff check . && pytest -q
python scripts/train.py --steps 2 --group-size 3 --max-length 5
```

No dataset download or API key is used. The eight local examples are original GSM8K-style arithmetic word problems, not GSM8K test examples.

## Limitations

- The policy is a hashed-prompt GRU with a 17-token answer vocabulary, not a pretrained language model.
- Sparse random rollout rewards often produce zero-variance groups; the smoke run is a systems check, not evidence of mathematical learning.
- The verifier checks the supplied ground-truth number and does not validate free-form reasoning steps or execute arbitrary model-generated code.
- GSM8K and MATH accuracy, reward curves for an LLM, and distributed/GPU throughput are pending.

## References

- Z. Shao et al. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* 2024. arXiv:2402.03300 (GRPO)
- K. Cobbe et al. *Training Verifiers to Solve Math Word Problems.* 2021. arXiv:2110.14168 (GSM8K)
- J. Schulman et al. *Proximal Policy Optimization Algorithms.* 2017. arXiv:1707.06347

## License

MIT
