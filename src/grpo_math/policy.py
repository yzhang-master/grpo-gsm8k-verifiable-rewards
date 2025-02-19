import torch
from torch import nn

ALPHABET = list("0123456789-.# =") + ["<bos>", "<eos>"]


class Tokenizer:
    def __init__(self) -> None:
        self.tokens = {token: index for index, token in enumerate(ALPHABET)}
        self.bos = self.tokens["<bos>"]
        self.eos = self.tokens["<eos>"]

    def encode(self, text: str) -> list[int]:
        return [self.tokens[char] for char in text if char in self.tokens]

    def decode(self, ids: list[int]) -> str:
        characters = []
        for index in ids:
            if index == self.eos:
                break
            token = ALPHABET[index]
            if token != "<bos>":
                characters.append(token)
        return "".join(characters)


class TinyPolicy(nn.Module):
    """Recurrent token policy conditioned on a deterministic hashed prompt bag."""

    def __init__(self, vocab_size: int = len(ALPHABET), hidden_size: int = 64) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.prompt_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, hidden_size)
        )
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(128, hidden_size)
        self.cell = nn.GRUCell(2 * hidden_size, hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)

    def prompt_features(self, prompt: str) -> torch.Tensor:
        features = torch.zeros(self.hidden_size)
        for index, byte in enumerate(prompt.encode()):
            features[(byte + 31 * index) % self.hidden_size] += 1.0
        return features / max(1, len(prompt))

    def initial_state(self, prompt: str) -> torch.Tensor:
        return self.prompt_encoder(self.prompt_features(prompt))

    def step(
        self, state: torch.Tensor, previous: torch.Tensor, position: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.cat((self.token_embedding(previous), self.position_embedding.weight[position]))
        state = self.cell(inputs, state)
        return self.head(state), state

    def rollout(
        self, prompt: str, tokenizer: Tokenizer, max_length: int, generator: torch.Generator
    ) -> tuple[list[int], torch.Tensor]:
        state = self.initial_state(prompt)
        previous = torch.tensor(tokenizer.bos)
        ids, log_probs = [], []
        for position in range(max_length):
            logits, state = self.step(state, previous, position)
            probabilities = torch.softmax(logits, dim=-1)
            action = int(torch.multinomial(probabilities, 1, generator=generator))
            log_probs.append(torch.log_softmax(logits, dim=-1)[action])
            if action == tokenizer.eos:
                break
            ids.append(action)
            previous = torch.tensor(action)
        if len(log_probs) == len(ids):
            logits, _ = self.step(state, previous, len(ids))
            log_probs.append(torch.log_softmax(logits, dim=-1)[tokenizer.eos])
        return ids, torch.stack(log_probs)

    def evaluate(
        self, prompt: str, ids: list[int], tokenizer: Tokenizer
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = self.initial_state(prompt)
        previous = torch.tensor(tokenizer.bos)
        log_probs, entropies, all_log_probs = [], [], []
        for position, action in enumerate(ids + [tokenizer.eos]):
            logits, state = self.step(state, previous, position)
            token_log_probs = torch.log_softmax(logits, dim=-1)
            log_probs.append(token_log_probs[action])
            probabilities = token_log_probs.exp()
            entropies.append(-(probabilities * token_log_probs).sum())
            all_log_probs.append(token_log_probs)
            previous = torch.tensor(action)
        return torch.stack(log_probs), torch.stack(entropies), torch.stack(all_log_probs)
