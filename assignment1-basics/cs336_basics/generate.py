import torch
import itertools

from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.functional import softmax

class Generator:
    def __init__(self, model: TransformerLM, tokenizer: Tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer
    
    def _generate_single_token(self, input_ids: list[int], temperature: float, threshold: float) -> int:
        logits = self.model(torch.Tensor(input_ids).to(next(self.model.parameters()).device))
        probs = softmax(logits[-1, :], temperature=temperature)
        # Truncate to top p
        sorted, indexs = torch.sort(probs, dim=-1)
        cs_sorted = torch.cumsum(sorted, dim=-1)
        mask = cs_sorted < threshold
        probs[indexs[~mask]] = 0
        # Sample from distribution
        out_id = torch.multinomial(probs, num_samples=1)
        return int(out_id.item())
    
    def generate(self, input: str, max_generated_tokens: int | None = None, temperature: float = 0.9, threshold: float = 0.1) -> str:
        input_ids = self.tokenizer.encode(input)
        output_ids = []
        eos_id = self.tokenizer.encode("<|endoftext|>")
        assert len(eos_id) == 1, "This shouldn't happen"
        eos_id = eos_id[0] # Get the token id int
        for _ in range(max_generated_tokens) if max_generated_tokens is not None else itertools.count():
            new_token_id = self._generate_single_token(input_ids, temperature, threshold)
            output_ids.append(new_token_id)
            if new_token_id == eos_id:
                break
        return self.tokenizer.decode(output_ids)

