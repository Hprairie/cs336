import pickle
import regex as re
from typing import Iterable, Iterator

class Tokenizer:
    def __init__(
        self, 
        vocab: dict[int, bytes], 
        merges: list[tuple[bytes, bytes]], 
        special_tokens: list[str] | None = None
    ) -> None:
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
    
    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str, 
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        vocab = pickle.load(vocab_filepath)
        merges = pickle.load(merges_filepath)
        return Tokenizer(vocab, merges, special_tokens)
    
    def _merge_pretoken(self, pretoken: str) -> list[int]:
        pretoken = list(bytes([b]) for b in pretoken.encode("UTF-8"))
        while True:
            merge_index = len(self.merges)
            for pair in zip(pretoken, pretoken[1:]):
                try:
                    tmp_merge_index = self.merges.index(pair)
                except ValueError:
                    tmp_merge_index = len(self.merges)
                if tmp_merge_index < merge_index:
                    merge_index = tmp_merge_index
            # If no new merges found then break out and return current tokens
            if merge_index == len(self.merges):
                break
            merge_key = self.merges[merge_index]
            new_pretoken = []
            while i < len(pretoken) - 1:
                if (pretoken[i], pretoken[i+1]) == merge_key:
                    new_pretoken.append(pretoken[i] + pretoken[i+1])
                    i += 2
                else:
                    new_pretoken.append(pretoken[i])
                    i += 1 
            # Add last token if not merged
            if i == len(pretoken) - 1:
                new_pretoken.append(pretoken[i])
            pretoken = new_pretoken
        return pretoken
    
    def encode(self, text: str) -> list[int]:
        # Pretokenize
        special_pattern = re.compile("|".join(re.escape(tok) for tok in self.special_tokens)) if self.special_tokens else None
        pretokenized_text = special_pattern.split(text) if special_pattern else [text]
        # Merge pretokens and encode into input ids
        input_ids = []
        for pretoken in pretokenized_text:
            input_ids += [self.vocab[tok] for tok in self._merge_pretoken(pretoken)]
        return input_ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        ...
    
    def decode(self, ids: list[int]) -> str:
        return sum([self.vocab[id] for id in ids], "").decode("UTF-8")