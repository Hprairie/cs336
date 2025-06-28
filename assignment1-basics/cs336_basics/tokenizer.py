import pickle
import regex as re
from typing import Iterable, Iterator

from cs336_basics.utils.tokenizer import PAT

class Tokenizer:
    def __init__(
        self, 
        vocab: dict[int, bytes], 
        merges: list[tuple[bytes, bytes]], 
        special_tokens: list[str] | None = None
    ) -> None:
        self.vocab = vocab
        self.reverse_vocab = {v:k for k,v in self.vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens
        # Add special tokens to dictionary if necissary
        if special_tokens:
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)
            self.special_pattern = "(" + "|".join(re.escape(k) for k in self.special_tokens) + ")"
            vocab_size = len(self.vocab)
            for tok in special_tokens:
                tok = tok.encode("UTF-8")
                if tok not in self.reverse_vocab.keys():
                    self.vocab[vocab_size] = tok
                    self.reverse_vocab[tok] = vocab_size
                    vocab_size += 1
        else:
            self.special_tokens = None
            self.special_pattern = None 
    
    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str, 
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        with open(vocab_filepath, "rb") as handle:
            vocab = pickle.load(handle)
        with open(merges_filepath, "rb") as handle:
            merges = pickle.load(handle)
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
            i = 0
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
        pretokenized_text = special_chunks = re.split(self.special_pattern, text) if self.special_pattern else [text]
        # Merge pretokens and encode into input ids
        input_ids = []
        for chunk in pretokenized_text:
            # Check if special token
            if self.special_pattern is not None and chunk in self.special_tokens:
                input_ids.append(self.reverse_vocab[chunk.encode("UTF-8")])
                continue
            # Normal logic if not special token
            for pretoken in re.finditer(PAT, chunk):
                input_ids += [self.reverse_vocab[tok] for tok in self._merge_pretoken(pretoken.group())]
        return input_ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)
    
    def decode(self, ids: list[int]) -> str:
        return b"".join([self.vocab[id] for id in ids]).decode("UTF-8", errors="replace")
    
    def get_vocab_size(self) -> int:
        return len(self.vocab)


if __name__ == "__main__":
    tokenizer = Tokenizer.from_files(
       vocab_filepath="state_dicts/tinystories_v2_tokenizer_vocab.pkl", 
       merges_filepath="state_dicts/tinystories_v2_tokenizer_merges.pkl",
       special_tokens=["<|endoftext|>", "<|endoftext|><|endoftext|>"],
    )
    encodings = tokenizer.encode("Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>")
    print(encodings)
    text = tokenizer.decode(encodings)
    print(text)