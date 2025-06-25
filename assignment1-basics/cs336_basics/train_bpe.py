import os
import regex as re
import multiprocessing as mp
from typing import BinaryIO
from collections import Counter
from copy import deepcopy

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )
    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    chunk_size = file_size // desired_num_chunks
    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size
    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk
            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size
    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pre_tokenize_and_count(
        chunk: str,
        special_pattern: str, 
    ) -> dict[tuple[bytes], int]:
    subchunks = special_pattern.split(chunk) if special_pattern else [chunk]
    pretoken_count = Counter()
    for subchunk in subchunks:
        for text in re.finditer(PAT, subchunk):
            pretoken_count[tuple(bytes([b]) for b in text.group().encode("UTF-8"))] += 1
    return pretoken_count

def compute_merge(
    pretokenize_count: dict[tuple[bytes], int],
) -> tuple[dict[tuple[bytes], int], tuple[bytes, bytes]]:
    # Count each pair of tokens
    paircount = Counter()
    for pretoken,freq in pretokenize_count.items():
        for idx in range(len(pretoken) - 1):
            paircount[(pretoken[idx], pretoken[idx + 1])] += freq
    # Select the highest count wich is also lexigraphically largest
    max_key = max(paircount, key=lambda k: (paircount[k], k))
    # Update pretokens
    new_pretokenize_count = Counter()
    for pretoken, freq in pretokenize_count.items():
        i = 0
        merged_token = []
        while i < len(pretoken) - 1:
            if (pretoken[i], pretoken[i+1]) == max_key:
                merged_token.append(pretoken[i] + pretoken[i+1])
                i += 2
            else:
                merged_token.append(pretoken[i])
                i += 1 
        # Add last token if not merged
        if i == len(pretoken) - 1:
            merged_token.append(pretoken[i])
        new_pretokenize_count[tuple(merged_token)] = freq
    return new_pretokenize_count, max_key

def pretokenize(input_path: str, special_tokens: list[str]) -> Counter[tuple]:
    special_pattern = re.compile("|".join(re.escape(tok) for tok in special_tokens)) if special_tokens else None
    num_processes = mp.cpu_count()
    pool = mp.Pool(processes=num_processes)
    pretokenized_count_pool = []
    # Load in the text file
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        # The following is a serial implementation, but you can parallelize this 
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            pretokenized_count_pool.append(pool.apply_async(pre_tokenize_and_count, (chunk, special_pattern)))
    pool.close(); pool.join()
    return sum([pretokenized_count.get() for pretokenized_count in pretokenized_count_pool], Counter())


def train_bpe(
    input_path: str, 
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) ->  tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # Create vocab and merge databases
    merges: list[tuple[bytes, bytes]] = []  # index1, index2 => merged index
    initial_tokens = [x.encode("UTF-8") for x in special_tokens] + [bytes([x]) for x in range(256)]
    vocab: dict[int, bytes] = {i: x for i,x in enumerate(initial_tokens)}  # index -> bytes
    # Pretokenize and get paircount
    pretokenize_count = pretokenize(input_path=input_path, special_tokens=special_tokens)
    # Merge loop
    while len(vocab) < vocab_size:
        pretokenize_count, merge = compute_merge(pretokenize_count=pretokenize_count)
        merges.append(merge)
        vocab[len(vocab)] = merge[0] + merge[1]
    return vocab, merges