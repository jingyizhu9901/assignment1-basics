from multiprocessing import Pool
import os
import regex as re
from typing import BinaryIO
from collections import Counter, defaultdict

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

def process_chunk_worker(
        filename: str,
        special_tokens: list[bytes],
        start: int,
        end: int
) -> dict[tuple[bytes],int]:
    # Escape each decoded special token, join with | (OR) and encode
    escaped_tokens = [re.escape(tok.decode("utf-8", errors="ignore")) for tok in special_tokens]
    pattern = "|".join(escaped_tokens)
    regex = re.compile(pattern)
    result = {}

    with open(filename, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        
        # Split the file chunk on special tokens
        docs = re.split(regex, chunk)

        # Tokenize each doc and update counts
        for doc in docs:
            for match in re.finditer(PAT, doc):
                token = match.group()
                token_bytes = token.encode('utf-8')
                token_tuple = tuple(bytes([b]) for b in token_bytes)
                result[token_tuple] = result.get(token_tuple, 0) + 1
    return result

def pre_tokenization(
        special_tokens: list[str],
        num_workers: int,
        filename: str = None,
        text: str = None
) -> dict[tuple[bytes],int]:
    if not filename and not text:
        raise ValueError("Either filename or text must be provided")
    
    special_tokens_list_bytes = [t.encode("utf-8") for t in special_tokens]
    if filename:
        with open(filename, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_workers, special_tokens_list_bytes[0])
    
    args = [(filename, special_tokens_list_bytes, start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]

    with Pool(num_workers) as pool:
        results = pool.starmap(process_chunk_worker, args)
    
    merged = Counter()
    for d in results:
        merged.update(d)
    return dict(merged)

def pre_tokenize(text: str, special_tokens: list[str]) -> list[list[bytes]]:
    """
    Pre-tokenizes the input text by splitting into chunks by special tokens, then tokens by PAT regex.
    Returns a list of lists, where each inner list contains bytes representing the token.
    """
    if not special_tokens:
        result = []
        for match in re.finditer(PAT, text):
            token = match.group()
            token_bytes = token.encode('utf-8')
            token_list = [bytes([b]) for b in token_bytes]
            result.append(token_list)
        return result
    
    # Sort special tokens by length (descending) to prioritize longer matches
    sorted_special_tokens = sorted(special_tokens, key=len, reverse=True)
    pattern = "|".join(re.escape(token) for token in sorted_special_tokens)
    parts = re.split(f'({pattern})', text)
    
    result = []
    
    for part in parts:
        if part in special_tokens:
            # If part is a special token, encode it as bytes
            result.append([part.encode('utf-8')])
        elif part:  # Skip empty parts
            for match in re.finditer(PAT, part):
                token = match.group()
                token_bytes = token.encode('utf-8')
                token_list = [bytes([b]) for b in token_bytes]
                result.append(token_list)
    
    return result