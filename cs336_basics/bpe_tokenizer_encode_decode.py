import json
from typing import Iterable

from cs336_basics.pretokenization_example import pre_tokenize


class Tokenizer:
    """
    A class to handle tokenization and detokenization of text using Byte Pair Encoding (BPE).
    """

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] = None):
        """
        Initializes the Tokenizer with a vocabulary and merge rules.

        Args:
            vocab (dict): A dictionary mapping tokens to their indices.
            merges (list): A list of tuples representing merge operations.
            special_tokens (list[str], optional): A list of special tokens to be used in the tokenizer.
                Defaults to None, which means no special tokens are used.
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []
        self.token_to_id: dict[bytes, int] = {v: k for k, v in self.vocab.items()}
        
        # Create merge priority lookup for efficient BPE
        self.merge_priority = {pair: i for i, pair in enumerate(self.merges)}
    
    @classmethod
    def from_files(cls, vocab_path: str, merges_path: str, special_tokens: list[str] = None):
        """
        Creates a Tokenizer instance from vocabulary and merge files.

        Args:
            vocab_path (str): Path to the vocabulary file.
            merges_path (str): Path to the merges file.
            special_tokens (list[str], optional): A list of special tokens to be used in the tokenizer.
                Defaults to None.

        Returns:
            Tokenizer: An instance of the Tokenizer class.
        """
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
        
        with open(merges_path, 'r') as f:
            merges = [tuple(line.strip().split()) for line in f]
        
        return cls(vocab, merges, special_tokens)
    
    def encode(self, text: str) -> list[int]:
        """
        Encodes a string into a sequence of token indices.

        Args:
            text (str): The input text to be tokenized.

        Returns:
            list[int]: A list of token indices corresponding to the input text.
        """
        # 1. pre-tokenization
        pre_tokenization_list = pre_tokenize(text, self.special_tokens)

        # 2. Apply merges on each byte sequence
        token_ids = []
        for bytes_list in pre_tokenization_list:
            # Check if this is a special token (single bytes object)
            if len(bytes_list) == 1 and bytes_list[0] in [token.encode('utf-8') for token in self.special_tokens]:
                # This is a special token, add it directly without BPE processing
                special_token_bytes = bytes_list[0]
                if special_token_bytes in self.token_to_id:
                    token_ids.append(self.token_to_id[special_token_bytes])
                continue
            
            # Convert to list of bytes if not already
            if isinstance(bytes_list, (str, bytes)):
                if isinstance(bytes_list, str):
                    bytes_list = bytes_list.encode('utf-8')
                bytes_list = [bytes([b]) for b in bytes_list]
            
            # For regular tokens, iteratively apply merges until no more merges are possible
            while True:
                # Find the best merge (earliest in merge list) among all possible pairs
                best_merge = None
                best_pos = -1
                best_priority = len(self.merges)  # Higher than any real priority
                
                for i in range(len(bytes_list) - 1):
                    pair = (bytes_list[i], bytes_list[i + 1])
                    if pair in self.merge_priority:
                        priority = self.merge_priority[pair]
                        if priority < best_priority:
                            best_priority = priority
                            best_merge = pair
                            best_pos = i
                
                # If no merge found, break
                if best_merge is None:
                    break
                
                # Apply the best merge
                merged = bytes_list[best_pos] + bytes_list[best_pos + 1]
                bytes_list = bytes_list[:best_pos] + [merged] + bytes_list[best_pos + 2:]
            
            # 3. Convert bytes to token indices
            for b in bytes_list:
                if b in self.token_to_id:
                    token_ids.append(self.token_to_id[b])
                else:
                    # Handle unknown token - you might want to add proper UNK handling
                    # For now, skip unknown tokens or handle as per your vocab
                    pass
                    
        return token_ids
    
    def encode_iterable(self, texts: Iterable[str]) -> Iterable[int]:
        """
        Encodes an iterable of strings into a generator that yields individual token IDs.

        Args:
            texts (Iterable[str]): An iterable containing strings to be tokenized.

        Returns:
            Iterable[int]: A generator that lazily yields individual token IDs.
        """
        for text in texts:
            for token_id in self.encode(text):
                yield token_id
    
    def decode(self, token_ids: list[int]) -> str:
        """
        Decodes a sequence of token indices back into a string.

        Args:
            token_ids (list[int]): A list of token indices to be detokenized.

        Returns:
            str: The detokenized string.
        """
        text_bytes = b""
        for token_id in token_ids:
            if token_id in self.vocab:
                text_bytes += self.vocab[token_id]
            
        return text_bytes.decode('utf-8', errors='ignore')