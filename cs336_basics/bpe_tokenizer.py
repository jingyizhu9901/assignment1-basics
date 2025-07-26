from collections import Counter
from cs336_basics.pretokenization_example import pre_tokenization

def convert_to_id_seq(data, vocab):
    byte_to_id = {v: k for k, v in vocab.items()}
    id_seq = []
    for tup, freq in data.items():
        ids = [byte_to_id[b] for b in tup] # convert each byte in tuple to id
        id_seq.append((ids, freq))
    return id_seq

def get_pair_counts(id_seq):
    pair_counts = Counter()
    for seq, freq in id_seq:
        for i in range(len(seq)-1):
            pair_counts[(seq[i], seq[i+1])] += freq
    return pair_counts

def merge_sequences(id_seq, pair, new_id):
    new_seq = []
    for seq, freq in id_seq:
        merged = []
        i = 0
        while i < len(seq):
            if i < len(seq) - 1 and (seq[i], seq[i+1]) == pair:
                merged.append(new_id)
                i += 2
            else:
                merged.append(seq[i])
                i += 1
        new_seq.append((merged, freq))
    return new_seq

def bpe_tokenizer (
        input_path: str,
        vocab_size: int,
        special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # 1. init vocab dict[int, bytes]
    vocab = {}
    for i, tok in enumerate(special_tokens):
        vocab[i] = tok.encode("utf-8")

    offset = len(special_tokens)
    for i in range(256):
        vocab[offset+i] = bytes([i])

    next_id = offset + 256
    num_merges = vocab_size - next_id
        
    # 2. convert dict[(bytes...), freq] → dict[sequence_of_ids, freq].
    pre_tokenization_dict = pre_tokenization(input_path, special_tokens, 3)
    id_seq = convert_to_id_seq(pre_tokenization_dict, vocab)

    # 3. merge
    merges = []
    for _ in range(num_merges):
        pair_counts = get_pair_counts(id_seq)
        if not pair_counts:
            break
        # Find the best pair: max frequency, tie-break by lexicographically greater pair
        max_freq = max(pair_counts.values())
        candidates = [pair for pair, freq in pair_counts.items() if freq == max_freq]
        best_pair = max(candidates, key=lambda x: (vocab[x[0]], vocab[x[1]]))
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))
        vocab[next_id] = vocab[best_pair[0]] + vocab[best_pair[1]]

        id_seq = merge_sequences(id_seq, best_pair, next_id)

        next_id += 1
    return vocab, merges