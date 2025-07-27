from collections import Counter
from cs336_basics.pretokenization_example import pre_tokenization

def convert_to_id_seq(data, vocab):
    byte_to_id = {v: k for k, v in vocab.items()}
    id_seq = []
    for tup, freq in data.items():
        ids = [byte_to_id[b] for b in tup]
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
    changed_sequences = []  # Track which sequences changed
    
    for seq_idx, (seq, freq) in enumerate(id_seq):
        merged = []
        i = 0
        had_change = False
        
        while i < len(seq):
            if i < len(seq) - 1 and (seq[i], seq[i+1]) == pair:
                merged.append(new_id)
                i += 2
                had_change = True
            else:
                merged.append(seq[i])
                i += 1
        
        new_seq.append((merged, freq))
        if had_change:
            changed_sequences.append(seq_idx)
    
    return new_seq, changed_sequences

def update_pair_counts_incremental(old_id_seq, new_id_seq, changed_sequences, pair_counts, merged_pair):
    """Update pair counts by removing old contributions and adding new ones for changed sequences only"""
    
    # Remove old pair contributions from changed sequences
    for seq_idx in changed_sequences:
        old_seq, freq = old_id_seq[seq_idx]
        for i in range(len(old_seq) - 1):
            old_pair = (old_seq[i], old_seq[i+1])
            pair_counts[old_pair] -= freq
            if pair_counts[old_pair] <= 0:
                del pair_counts[old_pair]
    
    # Add new pair contributions from changed sequences  
    for seq_idx in changed_sequences:
        new_seq, freq = new_id_seq[seq_idx]
        for i in range(len(new_seq) - 1):
            new_pair = (new_seq[i], new_seq[i+1])
            pair_counts[new_pair] += freq

def bpe_tokenizer_training(
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
        vocab[offset + i] = bytes([i])
    next_id = offset + 256
    num_merges = vocab_size - next_id
        
    # 2. convert dict[(bytes...), freq] → dict[sequence_of_ids, freq].
    pre_tokenization_dict = pre_tokenization(input_path, special_tokens, 3)
    id_seq = convert_to_id_seq(pre_tokenization_dict, vocab)
    
    # 3. Get initial pair counts
    pair_counts = get_pair_counts(id_seq)
    
    # 4. merge
    merges = []
    for _ in range(num_merges):
        if not pair_counts:
            break
            
        # Find the best pair: max frequency, tie-break by lexicographically greater pair
        max_freq = max(pair_counts.values())
        candidates = [pair for pair, freq in pair_counts.items() if freq == max_freq]
        best_pair = max(candidates, key=lambda x: (vocab[x[0]], vocab[x[1]]))
        
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))
        vocab[next_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
        
        # Keep reference to old sequences for incremental update
        old_id_seq = id_seq
        
        # Perform merge and track which sequences changed
        id_seq, changed_sequences = merge_sequences(id_seq, best_pair, next_id)
        
        # Update pair counts incrementally
        update_pair_counts_incremental(old_id_seq, id_seq, changed_sequences, pair_counts, best_pair)
        
        next_id += 1
    
    return vocab, merges

# if __name__ == "__main__":
#     # Example usage
#     input_path = "./data/TinyStoriesV2-GPT4-train.txt"
#     vocab_size = 10000
#     special_tokens = ["<|endoftext|>"]

#     vocab, merges = bpe_tokenizer(input_path, vocab_size, special_tokens)
#     print("Vocabulary:", vocab)
#     print("Merges:", merges)