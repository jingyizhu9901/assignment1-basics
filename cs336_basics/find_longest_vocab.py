import sys
import ast

def find_longest_vocab(output_file):
    """Find the longest vocabulary entry from print output"""
    vocab = None
    
    with open(output_file, 'r') as f:
        content = f.read()
    
    # Find the vocabulary line
    lines = content.split('\n')
    for line in lines:
        if line.startswith('Vocabulary:'):
            vocab_str = line.replace('Vocabulary: ', '', 1)
            vocab = ast.literal_eval(vocab_str)
            break
    
    if not vocab:
        print("No vocabulary found in file")
        return None
    
    longest_entry = None
    max_length = 0
    
    for vocab_id, vocab_bytes in vocab.items():
        length = len(vocab_bytes)
        
        if length > max_length:
            max_length = length
            longest_entry = (vocab_id, vocab_bytes, length)
    
    return longest_entry

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py output.txt")
        sys.exit(1)
    
    output_file = sys.argv[1]
    longest = find_longest_vocab(output_file)
    
    if longest:
        vocab_id, vocab_bytes, length = longest
        print(f"Longest vocabulary entry:")
        print(f"ID: {vocab_id}")
        print(f"Content: {repr(vocab_bytes)}")
        print(f"Length: {length} bytes")
    else:
        print("No vocabulary entries found")