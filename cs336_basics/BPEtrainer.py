import os
from typing import BinaryIO
import multiprocessing as mp
from multiprocessing import shared_memory
import regex as re

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


def process_chunk(args):
    shm_name, start, end = args

    try:
        # Connect to shared memory and get chunk data
        shm = shared_memory.SharedMemory(name=shm_name)
        chunk_data = bytes(shm.buf[start:end]).decode("utf-8", errors="ignore")
        # print(f"Processing chunk (bytes {start}-{end}):")
        # print(f"Length: {len(chunk_data)} characters")
        # print(f"Preview: {chunk_data[:100]}...")  # Show first 100 chars
        
        # Step 1:  Split chunk into documents (removes special tokens)
        special_tokens = ["<|endoftext|>"]  # Make it a list!
        split_pattern = "|".join(re.escape(token) for token in special_tokens)
        documents = re.split(split_pattern, chunk_data)

        # Step 2: Filter out empty documents
        documents = [doc.strip() for doc in documents if doc.strip()]
        
        # Step 3: Pre-tokenize each document separately
        all_tokens = []
        token_counts = {}

        pretokenization_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for doc in documents:
            # Pre-tokenize this document
            matches = list(re.finditer(pretokenization_pattern, doc))
            doc_tokens = [match.group() for match in matches]
            
            all_tokens.extend(doc_tokens)
            
            # Count tokens (for BPE training later)
            for token in doc_tokens:
                token_counts[token] = token_counts.get(token, 0) + 1
        
        shm.close()    # Close in every process that used it
        results = {
        'start_byte': start,
        'end_byte': end,
        'num_documents': len(documents),
        'total_tokens': len(all_tokens),
        'unique_tokens': len(token_counts),
        'token_counts': token_counts,
        'sample_tokens': all_tokens[:10]  # First 10 tokens for debugging
        }

        print(f"Processed chunk {start}-{end}: {len(documents)} docs, {len(all_tokens)} tokens")

        return results
    except Exception as e:
        print(f"Error processing chunk {start}-{end}: {e}")
        return {'error': str(e), 'start_byte': start, 'end_byte': end}
    
def get_word_tokens(word: str):
    """Convert a word into a list of character tokens (initial state for BPE)"""
    return list(word)

def get_byte_pair(word_tokens: list[str]):
    pairs = []
    for i in range(len(word_tokens) - 1):
        pairs.append((word_tokens[i], word_tokens[i + 1])) 
    return pairs

def count_pairs_from_tokenizations(word_to_tokens: dict[str, list[str]], token_counts: dict[str, int]) -> dict[tuple[str, str], int]:
    byte_pair_dict = {}

    for word, count in token_counts.items():
        word_tokens = word_to_tokens[word]
        byte_pairs = get_byte_pair(word_tokens)
        for byte_pair in byte_pairs:
            byte_pair_dict[byte_pair] = byte_pair_dict.get(byte_pair, 0) + count

    return byte_pair_dict

def merge_tokens(tokens: list[str], pair: tuple[str, str]) -> list[str]:
    new_tokens = []
    n = len(tokens)
    i = 0
    while i<n:
        if i<n-1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]:
            new_tokens.append(pair[0] + pair[1])
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    return new_tokens

def bpe_vocab_builder(merges: list[tuple[str, str]], special_tokens: list[str]) -> dict[int, str]:
    # Start with special tokens and byte characters
    token_id = -1
    vocab = {}
    for token in special_tokens:
        token_id += 1
        vocab[token_id] = token
    
    # Add all 256 byte characters
    for i in range(256):
        token_id += 1
        vocab[token_id] = chr(i)
        
    
    # Add merged tokens
    for merge in merges:
        merged_token = ''.join(merge)
        token_id += 1
        vocab[token_id] = merged_token
    
    return vocab

def bpe_trainer(token_counts: dict[str, int], vocab_size: int, special_tokens: list[str]) -> list[str]: 

    num_merges = vocab_size - len(special_tokens) - 256
    print(f"Starting BPE training with {len(token_counts)} unique words and {num_merges} merges")
    
    # Initialize: each word starts as character tokens
    word_to_tokens = {}
    for word in token_counts:
        word_to_tokens[word] = get_word_tokens(word)
    
    merges = []
    
    for merge_num in range(num_merges):
        if (merge_num + 1) % 100 == 0 or merge_num == 0:
            print(f"\n=== Merge {merge_num + 1}/{num_merges} ===")
        
        # Count all pairs across the current vocabulary
        pair_counts = count_pairs_from_tokenizations(word_to_tokens, token_counts)
        
        if not pair_counts:
            print("No more pairs to merge!")
            break
        
        # Find most frequent pair (with lexicographic tiebreaking)
        most_frequent_pair = max(pair_counts.items(), 
                               key=lambda x: (x[1], x[0]))  # Sort by count, then lexicographically
        
        pair_to_merge, frequency = most_frequent_pair
        if (merge_num + 1) % 100 == 0 or merge_num == 0:
            print(f"Most frequent pair: {pair_to_merge} (frequency: {frequency})")
        
        # Record the merge
        merge_tuple = (pair_to_merge[0], pair_to_merge[1])
        merges.append(merge_tuple)
        
        j = 0
        # Apply the merge to all words
        for word in word_to_tokens:
            word_to_tokens_set = set(word_to_tokens[word])
            if pair_to_merge[0] in word_to_tokens_set and pair_to_merge[1] in word_to_tokens_set:
                word_to_tokens[word] = merge_tokens(word_to_tokens[word], pair_to_merge)
                j += 1
                if j < 5 and ((merge_num + 1) % 100 == 0 or merge_num == 0):
                    print(f"  '{word}' -> {word_to_tokens[word]}")

    print(f"\nBPE training complete! Performed {len(merges)} merges.")
    # print(f"Final merges: {merges}")
    # merges: list[tuple[bytes, bytes]] A list of BPE merges produced from training. Each list item is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with <token2>. The merges should be ordered by order of creation.

    return merges


def train_bpe(input_path, vocab_size, special_tokens):

    # pretokenization in parallel
    num_processes = mp.cpu_count()

    print(f"Using {num_processes} processes")

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_processes, "<|endoftext|>".encode("utf-8"))
        
        print(f"Found {len(boundaries)-1} chunks with boundaries at: {boundaries}")

        file_data = f.read()
        file_size = len(file_data)
        
    print(f"Loaded file into memory: {file_size / (1024*1024):.1f} MB")
    
    # Create shared memory
    shm = shared_memory.SharedMemory(create=True, size=file_size)
    shm.buf[:len(file_data)] = file_data

    print(f"Shared memory created: {shm.name}, size: {shm.size}")

    # pretokenization in parallel
    chunk_args = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        chunk_args.append((shm.name, start, end))

    with mp.Pool(num_processes) as pool:
        results = pool.map(process_chunk, chunk_args)

    # Combine results from all chunks
    total_documents = sum(r.get('num_documents', 0) for r in results)
    total_tokens = sum(r.get('total_tokens', 0) for r in results)
    
    # Merge token counts from all chunks
    combined_token_counts = {}
    for result in results:
        if 'token_counts' in result:
            for token, count in result['token_counts'].items():
                combined_token_counts[token] = combined_token_counts.get(token, 0) + count
    
    print(f"\nProcessing complete!")
    print(f"Total documents: {total_documents}")
    print(f"Total tokens: {total_tokens}")
    print(f"Unique token types: {len(combined_token_counts)}")
    print(f"Most common tokens: {sorted(combined_token_counts.items(), key=lambda x: x[1], reverse=True)[:10]}")
    
    merges = bpe_trainer(combined_token_counts, vocab_size, special_tokens)
    vocab = bpe_vocab_builder(merges, special_tokens)
    # print(f"Vocab: {vocab}")

    # When done:
    shm.unlink()   # Unlink only once (usually in the main process)

    # return:
    # vocab: dict[int, bytes] The tokenizer vocabulary, a mapping from int (token ID in the vocabulary) to bytes (token bytes).
    # merges: list[tuple[bytes, bytes]] A list of BPE merges produced from training. Each list item is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with <token2>. The merges should be ordered by order of creation.


    return vocab, merges
        

if __name__ == "__main__":
    ## Usage
    input_path = "/Users/yuhangfang/Documents/learning/LLMfromScratch/assignment1-basics-main/cs336_basics/data/TinyStoriesV2-GPT4-valid.txt"
    vocab_size = 1000
    special_tokens = ["<|endoftext|>"]

    train_bpe(input_path, vocab_size, special_tokens)


