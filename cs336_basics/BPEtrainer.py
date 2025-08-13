import os
import json
import time
import gc
import psutil
from typing import BinaryIO
import multiprocessing as mp
import regex as re

def calculate_optimal_chunk_size(file_size_bytes: int, available_memory_gb: float = None) -> tuple[int, str]:
    """
    Calculate optimal chunk size based on file size and available memory.
    
    Args:
        file_size_bytes: Size of the input file in bytes
        available_memory_gb: Available RAM in GB (auto-detected if None)
    
    Returns:
        tuple: (optimal_chunk_size_mb, explanation)
    """
    if available_memory_gb is None:
        # Get available memory (leave some buffer for system)
        total_memory_gb = psutil.virtual_memory().total / (1024**3)
        available_memory_gb = total_memory_gb * 0.7  # Use 70% of total memory
    
    file_size_gb = file_size_bytes / (1024**3)
    num_cores = mp.cpu_count()
    
    # Memory considerations:
    # 1. Each chunk is loaded by a worker process
    # 2. Token counting creates dictionaries (roughly 2-3x text size in memory)
    # 3. BPE training needs to hold all unique tokens in memory
    # 4. Multiple processes run simultaneously
    
    # Estimate memory multiplier for tokenization overhead
    tokenization_overhead = 3.0  # Text -> tokens -> counts (rough estimate)
    
    # Memory needed per chunk = chunk_size * tokenization_overhead
    # Total memory for parallel processing = chunk_size * overhead * num_cores
    # Add buffer for BPE training phase
    safety_buffer = 0.5  # 50% buffer for BPE training and system overhead
    
    # Calculate max chunk size that fits in memory
    max_chunk_size_gb = (available_memory_gb * (1 - safety_buffer)) / (num_cores * tokenization_overhead)
    max_chunk_size_mb = max_chunk_size_gb * 1024
    
    # Practical considerations
    if file_size_gb < 1:  # Small files
        recommended_mb = min(100, max_chunk_size_mb)
        explanation = f"Small file ({file_size_gb:.1f}GB): using {recommended_mb:.0f}MB chunks"
    elif file_size_gb < 5:  # Medium files
        recommended_mb = min(500, max_chunk_size_mb)
        explanation = f"Medium file ({file_size_gb:.1f}GB): using {recommended_mb:.0f}MB chunks"
    else:  # Large files
        # For very large files, prioritize memory efficiency
        recommended_mb = min(300, max_chunk_size_mb)
        explanation = f"Large file ({file_size_gb:.1f}GB): using {recommended_mb:.0f}MB chunks for memory efficiency"
    
    # Ensure minimum viable chunk size
    if recommended_mb < 50:
        recommended_mb = 50
        explanation += f" (minimum 50MB enforced - consider upgrading RAM or reducing num_cores)"
    
    # Add memory usage details
    estimated_peak_memory = (recommended_mb * tokenization_overhead * num_cores) / 1024
    explanation += f"\nEstimated peak memory usage: ~{estimated_peak_memory:.1f}GB (Available: {available_memory_gb:.1f}GB)"
    
    return int(recommended_mb), explanation


def print_chunk_size_guide():
    """Print a guide for choosing chunk sizes manually."""
    print("""
=== CHUNK SIZE SELECTION GUIDE ===

The chunk size affects both memory usage and training performance:

📏 CHUNK SIZE FACTORS:
1. Available RAM - Each chunk uses ~3x its size in memory during processing
2. Number of CPU cores - All cores process chunks simultaneously  
3. File size - Larger files benefit from smaller chunks

💾 MEMORY CALCULATION:
   Memory per chunk = chunk_size_mb × 3 (tokenization overhead)
   Total memory = memory_per_chunk × number_of_cpu_cores
   
🎯 RECOMMENDED SIZES:
   • Small files (<1GB):   50-100MB chunks
   • Medium files (1-5GB): 200-500MB chunks  
   • Large files (>5GB):   100-300MB chunks
   • Very large (>20GB):   50-200MB chunks

⚠️  WARNING SIGNS:
   • Out of memory errors → Use smaller chunks
   • Very slow processing → May need larger chunks (if RAM allows)
   • Swap usage → Definitely use smaller chunks

🔧 MANUAL OVERRIDE:
   train_bpe(file, vocab_size, tokens, max_chunk_size_mb=YOUR_SIZE)

🤖 AUTO-CALCULATION (recommended):
   train_bpe(file, vocab_size, tokens)  # Automatically calculates optimal size
""")


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


def process_chunk_streaming(args):
    file_path, start, end = args

    try:
        with open(file_path, "rb") as f:
            f.seek(start)
            chunk_size = end - start
            chunk_data = f.read(chunk_size).decode("utf-8", errors="ignore")
        
        # Step 1:  Split chunk into documents (removes special tokens)
        special_tokens = ["<|endoftext|>"]  # Make it a list!
        split_pattern = "|".join(re.escape(token) for token in special_tokens)
        documents = re.split(split_pattern, chunk_data)

        # Step 2: Filter out empty documents but preserve internal whitespace
        documents = [doc for doc in documents if doc.strip()]
        
        # Step 3: Pre-tokenize each document separately
        token_counts = {}

        pretokenization_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        total_tokens = 0
        
        for doc in documents:
            # Pre-tokenize this document
            matches = list(re.finditer(pretokenization_pattern, doc))
            doc_tokens = [match.group() for match in matches]
            
            total_tokens += len(doc_tokens)
            
            # Count tokens (for BPE training later)
            for token in doc_tokens:
                token_counts[token] = token_counts.get(token, 0) + 1
        
        results = {
            'start_byte': start,
            'end_byte': end,
            'num_documents': len(documents),
            'total_tokens': total_tokens,
            'unique_tokens': len(token_counts),
            'token_counts': token_counts,
        }

        print(f"Processed chunk {start}-{end}: {len(documents)} docs, {total_tokens} tokens")

        return results
    except Exception as e:
        print(f"Error processing chunk {start}-{end}: {e}")
        return {'error': str(e), 'start_byte': start, 'end_byte': end}
    
def get_word_tokens(word: str):
    """Convert a word into a list of character tokens (initial state for BPE)"""
    word_bytes = word.encode('utf-8')
    return [bytes([b]) for b in word_bytes]

def get_byte_pair(word_tokens: list[bytes]):
    pairs = []
    for i in range(len(word_tokens) - 1):
        pairs.append((word_tokens[i], word_tokens[i + 1])) 
    return pairs

def count_pairs_from_tokenizations(word_to_tokens: dict[str, list[bytes]], token_counts: dict[str, int]) -> dict[tuple[bytes, bytes], int]:
    byte_pair_dict = {}

    for word, count in token_counts.items():
        word_tokens = word_to_tokens[word]
        byte_pairs = get_byte_pair(word_tokens)
        for byte_pair in byte_pairs:
            byte_pair_dict[byte_pair] = byte_pair_dict.get(byte_pair, 0) + count

    return byte_pair_dict

def merge_tokens(tokens: list[bytes], pair: tuple[bytes, bytes]) -> list[bytes]:
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

def bpe_vocab_builder(merges: list[tuple[bytes, bytes]], special_tokens: list[str]) -> dict[int, bytes]:
    # Start with special tokens and byte characters
    token_id = -1
    vocab = {}
    for token in special_tokens:
        token_id += 1
        vocab[token_id] = token.encode('utf-8')
    
    # Add all 256 byte characters
    for i in range(256):
        token_id += 1
        vocab[token_id] = bytes([i])
        
    
    # Add merged tokens
    for merge in merges:
        merged_token = b''.join(merge)
        token_id += 1
        vocab[token_id] = merged_token
    
    return vocab

def bpe_trainer(token_counts: dict[str, int], vocab_size: int, special_tokens: list[str]) -> list[tuple[bytes, bytes]]: 

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
        
        # Clear pair_counts to free memory
        del pair_counts
        
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


def train_bpe(input_path, vocab_size, special_tokens, max_chunk_size_mb=None):
    """
    Train BPE tokenizer with memory-efficient streaming processing.
    
    Args:
        input_path: Path to the input text file
        vocab_size: Target vocabulary size 
        special_tokens: List of special tokens
        max_chunk_size_mb: Maximum chunk size in MB. If None, will be auto-calculated based on available memory
    """
    
    # Get file size without loading into memory
    file_size = os.path.getsize(input_path)
    file_size_gb = file_size / (1024*1024*1024)
    print(f"File size: {file_size_gb:.2f} GB")

    # Auto-calculate optimal chunk size if not provided
    if max_chunk_size_mb is None:
        max_chunk_size_mb, explanation = calculate_optimal_chunk_size(file_size)
        print(f"\nAuto-calculated chunk size: {explanation}")
    else:
        print(f"Using manual chunk size: {max_chunk_size_mb} MB")

    # Calculate optimal number of chunks based on file size and memory limit
    max_chunk_size_bytes = max_chunk_size_mb * 1024 * 1024
    min_chunks = max(1, int(file_size / max_chunk_size_bytes))
    num_processes = mp.cpu_count()
    
    # Use more chunks for larger files to keep memory usage reasonable
    desired_chunks = max(num_processes, min_chunks)
    
    print(f"Using {num_processes} processes with {desired_chunks} chunks")
    print(f"Target chunk size: ~{max_chunk_size_mb} MB")

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, desired_chunks, "<|endoftext|>".encode("utf-8"))
        
        print(f"Found {len(boundaries)-1} chunks with boundaries at: {boundaries[:5]}...")

    # Stream processing without shared memory
    chunk_args = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        chunk_size_mb = (end - start) / (1024*1024)
        chunk_args.append((input_path, start, end))
        if len(chunk_args) <= 5:  # Print first few chunk sizes
            print(f"Chunk {len(chunk_args)}: {chunk_size_mb:.1f} MB")

    print("Starting streaming parallel processing...")
    with mp.Pool(num_processes) as pool:
        results = pool.map(process_chunk_streaming, chunk_args)

    # Combine results from all chunks
    total_documents = sum(r.get('num_documents', 0) for r in results)
    total_tokens = sum(r.get('total_tokens', 0) for r in results)
    
    # Merge token counts from all chunks
    print("Merging token counts from all chunks...")
    combined_token_counts = {}
    for i, result in enumerate(results):
        if 'token_counts' in result:
            for token, count in result['token_counts'].items():
                combined_token_counts[token] = combined_token_counts.get(token, 0) + count
        # Clear processed result to save memory
        results[i] = None
        
    # Force garbage collection after processing
    del results
    gc.collect()
    
    print(f"\nProcessing complete!")
    print(f"Total documents: {total_documents}")
    print(f"Total tokens: {total_tokens}")
    print(f"Unique token types: {len(combined_token_counts)}")
    print(f"Most common tokens: {sorted(combined_token_counts.items(), key=lambda x: x[1], reverse=True)[:10]}")
    
    print("Starting BPE training...")
    merges = bpe_trainer(combined_token_counts, vocab_size, special_tokens)
    vocab = bpe_vocab_builder(merges, special_tokens)
    
    # Clean up memory
    del combined_token_counts
    gc.collect()

    # return:
    # vocab: dict[int, bytes] The tokenizer vocabulary, a mapping from int (token ID in the vocabulary) to bytes (token bytes).
    # merges: list[tuple[bytes, bytes]] A list of BPE merges produced from training. Each list item is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with <token2>. The merges should be ordered by order of creation.

    return vocab, merges


def apply_bpe_merges(word: str, merges: list[tuple[bytes, bytes]], vocab_idx: dict[bytes, int]) -> list[bytes]:
    word_bytes_list = get_word_tokens(word)
    word_bytes_set = set(word_bytes_list)
    word_indices = []
    
    for merge_pair in merges:
        # print(f"Merge pair: {merge_pair}")
        if merge_pair[0] in word_bytes_set and merge_pair[1] in word_bytes_set:
            word_bytes_list = merge_tokens(word_bytes_list, merge_pair)

    print(f"Word bytes list: {word_bytes_list}")
    
    for word_byte in word_bytes_list:        
        word_indices.append(vocab_idx.get(word_byte, -1))

    return word_indices


def tokenizer_encoder(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str], text: str) -> list[int]:
    # Step 1:  Split chunk into documents (removes special tokens)
    vocab_idx = {v: k for k, v in vocab.items()}
    split_pattern = "|".join(re.escape(token) for token in special_tokens)
    documents = re.split(split_pattern, text)

    tokens_ids = []
    endoftext_id = vocab_idx["<|endoftext|>".encode('utf-8')]
    
    for doc in documents:
        
        pretokenization_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        matches = list(re.finditer(pretokenization_pattern, doc))
        
        doc_tokens = [match.group() for match in matches]

        for word in doc_tokens:
            word_token_ids = apply_bpe_merges(word, merges, vocab_idx)
            tokens_ids.extend(word_token_ids)
            
        tokens_ids.append(endoftext_id)

    return tokens_ids
   


def save_to_json(vocab, merges, input_path, vocab_size):
    """
    Save vocabulary and merges to JSON files named after the training data.
    """
    # Extract filename from input path (without extension)
    filename = os.path.splitext(os.path.basename(input_path))[0]
    
    # Get the directory where the input file is located
    input_dir = os.path.dirname(input_path)
    
    # Create filenames based on training data and vocab size
    vocab_filename = f"{filename}_vocab_{vocab_size}.json"
    merges_filename = f"{filename}_merges_{vocab_size}.json"
    
    # Create full paths in the same directory as input file
    vocab_filepath = os.path.join(input_dir, vocab_filename)
    merges_filepath = os.path.join(input_dir, merges_filename)
    
    # Convert vocab (bytes values) to JSON-serializable format
    # Use base64 encoding to preserve all byte values exactly
    import base64
    vocab_json = {str(k): base64.b64encode(v).decode('ascii') for k, v in vocab.items()}
    
    # Convert merges (bytes tuples) to JSON-serializable format  
    merges_json = [[base64.b64encode(merge[0]).decode('ascii'), 
                    base64.b64encode(merge[1]).decode('ascii')] for merge in merges]
    
    # Save vocab to JSON
    with open(vocab_filepath, 'w', encoding='utf-8') as f:
        json.dump(vocab_json, f, indent=2, ensure_ascii=False)
    print(f"Vocabulary saved to: {vocab_filepath}")
    
    # Save merges to JSON
    with open(merges_filepath, 'w', encoding='utf-8') as f:
        json.dump(merges_json, f, indent=2, ensure_ascii=False)
    print(f"Merges saved to: {merges_filepath}")


if __name__ == "__main__":
    ## train_bpe
    # input_path = "/Users/yuhangfang/Documents/learning/LLMfromScratch/assignment1-basics-main/cs336_basics/data/TinyStoriesV2-GPT4-train.txt"
    # vocab_size = 10000

    input_path = "/Users/yuhangfang/Documents/learning/LLMfromScratch/assignment1-basics-main/cs336_basics/data/owt_train.txt"
    vocab_size = 32000
    special_tokens = ["<|endoftext|>"]

    print("Training BPE tokenizer with intelligent memory optimization...")
    start_time = time.time()
    
    # Auto-calculate optimal chunk size based on available memory
    # You can also manually override: max_chunk_size_mb=300
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    
    end_time = time.time()
    print(f"Training time: {(end_time - start_time)/60:.2f} minutes")
    print(f"\nTraining complete!")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")
    
    # Save to JSON files
    save_to_json(vocab, merges, input_path, vocab_size)
    




