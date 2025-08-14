import os
import json
import time
import gc
import psutil
import sys
from typing import BinaryIO
import multiprocessing as mp
import regex as re
from datetime import datetime, timedelta

class ProgressTracker:
    """Enhanced progress tracking for BPE training."""
    
    def __init__(self, total_steps, phase_name, show_memory=True):
        self.total_steps = total_steps
        self.phase_name = phase_name
        self.show_memory = show_memory
        self.start_time = time.time()
        self.current_step = 0
        self.last_update = 0
        
    def update(self, step=None, extra_info=""):
        """Update progress and display status."""
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
            
        current_time = time.time()
        
        # Update every second or on significant steps
        if current_time - self.last_update >= 1.0 or self.current_step % 100 == 0 or self.current_step == self.total_steps:
            self.last_update = current_time
            self._display_progress(extra_info)
    
    def _display_progress(self, extra_info=""):
        """Display formatted progress information."""
        elapsed = time.time() - self.start_time
        progress = self.current_step / self.total_steps
        
        # Calculate ETA
        if progress > 0:
            total_estimated = elapsed / progress
            eta = total_estimated - elapsed
            eta_str = self._format_time(eta)
        else:
            eta_str = "calculating..."
        
        # Create progress bar
        bar_length = 30
        filled = int(bar_length * progress)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        # Memory info
        memory_str = ""
        if self.show_memory:
            memory = psutil.virtual_memory()
            memory_str = f" | RAM: {memory.percent:.1f}%"
        
        # Format output
        elapsed_str = self._format_time(elapsed)
        print(f"\r🔥 {self.phase_name}: [{bar}] {progress:.1%} "
              f"({self.current_step:,}/{self.total_steps:,}) | "
              f"⏱️  {elapsed_str} | ETA: {eta_str}{memory_str} {extra_info}", 
              end="", flush=True)
        
        if self.current_step == self.total_steps:
            print()  # New line when complete
    
    def _format_time(self, seconds):
        """Format seconds into readable time string."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds//60:.0f}m {seconds%60:.0f}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours:.0f}h {minutes:.0f}m"

def print_phase_header(phase_name, details=""):
    """Print a formatted phase header."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"\n{'='*60}")
    print(f"🚀 [{timestamp}] {phase_name}")
    if details:
        print(f"   {details}")
    print(f"{'='*60}")

def print_system_stats():
    """Print current system statistics."""
    # CPU info
    cpu_percent = psutil.cpu_percent(interval=0.1)
    cpu_count = psutil.cpu_count()
    
    # Memory info
    memory = psutil.virtual_memory()
    memory_gb = memory.total / (1024**3)
    used_gb = memory.used / (1024**3)
    available_gb = memory.available / (1024**3)
    
    print(f"💻 System Status:")
    print(f"   CPU: {cpu_percent:.1f}% ({cpu_count} cores)")
    print(f"   RAM: {used_gb:.1f}GB / {memory_gb:.1f}GB ({memory.percent:.1f}%)")
    print(f"   Available: {available_gb:.1f}GB")

def print_training_summary(phase_results):
    """Print a summary of training results."""
    print(f"\n{'='*60}")
    print(f"📊 TRAINING SUMMARY")
    print(f"{'='*60}")
    
    for phase, result in phase_results.items():
        duration = result.get('duration', 0)
        print(f"{phase}: {duration:.1f}s ({duration/60:.1f}m)")
    
    total_time = sum(r.get('duration', 0) for r in phase_results.values())
    print(f"{'─'*40}")
    print(f"Total Training Time: {total_time:.1f}s ({total_time/60:.1f}m)")

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
    """Full pair counting - used for initial count only."""
    byte_pair_dict = {}

    for word, count in token_counts.items():
        word_tokens = word_to_tokens[word]
        byte_pairs = get_byte_pair(word_tokens)
        for byte_pair in byte_pairs:
            byte_pair_dict[byte_pair] = byte_pair_dict.get(byte_pair, 0) + count

    return byte_pair_dict

def apply_merge_and_update_pairs(
    word_to_tokens: dict[str, list[bytes]], 
    token_counts: dict[str, int],
    pair_counts: dict[tuple[bytes, bytes], int],
    merged_pair: tuple[bytes, bytes]
) -> tuple[dict[tuple[bytes, bytes], int], int]:
    """Apply merge to all relevant words and update pair counts efficiently."""
    
    words_affected = 0
    words_to_update = []
    
    # First pass: find words that contain the pair and collect their old pairs
    for word in word_to_tokens:
        old_tokens = word_to_tokens[word]
        # Check if this word contains the pair to merge
        contains_pair = False
        for i in range(len(old_tokens) - 1):
            if old_tokens[i] == merged_pair[0] and old_tokens[i+1] == merged_pair[1]:
                contains_pair = True
                break
        
        if contains_pair:
            words_to_update.append(word)
            words_affected += 1
            
            # Remove old pairs from counts
            old_pairs = get_byte_pair(old_tokens)
            word_count = token_counts[word]
            for pair in old_pairs:
                if pair in pair_counts:
                    pair_counts[pair] -= word_count
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]
    
    # Second pass: apply merges and add new pairs
    for word in words_to_update:
        old_tokens = word_to_tokens[word]
        new_tokens = merge_tokens(old_tokens, merged_pair)
        word_to_tokens[word] = new_tokens
        
        # Add new pairs from the updated word
        new_pairs = get_byte_pair(new_tokens)
        word_count = token_counts[word]
        for pair in new_pairs:
            pair_counts[pair] = pair_counts.get(pair, 0) + word_count
    
    return pair_counts, words_affected

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
    
    print_phase_header("BPE MERGE TRAINING", 
                      f"{len(token_counts):,} unique words → {num_merges:,} merges")
    
    # Initialize: each word starts as character tokens
    print("🔧 Initializing word tokenizations...")
    init_start = time.time()
    word_to_tokens = {}
    init_progress = ProgressTracker(len(token_counts), "Initialization", show_memory=False)
    
    for i, word in enumerate(token_counts):
        word_to_tokens[word] = get_word_tokens(word)
        if i % 1000 == 0:
            init_progress.update(i)
    init_progress.update(len(token_counts))
    
    init_time = time.time() - init_start
    print(f"✅ Initialization complete in {init_time:.1f}s")
    
    merges = []
    merge_progress = ProgressTracker(num_merges, "BPE Merges")
    
    # Initial pair counting (only done once)
    print("🔧 Computing initial pair counts...")
    initial_pair_start = time.time()
    pair_counts = count_pairs_from_tokenizations(word_to_tokens, token_counts)
    initial_pair_time = time.time() - initial_pair_start
    print(f"✅ Initial pair counting complete in {initial_pair_time:.1f}s ({len(pair_counts):,} unique pairs)")
    
    for merge_num in range(num_merges):
        merge_start = time.time()
        
        if not pair_counts:
            print("\n⚠️  No more pairs to merge!")
            break
        
        # Find most frequent pair (with lexicographic tiebreaking)
        most_frequent_pair = max(pair_counts.items(), 
                               key=lambda x: (x[1], x[0]))  # Sort by count, then lexicographically
        
        pair_to_merge, frequency = most_frequent_pair
        
        # Record the merge
        merge_tuple = (pair_to_merge[0], pair_to_merge[1])
        merges.append(merge_tuple)
        
        # Remove the merged pair from consideration
        del pair_counts[pair_to_merge]
        
        # Apply merge and update pair counts efficiently
        pair_counts, words_affected = apply_merge_and_update_pairs(
            word_to_tokens, token_counts, pair_counts, pair_to_merge
        )
        
        # Update progress with detailed info
        merge_time = time.time() - merge_start
        extra_info = f"| freq: {frequency:,} | affected: {words_affected:,} | {merge_time:.2f}s"
        merge_progress.update(merge_num + 1, extra_info)
        
        # Detailed logging every 1000 merges
        if (merge_num + 1) % 1000 == 0:
            pair_str = f"{pair_to_merge[0]} + {pair_to_merge[1]}"
            print(f"\n   Merge #{merge_num + 1}: {pair_str} (freq: {frequency:,}, affected: {words_affected:,} words)")
            
        # Optional memory check and early termination
        if (merge_num + 1) % 5000 == 0:
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                print(f"\n⚠️  High memory usage ({memory.percent:.1f}%) - consider stopping or reducing chunk size")
            print(f"   Remaining pairs to consider: {len(pair_counts):,}")
            
        # Early termination if frequency gets very low (optional optimization)
        if frequency < 2 and merge_num > num_merges * 0.8:
            print(f"\n💡 Early termination: frequency dropped to {frequency} at merge {merge_num + 1}")
            print(f"   Continuing might not provide significant compression gains")
            break

    print(f"\n✅ BPE training complete! Performed {len(merges):,} merges.")
    
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
    
    training_start = time.time()
    phase_results = {}
    
    # Print header
    print_phase_header("BPE TOKENIZER TRAINING", 
                      f"File: {os.path.basename(input_path)} | Target vocab: {vocab_size:,}")
    print_system_stats()
    
    # Get file size without loading into memory
    file_size = os.path.getsize(input_path)
    file_size_gb = file_size / (1024*1024*1024)
    print(f"\n📁 Input file: {file_size_gb:.2f} GB ({file_size:,} bytes)")

    # Auto-calculate optimal chunk size if not provided
    if max_chunk_size_mb is None:
        max_chunk_size_mb, explanation = calculate_optimal_chunk_size(file_size)
        print(f"🧠 {explanation}")
    else:
        print(f"🔧 Using manual chunk size: {max_chunk_size_mb} MB")

    # Calculate optimal number of chunks based on file size and memory limit
    max_chunk_size_bytes = max_chunk_size_mb * 1024 * 1024
    min_chunks = max(1, int(file_size / max_chunk_size_bytes))
    num_processes = mp.cpu_count()
    
    # Use more chunks for larger files to keep memory usage reasonable
    desired_chunks = max(num_processes, min_chunks)
    
    print(f"⚙️  Processing: {num_processes} CPU cores, {desired_chunks} chunks (~{max_chunk_size_mb} MB each)")

    # Phase 1: Chunk boundary detection
    print_phase_header("PHASE 1: CHUNK BOUNDARY DETECTION")
    boundary_start = time.time()
    
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, desired_chunks, "<|endoftext|>".encode("utf-8"))
    
    boundary_time = time.time() - boundary_start
    phase_results["Boundary Detection"] = {"duration": boundary_time}
    print(f"✅ Found {len(boundaries)-1} chunks in {boundary_time:.1f}s")

    # Phase 2: Parallel text processing
    print_phase_header("PHASE 2: PARALLEL TEXT PROCESSING", 
                      f"{len(boundaries)-1} chunks across {num_processes} cores")
    
    chunk_args = []
    total_chunk_size = 0
    for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        chunk_size_mb = (end - start) / (1024*1024)
        total_chunk_size += chunk_size_mb
        chunk_args.append((input_path, start, end))
        if i < 3:  # Print first few chunk sizes
            print(f"   Chunk {i+1}: {chunk_size_mb:.1f} MB")
    
    print(f"   Average chunk size: {total_chunk_size/len(chunk_args):.1f} MB")

    processing_start = time.time()
    with mp.Pool(num_processes) as pool:
        results = pool.map(process_chunk_streaming, chunk_args)
    processing_time = time.time() - processing_start
    phase_results["Parallel Processing"] = {"duration": processing_time}

    # Phase 3: Result merging
    print_phase_header("PHASE 3: MERGING RESULTS")
    merge_start = time.time()
    
    total_documents = sum(r.get('num_documents', 0) for r in results)
    total_tokens = sum(r.get('total_tokens', 0) for r in results)
    
    # Merge token counts from all chunks
    print("🔄 Merging token counts from all chunks...")
    combined_token_counts = {}
    merge_progress = ProgressTracker(len(results), "Token Merge", show_memory=False)
    
    for i, result in enumerate(results):
        if 'token_counts' in result:
            for token, count in result['token_counts'].items():
                combined_token_counts[token] = combined_token_counts.get(token, 0) + count
        # Clear processed result to save memory
        results[i] = None
        merge_progress.update(i + 1)
        
    # Force garbage collection after processing
    del results
    gc.collect()
    
    merge_time = time.time() - merge_start
    phase_results["Result Merging"] = {"duration": merge_time}
    
    print(f"\n📊 Processing Results:")
    print(f"   Documents: {total_documents:,}")
    print(f"   Total tokens: {total_tokens:,}")
    print(f"   Unique tokens: {len(combined_token_counts):,}")
    print(f"   Tokens/second: {total_tokens/processing_time:,.0f}")
    
    # Show top tokens
    top_tokens = sorted(combined_token_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"   Top tokens: {[(token[:10], count) for token, count in top_tokens]}")
    
    # Phase 4: BPE training
    bpe_start = time.time()
    merges = bpe_trainer(combined_token_counts, vocab_size, special_tokens)
    bpe_time = time.time() - bpe_start
    phase_results["BPE Training"] = {"duration": bpe_time}
    
    # Phase 5: Vocabulary building
    print_phase_header("PHASE 5: VOCABULARY BUILDING")
    vocab_start = time.time()
    vocab = bpe_vocab_builder(merges, special_tokens)
    vocab_time = time.time() - vocab_start
    phase_results["Vocabulary Building"] = {"duration": vocab_time}
    
    print(f"✅ Built vocabulary with {len(vocab):,} tokens in {vocab_time:.1f}s")
    
    # Clean up memory
    del combined_token_counts
    gc.collect()

    # Final summary
    total_training_time = time.time() - training_start
    phase_results["TOTAL"] = {"duration": total_training_time}
    print_training_summary(phase_results)
    
    print(f"\n🎉 BPE training completed successfully!")
    print(f"   Final vocabulary size: {len(vocab):,}")
    print(f"   Total merges performed: {len(merges):,}")
    print(f"   Training time: {total_training_time/60:.1f} minutes")

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

    print("🚀 Starting BPE Tokenizer Training")
    print("=" * 50)
    print(f"📁 Input: {os.path.basename(input_path)}")
    print(f"🎯 Target vocabulary size: {vocab_size:,}")
    print(f"🏷️  Special tokens: {special_tokens}")
    print("=" * 50)
    
    try:
        # Auto-calculate optimal chunk size based on available memory
        # You can also manually override: max_chunk_size_mb=300
        vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
        
        # Save to JSON files
        print_phase_header("SAVING RESULTS")
        save_start = time.time()
        save_to_json(vocab, merges, input_path, vocab_size)
        save_time = time.time() - save_start
        print(f"✅ Results saved in {save_time:.1f}s")
        
        print(f"\n🎉 SUCCESS! BPE tokenizer training completed!")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        raise
    




