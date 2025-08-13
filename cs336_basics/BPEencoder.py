from collections.abc import Iterator, Iterable
import json
import os
import regex as re

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str]=None):
        self.vocab = vocab.copy()  # dict[int, bytes]
        self.merges = merges.copy()  # list[tuple[bytes, bytes]]
        self.special_tokens = special_tokens or []
        
        # Handle special tokens
        if special_tokens:
            self._add_special_tokens(special_tokens)
        
        # Create reverse mapping for encoding
        self.token_to_id = {token: id for id, token in self.vocab.items()}
    
    def _add_special_tokens(self, special_tokens):
        """Add special tokens to vocab if not already present"""
        next_id = max(self.vocab.keys()) + 1 if self.vocab else 0
        
        for token in special_tokens:
            token_bytes = token.encode('utf-8')
            # Check if token already exists in vocab values
            if token_bytes not in self.vocab.values():
                self.vocab[next_id] = token_bytes
                next_id += 1


    def get_word_tokens(self, word: str):
        """Convert a word into a list of character tokens (initial state for BPE)"""
        word_bytes = word.encode('utf-8')
        return [bytes([b]) for b in word_bytes]

    def merge_tokens(self, tokens: list[bytes], pair: tuple[bytes, bytes]) -> list[bytes]:
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

    def apply_bpe_merges(self, word: str, merges: list[tuple[bytes, bytes]], vocab_idx: dict[bytes, int]) -> list[int]:
        """Apply BPE merges to a word and return token IDs"""
        word_bytes_list = self.get_word_tokens(word)
        word_bytes_set = set(word_bytes_list)
        
        # Apply merges in order
        for merge_pair in merges:
            if merge_pair[0] in word_bytes_set and merge_pair[1] in word_bytes_set:
                word_bytes_list = self.merge_tokens(word_bytes_list, merge_pair)
                word_bytes_set = set(word_bytes_list)

            # word_bytes_list = self.merge_tokens(word_bytes_list, merge_pair)
        
        # Convert to token IDs
        word_indices = []
        for word_byte in word_bytes_list:        
            token_id = vocab_idx.get(word_byte, -1)
            if token_id == -1:
                # If token not found, this shouldn't happen with proper vocab
                raise ValueError(f"Token {word_byte} not found in vocabulary")
            word_indices.append(token_id)

        return word_indices



    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str]=None):
        """Load tokenizer from saved vocab and merges files"""
        import base64
        
        # Load vocab
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            vocab_json = json.load(f)
            vocab = {int(k): base64.b64decode(v.encode('ascii')) for k, v in vocab_json.items()}
        
        # Load merges  
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            merges_json = json.load(f)
            merges = [(base64.b64decode(m[0].encode('ascii')), base64.b64decode(m[1].encode('ascii'))) for m in merges_json]
        
        # Create and return tokenizer instance using __init__
        return cls(vocab, merges, special_tokens)
    
    
    def encode(self, text: str) -> list[int]:
        # Handle empty string case
        if not text:
            return []
            
        vocab_idx = {v: k for k, v in self.vocab.items()}
        tokens_ids = []
        
        # Handle special tokens if they exist
        if self.special_tokens:
            # Sort special tokens by length (longest first) to handle overlapping tokens
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            # Split text while keeping delimiters (special tokens)
            split_pattern = f"({'|'.join(re.escape(token) for token in sorted_special_tokens)})"
            parts = re.split(split_pattern, text)
            
            for part in parts:
                if part in self.special_tokens:
                    # Add special token
                    special_token_bytes = part.encode('utf-8')
                    if special_token_bytes in vocab_idx:
                        tokens_ids.append(vocab_idx[special_token_bytes])
                elif part:  # Non-empty regular text
                    tokens_ids.extend(self._encode_text_chunk(part, vocab_idx))
        else:
            # No special tokens, just encode the whole text
            tokens_ids.extend(self._encode_text_chunk(text, vocab_idx))

        return tokens_ids
    
    def _encode_text_chunk(self, text: str, vocab_idx: dict[bytes, int]) -> list[int]:
        """Encode a chunk of text without special tokens"""
        if not text:
            return []
            
        pretokenization_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        matches = list(re.finditer(pretokenization_pattern, text))
        
        chunk_tokens = []
        for match in matches:
            word = match.group()
            word_token_ids = self.apply_bpe_merges(word, self.merges, vocab_idx)
            chunk_tokens.extend(word_token_ids)
            
        return chunk_tokens
   


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is
        required for memory-efficient tokenization of large files that we cannot directly load into
        memory.
        """
        for text_line in iterable:
            # Don't strip or modify the text - process it exactly as is
            token_ids = self.encode(text_line)
            for token_id in token_ids:
                yield token_id

    def decode(self, ids: list[int]) -> str:
        # Concatenate all token bytes first, then decode as UTF-8
        all_bytes = b""
        for token_id in ids:
            token_bytes = self.vocab[token_id]
            all_bytes += token_bytes
        
        # Use 'replace' error handling to match tiktoken behavior for individual tokens
        # that may contain partial UTF-8 sequences
        return all_bytes.decode('utf-8', errors='replace')
        

## Usage
if __name__ == "__main__":

    special_tokens = ["<|endoftext|>"]
    vocab_size = 10000
    train_path = "/Users/yuhangfang/Documents/learning/LLMfromScratch/assignment1-basics-main/cs336_basics/data/TinyStoriesV2-GPT4-train.txt"
    # Extract filename from input path (without extension)
    filename = os.path.splitext(os.path.basename(train_path))[0]
    
    # Create filenames based on training data and vocab size
    vocab_filename = f"{filename}_vocab_{vocab_size}.json"
    merges_filename = f"{filename}_merges_{vocab_size}.json"
    
    vocab_filepath = f"/Users/yuhangfang/Documents/learning/LLMfromScratch/assignment1-basics-main/cs336_basics/data/{vocab_filename}"
    merges_filepath = f"/Users/yuhangfang/Documents/learning/LLMfromScratch/assignment1-basics-main/cs336_basics/data/{merges_filename}"

    tokener = Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)

    valid_path = "/Users/yuhangfang/Documents/learning/LLMfromScratch/assignment1-basics-main/cs336_basics/data/TinyStoriesV2-GPT4-valid.txt"
    with open(valid_path, 'r', encoding='utf-8') as f:
        valid_text = f.read()

    tokenized = tokener.encode(valid_text)
    print('tokenized', tokenized[:100])

    detokenized = tokener.decode(tokenized)
    # compare with the valid_text
    if detokenized == valid_text:
        print('detokenized == valid_text')
    else:
        print('detokenized != valid_text')
        # print the first 100 characters of the difference
        print(detokenized[:100], valid_text[:100])
    