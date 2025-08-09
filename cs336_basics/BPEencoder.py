from collections.abc import Iterator, Iterable
import json
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

    def apply_bpe_merges(self, word: str, merges: list[tuple[bytes, bytes]], vocab_idx: dict[bytes, int]) -> list[bytes]:
        word_bytes_list = self.get_word_tokens(word)
        word_bytes_set = set(word_bytes_list)
        word_indices = []
        
        for merge_pair in merges:
            # print(f"Merge pair: {merge_pair}")
            if merge_pair[0] in word_bytes_set and merge_pair[1] in word_bytes_set:
                word_bytes_list = self.merge_tokens(word_bytes_list, merge_pair)

        print(f"Word bytes list: {word_bytes_list}")
        
        for word_byte in word_bytes_list:        
            word_indices.append(vocab_idx.get(word_byte, -1))

        return word_indices



    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str]=None):
        """Load tokenizer from saved vocab and merges files"""
        # Load vocab
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            vocab_json = json.load(f)
            vocab = {int(k): v.encode('utf-8') for k, v in vocab_json.items()}
        
        # Load merges  
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            merges_json = json.load(f)
            merges = [(m[0].encode('utf-8'), m[1].encode('utf-8')) for m in merges_json]
        
        # Create and return tokenizer instance using __init__
        return cls(vocab, merges, special_tokens)
    
    
    def encode(self, text: str) -> list[int]:
        # Step 1:  Split chunk into documents (removes special tokens)
        vocab_idx = {v: k for k, v in self.vocab.items()}
        split_pattern = "|".join(re.escape(token) for token in self.special_tokens)
        documents = re.split(split_pattern, text)

        tokens_ids = []
        endoftext_id = vocab_idx["<|endoftext|>".encode('utf-8')]
        
        for doc in documents:
            
            pretokenization_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            matches = list(re.finditer(pretokenization_pattern, doc))
            
            doc_tokens = [match.group() for match in matches]

            for word in doc_tokens:
                word_token_ids = self.apply_bpe_merges(word, self.merges, vocab_idx)
                tokens_ids.extend(word_token_ids)
                
            if len(documents)>1:
                tokens_ids.append(endoftext_id)

        return tokens_ids
   


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is
        required for memory-efficient tokenization of large files that we cannot directly load into
        memory.
        """
        for text_line in iterable:
            # Use the existing encode method for each line
            if text_line.strip():  # Only process non-empty lines
                token_ids = self.encode(text_line.rstrip('\n\r'))  # Remove line endings but keep other whitespace
                for token_id in token_ids:
                    yield token_id

    def decode(self, ids: list[int]) -> str:
        decoded_string = ""
        for token_id in ids:
            token_bytes  = self.vocab[token_id]
            decoded_string += token_bytes.decode('utf-8')

        return decoded_string
        

## Usage
if __name__ == "__main__":

    special_tokens = ["<|endoftext|>"]
    vocab_filepath = "/Users/yuhangfang/Documents/learning/LLMfromScratch/assignment1-basics-main/cs336_basics/data/vocab.json"
    merges_filepath = "/Users/yuhangfang/Documents/learning/LLMfromScratch/assignment1-basics-main/cs336_basics/data/merges.json"

    tokener = Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)
    tokenized = tokener.encode("I am a boy.<|endoftext|> She is a girl.")
    print('tokenized', tokenized)

    detokenized = tokener.decode(tokenized)
    print('detokenized', detokenized)